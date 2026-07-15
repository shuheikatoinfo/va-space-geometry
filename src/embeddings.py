"""Speaker-embedding extractors for all six models in the study.

Each extractor exposes the same interface:

    ext = build_extractor(name, device)
    vec = ext.embed(wav)          # wav: 1-D float32 torch.Tensor, 16 kHz mono
    vec                            # 1-D float32 numpy array (L2 length preserved)

Models span three axes (see docs/research_plan_updated.md):
  generation (x-vector -> ECAPA -> ReDimNet), paradigm (SV-trained vs SSL),
  and training language (EN / ZH / JP).

torchaudio>=2.9 removed `list_audio_backends`, which SpeechBrain calls at import;
we shim it before importing SpeechBrain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

SAMPLE_RATE = 16000

# --- SpeechBrain torchaudio compatibility shim --------------------------------
import torchaudio  # noqa: E402

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile", "ffmpeg"]

# --- huggingface_hub compatibility shim ---------------------------------------
# huggingface_hub>=1.0 removed the `use_auth_token` kwarg (renamed to `token`),
# but SpeechBrain 1.0.x still passes it. Translate/drop it transparently.
import huggingface_hub  # noqa: E402

if "use_auth_token" not in __import__("inspect").signature(huggingface_hub.hf_hub_download).parameters:
    import requests as _requests
    from huggingface_hub.errors import EntryNotFoundError as _EntryNotFoundError

    # huggingface_hub>=1.0 moved to httpx, so a missing-file 404 is no longer a
    # requests.HTTPError (which SpeechBrain's fetch() catches) -- but transformers
    # still relies on catching EntryNotFoundError for optional files. Raise a type
    # that is BOTH so each library's existing handler works, and the "404 Client
    # Error" text SpeechBrain greps for is preserved.
    class _CompatEntryNotFound(_requests.exceptions.HTTPError, _EntryNotFoundError):
        pass

    _orig_hf_hub_download = huggingface_hub.hf_hub_download

    def _compat_hf_hub_download(*args, **kwargs):
        # Drop the removed `use_auth_token` kwarg (renamed to `token`).
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", tok if not isinstance(tok, bool) else None)
        try:
            return _orig_hf_hub_download(*args, **kwargs)
        except _EntryNotFoundError as exc:
            new = _CompatEntryNotFound(f"404 Client Error: {exc}")
            new.response = getattr(exc, "response", None)
            raise new from exc

    huggingface_hub.hf_hub_download = _compat_hf_hub_download


@dataclass
class ModelSpec:
    name: str
    paradigm: str       # "sv" or "ssl"
    train_lang: str     # "en" / "zh" / "ja"
    generation: str     # human-readable generation label


MODEL_SPECS = {
    "xvector": ModelSpec("xvector", "sv", "en", "x-vector (TDNN, 2018)"),
    "ecapa": ModelSpec("ecapa", "sv", "en", "ECAPA-TDNN (2020)"),
    "wavlm": ModelSpec("wavlm", "ssl", "en", "WavLM-base-plus-sv (2021)"),
    "campp": ModelSpec("campp", "sv", "zh", "CAM++ (zh-cn 200k-common, 2023)"),
    "redimnet": ModelSpec("redimnet", "sv", "en", "ReDimNet (2024)"),
    "jhubert": ModelSpec("jhubert", "ssl", "ja", "Japanese HuBERT (layer-mean)"),
    "jxvector": ModelSpec("jxvector", "sv", "ja", "x-vector JTubeSpeech (JA-trained)"),
    "animeva": ModelSpec("animeva", "sv", "ja", "ECAPA anime-VA (JA voice actors)"),
}


class BaseExtractor:
    spec: ModelSpec

    def embed(self, wav: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


# --- SpeechBrain: x-vector and ECAPA-TDNN -------------------------------------
class SpeechBrainExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, hub_source: str, device: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self.spec = spec
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=hub_source,
            savedir=os.path.join("pretrained_models", hub_source.split("/")[-1]),
            run_opts={"device": device},
        )

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        wav = wav.to(self.device).unsqueeze(0)  # (1, T)
        emb = self.model.encode_batch(wav)      # (1, 1, D)
        return emb.squeeze().detach().cpu().float().numpy()


# --- HuggingFace WavLM speaker-verification model -----------------------------
class WavLMExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: str):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.spec = spec
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        inputs = self.fe(
            wav.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.embeddings.squeeze().detach().cpu().float().numpy()


# --- Japanese HuBERT: SSL layer-mean embedding (WavLM-SSL-style) ---------------
class JHubertExtractor(BaseExtractor):
    """Mean-pooled mean-over-layers hidden states as a speaker embedding.

    reazon-research/japanese-hubert-base-k2 is not fine-tuned for speaker verification, so we
    follow the common SSL recipe: average all transformer layer outputs, then
    mean-pool over time. This is the JP-SSL contrast point, treated as exploratory.
    """

    def __init__(self, spec: ModelSpec, device: str):
        from transformers import AutoFeatureExtractor, AutoModel

        self.spec = spec
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained("reazon-research/japanese-hubert-base-k2")
        self.model = AutoModel.from_pretrained("reazon-research/japanese-hubert-base-k2").to(device).eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        inputs = self.fe(wav.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)   # (L+1, 1, T, H)
        emb = hs.mean(dim=0).mean(dim=1).squeeze()   # mean over layers, then time
        return emb.detach().cpu().float().numpy()


# --- CAM++ via 3D-Speaker / modelscope ----------------------------------------
class CamPPExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: str):
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks

        self.spec = spec
        self.device = device
        self.pipe = ms_pipeline(
            task=Tasks.speaker_verification,
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=device,
        )

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        # The pipeline accepts a float32 numpy array at 16 kHz.
        emb = self.pipe([wav.cpu().numpy()], output_emb=True)["embs"][0]
        return np.asarray(emb, dtype=np.float32).squeeze()


# --- ReDimNet via torch.hub ---------------------------------------------------
class ReDimNetExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self.model = torch.hub.load(
            "IDRnD/ReDimNet", "ReDimNet", model_name="b2", train_type="ptn",
            dataset="vox2", trust_repo=True,
        ).to(device).eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        emb = self.model(wav.to(self.device).unsqueeze(0))  # (1, D)
        return emb.squeeze().detach().cpu().float().numpy()


# --- Japanese-trained speaker-verification models (language-confound controls) -
class JXVectorExtractor(BaseExtractor):
    """sarulab-speech/xvector_jtubespeech -- x-vector trained on JTubeSpeech (JA).

    Same TDNN x-vector architecture as the English VoxCeleb x-vector, so EN vs JA
    is a clean same-architecture language contrast. Uses a 24-dim Kaldi MFCC front
    end at 16 kHz.
    """

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self.model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector",
                                    trust_repo=True).to(device).eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        from torchaudio.compliance import kaldi
        mfcc = kaldi.mfcc(wav.unsqueeze(0).cpu(), num_ceps=24, num_mel_bins=24)
        mfcc = mfcc.unsqueeze(0).to(self.device)
        emb = self.model.vectorize(mfcc)
        return emb.squeeze().detach().cpu().float().numpy()


class AnimeVAExtractor(BaseExtractor):
    """litagin/anime_speaker_embedding (VA variant) -- ECAPA (GroupNorm) trained to
    separate ~989 Japanese voice actors. Domain-matched JA speaker embedding.
    """

    def __init__(self, spec: ModelSpec, device: str):
        import torch.nn.functional as F

        from anime_speaker_embedding import AnimeSpeakerEmbedding

        self.spec = spec
        self.device = device
        self._F = F
        self.model = AnimeSpeakerEmbedding(variant="va", device=device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> np.ndarray:
        out = self.model(wav.unsqueeze(0).to(self.device))  # (1, 1, 192)
        emb = out.squeeze(0).squeeze(0)
        emb = self._F.normalize(emb, dim=0)
        return emb.detach().cpu().float().numpy()


def build_extractor(name: str, device: str = "cuda") -> BaseExtractor:
    spec = MODEL_SPECS[name]
    if name == "xvector":
        return SpeechBrainExtractor(spec, "speechbrain/spkrec-xvect-voxceleb", device)
    if name == "ecapa":
        return SpeechBrainExtractor(spec, "speechbrain/spkrec-ecapa-voxceleb", device)
    if name == "wavlm":
        return WavLMExtractor(spec, device)
    if name == "jhubert":
        return JHubertExtractor(spec, device)
    if name == "campp":
        return CamPPExtractor(spec, device)
    if name == "redimnet":
        return ReDimNetExtractor(spec, device)
    if name == "jxvector":
        return JXVectorExtractor(spec, device)
    if name == "animeva":
        return AnimeVAExtractor(spec, device)
    raise ValueError(f"Unknown model: {name}")
