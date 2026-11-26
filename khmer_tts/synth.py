"""Speech synthesis backends.

VoxPocket (Khmer fine-tune of VoxCPM 1.5) clones a voice from a short reference clip.
The plain VITS model (khmerttsopensource/khmer-tts) is kept as a fast, single-voice baseline.
Heavy imports happen inside the classes so the CLI starts quickly for cheap commands.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from .lexicon import speakable
from .voices import Voice

VOXPOCKET_REPO = "muyleanging/VoxPocket-TTS-Khmer"
VITS_REPO = "khmerttsopensource/khmer-tts"


def _patch_torchaudio_load() -> None:
    """torchaudio.load goes through torchcodec, which fails here (missing libnppicc.so.12); use soundfile."""
    import torchaudio

    def _sf_load(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _sf_load


@dataclass
class Result:
    audio: np.ndarray
    sample_rate: int
    seconds_taken: float

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class VoxPocket:
    def __init__(self, cfg: float = 2.0, steps: int = 10):
        _patch_torchaudio_load()
        from huggingface_hub import snapshot_download
        from voxcpm.core import VoxCPM

        # Skip the repo's extra training checkpoints/ (~10 GB) and images.
        path = snapshot_download(VOXPOCKET_REPO, ignore_patterns=["checkpoints/*", "images/*"])
        self.model = VoxCPM.from_pretrained(hf_model_id=path, load_denoiser=False)
        self.sample_rate = self.model.tts_model.sample_rate
        self.cfg, self.steps = cfg, steps

    def generate(self, text: str, voice: Voice, seed: int = 0) -> Result:
        torch.manual_seed(seed)
        t0 = time.time()
        audio = self.model.generate(
            text=speakable(text),  # English words → Khmer phonetic spelling (lexicon.txt)
            prompt_wav_path=str(voice.wav),
            prompt_text=voice.text,
            cfg_value=self.cfg,
            inference_timesteps=self.steps,
        )
        return Result(np.asarray(audio, dtype=np.float32), self.sample_rate, time.time() - t0)


class Vits:
    def __init__(self):
        from transformers import AutoTokenizer, VitsModel

        self.tokenizer = AutoTokenizer.from_pretrained(VITS_REPO)
        self.model = VitsModel.from_pretrained(VITS_REPO).eval()
        self.sample_rate = self.model.config.sampling_rate

    def generate(self, text: str, seed: int = 0) -> Result:
        torch.manual_seed(seed)  # VITS is stochastic
        inputs = self.tokenizer(text, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            audio = self.model(**inputs).waveform.squeeze().cpu().numpy()
        return Result(audio.astype(np.float32), self.sample_rate, time.time() - t0)


def save(result: Result, path) -> None:
    sf.write(str(path), result.audio, result.sample_rate)
