"""Voice library: each voice is a folder under voices/ holding a short reference clip.

    voices/<name>/
        reference.wav   4-9 s mono clip of the speaker (the cloning prompt)
        reference.txt   exact words spoken in reference.wav
        source.*        original recording (optional, kept for re-cutting)
        source.txt      transcript of the original recording (optional)
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"

REFERENCE_WAV = "reference.wav"
REFERENCE_TXT = "reference.txt"
SAMPLE_RATE = 44100
MIN_SECONDS, MAX_SECONDS = 3.0, 10.0


@dataclass(frozen=True)
class Voice:
    name: str
    wav: Path
    text: str

    @classmethod
    def load(cls, name: str) -> "Voice":
        folder = VOICES_DIR / name
        wav, txt = folder / REFERENCE_WAV, folder / REFERENCE_TXT
        if not wav.is_file() or not txt.is_file():
            available = ", ".join(list_voices()) or "none"
            raise SystemExit(f"Unknown voice '{name}' (expected {wav} and {txt}). Available: {available}")
        return cls(name=name, wav=wav, text=txt.read_text(encoding="utf-8").strip())

    @property
    def duration(self) -> float:
        return probe_duration(self.wav)


def list_voices() -> list[str]:
    if not VOICES_DIR.is_dir():
        return []
    return sorted(p.name for p in VOICES_DIR.iterdir() if (p / REFERENCE_WAV).is_file())


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def add_voice(
    name: str,
    source: Path,
    text: str,
    start: float = 0.0,
    duration: float = 6.0,
    transcript: Path | None = None,
    overwrite: bool = False,
) -> Voice:
    """Cut a reference clip from `source` and register it as voices/<name>/."""
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise SystemExit(f"--duration should be {MIN_SECONDS:g}-{MAX_SECONDS:g} s (the model works best with 4-9 s)")
    folder = VOICES_DIR / name
    if folder.exists() and not overwrite:
        raise SystemExit(f"Voice '{name}' already exists at {folder}; pass --overwrite to replace it")
    folder.mkdir(parents=True, exist_ok=True)

    fade = min(0.15, duration / 10)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{duration}", "-i", str(source),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-af", f"afade=t=out:st={duration - fade}:d={fade}",
         str(folder / REFERENCE_WAV)],
        check=True,
    )
    (folder / REFERENCE_TXT).write_text(text.replace("​", "").strip() + "\n", encoding="utf-8")

    kept_source = folder / f"source{source.suffix.lower()}"
    if source.resolve() != kept_source.resolve():
        shutil.copy2(source, kept_source)
    if transcript is not None:
        shutil.copy2(transcript, folder / "source.txt")
    return Voice.load(name)
