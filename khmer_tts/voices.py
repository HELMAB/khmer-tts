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


def find_pauses(source: Path, noise_db: float = -30, min_pause: float = 0.18) -> list[tuple[float, float]]:
    """Return (start, end) seconds of silent gaps, i.e. good places to cut a reference clip."""
    import re

    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(source), "-af", f"silencedetect=noise={noise_db}dB:d={min_pause}",
         "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    return list(zip(starts, ends))


def suggest_clips(source: Path, lo: float = 4.0, hi: float = 9.0) -> list[tuple[float, float]]:
    """Speech stretches between pauses (or from the start) that are lo-hi seconds long: (start, duration)."""
    pauses = find_pauses(source)
    edges = [0.0] + [end for _, end in pauses]
    stops = [start for start, _ in pauses] + [probe_duration(source)]
    return [(a, round(b - a, 2)) for a, b in zip(edges, stops) if lo <= b - a <= hi]


# Gentle on purpose: the model clones processing artefacts along with the voice.
# No pitch/EQ/compression changes, so the speaker's timbre is left alone.
CLEAN_HIGHPASS_HZ = 70           # rumble, handling noise, mains hum
CLEAN_DENOISE = "afftdn=nr=10:nf=-50:tn=1"  # 10 dB reduction, adaptive noise tracking
CLEAN_TARGET_LUFS = -20.0
CLEAN_TRUE_PEAK = -2.0


def clean_filters(source: Path, start: float, duration: float) -> list[str]:
    """ffmpeg filters for --clean: high-pass, light denoise, then 2-pass linear loudness normalisation."""
    import json

    pre = [f"highpass=f={CLEAN_HIGHPASS_HZ}", CLEAN_DENOISE]
    measure = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", f"{start}", "-t", f"{duration}", "-i", str(source), "-ac", "1",
         "-af", ",".join(pre + [f"loudnorm=I={CLEAN_TARGET_LUFS}:TP={CLEAN_TRUE_PEAK}:print_format=json"]),
         "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    m = json.loads(measure[measure.rindex("{"):measure.rindex("}") + 1])
    loudnorm = (
        f"loudnorm=I={CLEAN_TARGET_LUFS}:TP={CLEAN_TRUE_PEAK}:linear=true"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
        f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
    )
    return pre + [loudnorm]


def add_voice(
    name: str,
    source: Path,
    text: str,
    start: float = 0.0,
    duration: float = 6.0,
    transcript: Path | None = None,
    overwrite: bool = False,
    clean: bool = False,
) -> Voice:
    """Cut a reference clip from `source` and register it as voices/<name>/ (optionally cleaned)."""
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise SystemExit(f"--duration should be {MIN_SECONDS:g}-{MAX_SECONDS:g} s (the model works best with 4-9 s)")
    folder = VOICES_DIR / name
    if folder.exists() and not overwrite:
        raise SystemExit(f"Voice '{name}' already exists at {folder}; pass --overwrite to replace it")
    folder.mkdir(parents=True, exist_ok=True)

    fade = min(0.15, duration / 10)
    filters = [f"afade=t=out:st={duration - fade}:d={fade}"]
    if clean:
        filters = clean_filters(source, start, duration) + filters
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{duration}", "-i", str(source),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-af", ",".join(filters), str(folder / REFERENCE_WAV)],
        check=True,
    )
    (folder / REFERENCE_TXT).write_text(text.replace("​", "").strip() + "\n", encoding="utf-8")

    kept_source = folder / f"source{source.suffix.lower()}"
    if source.resolve() != kept_source.resolve():
        shutil.copy2(source, kept_source)
    if transcript is not None:
        shutil.copy2(transcript, folder / "source.txt")
    return Voice.load(name)
