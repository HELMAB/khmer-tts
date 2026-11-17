"""Read a prepared article (one short chunk per line) and join the lines into one file.

Prepare the text first: numbers written as Khmer words, ៗ written out, lines kept short
(VoxPocket reads long paragraphs badly). Lines ending in ។ ! ? get a sentence pause, others a short one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .synth import Result, VoxPocket
from .voices import Voice

# Normal reading is ~0.06 s per character; much faster usually means the model cut the line short.
MIN_SEC_PER_CHAR = 0.045
SENTENCE_ENDINGS = ("។", "!", "?", "៕")


@dataclass
class ArticleReport:
    output: Path
    duration: float
    seconds_taken: float
    flagged_lines: list[int] = field(default_factory=list)


def read_lines(text_file: Path) -> list[str]:
    return [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def looks_complete(result: Result, line: str) -> bool:
    return result.duration >= MIN_SEC_PER_CHAR * len(line)


def read_article(
    tts: VoxPocket,
    voice: Voice,
    text_file: Path,
    out_dir: Path,
    tries: int = 3,
    short_pause: float = 0.25,
    long_pause: float = 0.6,
) -> ArticleReport:
    """Write out_dir/article.wav plus out_dir/parts/NN.wav (one per line, for re-doing single lines)."""
    lines = read_lines(text_file)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    sr = tts.sample_rate

    pieces, flagged, taken = [], [], 0.0
    for i, line in enumerate(lines, 1):
        best: Result | None = None
        for seed in range(tries):
            result = tts.generate(line, voice, seed=seed)
            taken += result.seconds_taken
            if best is None or result.duration > best.duration:
                best = result
            if looks_complete(result, line):
                break
        ok = looks_complete(best, line)
        if not ok:
            flagged.append(i)
        sf.write(parts_dir / f"{i:02d}.wav", best.audio, sr)
        note = "" if ok else "  (maybe cut short)"
        print(f"[{i:2d}/{len(lines)}] {best.duration:5.2f}s seed={seed}{note}  {line[:40]}…", flush=True)

        pause = long_pause if line.endswith(SENTENCE_ENDINGS) else short_pause
        pieces += [best.audio, np.zeros(int(pause * sr), dtype=np.float32)]

    output = out_dir / "article.wav"
    audio = np.concatenate(pieces)
    sf.write(output, audio, sr)
    return ArticleReport(output, len(audio) / sr, taken, flagged)
