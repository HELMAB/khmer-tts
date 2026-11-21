"""Read a prepared article (one short chunk per line) and join the lines into one file.

Prepare the text first: numbers written as Khmer words, ៗ written out, lines kept short
(VoxPocket reads long paragraphs badly). Lines ending in ។ ! ? get a sentence pause, others a short one.
"""
from __future__ import annotations

import json
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
    reused: int = 0


def read_lines(text_file: Path) -> list[str]:
    return [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def looks_complete(result: Result, line: str) -> bool:
    return result.duration >= MIN_SEC_PER_CHAR * len(line)


def previous_parts(out_dir: Path, voice: Voice) -> dict[str, np.ndarray]:
    """Audio of lines already read into out_dir by the same voice, keyed by line text (from lines.json)."""
    meta = out_dir / "lines.json"
    if not meta.exists():
        return {}
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("voice") != voice.name or info.get("reference") != voice.text:
        return {}
    found = {}
    for i, line in enumerate(info["lines"], 1):
        wav = out_dir / "parts" / f"{i:02d}.wav"
        if wav.exists() and line not in found:
            found[line] = sf.read(wav, dtype="float32")[0]
    return found


def read_article(
    tts,
    voice: Voice,
    text_file: Path,
    out_dir: Path,
    tries: int = 3,
    short_pause: float = 0.25,
    long_pause: float = 0.6,
    reuse: bool = True,
) -> ArticleReport:
    """Write out_dir/article.wav plus out_dir/parts/NN.wav (one per line, for re-doing single lines).

    Lines whose text is unchanged since the last run with the same voice are reused, not regenerated,
    so editing one line of a long text costs one line. `tts` may be a VoxPocket or a function returning
    one: it is only called (loading the model) if some line needs generating."""
    lines = read_lines(text_file)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    old = previous_parts(out_dir, voice) if reuse else {}
    model = None

    def get_model() -> VoxPocket:
        nonlocal model
        if model is None:
            model = tts() if callable(tts) and not isinstance(tts, VoxPocket) else tts
        return model

    pieces, flagged, taken, reused, sr = [], [], 0.0, 0, None
    for i, line in enumerate(lines, 1):
        if line in old:
            audio, reused = old[line], reused + 1
            sr = sr or sf.info(next(parts_dir.glob("*.wav"))).samplerate
            print(f"[{i:2d}/{len(lines)}] reused  {line[:40]}…", flush=True)
        else:
            m = get_model()
            sr = m.sample_rate
            best: Result | None = None
            for seed in range(tries):
                result = m.generate(line, voice, seed=seed)
                taken += result.seconds_taken
                if best is None or result.duration > best.duration:
                    best = result
                if looks_complete(result, line):
                    break
            ok = looks_complete(best, line)
            if not ok:
                flagged.append(i)
            audio = best.audio
            note = "" if ok else "  (maybe cut short)"
            print(f"[{i:2d}/{len(lines)}] {best.duration:5.2f}s seed={seed}{note}  {line[:40]}…", flush=True)
        sf.write(parts_dir / f"{i:02d}.wav", audio, sr)
        pause = long_pause if line.endswith(SENTENCE_ENDINGS) else short_pause
        pieces += [audio, np.zeros(int(pause * sr), dtype=np.float32)]

    for extra in parts_dir.glob("*.wav"):  # parts left over from a longer earlier version
        if int(extra.stem) > len(lines):
            extra.unlink()
    output = out_dir / "article.wav"
    audio = np.concatenate(pieces)
    sf.write(output, audio, sr)
    (out_dir / "lines.json").write_text(json.dumps({"voice": voice.name, "reference": voice.text, "lines": lines},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    return ArticleReport(output, len(audio) / sr, taken, flagged, reused)
