"""Song tools for Facebook Story videos: fetch a song from a URL, split off its vocals, find the sung phrases,
and force-align lyric lines to the singing so the karaoke fill follows the singer.

    fetch_song(url, out_dir)   song.wav + description.txt (YouTube uploaders often post the lyrics there)
    separate_vocals(song)      vocals.wav next to it (Demucs htdemucs, two stems)
    vocal_phrases(vocals)      [(start, end), ...] stretches of singing between breaths
    align_lyrics(vocals, ...)  per-part start/end times for each lyric line (torchaudio MMS_FA + uroman)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def fetch_song(url: str, out_dir: Path) -> tuple[Path, Path]:
    """Download the audio as song.wav and the page description as description.txt (yt-dlp via uvx)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ytdlp = ["yt-dlp"] if shutil.which("yt-dlp") else ["uvx", "yt-dlp"]
    common = ytdlp + ["--js-runtimes", "node", "--no-playlist"]
    desc = out_dir / "description.txt"
    r = subprocess.run(common + ["--skip-download", "--print", "%(title)s\n\n%(description)s", url],
                       capture_output=True, text=True)
    desc.write_text(r.stdout, encoding="utf-8")
    subprocess.run(common + ["-f", "bestaudio", "-x", "--audio-format", "wav", "-o", str(out_dir / "song.%(ext)s"), url],
                   check=True, stdout=subprocess.DEVNULL)
    return out_dir / "song.wav", desc


def separate_vocals(song: Path) -> Path:
    """Write vocals.wav next to the song with Demucs (~30 s for a 3.5 min song on an M-series Mac)."""
    tmp = song.parent / "demucs"
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(tmp), str(song)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vocals = song.parent / "vocals.wav"
    shutil.move(tmp / "htdemucs" / song.stem / "vocals.wav", vocals)
    shutil.rmtree(tmp)
    return vocals


def _envelope_db(vocals: Path, hop_s: float = 0.05) -> np.ndarray:
    x, sr = sf.read(vocals, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(sr * hop_s)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return np.convolve(20 * np.log10(rms + 1e-9), np.ones(3) / 3, "same")


def vocal_phrases(vocals: Path, threshold_db: float = -24, min_gap: float = 0.2, min_len: float = 0.3):
    """Stretches of singing in the separated vocals; gaps shorter than `min_gap` are joined.
    Separated vocals keep some bleed, so the threshold sits well above silence (-24 dB worked for a 1960s mix)."""
    db, hop = _envelope_db(vocals), 0.05
    on = db > threshold_db
    out, start, gap = [], None, 0
    for i, v in enumerate(on):
        if v:
            start = i if start is None else start
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap * hop > min_gap:
                out.append((start * hop, (last + 1) * hop))
                start = None
    if start is not None:
        out.append((start * hop, (last + 1) * hop))
    return [(round(a, 2), round(b, 2)) for a, b in out if b - a >= min_len]


def align_lyrics(vocals: Path, lines: list[list[str]], t_from: float, t_to: float) -> list[list[float]]:
    """Force-align lyric lines (each a list of parts, e.g. the two halves of a Khmer couplet) to the vocals
    between t_from and t_to (song seconds). Returns per line [start of part 1, start of part 2, ..., end].
    The text is romanized with uroman, so any language works; keep the window to the sung lines plus ~1 s."""
    import torch
    import torchaudio
    import uroman
    from torchaudio.pipelines import MMS_FA as bundle

    x, sr = sf.read(vocals, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    wav = torch.from_numpy(x[int(t_from * sr):int(t_to * sr)].copy())[None]
    wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)

    rom, vocab = uroman.Uroman(), bundle.get_dict()
    words, owner = [], []  # each romanized word, and the (line, part) it belongs to
    for li, parts in enumerate(lines):
        for pi, part in enumerate(parts):
            r = "".join(c for c in rom.romanize_string(part).lower() if c in vocab or c == " ")
            for w in r.split():
                words.append(w)
                owner.append((li, pi))
            if not r.split():
                raise ValueError(f"nothing to align in lyric part {part!r}")

    model = bundle.get_model()
    with torch.inference_mode():
        emission, _ = model(wav)
        spans = bundle.get_aligner()(emission[0], bundle.get_tokenizer()(words))
    sec = wav.shape[1] / emission.shape[1] / bundle.sample_rate

    out, ends = [], []
    for li, parts in enumerate(lines):
        times = []
        for pi in range(len(parts)):
            idx = [i for i, o in enumerate(owner) if o == (li, pi)]
            times.append(round(t_from + spans[idx[0]][0].start * sec, 2))
        last = [i for i, o in enumerate(owner) if o[0] == li][-1]
        ends.append((t_from + spans[last][0].start * sec, t_from + spans[last][-1].end * sec))
        out.append(times)
    # The aligner under-times held final notes ("…អឺយ") and stretches the very last word over any silence after
    # it. So a line lasts until just before the next one, but stops where the singing really stops (a gap of
    # 0.5 s or more: breaths inside a held note are shorter).
    phrases = vocal_phrases(vocals, min_gap=0.5)
    for li, (times, (last_start, raw_end)) in enumerate(zip(out, ends)):
        end = out[li + 1][0] - 0.3 if li + 1 < len(out) else raw_end
        end = min([end] + [b for a, b in phrases if a <= last_start < b])
        times.append(round(max(end, last_start + 0.5), 2))
    return out
