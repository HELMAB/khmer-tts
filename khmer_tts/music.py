"""Background music for videos: original, synthesized beds ("ambient", "khmer"; no licensing issues), and the mixer
helper that lays any music track under the narration with ducking.

The ambient bed is in A minor pentatonic (A C D E G), the five-note colour of much Khmer music: slow
crossfading pad chords, a low drone and a sparse mallet motif (roneat-like inharmonic partials), with reverb.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

CHORDS = [  # Hz; changes every CHORD_S seconds, all notes from the pentatonic scale
    (220.00, 261.63, 329.63),  # A C E
    (261.63, 329.63, 392.00),  # C E G
    (196.00, 293.66, 392.00),  # G D G (open)
    (164.81, 220.00, 293.66),  # E A D (suspended)
]
MOTIF = [440.00, 523.25, 587.33, 659.26, 783.99, 880.00]  # A C D E G A, an octave above the pad
CHORD_S, XFADE_S = 8.0, 3.0


def _pad(t: np.ndarray, sr: int) -> np.ndarray:
    out = np.zeros_like(t)
    n_chords = int(t[-1] // CHORD_S) + 2
    for k in range(n_chords):
        start = k * CHORD_S - XFADE_S / 2
        seg = (t >= start) & (t < start + CHORD_S + XFADE_S)
        if not seg.any():
            continue
        u = t[seg] - start
        env = np.clip(np.minimum(u / XFADE_S, (CHORD_S + XFADE_S - u) / XFADE_S), 0, 1)
        env = np.sin(env * np.pi / 2) ** 2  # equal-power-ish crossfade between chords
        tone = np.zeros_like(u)
        for f in CHORDS[k % len(CHORDS)]:
            for detune in (-0.0015, 0.0, 0.0015):  # three slightly detuned voices per note -> chorus
                ph = 2 * np.pi * f * (1 + detune) * t[seg]
                tone += np.sin(ph) + 0.12 * np.sin(2 * ph)
        out[seg] += env * tone / 9
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * t / 11)  # slow swell
    return out * lfo


def _mallet(f: float, sr: int, dur: float = 2.2) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    body = np.sin(2 * np.pi * f * t) * np.exp(-t * 2.6)
    bright = 0.35 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t * 9)  # inharmonic bar partial
    click = 0.15 * np.sin(2 * np.pi * f * 9.2 * t) * np.exp(-t * 40)
    return (body + bright + click) * np.minimum(1, t * 600)


def _reverb(x: np.ndarray, sr: int, seconds: float = 2.6, mix: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / sr * (6.9 / seconds))
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return (1 - mix) * x + mix * wet


def ambient(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    """Mono ambient bed, peak-normalised to 0.9."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    music = 0.55 * _pad(t, sr)
    music += 0.18 * np.sin(2 * np.pi * 55 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * t / 7))  # low A drone
    pos = 1.5
    while pos < seconds - 2:  # sparse motif: phrases of 1-3 notes with breathing gaps
        for _ in range(rng.integers(1, 4)):
            note = _mallet(rng.choice(MOTIF), sr)
            i = int(pos * sr)
            seg = note[: max(0, n - i)]
            music[i: i + len(seg)] += 0.22 * rng.uniform(0.6, 1.0) * seg
            pos += rng.choice([0.5, 0.75, 1.0])
        pos += rng.uniform(1.5, 3.5)
    music = _reverb(music, sr, seed=seed)
    return (0.9 * music / (np.abs(music).max() + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- "khmer": sad -> hopeful, traditional colour
# An original piece in the manner of a small mahori ensemble: tro (bowed fiddle) melody with slides, roneat
# (xylophone) answering an octave up with rolled tremolo, khim (hammered dulcimer) ostinato, chhing (finger
# cymbals: open "chhing" on beat 1, closed "chhap" on beat 3) and a soft skor (drum). D minor pentatonic
# (D F G A C); the last third re-centres on F over the same notes, which turns the colour hopeful.
# Sections run by fraction of the video: sad 0-0.35 (drone, fiddle, sparse roneat), building 0.35-0.7 (+khim,
# chhing, higher melody), hopeful 0.7-1 (+skor, F/C drone, brighter, busier roneat).

SCALE = [146.83, 174.61, 196.00, 220.00, 261.63]  # D3 F3 G3 A3 C4; degree k -> SCALE[k % 5] * 2 ** (k // 5)
BEAT = 60 / 66  # seconds per beat (66 bpm, unhurried)


def _deg(k: int) -> float:
    return SCALE[k % 5] * 2 ** (k // 5)


def _bowed(freqs: np.ndarray, sr: int) -> np.ndarray:
    """Fiddle-like tone following a per-sample frequency curve (vibrato and bow noise added here)."""
    n = len(freqs)
    t = np.arange(n) / sr
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.6)
    phase = 2 * np.pi * np.cumsum(freqs * vib) / sr
    tone = sum(a * np.sin(k * phase) for k, a in enumerate((1, .55, .35, .25, .16, .1, .06), 1))
    noise = np.random.default_rng(n).standard_normal(n) * 0.02
    return (tone + noise).astype(np.float32)


def _pluck(f: float, sr: int, dur: float = 1.6) -> np.ndarray:
    """Khim string: bright attack, upper harmonics fade first."""
    t = np.arange(int(dur * sr)) / sr
    out = sum((0.8 ** k) * np.sin(2 * np.pi * f * k * t) * np.exp(-t * (2.5 + 2.2 * k)) for k in range(1, 7))
    return (out * np.minimum(1, t * 900)).astype(np.float32)


def _chhing(sr: int, open_: bool) -> np.ndarray:
    dur = 1.4 if open_ else 0.12
    t = np.arange(int(dur * sr)) / sr
    decay = 3.0 if open_ else 40.0
    out = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((2630, 1), (3910, .7), (5170, .5), (6940, .3)))
    return (out * np.exp(-t * decay) * np.minimum(1, t * 3000)).astype(np.float32)


def _skor(sr: int, soft: bool) -> np.ndarray:
    t = np.arange(int(0.6 * sr)) / sr
    f = 55 + 45 * np.exp(-t * 18)
    out = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 7)
    return (out * (0.5 if soft else 1.0)).astype(np.float32)


def _add(buf: np.ndarray, x: np.ndarray, at: float, sr: int, gain: float) -> None:
    i = int(at * sr)
    if i >= len(buf) or gain <= 0:
        return
    seg = x[: len(buf) - i]
    buf[i: i + len(seg)] += gain * seg


def _phrase(rng, centre: int, beats: int) -> list[tuple[int, float]]:
    """A melodic phrase: (degree, beats) pairs, mostly stepwise, ending long on the centre note."""
    notes, k, left = [], centre + int(rng.choice([0, 1, 2, -1])), beats
    while left > 2:
        d = float(rng.choice([1, 1, 2, 2, 0.5, 3]))
        d = min(d, left - 2)
        notes.append((k, d))
        left -= d
        k += int(rng.choice([-2, -1, -1, 1, 1, 2]))
        k = int(np.clip(k, centre - 3, centre + 4))
    notes.append((centre, left))
    return notes


def khmer(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    """Mono bed, peak-normalised to 0.9: sad, then building, then hopeful (see the notes above)."""
    from scipy.signal import oaconvolve
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    prog = t / max(seconds, 1)
    sad = np.clip((0.42 - prog) / 0.14, 0, 1)            # 1 in the opening, fades out 0.28-0.42
    hope = np.clip((prog - 0.62) / 0.14, 0, 1)           # fades in 0.62-0.76
    build = np.clip((prog - 0.3) / 0.1, 0, 1)            # from 0.3 on
    fiddle, roneat, khim, perc = (np.zeros(n, np.float32) for _ in range(4))

    # drone: D + A (sad), moving to F + C (hopeful), soft and slowly breathing
    breathe = 0.8 + 0.2 * np.sin(2 * np.pi * t / 9)
    drone = (1 - hope) * (np.sin(2 * np.pi * 73.42 * t) + 0.5 * np.sin(2 * np.pi * 110 * t)) \
        + hope * (np.sin(2 * np.pi * 87.31 * t) + 0.5 * np.sin(2 * np.pi * 130.81 * t))
    drone = (drone * breathe * 0.11).astype(np.float32)

    # melody: phrases of 8 beats with breathing gaps; centre D (sad) -> A (building) -> F an octave up (hope)
    pos = 2 * BEAT
    while pos < seconds - 4:
        p = pos / seconds
        centre = 5 if p < 0.35 else 8 if p < 0.7 else 11   # D4, A4, F5
        phrase = _phrase(rng, centre, 8)
        dur = sum(b for _, b in phrase) * BEAT
        m = int(dur * sr)
        curve, i = np.empty(m, np.float32), 0
        prev = _deg(phrase[0][0])
        for k, b in phrase:  # slide into each note over ~0.12 s, the fiddle's portamento
            f, L = _deg(k), int(b * BEAT * sr)
            L = min(L, m - i)
            g = min(L, int(0.12 * sr))
            curve[i: i + g] = np.linspace(prev, f, g) if g else []
            curve[i + g: i + L] = f
            i += L
            prev = f
        curve[i:] = prev
        env = np.minimum(1, np.arange(m) / (0.35 * sr)) * np.minimum(1, (m - np.arange(m)) / (0.8 * sr))
        _add(fiddle, _bowed(curve, sr) * env.astype(np.float32), pos, sr, 1.0)
        # roneat answers the phrase an octave up: struck notes, long ones rolled (tremolo)
        if rng.random() < (0.55 if p < 0.35 else 0.8):
            at = pos + dur * 0.55
            for k, b in phrase[-3:]:
                f = _deg(k) * 2
                if b >= 2:
                    for r in range(int(b * BEAT / 0.095)):
                        _add(roneat, _mallet(f, sr, 0.8), at + r * 0.095, sr, 0.55 * 0.97 ** r)
                else:
                    _add(roneat, _mallet(f, sr), at, sr, 0.9)
                at += b * BEAT * 0.5
        pos += dur + float(rng.choice([1, 2])) * BEAT

    # khim ostinato in eighth notes on the drone's chord tones, from the building section on
    beat_t = 0.0
    pattern_sad, pattern_hope = [5, 8, 7, 8], [6, 9, 8, 10]  # D4 A4 G4 A4 / F4 C5 A4 D5
    while beat_t < seconds - 1:
        p = beat_t / seconds
        if p > 0.3:
            pat = pattern_hope if p > 0.7 else pattern_sad
            for e in range(2):
                k = pat[(int(beat_t / BEAT) * 2 + e) % 4]
                _add(khim, _pluck(_deg(k), sr), beat_t + e * BEAT / 2, sr, 0.35 if e else 0.5)
        # chhing: open on beat 1, closed on beat 3 (from 0.35); skor on beat 1 (and soft on 3) in the hopeful part
        b = int(round(beat_t / BEAT)) % 4
        if p > 0.35 and b in (0, 2):
            _add(perc, _chhing(sr, b == 0), beat_t, sr, 0.10)
        if p > 0.7 and b in (0, 2):
            _add(perc, _skor(sr, b == 2), beat_t, sr, 0.6)
        beat_t += BEAT

    mix = (drone + 0.42 * fiddle * (0.85 + 0.15 * build) + 0.20 * roneat * (0.8 + 0.4 * hope)
           + 0.16 * khim * build + 0.5 * perc) * (0.9 + 0.1 * sad)
    ir_n = int(2.4 * sr)
    ir = rng.standard_normal(ir_n).astype(np.float32) * np.exp(-np.arange(ir_n) / sr * (6.9 / 2.4)).astype(np.float32)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = oaconvolve(mix, ir)[:n].astype(np.float32)
    out = 0.7 * mix + 0.3 * wet
    return (0.9 * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


SYNTHS = {"ambient": ambient, "khmer": khmer}


def load_track(path: Path, sr: int) -> np.ndarray:
    """Any audio file as mono float32 at `sr` (via ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def music_bed(spec: dict, voice: np.ndarray, sr: int, length: int, root: Path) -> np.ndarray:
    """The music for a storyboard `music` spec, looped/cut to `length`, levelled to `level_db` (RMS, dBFS),
    ducked by `duck_db` while the voice speaks, with a fade-in and a fade-out at the end."""
    if "file" not in spec:  # "synth": "ambient" (default) or "khmer"
        name = spec.get("synth", "ambient")
        if name not in SYNTHS:
            raise SystemExit(f"unknown music synth {name!r} ({', '.join(SYNTHS)})")
        track = SYNTHS[name](length / sr + 1, sr, spec.get("seed", 0))
    else:
        track = load_track(root / spec["file"], sr)
    if len(track) < length:
        track = np.tile(track, length // len(track) + 1)
    track = track[:length]
    rms = np.sqrt(np.mean(track ** 2)) + 1e-9
    track *= 10 ** (spec.get("level_db", -24) / 20) / rms

    # ducking: speech activity in 10 ms frames -> smoothed gain (fast attack, slow release)
    hop = sr // 100
    v = np.zeros(length, np.float32)
    v[: min(length, len(voice))] = voice[:length]
    frames = np.sqrt(np.mean(v[: len(v) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    active = np.convolve(frames > 10 ** (-42 / 20), np.ones(25) / 25, mode="same") > 0  # bridge short gaps
    duck = 10 ** (-spec.get("duck_db", 7) / 20)
    target = np.where(active, duck, 1.0)
    gain, g = np.empty_like(target), 1.0
    for i, x in enumerate(target):
        g += (x - g) * (0.25 if x < g else 0.03)  # attack ~40 ms, release ~330 ms
        gain[i] = g
    gain = np.interp(np.arange(length), np.arange(len(gain)) * hop, gain)
    fade_in, fade_out = int(spec.get("fade_in", 1.5) * sr), int(spec.get("fade_out", 2.5) * sr)
    env = np.ones(length)
    env[:fade_in] = np.linspace(0, 1, fade_in)
    env[-fade_out:] = np.minimum(env[-fade_out:], np.linspace(1, 0, fade_out))
    return (track * gain * env).astype(np.float32)
