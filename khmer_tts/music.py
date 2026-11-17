"""Background music for videos: an original, synthesized ambient bed (no licensing issues), and the mixer
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


def load_track(path: Path, sr: int) -> np.ndarray:
    """Any audio file as mono float32 at `sr` (via ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def music_bed(spec: dict, voice: np.ndarray, sr: int, length: int, root: Path) -> np.ndarray:
    """The music for a storyboard `music` spec, looped/cut to `length`, levelled to `level_db` (RMS, dBFS),
    ducked by `duck_db` while the voice speaks, with a fade-in and a fade-out at the end."""
    if spec.get("synth", "ambient") == "ambient" and "file" not in spec:
        track = ambient(length / sr + 1, sr, spec.get("seed", 0))
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
