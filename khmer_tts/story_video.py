"""Render a 1-minute Facebook Story (1080x1920) from photo(s) and a song: karaoke lyrics, a sound visualizer
and camera moves inside the photo, one shot per lyric line.

Storyboard (see videos/hometown_story.json):
    style      "story"
    song       audio file (from `khmer_tts song URL`), cut from `start` for `seconds` (default 60)
    title / artist                     shown over the instrumental intro, until the first lyric
    badge / date                       label on the photo card (e.g. "វិស្សមកាលនៅស្រុកកំណើត" / "22 · 11 · 2026")
    photo      default photo; a shot may name another with "photo"
    lyrics     [{"t": [start, part2 start, ..., end], "text": [part, ...]}, ...] in SONG seconds
               (one row per part; from `khmer_tts song --align`)
    shots      optional [{"from": [x, y, zoom], "to": [x, y, zoom], "photo"?}, ...]: the intro, then one per
               lyric line; x/y in photo pixels, zoom 1 = the largest card-shaped crop of the photo
    mood       "warm" (default: golden light, sun flare, rising sparkles, lively float and beat pulse) or "sad"
               (cool faded grade, dark vignette, a few faint drifting dust specks, slow calm motion, dissolves);
               pick it from what the song says, not its tempo
    grade      photo colour grade, defaults to the mood's: "warm", "cool" or "none"
    title_at   "start" (default: over the intro, until the first lyric), "end" (after the last lyric, for a
               clip that opens on the singing, e.g. a chorus) or "none" (singing from start to end: put the
               song and artist in badge/date instead)
    fade_out   seconds of fade to black (and audio fade) at the end, default 2.5
    lyrics_style  "plain" (default: each line shown whole in one soft colour, fading in and out; the user's
               choice) or "karaoke" (dim line filled left→right as it is sung)
All text is Google Sans (the user's rule for every video).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .video import default_workers, load_storyboard, text_image
from .voices import PROJECT_ROOT

W, H, FPS = 1080, 1920, 30
# Facebook Stories: the profile bar covers the top ~250 px, the reply bar the bottom ~300 px
CARD_W, CARD_H, CARD_CY = 690, 862, 730
VIS_Y, VIS_W, BARS = 1320, 860, 56
LYRIC_CY = 1500
LYRIC_FADE = 0.35  # a lyric line fades in this long before it is sung, and out after
FADE_OUT = 2.5
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"

GOLD, LEAF, CREAM = (255, 204, 77), (128, 214, 120), (255, 248, 230)

# Everything the mood changes. "motion" scales the float, tilt and beat pulse; "cut" is the half-length of the
# shot transition ("dissolve" = plain cross-fade, no zoom blur).
MOODS = {
    "warm": dict(accent=GOLD, low=LEAF, border=CREAM, text=CREAM, badge_bg=(12, 20, 14, 170), grade="warm",
                 bg_dim=0.62, bg_tint=(34, 40, 18), flare=True, weather="sparkles", motion=1.0, vignette=0.0,
                 cut=0.3, dissolve=False, pop=True, unsung=120, lyric=(255, 236, 196)),
    "sad": dict(accent=(214, 232, 255), low=(84, 104, 140), border=(214, 219, 227), text=(226, 231, 238),
                badge_bg=(12, 16, 26, 180), grade="cool", bg_dim=0.40, bg_tint=(10, 16, 28), flare=False,
                weather="dust", motion=0.4, vignette=0.6, cut=0.6, dissolve=True, pop=False, unsung=62,
                lyric=(228, 234, 243)),
}


# ---------------------------------------------------------------- easing

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(p):
    return 1 - (1 - clamp(p)) ** 3


def ease_in_out(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def back_out(p, s=1.6):
    p = clamp(p) - 1
    return 1 + p * p * ((s + 1) * p + s)


def with_alpha(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    img = img.copy()
    img.putalpha(img.getchannel("A").point(lambda v: int(v * clamp(a))))
    return img


def paste_center(frame: Image.Image, img: Image.Image, cx: float, cy: float, alpha=1.0):
    if alpha <= 0.01:
        return
    frame.alpha_composite(with_alpha(img, alpha), (int(cx - img.width / 2), int(cy - img.height / 2)))


# ---------------------------------------------------------------- audio analysis

def analyse(audio: np.ndarray, sr: int, n_frames: int):
    """Per-video-frame spectrum bars (0..1, smoothed with fast attack / slow release) and a bass pulse."""
    win = 4096
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(55, 11000, BARS + 1)
    band_idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in zip(edges[:-1], edges[1:])]
    bass_idx = np.where((freqs >= 40) & (freqs < 160))[0]
    pad = np.pad(audio, win)
    raw, bass = np.zeros((n_frames, BARS)), np.zeros(n_frames)
    for n in range(n_frames):
        c = int(n / FPS * sr) + win
        spec = np.abs(np.fft.rfft(pad[c - win // 2:c + win // 2] * hann))
        raw[n] = [spec[i].mean() if len(i) else spec[np.argmin(abs(freqs - lo))]
                  for i, lo in zip(band_idx, edges)]
        bass[n] = spec[bass_idx].mean()
    db = 20 * np.log10(raw + 1e-6)
    lo, hi = np.percentile(db, 8, axis=0), np.percentile(db, 99.5, axis=0)  # per-band, so treble moves too
    level = np.clip((db - lo) / (hi - lo + 1e-6), 0, 1) ** 1.6
    bars = np.zeros_like(level)
    for n in range(n_frames):
        prev = bars[n - 1] if n else level[0]
        k = np.where(level[n] > prev, 0.65, 0.18)
        bars[n] = prev + (level[n] - prev) * k
    bdb = 20 * np.log10(bass + 1e-6)
    pulse = np.clip((bdb - np.percentile(bdb, 30)) / (np.percentile(bdb, 99) - np.percentile(bdb, 30)), 0, 1)
    pulse = np.convolve(pulse, [0.5, 0.3, 0.2], "full")[:n_frames]
    return bars, pulse


# ---------------------------------------------------------------- static layers

def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    s = max(w / img.width, h / img.height)
    img = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def grade(photo: Image.Image) -> Image.Image:
    """Warm 'golden afternoon' look: richer colour, a touch of contrast, warmer whites, softer blue."""
    img = ImageEnhance.Color(photo).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Brightness(img).enhance(1.04)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * np.array([1.06, 1.015, 0.9], np.float32) + np.array([6, 3, 0], np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def grade_cool(photo: Image.Image) -> Image.Image:
    """Sad 'grey overcast day' look: drained colour, soft contrast, lifted (faded) blacks, a cold blue cast."""
    img = ImageEnhance.Color(photo).enhance(0.5)
    img = ImageEnhance.Contrast(img).enhance(0.92)
    img = ImageEnhance.Brightness(img).enhance(0.9)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * np.array([0.88, 0.95, 1.06], np.float32) + np.array([10, 16, 28], np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


GRADES = {"warm": grade, "cool": grade_cool, "none": lambda im: im}


def make_vignette(strength: float) -> Image.Image | None:
    if strength <= 0:
        return None
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H * 0.45) / (H * 0.62)) ** 2)
    a = np.clip((d - 0.45) / 0.75, 0, 1) ** 1.5 * 255 * strength
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def make_background(photo: Image.Image, view, dim: float, tint) -> Image.Image:
    """Blurred photo around the first shot's view (so whatever the shots avoid stays out of it too),
    25 % larger than the frame so it can slowly zoom and drift."""
    bw, bh = int(W * 1.25), int(H * 1.25)
    x, y, _ = view
    half = min(x, photo.width - x, photo.width / 2)
    side = photo.crop((int(x - half), 0, int(x + half), photo.height))
    bg = cover(side, bw, bh).filter(ImageFilter.GaussianBlur(38)).convert("RGB")
    arr = np.asarray(bg).astype(np.float32) * dim
    yy = np.linspace(0, 1, bh)[:, None, None]
    tint = np.array(tint, np.float32)  # the mood's colour toward the bottom, behind the lyrics
    arr = arr * (1 - 0.5 * yy) + tint * 0.5 * yy
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")


def round_mask(w, h, r) -> Image.Image:
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def make_badge(text: str, date: str, mood: dict) -> Image.Image | None:
    if not text and not date:
        return None
    rows = [r for r in (text_image(text, 34, "Bold", mood["text"], max_w=560) if text else None,
                        text_image(date, 26, "Medium", mood["accent"], max_w=560) if date else None) if r]
    pw, ph = max(r.width for r in rows) + 56, sum(r.height for r in rows) - 6 * (len(rows) - 1) + 8
    b = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    ImageDraw.Draw(b).rounded_rectangle((0, 0, pw - 1, ph - 1), 26, fill=mood["badge_bg"])
    y = 4
    for r in rows:
        b.alpha_composite(r, (28, y))
        y += r.height - 6
    return b


def make_shadow(w, h, r) -> Image.Image:
    pad = 70
    s = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(s).rounded_rectangle((pad, pad + 24, pad + w, pad + h + 24), r, fill=(0, 0, 0, 150))
    return s.filter(ImageFilter.GaussianBlur(30))


def dot_sprite(r: int, color=(255, 240, 190)) -> Image.Image:
    s = r * 4
    d = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(d).ellipse((r, r, 3 * r, 3 * r), fill=color + (255,))
    return d.filter(ImageFilter.GaussianBlur(r * 0.6))


def make_sunflare() -> Image.Image:
    s = 1300
    y, x = np.mgrid[0:s, 0:s] / s - 0.5
    a = np.clip(1 - np.sqrt(x * x + y * y) * 2, 0, 1) ** 2.2
    img = np.zeros((s, s, 4), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 255, 190, 90
    img[..., 3] = (a * 120).astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def title_text(text: str, size: int, color) -> Image.Image:
    """Google Sans Bold title with a soft dark glow behind it for legibility."""
    fg = text_image(text, size, "Bold", color, max_w=980)
    pad = size // 2
    img = Image.new("RGBA", (fg.width + pad * 2, fg.height + pad * 2), (0, 0, 0, 0))
    img.alpha_composite(text_image(text, size, "Bold", (0, 0, 0, 170), max_w=980), (pad, pad))
    img = img.filter(ImageFilter.GaussianBlur(10))
    img.alpha_composite(fg, (pad, pad))
    return img


def fit_title(text: str, color) -> Image.Image:
    """The title at 96 px, shrunk (down to 60 px) until it fits one line; a long title still wraps after that."""
    size = 96
    while size > 60 and text_image(text, size, "Bold", color, max_w=10_000).width > 960:
        size -= 4
    return title_text(text, size, color)


# ---------------------------------------------------------------- per-frame state (filled before fork)

_ST: dict = {}


@lru_cache(maxsize=32)
def lyric_layers(text: str) -> tuple[Image.Image, Image.Image]:
    """(dim base with a dark glow, accent fill) of one lyric row; the fill is revealed left→right as sung."""
    if not text:
        text = "•   •   •"
    base = text_image(text, 56, "Bold", (255, 255, 255, _ST["mood"]["unsung"]), max_w=900, line_gap=1.45)
    fill = text_image(text, 56, "Bold", _ST["mood"]["accent"], max_w=900, line_gap=1.45)
    glow = text_image(text, 56, "Bold", (0, 0, 0, 200), max_w=900, line_gap=1.45).filter(ImageFilter.GaussianBlur(10))
    glow.alpha_composite(base)
    return glow, fill


@lru_cache(maxsize=32)
def lyric_plain(text: str) -> Image.Image:
    """One lyric row shown whole: the mood's lyric colour over a soft dark glow."""
    fg = text_image(text, 56, "Bold", _ST["mood"]["lyric"], max_w=900, line_gap=1.45)
    glow = text_image(text, 56, "Bold", (0, 0, 0, 200), max_w=900, line_gap=1.45).filter(ImageFilter.GaussianBlur(10))
    glow.alpha_composite(fg)
    return glow


def draw_visualizer(frame: Image.Image, levels: np.ndarray, alpha: float):
    layer = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    mid, step = 150, VIS_W / BARS
    bw, x0 = step * 0.56, (W - VIS_W) / 2
    for i, v in enumerate(levels):
        t = i / (BARS - 1)
        lo, hi = _ST["mood"]["low"], _ST["mood"]["accent"]
        col = tuple(int(lo[k] + (hi[k] - lo[k]) * t) for k in range(3))
        h = 6 + v * 118
        x = x0 + i * step + (step - bw) / 2
        d.rounded_rectangle((x, mid - h, x + bw, mid), bw / 2, fill=col + (255,))
        d.rounded_rectangle((x, mid + 4, x + bw, mid + 4 + h * 0.42), bw / 2, fill=col + (70,))  # reflection
    out = Image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(14)), layer)
    frame.alpha_composite(with_alpha(out, alpha), (0, VIS_Y - mid))


def shot_view(i: int, t: float):
    sh = _ST["shots"][i]
    (x0, y0, z0), (x1, y1, z1) = sh["from"], sh["to"]
    s0, s1 = _ST["cuts"][i], _ST["cuts"][i + 1]
    p = ease_in_out((t - s0) / (s1 - s0))
    return sh["photo"], (x0 + (x1 - x0) * p, y0 + (y1 - y0) * p, z0 + (z1 - z0) * p)


def shot_image(photo_key: str, view) -> Image.Image:
    """Crop the (upscaled, graded) photo at a view and size it to the card."""
    src, k, (pw, ph) = _ST["photos"][photo_key]
    cx, cy, z = view
    cw = min(pw, ph * CARD_W / CARD_H) / z
    ch = cw * CARD_H / CARD_W
    cx, cy = clamp(cx, cw / 2, pw - cw / 2), clamp(cy, ch / 2, ph - ch / 2)
    box = (k * (cx - cw / 2), k * (cy - ch / 2), k * (cx + cw / 2), k * (cy + ch / 2))
    return src.resize((CARD_W, CARD_H), Image.BICUBIC, box=box)


def zoom_blur(img: Image.Image, k: float) -> Image.Image:
    """Cheap radial zoom blur: average of the image and two slightly enlarged copies."""
    if k < 0.04:
        return img
    out, (w, h) = img, img.size
    for i, sc in enumerate((1 + 0.045 * k, 1 + 0.09 * k), start=2):
        cw, ch = w / sc, h / sc
        big = img.resize((w, h), Image.BILINEAR, box=((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2))
        out = Image.blend(out, big, 1 / i)
    return out


def card_photo(t: float) -> Image.Image:
    cuts = _ST["cuts"]
    i = max(j for j in range(len(_ST["shots"])) if cuts[j] <= t)
    img = shot_image(*shot_view(i, t))
    cut = _ST["mood"]["cut"]
    near = min(cuts[1:-1], key=lambda c: abs(c - t), default=None)
    if near is not None and abs(t - near) < cut:
        j = cuts.index(near)  # shot j starts at this cut
        a = ease_in_out((t - near + cut) / (2 * cut))
        img = Image.blend(shot_image(*shot_view(j - 1, t)), shot_image(*shot_view(j, t)), a)
        if not _ST["mood"]["dissolve"]:
            img = zoom_blur(img, 1 - abs(t - near) / cut)
    return img


def render_frame(n: int) -> bytes:
    st, t = _ST, n / FPS
    mood = st["mood"]
    m = mood["motion"]
    pulse = float(st["pulse"][n]) * m
    intro = ease_out(t / 1.2)
    outro = clamp((st["total"] - t) / st["fade_out"])

    # background: slow zoom + drift, nudged by the bass
    bg = st["bg"]
    z = 1.0 + 0.10 * (t / st["total"]) + 0.012 * pulse
    cw, ch = W / z, H / z
    cx = bg.width / 2 + 40 * math.sin(t * 0.13)
    cy = bg.height / 2 + 60 * math.sin(t * 0.09 + 1)
    frame = bg.resize((W, H), Image.BILINEAR, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))

    if mood["flare"]:  # warm sun flare drifting across the top
        fx = W * 0.5 + W * 0.55 * math.sin(t * 0.21)
        paste_center(frame, st["flare"], fx, 260 + 80 * math.cos(t * 0.17), 0.55 + 0.35 * pulse)

    if mood["weather"] == "sparkles":  # rising dust/pollen particles
        for (x0, spd, r, ph) in st["particles"]:
            y = (H + 60) - ((t * spd + ph * 977) % (H + 120))
            x = x0 + 24 * math.sin(t * 0.7 + ph * 6)
            a = 0.35 + 0.45 * (0.5 + 0.5 * math.sin(t * 2.3 + ph * 11))
            paste_center(frame, st["dots"][r], x, y, a * intro)
    elif mood["weather"] == "dust":  # a few faint specks floating slowly in the air, no direction
        for (x0, spd, r, ph) in st["particles"][:16]:
            x = x0 + 60 * math.sin(t * 0.11 + ph * 9)
            y = (ph * 1.7 % 1) * H + 80 * math.sin(t * 0.08 + ph * 5)
            a = 0.18 + 0.17 * (0.5 + 0.5 * math.sin(t * 0.6 + ph * 11))
            paste_center(frame, st["dots"][r], x, y, a * intro)

    # photo card: a camera move per lyric line, float, tilt and beat pulse
    card = Image.new("RGBA", (CARD_W + 16, CARD_H + 16), (0, 0, 0, 0))
    card.paste(Image.new("RGBA", card.size, mood["border"] + (255,)), (0, 0), st["border_mask"])
    card.paste(card_photo(t), (8, 8), st["card_mask"])
    if st["badge"]:
        card.alpha_composite(st["badge"], (34, CARD_H - st["badge"].height - 20))
    entry = back_out(t / 1.3) if mood["pop"] else ease_out(t / 2.0)  # sad: settles in, no overshoot
    scale = (0.86 + 0.14 * entry) * (1 + 0.012 * pulse)
    shadow = st["shadow"]
    lay = shadow.copy()
    lay.alpha_composite(card, ((shadow.width - card.width) // 2, (shadow.height - card.height) // 2))
    lay = lay.resize((int(lay.width * scale), int(lay.height * scale)), Image.BICUBIC)
    lay = lay.rotate(1.6 * m * math.sin(t * 0.55 * (0.5 + m / 2)), Image.BICUBIC, expand=True)
    paste_center(frame, lay, W / 2, CARD_CY + 14 * m * math.sin(t * 0.8 * (0.5 + m / 2)) + (1 - intro) * 80, intro)

    draw_visualizer(frame, st["bars"][n], ease_out((t - 0.4) / 1.0) * (1.0 if mood["pop"] else 0.85))
    if st["vignette"]:
        frame.alpha_composite(st["vignette"])

    # title over the instrumental intro (or after the last lyric), and karaoke lyrics
    t0, end = st["title_window"]
    if t0 <= t < end:
        out_a = clamp((end - t) / 0.5)
        a1 = ease_out((t - t0 - 0.6) / 0.8) * out_a
        a2 = ease_out((t - t0 - 1.2) / 0.8) * out_a
        s = 0.9 + 0.1 * (back_out if mood["pop"] else ease_out)((t - t0 - 0.6) / 0.9)
        title = st["title"]
        th = title.height if title else 0
        extra = max(0, th - 290)  # a title that wraps to two rows grows upward and pushes the artist down
        ty = LYRIC_CY - 30 - extra * 0.55
        if title:
            tt = title.resize((int(title.width * s), int(title.height * s)), Image.BICUBIC)
            paste_center(frame, tt, W / 2, ty - (1 - out_a) * 30, a1)
        if st["artist"]:
            ay = ty + th / 2 - 30 if title else LYRIC_CY
            paste_center(frame, st["artist"], W / 2, ay + (1 - a2) * 24, a2)

    for times, parts, show, hide in st["lyrics"]:
        if not (show <= t <= hide):
            continue
        enter = ease_out((t - show) / LYRIC_FADE)
        leave = clamp((hide - t) / LYRIC_FADE)
        rows = []
        for i, part in enumerate(parts):  # each half-line on its own row
            if not st["karaoke"]:
                rows.append(lyric_plain(part))
                continue
            base, fill = lyric_layers(part)  # karaoke: the fill sweeps across while the part is sung
            row = base.copy()
            reveal = int(fill.width * clamp((t - times[i]) / (times[i + 1] - times[i])))
            if reveal > 0:
                row.alpha_composite(fill.crop((0, 0, reveal, fill.height)),
                                    ((base.width - fill.width) // 2, (base.height - fill.height) // 2))
            rows.append(row)
        gap = -18
        total_h = sum(r.height for r in rows) + gap * (len(rows) - 1)
        y = LYRIC_CY - total_h / 2 + (1 - enter) * 40 - (1 - leave) * 30
        for row in rows:
            paste_center(frame, row, W / 2, y + row.height / 2, enter * leave)
            y += row.height + gap

    if outro < 1:
        frame = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), frame, outro ** 1.5)
    return frame.convert("RGB").tobytes()


# ---------------------------------------------------------------- setup and render

def auto_shots(n: int, size) -> list[dict]:
    """Gentle alternating push-in / pull-out around the photo centre when the storyboard gives no shots."""
    pw, ph = size
    c = (pw / 2, ph / 2)
    return [{"from": [c[0], c[1] + (20 if i % 2 else -20), 1.0 if i % 2 == 0 else 1.15],
             "to": [c[0], c[1] - (20 if i % 2 else -20), 1.15 if i % 2 == 0 else 1.0]} for i in range(n)]


def load_photo(path: Path, grade_name: str):
    """Graded photo upscaled for sharp close-ups (long side ~2600 px): (image, scale, original size)."""
    photo = GRADES[grade_name](Image.open(path).convert("RGB"))
    k = max(1, round(2600 / max(photo.size)))
    big = photo.resize((photo.width * k, photo.height * k), Image.LANCZOS)
    if k > 1:
        big = big.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    return big.convert("RGBA"), k, photo.size, photo


def prepare(sb: dict, root: Path, total: float) -> int:
    start = float(sb.get("start", 0))
    audio, sr = sf.read(root / sb["song"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n_frames = int(total * FPS)
    bars, pulse = analyse(audio[int(start * sr):int((start + total) * sr)], sr, n_frames)

    lyrics = [([float(v) - start for v in ln["t"]], list(ln["text"])) for ln in sb.get("lyrics", [])]
    for times, parts in lyrics:
        if len(times) != len(parts) + 1:
            raise SystemExit(f"lyric {parts}: {len(parts)} part(s) need {len(parts) + 1} times, got {times}")
    # on-screen window per line: fade in just before it is sung, out after; when lines follow closely,
    # the old one leaves before the next fades in so the two never overlap
    shown = []
    for i, (times, parts) in enumerate(lyrics):
        hide = times[-1] + LYRIC_FADE
        if i + 1 < len(lyrics):
            hide = min(hide, lyrics[i + 1][0][0] - LYRIC_FADE)
        shown.append((times, parts, times[0] - LYRIC_FADE, hide))

    mood = MOODS[sb.get("mood", "warm")]
    _ST["mood"] = mood  # lyric_layers and draw_visualizer read it, before and after the fork
    grade_name = sb.get("grade", mood["grade"])
    default = sb["photo"]
    photos = {}
    for f in {default, *(s.get("photo") for s in sb.get("shots", []) if s.get("photo"))}:
        photos[f] = load_photo(root / f, grade_name)
    shots = sb.get("shots") or auto_shots(len(lyrics) + 1, photos[default][2])
    shots = [{"photo": s.get("photo", default), "from": s["from"], "to": s["to"]} for s in shots]
    starts = [times[0] for times, _ in lyrics if 0 < times[0] < total]
    cuts = [0.0] + [s - 0.25 for s in starts[:len(shots) - 1]]  # cut just before each line is sung
    shots = shots[:len(cuts)]
    cuts.append(total)

    first_lyric = min((times[0] for times, _ in lyrics if times[0] > 0), default=5.0)
    rng = np.random.default_rng(7)
    _ST.update(
        total=total, bars=bars, pulse=pulse, lyrics=shown, shots=shots, cuts=cuts,
        photos={f: v[:3] for f, v in photos.items()},
        bg=make_background(photos[shots[0]["photo"]][3], shots[0]["from"], mood["bg_dim"], mood["bg_tint"]),
        vignette=make_vignette(mood["vignette"]),
        card_mask=round_mask(CARD_W, CARD_H, 36),
        border_mask=round_mask(CARD_W + 16, CARD_H + 16, 44),
        shadow=make_shadow(CARD_W + 16, CARD_H + 16, 44),
        badge=make_badge(sb.get("badge", ""), sb.get("date", ""), mood),
        flare=make_sunflare(),
        dots={r: dot_sprite(r, (255, 240, 190) if mood["weather"] == "sparkles" else (190, 205, 225))
              for r in (2, 3, 4, 6)},
        particles=[(rng.uniform(0, W), rng.uniform(18, 55), int(rng.choice([2, 3, 4, 6])), rng.uniform())
                   for _ in range(42)],
        title=fit_title(sb["title"], mood["accent"]) if sb.get("title") else None,
        artist=text_image(sb["artist"], 42, "Medium", mood["text"], max_w=900) if sb.get("artist") else None,
        title_window={"start": (0.0, max(2.0, first_lyric - 0.4)),
                      "end": (max((h for *_, h in shown), default=0.0) - 0.4, total + 1),
                      "none": (0.0, 0.0)}[sb.get("title_at", "start")],
        fade_out=float(sb.get("fade_out", FADE_OUT)),
        karaoke=sb.get("lyrics_style", "plain") == "karaoke",
    )
    return n_frames


def render(storyboard_path: Path, out: Path | None = None, workers: int | None = None, seconds: float | None = None,
           narration: str | None = None):
    root = PROJECT_ROOT
    sb = load_storyboard(storyboard_path)
    out = out or root / sb["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    total = min(float(sb.get("seconds", 60)), seconds or 1e9)
    n_frames = prepare(sb, root, total)
    start = float(sb.get("start", 0))
    fade_out = float(sb.get("fade_out", FADE_OUT))
    fade_at = max(0.0, total - fade_out)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-ss", str(start), "-t", str(total), "-i", str(root / sb["song"]),
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-af", f"afade=t=in:d=0.6,afade=t=out:st={fade_at}:d={fade_out},{LOUDNORM}",
         "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    ctx = mp.get_context("fork")  # workers inherit the prebuilt layers in _ST
    with ctx.Pool(workers or default_workers()) as pool:
        for i, frame in enumerate(pool.imap(render_frame, range(n_frames), chunksize=6)):
            ff.stdin.write(frame)
            if i % (FPS * 10) == 0:
                print(f"  frame {i}/{n_frames} ({i / FPS:.0f}s)", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    return out, total, 0


def render_still(storyboard_path: Path, t: float, out: Path, narration: str | None = None) -> None:
    sb = load_storyboard(storyboard_path)
    prepare(sb, PROJECT_ROOT, float(sb.get("seconds", 60)))
    Image.frombytes("RGB", (W, H), render_frame(int(t * FPS))).save(out)
