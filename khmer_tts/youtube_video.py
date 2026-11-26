"""Landscape (YouTube, 1920x1080) narrated video: an article's images with slow camera moves, text on the left,
an audio visualizer, captions, and the music bed under the narration. No sound effects.

Look (after Khmer storytelling channels such as វិភាគរឿង): a dark, smoky left side made from the current image
blurred, the image itself as a large panel on the right whose left edge dissolves into the dark, a small brand
mark top-left, and the section's kicker / big title / points (or a quote) revealed as they are spoken.

Storyboard ("style": "youtube"; example videos/dog_barking_youtube.json):
    tts_text, narration, output, accent, music, lockup {name, tag}
    images     {key: file}
    captions   one per TTS line (burned in unless "burn_captions": false; an .srt is always written)
    layout     "points" (default: kicker, title, numbered points, bottom captions, full-width visualizer),
               "paragraph" (the narration as running pages in the left column, unread lines gray, small visualizer) or
               "minimal" (left side empty, slim visualizer under the brand mark, one-line caption at the bottom)
    scenes     {"type": "headline" | "quote", "line", ... as in the explainer, plus
                "shot": {"image": key, "from": [x, y, zoom], "to": [x, y, zoom]}   (cover crop, image pixels)
                    or  {"image": key, "card": true}                               (whole image as a card)}
    tail       seconds after the narration (default 4)
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

from .video import (LATIN, TEXT, default_workers, font, hex_rgb, line_timings, load_storyboard, mix_audio,
                    text_image, with_alpha, wrap)
from .article import read_lines
from .voices import PROJECT_ROOT

W, H, FPS = 1920, 1080, 30
PANEL_X = 700                     # image panel: x PANEL_X..W, full height
PANEL_W = W - PANEL_X
FEATHER = 460                     # px over which the panel's left edge dissolves
COL_X, COL_W = 110, 780           # text column
KICKER_Y = 210
CAPTION_CY = 975
VIS_BASE, VIS_H, VIS_BARS = 1074, 64, 96
ANIM, XFADE = 0.8, 1.2            # element entrance, scene cross-fade (seconds)
MAX_ZOOM = 1.8
# "layout": "paragraph" — the narration as a page of running text in the left column (whole sentences, as many as
# fit; spoken and current lines white, the rest gray, each line turning white as it is read), and the visualizer as a
# compact mirrored strip in a small box under it (the user's mark-ups of the first two cuts)
PAGE_Y, PAGE_BOTTOM = 178, 742   # the page's text area (top-aligned)
PAGE_SIZE, PAGE_GAP = 50, 1.55   # text size, line height factor
UNREAD = (118, 118, 126)
LINE_FADE, PAGE_FADE = 0.35, 0.4
BOX_VIS = (COL_X, 778, 690, 92)  # x, y, w, h of the small visualizer box
# "layout": "minimal" — the left side left empty (just the smoky backdrop), the visualizer in a slim box under the
# brand mark, and the one-line caption centred at the bottom (the user's third mark-up)
TOP_VIS = (COL_X, 150, 0, 60)  # width 0: as wide as the brand mark above it
BOX_BARS = 46
MUTED = (175, 178, 188)
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


# ---------------------------------------------------------------- images

def panel_mask() -> Image.Image:
    x = np.arange(PANEL_W, dtype=np.float32)
    ramp = np.clip(x / FEATHER, 0, 1) ** 1.6
    return Image.fromarray(np.repeat((ramp * 255).astype(np.uint8)[None, :], H, 0), "L")


def smoky(img: Image.Image) -> Image.Image:
    """Full-frame dark backdrop: the image blurred and darkened, with a vignette and grain."""
    src = img.convert("RGB")
    s = max(W / src.width, H / src.height)
    bg = src.resize((math.ceil(src.width * s), math.ceil(src.height * s)), Image.LANCZOS)
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    bg = bg.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BICUBIC)
    a = np.asarray(bg, np.float32) * 0.22
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W * 0.35) / (W * 0.8)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    a *= np.clip(1.15 - 0.6 * r, 0.35, 1)[..., None]
    a += np.random.default_rng(0).normal(0, 1.6, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(a + 6, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


class Shot:
    """t -> the right-hand panel image (PANEL_W x H) for one scene."""

    def __init__(self, spec: dict, img: Image.Image, t0: float, t1: float):
        self.t0, self.t1, self.card = t0, t1, spec.get("card", False)
        src = img.convert("RGB")
        if self.card:  # whole image as a card on a blurred panel, slowly breathing in
            self.back = smoky(src).crop((PANEL_X, 0, W, H))
            ch = 760  # clear of the caption line
            cw = round(src.width * ch / src.height)
            card = src.resize((cw, ch), Image.LANCZOS).convert("RGBA")
            sh = Image.new("RGBA", (cw + 120, ch + 120), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rectangle((60, 76, 60 + cw, 76 + ch), fill=(0, 0, 0, 200))
            sh = sh.filter(ImageFilter.GaussianBlur(26))
            sh.alpha_composite(card, (60, 60))
            self.cardimg = sh
            return
        self.aspect = PANEL_W / H
        # pre-upscale so the closest zoom still has ~H source pixels of height
        k = max(1.0, H * MAX_ZOOM / src.height)
        self.src = src.resize((round(src.width * k), round(src.height * k)), Image.LANCZOS)
        self.k = k
        self.a, self.b = spec.get("from", [src.width / 2, src.height / 2, 1.0]), spec.get("to") or spec.get("from")

    def view(self, t: float):
        p = ease((t - self.t0) / max(1e-3, self.t1 - self.t0 + XFADE))
        x, y, z = (self.a[i] + (self.b[i] - self.a[i]) * p for i in range(3))
        z = min(MAX_ZOOM, max(1.0, z))
        sw, sh = self.src.width, self.src.height
        vh = min(sh, sw / self.aspect) / z
        vw = vh * self.aspect
        cx, cy = x * self.k, y * self.k
        x0 = min(max(0, cx - vw / 2), sw - vw)
        y0 = min(max(0, cy - vh / 2), sh - vh)
        return x0, y0, vw, vh

    def __call__(self, t: float) -> Image.Image:
        if self.card:
            out = self.back.copy()
            p = ease((t - self.t0) / max(1e-3, self.t1 - self.t0 + XFADE))
            s = 1.0 + 0.035 * p
            c = self.cardimg.resize((round(self.cardimg.width * s), round(self.cardimg.height * s)), Image.BILINEAR)
            out.alpha_composite(c, ((PANEL_W - c.width) // 2 + 60, (H - c.height) // 2 - 70))
            return out
        x0, y0, vw, vh = self.view(t)
        return self.src.resize((PANEL_W, H), Image.BILINEAR, box=(x0, y0, x0 + vw, y0 + vh)).convert("RGBA")


# ---------------------------------------------------------------- text column

def num_pill(n: int, color, ink) -> Image.Image:
    t = text_image(str(n), 34, "Bold", ink, max_w=10_000, line_gap=1.25)
    d = max(t.width + 34, t.height + 10)
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((0, 0, d - 1, d - 1), fill=color + (255,))
    img.alpha_composite(t, ((d - t.width) // 2, (d - t.height) // 2))
    return img


def page_rows(texts: list[str]) -> list[list[tuple[str, int]]]:
    """Wrap a page's lines as one running paragraph; each token is tagged with the index of its line."""
    f = font(PAGE_SIZE, "Bold")
    joined = " ".join(texts)
    ends, pos = [], 0
    for t in texts:
        pos += len(t)
        ends.append(pos)
        pos += 1
    rows, cur = [], 0
    for row in wrap(joined, f, COL_W):
        out = []
        for tok in row:
            k = joined.find(tok.replace("\u00a0", " "), cur)
            k = cur if k < 0 else k
            cur = k + len(tok)
            out.append((tok, next(i for i, e in enumerate(ends) if k < e + 1)))
        rows.append(out)
    return rows


def page_height(rows) -> int:
    return int(PAGE_SIZE * PAGE_GAP) * len(rows) + int(PAGE_SIZE * 0.45)


_PAGE_CACHE: dict = {}


def page_image(pi: int, read: int) -> Image.Image:
    """Page `pi` with lines 0..read white (English words in the accent) and the rest gray, with a soft shadow."""
    key = (pi, read)
    if key in _PAGE_CACHE:
        return _PAGE_CACHE[key]
    rows, accent = _ST["pages"][pi]["rows"], _ST["accent"]
    f = font(PAGE_SIZE, "Bold")
    space = f.getlength(" ")
    line_h = int(PAGE_SIZE * PAGE_GAP)
    pad = 20
    t = Image.new("RGBA", (COL_W + 8, page_height(rows)), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    for r, row in enumerate(rows):
        x, y = 4.0, r * line_h + PAGE_SIZE * 0.22 + PAGE_SIZE
        for tok, li in row:
            col = (accent if LATIN.search(tok) else TEXT) if li <= read else UNREAD
            d.text((x, y), tok, font=f, fill=col, anchor="ls")
            x += f.getlength(tok) + space
    out = Image.new("RGBA", (t.width + 2 * pad, t.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", t.size, (0, 0, 0, 0))
    sh.putalpha(t.getchannel("A").point(lambda v: v * 190 // 255))
    out.alpha_composite(sh, (pad, pad + 3))
    out = out.filter(ImageFilter.GaussianBlur(8))
    out.alpha_composite(t, (pad, pad))
    if len(_PAGE_CACHE) > 64:
        _PAGE_CACHE.clear()
    _PAGE_CACHE[key] = out
    return out


def build_pages(captions: list[str], spans, narration_end: float, scene_lines: set[int],
                sentence_end: list[bool]) -> list[dict]:
    """Greedy pages of whole sentences that fit the text area; a page never crosses a scene start. A sentence too
    long for one page is split at its line breaks."""
    room = PAGE_BOTTOM - PAGE_Y
    sentences, cur = [], []
    for i, c in enumerate(captions):
        if cur and (i + 1) in scene_lines:
            sentences.append(cur)
            cur = []
        cur.append(i)
        if sentence_end[i]:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    fits = lambda idx: page_height(page_rows([captions[j] for j in idx])) <= room  # noqa: E731
    units = []  # sentences, with any too long for one page split into balanced chunks at line breaks
    for sen in sentences:
        k = 1
        while True:
            size = -(-len(sen) // k)
            chunks = [sen[j:j + size] for j in range(0, len(sen), size)]
            if all(fits(c) for c in chunks) or size == 1:
                break
            k += 1
        units += chunks
    pages, page = [], []
    for u in units:
        if page and ((u[0] + 1) in scene_lines or not fits(page + u)):
            pages.append(page)
            page = []
        page += u
    if page:
        pages.append(page)
    out = []
    for k, lines in enumerate(pages):
        t1 = spans[pages[k + 1][0]][0] if k + 1 < len(pages) else narration_end + 3
        out.append({"lines": lines, "t0": spans[lines[0]][0], "t1": t1,
                    "starts": [spans[i][0] for i in lines], "rows": page_rows([captions[i] for i in lines])})
    return out


def draw_box_visualizer(frame: Image.Image, levels: np.ndarray, accent, box=BOX_VIS) -> None:
    """Mirrored bars (up and down from a centre line) inside `box`, sampling BOX_BARS of the analysed bands."""
    x0, y0, w, h = box
    bars = BOX_BARS if w >= 600 else max(20, w // 12)  # keep bars a comfortable width in a narrow box
    layer = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    idx = np.linspace(2, len(levels) - 12, bars).astype(int)  # skip the emptiest top bands
    step = w / bars
    bw = step * 0.52
    mid = 20 + h / 2
    top_col = tuple(min(255, c + 70) for c in accent)
    for i, v in enumerate(levels[idx]):
        half = 3 + v * (h / 2 - 3)
        x = 20 + i * step + (step - bw) / 2
        col = tuple(int(accent[k] + (top_col[k] - accent[k]) * min(1.0, v * 1.3)) for k in range(3))
        d.rounded_rectangle((x, mid - half, x + bw, mid + half), bw / 2, fill=col + (240,))
    out = Image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(8)), layer)
    frame.alpha_composite(out, (x0 - 20, y0 - 20))


def column(scene: dict, at, accent, ink) -> list[tuple[Image.Image, int, int, float]]:
    """(image, x, y, t_in) for the scene's kicker, title, subtitle, points or quote."""
    t0 = at(scene["line"])
    els, y = [], KICKER_Y
    if scene["type"] == "quote":
        mark = text_image("“", 150, "Bold", accent, line_gap=1.0)
        els.append((mark, COL_X - 8, y - 30, t0))
        y += 120
        bar_top = y
        for text, line in scene["phrases"]:
            ph = text_image(text, scene.get("size", 54) - 2, "Bold", TEXT, align="left", max_w=COL_W - 40, line_gap=1.42)
            els.append((ph, COL_X + 36, y, at(line)))
            y += ph.height + 6
        bar = Image.new("RGBA", (8, max(8, y - bar_top)), accent + (255,))
        els.append((bar, COL_X, bar_top, at(scene["phrases"][0][1]) if scene["phrases"] else t0))
        if who := scene.get("by"):
            nm = text_image(who["name"], 40, "Bold", accent, align="left", max_w=COL_W, line_gap=1.3)
            rl = text_image(who["role"], 30, "Medium", MUTED, align="left", max_w=COL_W, line_gap=1.3)
            els += [(nm, COL_X + 36, y + 26, at(who["line"])), (rl, COL_X + 36, y + 26 + nm.height, at(who["line"]) + 0.15)]
        return els
    kicker = text_image(scene["kicker"], 38, "Bold", accent, align="left", max_w=COL_W, line_gap=1.3)
    els.append((kicker, COL_X, y, t0))
    y += kicker.height + 4
    title = text_image(scene["headline"], 80, "Bold", TEXT, align="left", max_w=COL_W, line_gap=1.34)
    els.append((title, COL_X - 4, y, t0 + 0.15))
    y += title.height + 10
    if sub := scene.get("subtitle"):
        s = text_image(sub["text"], 44, "Medium", accent, align="left", max_w=COL_W, line_gap=1.35)
        els.append((s, COL_X, y, at(sub["line"])))
        y += s.height + 20
    rows = scene.get("points", [])
    gap = 22 if len(rows) <= 3 else 12
    for i, (text, line) in enumerate(rows, 1):
        pill = num_pill(i, accent, ink)
        txt = text_image(text, 44, "Medium", TEXT, align="left", max_w=COL_W - pill.width - 26, line_gap=1.35)
        row = Image.new("RGBA", (pill.width + 22 + txt.width, max(pill.height, txt.height)), (0, 0, 0, 0))
        row.alpha_composite(pill, (0, (row.height - pill.height) // 2))
        row.alpha_composite(txt, (pill.width + 22, (row.height - txt.height) // 2))
        els.append((row, COL_X, y, at(line) + 0.1 * (i - 1)))
        y += row.height + gap
    if y > CAPTION_CY - 70:
        print(f"  warning: scene at line {scene['line']} reaches y={y}, into the caption area")
    return els


def caption_image(text: str, accent) -> Image.Image:
    t = text_image(text, 46, "Bold", TEXT, max_w=1500, highlight=accent, line_gap=1.4)
    pad = 20
    out = Image.new("RGBA", (t.width + 2 * pad, t.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", t.size, (0, 0, 0, 0))
    sh.putalpha(t.getchannel("A").point(lambda v: v * 235 // 255))
    out.alpha_composite(sh, (pad, pad + 3))
    out = out.filter(ImageFilter.GaussianBlur(8))
    out.alpha_composite(t, (pad, pad))
    return out


def lockup(lk: dict, accent) -> Image.Image:
    name = text_image(lk["name"], 40, "Bold", TEXT, max_w=10_000, line_gap=1.25)
    tag = text_image(lk.get("tag", ""), 28, "Medium", accent, max_w=10_000, line_gap=1.25) if lk.get("tag") else None
    w = name.width + (tag.width + 34 if tag else 0)
    img = Image.new("RGBA", (w, max(name.height, tag.height if tag else 0)), (0, 0, 0, 0))
    img.alpha_composite(name, (0, (img.height - name.height) // 2))
    if tag:
        ImageDraw.Draw(img).line((name.width + 16, img.height * 0.3, name.width + 16, img.height * 0.7),
                                 fill=(255, 255, 255, 90), width=2)
        img.alpha_composite(tag, (name.width + 34, (img.height - tag.height) // 2))
    return img


# ---------------------------------------------------------------- audio visualizer

def analyse(audio: np.ndarray, sr: int, n_frames: int, bars: int = VIS_BARS) -> np.ndarray:
    """Per-frame spectrum levels 0..1 (log-spaced bands, per-band normalised, fast attack / slow release)."""
    win = 4096
    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(60, 9000, bars + 1)
    idx = np.clip(np.searchsorted(freqs, edges), 1, len(freqs) - 1)
    pad = np.pad(audio.astype(np.float32), win)
    raw = np.zeros((n_frames, bars), np.float32)
    for n in range(n_frames):
        c = int(n / FPS * sr) + win
        spec = np.abs(np.fft.rfft(pad[c - win // 2:c + win // 2] * hann))
        cs = np.concatenate([[0], np.cumsum(spec)])
        hi = np.maximum(idx[1:], idx[:-1] + 1)
        raw[n] = (cs[hi] - cs[idx[:-1]]) / (hi - idx[:-1])
    db = 20 * np.log10(raw + 1e-6)
    lo, hi = np.percentile(db, 10, axis=0), np.percentile(db, 99.5, axis=0)
    level = np.clip((db - lo) / (hi - lo + 1e-6), 0, 1) ** 1.5
    out = np.zeros_like(level)
    for n in range(n_frames):
        prev = out[n - 1] if n else level[0]
        out[n] = prev + (level[n] - prev) * np.where(level[n] > prev, 0.6, 0.16)
    return out


def draw_visualizer(frame: Image.Image, levels: np.ndarray, accent) -> None:
    layer = Image.new("RGBA", (W, VIS_H + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    step = W / VIS_BARS
    bw = step * 0.5
    base = VIS_H + 20
    top_col = tuple(min(255, c + 70) for c in accent)
    for i, v in enumerate(levels):
        h = 4 + v * VIS_H
        x = i * step + (step - bw) / 2
        mix = min(1.0, v * 1.3)
        col = tuple(int(accent[k] + (top_col[k] - accent[k]) * mix) for k in range(3))
        d.rounded_rectangle((x, base - h, x + bw, base), bw / 2, fill=col + (235,))
    glow = layer.filter(ImageFilter.GaussianBlur(10))
    out = Image.alpha_composite(glow, layer)
    frame.alpha_composite(out, (0, VIS_BASE - base))


# ---------------------------------------------------------------- frames

_ST: dict = {}


def scene_frame(i: int, t: float) -> Image.Image:
    st = _ST
    sc = st["scenes"][i]
    frame = st["backs"][sc["image"]].copy()
    panel = sc["shot"](t)
    frame.paste(panel.convert("RGB"), (PANEL_X, 0), st["mask"])
    frame.alpha_composite(st["scrim"])
    for img, x, y, t_in in sc["els"]:
        if t < t_in:
            continue
        p = ease((t - t_in) / ANIM)
        frame.alpha_composite(with_alpha(img, p), (x, y + int(18 * (1 - p))))
    return frame


def render_frame(n: int) -> bytes:
    st = _ST
    t = n / FPS
    scenes = st["scenes"]
    i = max(k for k, sc in enumerate(scenes) if sc["t0"] <= t + 1e-6) if t >= scenes[0]["t0"] else 0
    frame = scene_frame(i, t)
    nxt = i + 1
    if nxt < len(scenes) and t > scenes[nxt]["t0"] - XFADE:  # cross-fade into the next scene
        a = ease((t - (scenes[nxt]["t0"] - XFADE)) / XFADE)
        frame = Image.blend(frame, scene_frame(nxt, max(t, scenes[nxt]["t0"])), a)
    frame.alpha_composite(st["lockup"], (COL_X, 70))
    if st["layout"] == "paragraph":
        for pi, pg in enumerate(st["pages"]):
            if pg["t0"] - 0.01 <= t < pg["t1"]:
                read = max(k for k, s0 in enumerate(pg["starts"]) if s0 <= t + 1e-6) if t >= pg["starts"][0] else -1
                img = page_image(pi, read)
                since = t - pg["starts"][read] if read >= 0 else 1.0
                if read >= 0 and since < LINE_FADE:  # the line being started fades from gray to white
                    img = Image.blend(page_image(pi, read - 1), img, ease(since / LINE_FADE))
                a = min(ease((t - pg["t0"]) / PAGE_FADE) if pi else 1.0, ease((pg["t1"] - t) / PAGE_FADE))
                frame.alpha_composite(with_alpha(img, a), (COL_X - 24, PAGE_Y - 20))
                break
        draw_box_visualizer(frame, st["levels"][min(n, len(st["levels"]) - 1)], st["accent"])
    elif st["burn"]:  # one-line caption centred at the bottom ("points" and "minimal")
        for start, end, img in st["captions"]:
            if start <= t < end:
                a = ease((t - start) / 0.25)
                frame.alpha_composite(with_alpha(img, a), (W // 2 - img.width // 2, CAPTION_CY - img.height // 2))
                break
    if st["layout"] == "minimal":
        x, y, w, h = TOP_VIS
        draw_box_visualizer(frame, st["levels"][min(n, len(st["levels"]) - 1)], st["accent"],
                            (x, y, w or st["lockup"].width, h))
    elif st["layout"] == "points":
        draw_visualizer(frame, st["levels"][min(n, len(st["levels"]) - 1)], st["accent"])
    fade = min(1.0, (st["total"] - t) / 1.5)  # fade to black at the end
    if fade < 1:
        frame = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), frame, max(0.0, fade))
    return frame.convert("RGB").tobytes()


def srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def prepare(sb: dict, root: Path, seconds: float | None = None) -> tuple[Path, int]:
    spans, narration_end = line_timings(sb, root)
    at = lambda line: spans[line - 1][0]  # noqa: E731
    accent = hex_rgb(sb.get("accent", "#D4A72C"))
    lum = (0.299 * accent[0] + 0.587 * accent[1] + 0.114 * accent[2]) / 255
    ink = (20, 18, 12) if lum > 0.6 else (255, 255, 255)
    images = {k: Image.open(root / f) for k, f in sb["images"].items()}
    total = narration_end + sb.get("tail", 4)
    if seconds:
        total = min(total, seconds)
    starts = [at(s["line"]) for s in sb["scenes"]] + [total]
    layout = sb.get("layout", "points")
    scenes = []
    for k, s in enumerate(sb["scenes"]):
        if s["type"] not in ("headline", "quote"):
            raise SystemExit(f"youtube style supports headline and quote scenes, not {s['type']!r}")
        key = s["shot"]["image"]
        els = column(s, at, accent, ink) if layout == "points" else []
        scenes.append({"t0": starts[k] if k else 0.0, "image": key, "els": els,
                       "shot": Shot(s["shot"], images[key], starts[k], starts[k + 1])})
    ramp = np.clip((np.arange(H, dtype=np.float32) - (900 if layout == "paragraph" else 820)) / 260, 0, 1) ** 1.4 \
        * (120 if layout == "paragraph" else 215)
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    scrim.putalpha(Image.fromarray(np.repeat(ramp[:, None], W, 1).astype(np.uint8), "L"))
    pages = []
    if layout == "paragraph":
        ends = [line.endswith(("។", "?", "!", "៕")) for line in read_lines(root / sb["tts_text"])]
        pages = build_pages(sb["captions"], spans, narration_end, {s["line"] for s in sb["scenes"]}, ends)
        captions = []
    else:
        captions = [(s, spans[i + 1][0] if i + 1 < len(spans) else narration_end, caption_image(c, accent))
                    for i, (c, (s, _)) in enumerate(zip(sb["captions"], spans))]
    _ST.update(scenes=scenes, backs={k: smoky(v) for k, v in images.items()}, mask=panel_mask(), scrim=scrim,
               lockup=lockup(sb["lockup"], accent), captions=captions, burn=sb.get("burn_captions", True), layout=layout,
               pages=pages,
               accent=accent, total=total)
    return total, spans, narration_end


def write_srt(sb: dict, spans, narration_end: float, path: Path) -> None:
    rows = []
    for i, (c, (s, e)) in enumerate(zip(sb["captions"], spans), 1):
        end = min(spans[i][0] if i < len(spans) else narration_end, e + 0.6)
        rows.append(f"{i}\n{srt_time(s)} --> {srt_time(end)}\n{c}\n")
    path.write_text("\n".join(rows), encoding="utf-8")


def render(storyboard_path: Path, out: Path | None = None, workers: int | None = None, seconds: float | None = None,
           narration: str | None = None):
    root = PROJECT_ROOT
    sb = load_storyboard(storyboard_path, narration)
    out = out or root / sb["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    total, spans, narration_end = prepare(sb, root, seconds)
    audio = out.with_suffix(".audio.wav")
    mix_audio(root / sb["narration"] / "article.wav", [], total, audio, music=sb.get("music"))  # no sound effects
    mixed, sr = sf.read(str(audio), dtype="float32")
    n_frames = int(total * FPS)
    _ST["levels"] = analyse(mixed if mixed.ndim == 1 else mixed.mean(1), sr, n_frames)
    write_srt(sb, spans, narration_end, out.with_suffix(".srt"))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(audio),
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-af", LOUDNORM, "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    ctx = mp.get_context("fork")
    with ctx.Pool(workers or default_workers(1.2)) as pool:
        for i, frame in enumerate(pool.imap(render_frame, range(n_frames), chunksize=8)):
            ff.stdin.write(frame)
            if i % (FPS * 30) == 0:
                print(f"  frame {i}/{n_frames} ({i / FPS:.0f}s)", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    audio.unlink()
    return out, total, 0


def render_still(storyboard_path: Path, t: float, out: Path, narration: str | None = None) -> None:
    sb = load_storyboard(storyboard_path, narration)
    prepare(sb, PROJECT_ROOT)
    n = int(t * FPS)
    _ST["levels"] = np.tile(np.linspace(0.2, 0.8, VIS_BARS) * (0.6 + 0.4 * np.sin(np.arange(VIS_BARS) / 3)) ** 2,
                            (n + 1, 1))
    Image.frombytes("RGB", (W, H), render_frame(n)).save(out)


# ---------------------------------------------------------------- thumbnail (1280x720)

def make_thumbnail(sb: dict, root: Path = PROJECT_ROOT, out: Path | None = None) -> Path:
    """Storyboard "thumbnail": {image, view: [x0, y0, height] (image px; width follows 16:9), lines: [[text, size,
    "accent"|"white"], ...], badge, place}. The image fills the frame, a dark fade on the left carries a big bold
    title, the brand mark sits top-left and an optional badge pill sits under the title. "place": "bottom" puts the
    title along the bottom over a dark fade from below instead (for images whose subject stands on the left)."""
    th = sb["thumbnail"]
    TW, TH = 1280, 720
    accent = hex_rgb(sb.get("accent", "#D4A72C"))
    lum = (0.299 * accent[0] + 0.587 * accent[1] + 0.114 * accent[2]) / 255
    ink = (20, 18, 12) if lum > 0.6 else (255, 255, 255)
    src = Image.open(root / sb["images"][th["image"]]).convert("RGB")
    x0, y0, vh = th["view"]
    vw = vh * 16 / 9
    img = src.resize((TW, TH), Image.LANCZOS, box=(x0, y0, x0 + vw, y0 + vh))
    img = img.filter(ImageFilter.UnsharpMask(2, 70, 2)).convert("RGBA")
    x = np.arange(TW, dtype=np.float32) / TW
    y = np.arange(TH, dtype=np.float32)[:, None] / TH
    bottom = th.get("place") == "bottom"
    if bottom:  # dark from below (title band), plus a light top-left corner for the brand mark
        alpha = np.maximum(np.clip((y - 0.42) / 0.38, 0, 1) ** 1.2 * 235 + 0 * x[None, :],
                           np.clip((0.35 - x[None, :]) / 0.35, 0, 1) * np.clip((0.22 - y) / 0.22, 0, 1) * 150)
    else:
        fade = np.clip((0.66 - x) / 0.36, 0, 1) ** 1.3 * 235      # dark on the left, clear by ~66%
        alpha = np.maximum(fade[None, :], np.clip((y - 0.72) / 0.28, 0, 1) * 150)
    shade = Image.new("RGBA", (TW, TH), (8, 8, 10, 0))
    shade.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    img.alpha_composite(shade)
    mark = lockup(sb["lockup"], accent)
    mark = mark.resize((int(mark.width * 0.8), int(mark.height * 0.8)), Image.LANCZOS)
    img.alpha_composite(mark, (54, 40))
    titles = [(text_image(text, size, "Bold", accent if colour == "accent" else (255, 255, 255), align="left",
                          max_w=10_000, line_gap=1.18)) for text, size, colour in th["lines"]]
    tight = th.get("tight", 0.86)
    if bottom:
        ty = TH - 46 - int(sum(t.height * tight for t in titles[:-1])) - titles[-1].height
    else:
        ty = th.get("top", 120)
    for t in titles:
        # heavy soft shadow + thin dark outline so the title reads on any image, at any size
        sh = Image.new("RGBA", (t.width + 60, t.height + 60), (0, 0, 0, 0))
        sil = Image.new("RGBA", t.size, (0, 0, 0, 0))
        sil.putalpha(t.getchannel("A"))
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
            sh.alpha_composite(sil, (30 + dx, 30 + dy))
        glow = sh.filter(ImageFilter.GaussianBlur(10))
        glow.alpha_composite(sh)
        glow.alpha_composite(t, (30, 30))
        img.alpha_composite(glow, (54 - 30, ty - 30))
        ty += int(t.height * tight)
    if badge := th.get("badge"):
        b = text_image(badge, 34, "Bold", ink, max_w=10_000, line_gap=1.25)
        pill = Image.new("RGBA", (b.width + 52, b.height + 22), (0, 0, 0, 0))
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), radius=pill.height // 2,
                                               fill=accent + (255,))
        pill.alpha_composite(b, (26, 11))
        img.alpha_composite(pill, (54, ty + 18))
    out = out or (root / sb["output"]).with_name("thumbnail.jpg")
    img.convert("RGB").save(out, quality=92)
    return out
