"""Render a vertical (TikTok/Shorts) explainer video from a storyboard JSON and a narrated article.

The narration comes from `khmer_tts read`, whose parts/NN.wav give exact per-line timing, so captions
and on-screen items are synced to the voice without any alignment model. Sound effects are synthesized.

Storyboard (see videos/laravel_spring_boot.json):
    tts_text   prepared text that was read (one line per part)
    narration  read output folder (article.wav + parts/)
    captions   one caption per TTS line, in display spelling (English terms in Latin script)
    brands     a/b: name, color, tag
    scenes     "title" or "compare"; every element names the 1-based line at which it appears
    outro      seconds + text shown after the narration ends (or from TTS line `line`, a spoken sign-off)
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from .article import SENTENCE_ENDINGS, read_lines
from .voices import PROJECT_ROOT

W, H, FPS = 1080, 1920, 30
SHORT_PAUSE, LONG_PAUSE = 0.25, 0.6  # must match article.read_article defaults

# TikTok's UI, measured on an iPhone 14 Pro Max (19.5:9): the video is scaled to fill the height, cropping
# ~55 px off each side; the status + search bar cover the top ~225 px, the button rail covers x > ~905 for
# y ~900-1610, and the username/description cover everything below ~1620.
CROP_X, SAFE_TOP, RAIL_X, SAFE_BOTTOM = 55, 240, 880, 1600
# explainer content column: inside the side crop, left of the button rail; centred items use its centre CX
MARGIN, CONTENT_R = CROP_X + 45, RAIL_X
CONTENT_W, CX = CONTENT_R - MARGIN, (MARGIN + CONTENT_R) // 2
HEADER_Y, CARD_A_Y, CARD_B_Y, CARD_H = 330, 480, 850, 350
CAPTION_CY = 1340

BG_TOP, BG_BOTTOM = (8, 9, 15), (12, 14, 26)
CARD_BG = (24, 28, 48, 235)
TEXT, MUTED, HIGHLIGHT = (245, 246, 250), (160, 166, 190), (255, 210, 70)
ACCENT_INK = (20, 22, 35)  # text drawn on the accent colour (badges, counters)
ANIM = 0.4  # seconds for an element to slide/fade in
SCENE_FADE = 0.3
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"  # TikTok/Reels-level loudness

COENG = "្"
LATIN = re.compile(r"[A-Za-z0-9]")


# ---------------------------------------------------------------- text

@lru_cache(maxsize=None)
def font_path() -> str:
    out = subprocess.run(["fc-match", "-f", "%{file}", "Google Sans"], capture_output=True, text=True)
    path = out.stdout.strip()
    if "GoogleSans" not in path:
        raise SystemExit("Google Sans not found (fc-match). Install it to ~/.local/share/fonts.")
    return path


@lru_cache(maxsize=None)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    if not features.check("raqm"):  # without it Khmer clusters render unshaped (coeng stacks fall apart)
        raise SystemExit("Pillow has no Raqm text shaping. On macOS: brew install fribidi && "
                         "sudo ln -s /opt/homebrew/lib/libfribidi.dylib /usr/local/lib/")
    f = ImageFont.truetype(font_path(), size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_name(weight)
    return f


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def khmer_clusters(word: str) -> list[str]:
    """Split at grapheme-cluster starts (a base letter not attached by a coeng) so wraps never break a stack."""
    out = []
    for i, ch in enumerate(word):
        base = "ក" <= ch <= "ឳ" and not (i and word[i - 1] == COENG)
        if out and base:
            out.append(ch)
        elif out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def split_balanced(word: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Split a long unspaced Khmer run into equal-width pieces, between words where a Khmer word
    segmenter is installed (khmer-nltk), else at cluster boundaries (never inside a letter stack)."""
    from .prepare import khmer_words
    units = khmer_words(word)
    if len(units) < 2 or any(f.getlength(u) > max_w for u in units):
        units = khmer_clusters(word)
    clusters = units
    total = f.getlength(word)
    n = max(1, math.ceil(total / max_w))
    if n == 1:
        n = 2  # only called when the run doesn't fit the current line; halve it rather than orphan
    target, pieces, buf = total / n, [], ""
    for cl in clusters:
        if buf and len(pieces) < n - 1 and f.getlength(buf + cl) > target + f.getlength(cl) / 2:
            pieces.append(buf)
            buf = ""
        buf += cl
    pieces.append(buf)
    return pieces


def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[list[str]]:
    """Greedy wrap on spaces; Khmer runs wider than a line are split between clusters."""
    lines, cur = [], []
    space = f.getlength(" ")

    def width(tokens):
        return sum(f.getlength(t) for t in tokens) + space * max(0, len(tokens) - 1)

    words = []
    for word in text.split(" "):
        if words and LATIN.search(word) and LATIN.search(words[-1]) and len(words[-1]) + len(word) < 24:
            words[-1] += "\u00a0" + word
        elif word:
            words.append(word)
    for word in words:
        pieces = [word]
        room = max_w - (width(cur) + space if cur else 0)
        if f.getlength(word) > max_w or (cur and f.getlength(word) > room and f.getlength(word) > max_w * 0.6):
            pieces = split_balanced(word, f, max_w)
        for piece in pieces:
            if cur and width(cur + [piece]) > max_w:
                lines.append(cur)
                cur = []
            cur.append(piece)
    if cur:
        lines.append(cur)
    # Khmer has no spaces between words, so a long run may need to be split to balance two lines.
    if len(lines) == 1 and width(lines[0]) > max_w:
        lines = [[p] for p in lines[0]]
    return lines


def text_image(text: str, size: int, weight="Regular", color=TEXT, max_w=CONTENT_W,
               highlight=None, align="center", line_gap=1.55) -> Image.Image:
    """Render (wrapped) text to a tight RGBA image. Latin tokens get `highlight` colour if given."""
    if not text.strip():
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    f = font(size, weight)
    lines = wrap(text, f, max_w)
    space = f.getlength(" ")
    line_h = int(size * line_gap)
    widths = [sum(f.getlength(t) for t in ln) + space * (len(ln) - 1) for ln in lines]
    img_w = int(max(widths)) + 8
    pad = int(size * 0.45)  # Khmer stacks reach well above/below the Latin cap height/baseline
    img = Image.new("RGBA", (img_w, line_h * len(lines) + pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (ln, lw) in enumerate(zip(lines, widths)):
        x = {"center": (img_w - lw) / 2, "left": 4}[align]
        y = i * line_h + pad / 2 + size * 1.0
        for tok in ln:
            c = highlight if highlight and LATIN.search(tok) else color
            d.text((x, y), tok, font=f, fill=c, anchor="ls")
            x += f.getlength(tok) + space
    return img


def pill(text: str, size: int, fg, bg, weight="Medium", pad_x=26, pad_y=14, outline=None) -> Image.Image:
    t = text_image(text, size, weight, fg, max_w=10_000, line_gap=1.3)
    w, h = t.width + 2 * pad_x, t.height + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=bg,
                                          outline=outline, width=3 if outline else 0)
    img.alpha_composite(t, (pad_x, pad_y))
    return img


def check_icon(size: int, color) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, size - 1, size - 1), fill=color)
    w = max(4, size // 9)
    d.line([(size * .28, size * .52), (size * .44, size * .68), (size * .73, size * .34)],
           fill=(255, 255, 255), width=w, joint="curve")
    return img


# ---------------------------------------------------------------- timeline

@dataclass
class Element:
    img: Image.Image
    x: int
    y: int
    t_in: float
    dy: int = 40          # slide distance
    scale_in: bool = False


@dataclass
class Scene:
    start: float
    end: float
    elements: list
    counter: str = ""
    captions: bool = True  # quote scenes show the words themselves, so they hide the caption box


def line_timings(storyboard: dict, root: Path) -> tuple[list[tuple[float, float]], float]:
    lines = read_lines(root / storyboard["tts_text"])
    parts = root / storyboard["narration"] / "parts"
    t, spans = 0.0, []
    for i, line in enumerate(lines, 1):
        dur = sf.info(str(parts / f"{i:02d}.wav")).duration
        pause = LONG_PAUSE if line.endswith(SENTENCE_ENDINGS) else SHORT_PAUSE
        spans.append((t, t + dur))
        t += dur + pause
    if len(storyboard["captions"]) != len(lines):
        raise SystemExit(f"{len(storyboard['captions'])} captions but {len(lines)} TTS lines; they must match 1:1")
    return spans, t


def build_card(brand: dict, items: list, y: int, at, sfx: list, height: int = CARD_H, t_show: float | None = None) -> list[Element]:
    color = hex_rgb(brand["color"])
    card = Image.new("RGBA", (CONTENT_W, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, card.width - 1, height - 1), radius=36, fill=CARD_BG,
                        outline=color + (110,), width=3)
    d.rounded_rectangle((0, 0, 14, height - 1), radius=7, fill=color)
    name = text_image(brand["name"], 58, "Bold", color, line_gap=1.2)
    card.alpha_composite(name, (48, 18))
    tag = pill(brand["tag"], 30, TEXT, color + (60,), outline=color)
    card.alpha_composite(tag, (card.width - tag.width - 34, 34))

    t_card = t_show if t_show is not None else at(items[0][1]) if items else 0  # compare cards: shown from the scene start
    elements = [Element(card, MARGIN, y, t_card)]
    sfx.append((t_card, "swish"))

    # chips flow left-to-right, wrapping inside the card; shrink the text until every chip fits
    x0, max_x, top, bottom = MARGIN + 48, MARGIN + card.width - 36, y + 125, y + height - 22
    for size in (48, 44, 40, 36, 33, 30, 27, 24) if height > CARD_H else (42, 39, 36, 33, 30, 27, 24):
        chips = [_chip(text, size, color, "strong" in style) for text, line, *style in items]
        placed, cx, cy, row_h = [], x0, top, 0
        for chip in chips:
            if cx + chip.width > max_x and cx > x0:
                cx, cy, row_h = x0, cy + row_h + (22 if height > CARD_H else 16), 0
            placed.append((cx, cy))
            cx += chip.width + 16
            row_h = max(row_h, chip.height)
        if cy + row_h <= bottom:
            break
    for chip, (cx, cy), (text, line, *style) in zip(chips, placed, items):
        elements.append(Element(chip, cx, cy, at(line), dy=24, scale_in=True))
        sfx.append((at(line), "ding" if "strong" in style else "pop"))
    return elements


def _chip(text: str, size: int, color, strong: bool) -> Image.Image:
    if not strong:
        return pill(text, size, TEXT, (255, 255, 255, 18), outline=color + (150,))
    chip = pill(text, size + 2, (255, 255, 255), color + (255,), weight="Bold")
    icon = check_icon(chip.height, color + (255,))
    ImageDraw.Draw(icon).ellipse((0, 0, icon.width - 1, icon.height - 1), outline=(255, 255, 255), width=4)
    combo = Image.new("RGBA", (chip.width + icon.width + 12, chip.height), (0, 0, 0, 0))
    combo.alpha_composite(chip)
    combo.alpha_composite(icon, (chip.width + 12, 0))
    return combo


def build_header(scene: dict, t0: float, verdict: bool) -> list[Element]:
    num = scene["number"]
    if num == "✓":
        badge = check_icon(110, HIGHLIGHT + (255,))
    else:
        badge = Image.new("RGBA", (110, 110), (0, 0, 0, 0))
        ImageDraw.Draw(badge).rounded_rectangle((0, 0, 109, 109), radius=28, fill=HIGHLIGHT)
        t = text_image(num, 52, "Bold", ACCENT_INK, line_gap=1.2)
        badge.alpha_composite(t, ((110 - t.width) // 2, (110 - t.height) // 2))
    title = text_image(scene["title"], 54, "Bold", TEXT, max_w=CONTENT_W - 140, align="left", line_gap=1.4)
    sub = text_image(scene["subtitle"], 32, "Medium", MUTED, max_w=CONTENT_W - 140, align="left", line_gap=1.3)
    return [
        Element(badge, MARGIN, HEADER_Y + 20, t0, dy=0, scale_in=True),
        Element(title, MARGIN + 136, HEADER_Y - 4, t0 + 0.08),
        Element(sub, MARGIN + 136, HEADER_Y + title.height - 14, t0 + 0.16),
    ]


def build_title(scene: dict, brands: dict, at, sfx: list) -> list[Element]:
    t0 = at(scene["line"])
    a_col, b_col = hex_rgb(brands["a"]["color"]), hex_rgb(brands["b"]["color"])
    a = text_image(brands["a"]["name"], 150, "Bold", a_col, line_gap=1.2)
    vs = Image.new("RGBA", (150, 150), (0, 0, 0, 0))
    ImageDraw.Draw(vs).ellipse((0, 0, 149, 149), fill=HIGHLIGHT)
    vt = text_image("VS", 64, "Bold", ACCENT_INK, line_gap=1.2)
    vs.alpha_composite(vt, ((150 - vt.width) // 2, (150 - vt.height) // 2))
    b = text_image(brands["b"]["name"], 130, "Bold", b_col, line_gap=1.2)
    els = [
        Element(a, CX - a.width // 2, 330, t0),
        Element(vs, CX - 75, 560, t0 + 0.35, dy=0, scale_in=True),
        Element(b, CX - b.width // 2, 730, t0 + 0.6),
    ]
    sfx += [(t0, "whoosh"), (t0 + 0.35, "pop"), (t0 + 0.6, "swish")]
    if sub := scene.get("subtitle"):
        s = text_image(sub["text"], 50, "Medium", TEXT, highlight=HIGHLIGHT)
        els.append(Element(s, CX - s.width // 2, 960, at(sub["line"])))
    if topics := scene.get("topics"):
        t = at(topics["line"])
        chips = [pill(f"{i}. {x}", 34, TEXT, (255, 255, 255, 22), outline=HIGHLIGHT + (160,))
                 for i, x in enumerate(topics["items"], 1)]
        rows, row = [], []
        for c in chips:  # two rows max, centred
            if row and sum(r.width + 16 for r in row) + c.width > CONTENT_W:
                rows.append(row)
                row = []
            row.append(c)
        rows.append(row)
        for r_i, r in enumerate(rows):
            x = CX - (sum(c.width for c in r) + 16 * (len(r) - 1)) // 2
            for c in r:
                els.append(Element(c, x, 1070 + r_i * 86, t, dy=20, scale_in=True))
                sfx.append((t, "pop"))
                x += c.width + 16
                t += 0.12
    return els


def build_headline(scene: dict, brands: dict, at, sfx: list, root: Path = PROJECT_ROOT) -> list[Element]:
    """News-style intro: kicker label, big headline, subtitle, then numbered key points."""
    t0 = at(scene["line"])
    accent = hex_rgb(brands["b"]["color"])
    kicker = pill(scene["kicker"], 34, (255, 255, 255), accent + (255,), weight="Bold")
    head = text_image(scene["headline"], 84, "Bold", TEXT, align="left", highlight=HIGHLIGHT, line_gap=1.45)
    els = [Element(kicker, MARGIN, 330, t0, dy=0, scale_in=True), Element(head, MARGIN, 330 + kicker.height + 24, t0 + 0.15)]
    sfx += [(t0, "whoosh"), (t0 + 0.15, "swish")]
    y = 330 + kicker.height + 24 + head.height + 16
    if sub := scene.get("subtitle"):
        s = text_image(sub["text"], 48, "Medium", HIGHLIGHT, align="left")
        els.append(Element(s, MARGIN, y, at(sub["line"])))
        sfx.append((at(sub["line"]), "pop"))
        y += s.height + 26
    color = hex_rgb(brands["a"]["color"])
    for i, (text, line) in enumerate(scene.get("points", []), 1):
        num = pill(str(i), 34, (255, 255, 255), color + (255,), weight="Bold", pad_x=22)
        txt = text_image(text, 42, "Medium", TEXT, align="left", max_w=CONTENT_W - num.width - 24, line_gap=1.35)
        row = Image.new("RGBA", (num.width + 20 + txt.width, max(num.height, txt.height)), (0, 0, 0, 0))
        row.alpha_composite(num, (0, (row.height - num.height) // 2))
        row.alpha_composite(txt, (num.width + 20, (row.height - txt.height) // 2))
        els.append(Element(row, MARGIN, y, at(line) + 0.1 * (i - 1), dy=24))
        sfx.append((at(line) + 0.1 * (i - 1), "pop"))
        y += row.height + 14
    if icons := scene.get("icons"):  # a row of icon tiles under the text (place `media_y` below it)
        els += build_icon_row(icons, scene.get("icons_y", y + 20), at, sfx, root)
    return els


def portrait(path: Path, crop, size: int, ring) -> Image.Image:
    """Square photo with rounded corners, an accent ring and a soft shadow (small sources are upscaled)."""
    src = Image.open(path).convert("RGB")
    if crop:
        src = src.crop(tuple(crop))
    side = min(src.size)
    src = src.crop(((src.width - side) // 2, (src.height - side) // 2, (src.width + side) // 2, (src.height + side) // 2))
    src = src.resize((size, size), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 6, fill=255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(src, (0, 0), mask)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 6, outline=ring + (255,), width=6)
    return shadow(img)


PHOTO_Y = 330  # quote portrait header, below the series badge


def build_quote(scene: dict, brands: dict, at, sfx: list, root: Path = PROJECT_ROOT) -> list[Element]:
    """A direct quote revealed phrase by phrase, then the attribution. With `photo` ({file, crop}), the speaker's
    portrait and name open the scene as a header and the quote sits below it."""
    t0 = at(scene["line"])
    color = hex_rgb(brands["b"]["color"])
    who = scene.get("by")
    els: list[Element] = []
    top = 300
    if photo := scene.get("photo"):
        pic = portrait(root / photo["file"], photo.get("crop"), 230, HIGHLIGHT)
        els.append(Element(pic, MARGIN - SHADOW, PHOTO_Y - SHADOW, t0, dy=0, scale_in=True))
        if who:
            nx, w = MARGIN + 260, CONTENT_R - MARGIN - 260
            name = text_image(who["name"], 42, "Bold", HIGHLIGHT, align="left", max_w=w, line_gap=1.35)
            role = text_image(who["role"], 30, "Medium", MUTED, align="left", max_w=w, line_gap=1.3)
            ny = PHOTO_Y + (230 - name.height - role.height) // 2
            els += [Element(name, nx, ny, t0 + 0.15), Element(role, nx, ny + name.height, t0 + 0.25)]
        top, who = PHOTO_Y + 260, None
    mark = text_image("\u201c", 220 if not photo else 160, "Bold", color, line_gap=1.0)
    els.append(Element(mark, MARGIN - 6, top, t0, dy=0, scale_in=True))
    sfx.append((t0, "whoosh"))
    bar_top = y = top + (240 if not photo else 150)
    size = scene.get("size", 52)
    for text, line in scene["phrases"]:
        ph = text_image(text, size, "Bold", TEXT, align="left", max_w=CONTENT_W - 50, line_gap=1.45)
        els.append(Element(ph, MARGIN + 44, y, at(line), dy=20))
        y += ph.height + 4
    bar = Image.new("RGBA", (10, y - bar_top), color + (255,))
    t_bar = at(scene["phrases"][0][1]) if scene["phrases"] else t0  # the bar arrives with the first phrase
    els.insert(len(els) - len(scene["phrases"]), Element(bar, MARGIN, bar_top, t_bar, dy=0))
    if who:
        name = text_image(who["name"], 44, "Bold", color, align="left", line_gap=1.3)
        role = text_image(who["role"], 34, "Medium", MUTED, align="left", line_gap=1.3)
        t = at(who["line"])
        els += [Element(name, MARGIN + 44, y + 30, t), Element(role, MARGIN + 44, y + 30 + name.height, t + 0.1)]
        sfx.append((t, "pop"))
    return els


# ---------------------------------------------------------------- media windows (browser / code / terminal)

BAR_H, WIN_R = 54, 22
WIN_BG, WIN_BAR = (16, 18, 27), (28, 31, 44)
COMMENT = (110, 118, 140)  # comments are also set in italic
CODE_COLORS = [  # first match wins; (pattern, colour)
    (re.compile(r"^# .*"), COMMENT), (re.compile(r"^\$ "), (80, 220, 140)), (re.compile(r"^✓.*"), (80, 220, 140)),
    (re.compile(r"<!--.*?-->"), COMMENT), (re.compile(r'"[^"]*"'), (152, 224, 136)),
    (re.compile(r"</?[A-Za-z!][\w-]*|(?<![-=])/?>"), (255, 121, 168)), (re.compile(r"\b[a-z-]+(?==)"), (120, 190, 255)),
    (re.compile(r"\b(git|vercel|npm)\b"), (120, 190, 255)),
]


@lru_cache(maxsize=None)
def mono(size: int, style: str = "Regular") -> ImageFont.FreeTypeFont:
    """Code font: the first of MONO_FONTS that is installed (fc-match silently substitutes, so check the file)."""
    for family, file_tag in MONO_FONTS:
        out = subprocess.run(["fc-match", "-f", "%{file}", f"{family}:style={style}"], capture_output=True, text=True)
        if file_tag in out.stdout:
            return ImageFont.truetype(out.stdout.strip(), size, layout_engine=ImageFont.Layout.RAQM)
    out = subprocess.run(["fc-match", "-f", "%{file}", "Menlo"], capture_output=True, text=True)
    return ImageFont.truetype(out.stdout.strip(), size, index=1 if style == "Bold" else 0)


MONO_FONTS = [("Dank Mono", "DankMono"), ("Fira Code", "FiraCode")]  # (family, file-name tag), in preference order
SYMBOLS = "✓"  # missing from Dank Mono; drawn with Menlo instead


@lru_cache(maxsize=None)
def symbol_font(size: int) -> ImageFont.FreeTypeFont:
    """Menlo on macOS, DejaVu Sans Mono on Linux (fc-match substitutes silently, so check the glyph)."""
    for family in ("Menlo", "DejaVu Sans Mono", "Noto Sans Symbols 2"):
        path = subprocess.run(["fc-match", "-f", "%{file}", family], capture_output=True, text=True).stdout.strip()
        f = ImageFont.truetype(path, size)
        if all(f.getmask(c).getbbox() and f.getmask(c).size != f.getmask("￿").size for c in SYMBOLS):
            return f
    return f


LIGATURES = ["liga", "calt"]


def window_frame(w: int, h: int, title: str, url: bool = False) -> Image.Image:
    """macOS-style window: rounded body, title bar with traffic lights and a title or address pill."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=WIN_R, fill=WIN_BG, outline=(255, 255, 255, 40), width=2)
    d.rounded_rectangle((1, 1, w - 2, BAR_H), radius=WIN_R, fill=WIN_BAR)
    d.rectangle((1, BAR_H - WIN_R, w - 2, BAR_H), fill=WIN_BAR)
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((24 + i * 30, BAR_H // 2 - 8, 40 + i * 30, BAR_H // 2 + 8), fill=c)
    f = font(24, "Medium")
    tw = f.getlength(title)
    if url:  # address pill with a small lock
        pw = min(w - 260, tw + 70)
        px = (w - pw) // 2
        d.rounded_rectangle((px, 11, px + pw, BAR_H - 11), radius=16, fill=(12, 13, 20))
        d.rounded_rectangle((px + 20, BAR_H // 2 - 2, px + 32, BAR_H // 2 + 8), radius=2, fill=MUTED)
        d.arc((px + 21, BAR_H // 2 - 10, px + 31, BAR_H // 2 + 2), 180, 360, fill=MUTED, width=2)
        d.text((px + 44, BAR_H // 2), title, font=f, fill=TEXT, anchor="lm")
    else:
        d.text((w // 2, BAR_H // 2), title, font=f, fill=MUTED, anchor="mm")
    return img


def browser_window(shot: Image.Image, crop, w: int, url: str) -> tuple[Image.Image, float]:
    """Screenshot region in a browser frame; returns the window and the screenshot->window scale."""
    cx, cy, cw, ch = crop
    s = w / cw
    h = BAR_H + int(ch * s)
    img = window_frame(w, h, url, url=True)
    body = shot.crop((cx, cy, cx + cw, cy + ch)).resize((w - 4, h - BAR_H - 2), Image.LANCZOS)
    mask = Image.new("L", body.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, -WIN_R, body.width - 1, body.height - 1), radius=WIN_R - 2, fill=255)
    img.paste(body.convert("RGB"), (2, BAR_H), mask)
    return img, s


def code_line(text: str, size: int) -> Image.Image:
    f = mono(size)
    img = Image.new("RGBA", (int(f.getlength(text)) + 4, int(size * 1.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    colors = [TEXT] * len(text)
    for pat, col in reversed(CODE_COLORS):  # earlier patterns win, so paint them last
        for m in pat.finditer(text):
            colors[m.start():m.end()] = [col] * (m.end() - m.start())
    # draw runs of one colour together so the font's ligatures (->, &&, ...) can form inside them
    x, i = 0.0, 0
    while i < len(text):
        j = i
        while j < len(text) and colors[j] == colors[i] and (text[j] in SYMBOLS) == (text[i] in SYMBOLS):
            j += 1
        rf = symbol_font(size) if text[i] in SYMBOLS else mono(size, "Italic") if colors[i] == COMMENT else f
        d.text((x, img.height // 2), text[i:j], font=rf, fill=colors[i], anchor="lm", features=LIGATURES)
        x += rf.getlength(text[i:j], features=LIGATURES)
        i = j
    return img


def focus_ring(w: int, h: int) -> Image.Image:
    pad = 14
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((pad, pad, pad + w, pad + h), radius=14, outline=HIGHLIGHT + (255,), width=6)
    glow = img.filter(ImageFilter.GaussianBlur(8))
    glow.alpha_composite(img)
    return glow


def build_media(media: dict, top: int, max_h: int, at, sfx: list, root: Path) -> tuple[list[Element], int]:
    """One window in the content column starting at `top`; returns its elements and its bottom edge.
    kinds: browser {file, crop, url, focus, focus_line}, code/terminal {title, lines [[text, line], ...], min_h}."""
    t = at(media["line"])
    els: list[Element] = []
    if media["kind"] == "browser":
        shot = Image.open(root / media["file"])
        crop = media.get("crop", (0, 0, shot.width, shot.height))
        if BAR_H + crop[3] * CONTENT_W / crop[2] > max_h:  # too tall for the slot: trim the bottom
            crop = (crop[0], crop[1], crop[2], int((max_h - BAR_H) * crop[2] / CONTENT_W))
        win, s = browser_window(shot, crop, CONTENT_W, media["url"])
        els.append(Element(shadow(win), MARGIN - SHADOW, top - SHADOW, t, dy=50))
        if fx := media.get("focus"):
            ring = focus_ring(int(fx[2] * s), int(fx[3] * s))
            fl = at(media.get("focus_line", media["line"])) + 0.35
            els.append(Element(ring, MARGIN + 2 + int((fx[0] - crop[0]) * s) - 14,
                               top + BAR_H + int((fx[1] - crop[1]) * s) - 14, fl, dy=0, scale_in=True))
            sfx.append((fl, "pop"))
        bottom = top + win.height
    else:
        size = media.get("size", 24)
        lines = [(code_line(text, size), line) for text, line in media["lines"]]
        # min_h: as tall as a window it covers, so nothing underneath shows through
        h = min(max_h, max(media.get("min_h", 0), BAR_H + 34 + sum(img.height for img, _ in lines) + 20))
        win = window_frame(CONTENT_W, h, media["title"])
        els.append(Element(shadow(win), MARGIN - SHADOW, top - SHADOW, t, dy=50))
        y = top + BAR_H + 26
        for k, (img, line) in enumerate(lines):
            tl = max(at(line), t + 0.2) + 0.12 * k if line == media["line"] else at(line)
            els.append(Element(img, MARGIN + 30, y, tl, dy=10))
            y += img.height
        bottom = top + h
    sfx.append((t, "swish"))
    return els, bottom


SHADOW = 30


@lru_cache(maxsize=None)
def icon_tile(path: Path, size: int) -> Image.Image:
    """App-icon style tile: an icon (e.g. a PNG from IconScout) centred on a rounded dark card with an accent ring."""
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 4, fill=WIN_BAR + (255,),
                                           outline=HIGHLIGHT + (200,), width=4)
    icon = Image.open(path).convert("RGBA")
    icon.thumbnail((int(size * 0.62), int(size * 0.62)), Image.LANCZOS)
    tile.alpha_composite(icon, ((size - icon.width) // 2, (size - icon.height) // 2))
    return shadow(tile)


def build_icon_row(icons: list, y: int, at, sfx: list, root: Path, size: int = 120) -> list[Element]:
    """A centred row of icon tiles, each popping in at its line: [[file, line], ...]."""
    gap = 28
    x = CX - (len(icons) * size + (len(icons) - 1) * gap) // 2
    els = []
    for file, line in icons:
        els.append(Element(icon_tile(root / file, size), x - SHADOW, y - SHADOW, at(line), dy=0, scale_in=True))
        sfx.append((at(line), "pop"))
        x += size + gap
    return els


def shadow(img: Image.Image) -> Image.Image:
    out = Image.new("RGBA", (img.width + 2 * SHADOW, img.height + 2 * SHADOW), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(img.getchannel("A").point(lambda v: v * 170 // 255))
    out.alpha_composite(sh, (SHADOW, SHADOW + 12))
    out = out.filter(ImageFilter.GaussianBlur(16))
    out.alpha_composite(img, (SHADOW, SHADOW))
    return out


def build_step(scene: dict, brands: dict, at, sfx: list, root: Path) -> list[Element]:
    """Tutorial step: numbered header, one or more windows (each replaces the last) and a row of chips."""
    t0 = at(scene["line"])
    els = build_header(scene, t0, False)
    top, bottom = CARD_A_Y, CARD_A_Y
    media = scene.get("media", [])
    for m in media if isinstance(media, list) else [media]:
        e, b = build_media(m, top, scene.get("media_h", 600), at, sfx, root)
        els += e
        bottom = max(bottom, b)
    # stickers: icon tiles centred on the window's top edge at its right end (over the empty end of the title bar)
    size = 120
    x = CONTENT_R + 30 - size
    for file, line in scene.get("stickers", []):
        els.append(Element(icon_tile(root / file, size), x - SHADOW, top - size // 2 - SHADOW, at(line), dy=0, scale_in=True))
        sfx.append((at(line), "pop"))
        x -= size + 16
    color = hex_rgb(brands["a"]["color"])
    x, y = MARGIN, bottom + 26
    for text, line, *style in scene.get("items", []):
        chip = _chip(text, 30, color, "strong" in style)
        if x + chip.width > CONTENT_R and x > MARGIN:
            x, y = MARGIN, y + chip.height + 12
        els.append(Element(chip, x, y, at(line), dy=20, scale_in=True))
        sfx.append((at(line), "ding" if "strong" in style else "pop"))
        x += chip.width + 12
    return els


def set_accent(sb: dict) -> None:
    """Storyboard "accent" (hex, e.g. a brand colour) replaces the yellow highlight everywhere;
    badge text turns white on dark accents."""
    global HIGHLIGHT, ACCENT_INK
    HIGHLIGHT = hex_rgb(sb["accent"]) if sb.get("accent") else (255, 210, 70)
    lum = (0.299 * HIGHLIGHT[0] + 0.587 * HIGHLIGHT[1] + 0.114 * HIGHLIGHT[2]) / 255
    ACCENT_INK = (20, 22, 35) if lum > 0.6 else (255, 255, 255)


def build_timeline(sb: dict, root: Path):
    set_accent(sb)
    spans, narration_end = line_timings(sb, root)
    at = lambda line: spans[line - 1][0]  # noqa: E731
    sfx: list[tuple[float, str]] = []
    scenes: list[Scene] = []
    compare_total = sum(1 for s in sb["scenes"] if s["type"] in ("compare", "points", "step") and not s.get("verdict"))
    outro = sb.get("outro")
    # outro.line: the spoken sign-off's first TTS line; the outro card comes up as it starts
    t_outro = at(outro["line"]) if outro and outro.get("line") else narration_end
    starts = [at(s["line"]) for s in sb["scenes"]] + [t_outro]
    for i, s in enumerate(sb["scenes"]):
        t0, t1 = starts[i], starts[i + 1]
        kind, counter = s["type"], ""
        if kind == "title":
            els = build_title(s, sb["brands"], at, sfx)
        elif kind == "headline":
            els = build_headline(s, sb["brands"], at, sfx, root)
            if media := s.get("media"):  # a window under the headline, filling the space above the captions
                els += build_media(media, s.get("media_y", 720), 1210 - s.get("media_y", 720), at, sfx, root)[0]
        elif kind == "step":
            els = build_step(s, sb["brands"], at, sfx, root)
            counter = f"{s['number']} / {compare_total:02d}"
        elif kind == "quote":
            els = build_quote(s, sb["brands"], at, sfx, root)
        elif kind in ("compare", "points"):
            verdict = s.get("verdict", False)
            els = build_header(s, t0, verdict)
            if kind == "compare":
                # each card slides in when its brand is first named (its "line"), so it never sits empty
                for key, y, lag in (("a", CARD_A_Y, 0.2), ("b", CARD_B_Y, 0.45)):
                    t_card = max(t0 + lag, at(s[key]["line"]) - 0.15) if "line" in s[key] else t0 + lag
                    els += build_card(sb["brands"][key], s[key]["items"], y, at, sfx, t_show=t_card)
            else:  # one tall card filling both card slots
                brand = {"color": sb["brands"]["a"]["color"], **s["card"]}
                els += build_card(brand, s["card"]["items"], CARD_A_Y, at, sfx, height=CARD_B_Y + CARD_H - CARD_A_Y)
            sfx.append((t0, "whoosh"))
            counter = "" if verdict else f"{s['number']} / {compare_total:02d}"
        else:
            raise SystemExit(f"unknown scene type {kind!r} (title, headline, compare, points, step, quote)")
        scenes.append(Scene(t0, t1, els, counter, captions=s.get("captions", kind != "quote")))

    total = max(t_outro + (outro["seconds"] if outro else 0.5), narration_end + 0.5)
    if outro:
        t0 = t_outro
        big = text_image(outro["text"], 70, "Bold", TEXT)
        sub = text_image(outro["subtitle"], 40, "Medium", HIGHLIGHT)
        # outro.icon: an icon tile (e.g. a rocket) in place of the check mark
        heart = icon_tile(root / outro["icon"], 170) if outro.get("icon") else check_icon(150, HIGHLIGHT + (255,))
        scenes.append(Scene(t0, total + 1, [
            Element(heart, CX - heart.width // 2, 635 - heart.height // 2, t0, dy=0, scale_in=True),
            Element(big, CX - big.width // 2, 760, t0 + 0.15),
            Element(sub, CX - sub.width // 2, 760 + big.height + 10, t0 + 0.3),
        ]))
        sfx += [(t0, "whoosh"), (t0 + 0.15, "ding")]

    captions = [
        (s, (spans[i + 1][0] if i + 1 < len(spans) else narration_end),
         text_image(c, 50, "Bold", TEXT, max_w=CONTENT_W - 60, highlight=HIGHLIGHT, line_gap=1.5))
        for i, (c, (s, _)) in enumerate(zip(sb["captions"], spans))
    ]
    return scenes, captions, sfx, narration_end, total


# ---------------------------------------------------------------- frames

def make_background(glow_a=(255, 45, 32), glow_b=(109, 179, 63)) -> Image.Image:
    """Larger-than-frame backdrop: near-black, a fine grid fading out from the upper middle, two soft brand
    glows and a vignette; frames crop a slowly drifting window."""
    bw, bh = W + 160, H + 160
    y, x = np.mgrid[0:bh, 0:bw].astype(np.float32)
    base = np.array(BG_TOP, np.float32) * (1 - y / bh)[..., None] + np.array(BG_BOTTOM, np.float32) * (y / bh)[..., None]
    # grid lines, strongest around the content and fading towards the edges
    spot = np.exp(-(((x - bw * 0.45) / (bw * 0.55)) ** 2 + ((y - bh * 0.38) / (bh * 0.42)) ** 2))
    line = ((x % 72) < 1.5) | ((y % 72) < 1.5)
    base += (line * spot * 22)[..., None]
    # vignette
    r = np.sqrt(((x - bw / 2) / (bw * 0.75)) ** 2 + ((y - bh / 2) / (bh * 0.7)) ** 2)
    base *= np.clip(1.15 - 0.55 * r, 0.45, 1.0)[..., None]
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    glow = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    d.ellipse((-350, -250, 650, 650), fill=glow_a + (60,))
    d.ellipse((bw - 600, bh - 1000, bw + 350, bh - 100), fill=glow_b + (55,))
    bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(190)))
    return bg


def _dither(rgb: np.ndarray, seed: int = 0) -> Image.Image:
    """Quantise a float image with fine noise so smooth gradients don't band in 8-bit/H.264."""
    noise = np.random.default_rng(seed).normal(0, 1.4, rgb.shape[:2]).astype(np.float32)[..., None]
    return Image.fromarray(np.clip(rgb + noise, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def _blob(x, y, cx, cy, rx, ry):
    return np.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))


def make_spotlight(glow_a, glow_b) -> Image.Image:
    """Black stage, a soft accent light from the top centre, a faint dot grid under it (Vercel-like)."""
    bw, bh = W + 160, H + 160
    y, x = np.mgrid[0:bh, 0:bw].astype(np.float32)
    a, b = np.array(glow_a, np.float32), np.array(glow_b, np.float32)
    img = np.full((bh, bw, 3), 4.0, np.float32)
    img += a * 0.30 * _blob(x, y, bw * 0.5, -bh * 0.05, bw * 0.55, bh * 0.38)[..., None]
    img += b * 0.12 * _blob(x, y, bw * 0.5, bh * 1.02, bw * 0.7, bh * 0.25)[..., None]
    dots = (((x % 36) - 18) ** 2 + ((y % 36) - 18) ** 2 < 2.2).astype(np.float32)
    img += (dots * 42 * _blob(x, y, bw * 0.5, bh * 0.3, bw * 0.6, bh * 0.5))[..., None]
    return _dither(img)


def make_beams(glow_a, glow_b) -> Image.Image:
    """Deep navy gradient crossed by two soft diagonal light beams in the brand colours."""
    bw, bh = W + 160, H + 160
    y, x = np.mgrid[0:bh, 0:bw].astype(np.float32)
    a, b = np.array(glow_a, np.float32), np.array(glow_b, np.float32)
    t = (y / bh)[..., None]
    img = np.array((6, 8, 18), np.float32) * (1 - t) + np.array((3, 4, 10), np.float32) * t
    for col, off, k in ((a, 0.15, 0.40), (b, 0.75, 0.32)):
        d = (x * 0.6 - y * 0.8) / bh + off  # distance across a beam tilted ~37°
        img = img + col * k * np.exp(-(d / 0.11) ** 2)[..., None] * (1 - 0.6 * t)
    return _dither(img)


class Aurora:
    """Animated backdrop: a few large accent glows drifting on black, computed at 1/8 size and upscaled."""
    S = 8

    def __init__(self, glow_a, glow_b):
        self.y, self.x = np.mgrid[0:H // self.S, 0:W // self.S].astype(np.float32) * self.S
        self.cols = [np.array(c, np.float32) for c in (glow_a, glow_b, glow_a)]
        self.grain = np.random.default_rng(0).normal(0, 1.4, (H, W, 1)).astype(np.float32)

    def __call__(self, t: float) -> Image.Image:
        img = np.full(self.x.shape + (3,), 3.0, np.float32)
        for i, col in enumerate(self.cols):
            cx = W * (0.5 + 0.45 * math.sin(t / (9 + 4 * i) + i * 2.1))
            cy = H * (0.25 + 0.3 * i + 0.12 * math.cos(t / (11 + 3 * i) + i))
            img += col * (0.34 if i != 1 else 0.30) * _blob(self.x, self.y, cx, cy, W * 0.55, H * 0.28)[..., None]
        big = np.asarray(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").resize((W, H), Image.BICUBIC),
                         np.float32)
        return Image.fromarray(np.clip(big + self.grain, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def background(sb: dict):
    """Storyboard "background": grid (default), spotlight, beams (static, drifting) or aurora (animated).
    Returns a large still to crop from, or a callable t -> frame-sized image."""
    a, b = hex_rgb(sb["brands"]["a"]["color"]), hex_rgb(sb["brands"]["b"]["color"])
    style = sb.get("background", "grid")
    makers = {"grid": make_background, "spotlight": make_spotlight, "beams": make_beams, "aurora": Aurora}
    if style not in makers:
        raise SystemExit(f"unknown background {style!r} ({', '.join(makers)})")
    return makers[style](a, b)


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def back_out(p: float) -> float:
    c = 1.70158
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


def with_alpha(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    return out


def paste(frame: Image.Image, el: Element, t: float, scene_alpha: float) -> None:
    if t < el.t_in:
        return
    p = min(1.0, (t - el.t_in) / ANIM)
    img, x, y = el.img, el.x, el.y + int(el.dy * (1 - ease_out(p)))
    if el.scale_in and p < 1:
        s = max(0.05, back_out(p) * 0.4 + 0.6)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        x += (el.img.width - img.width) // 2
        y += (el.img.height - img.height) // 2
    a = min(1.0, p * 1.6) * scene_alpha
    if a <= 0.01:
        return
    img = with_alpha(img, a)
    # alpha_composite needs the source fully inside the frame
    if x < 0 or y < 0 or x + img.width > W or y + img.height > H:
        return
    frame.alpha_composite(img, (x, y))


_STATE: dict = {}


def render_frame(n: int) -> bytes:
    st = _STATE
    t = n / FPS
    ox = int(80 + 60 * math.sin(t / 7))
    oy = int(80 + 60 * math.cos(t / 9))
    frame = st["bg"](t) if callable(st["bg"]) else st["bg"].crop((ox, oy, ox + W, oy + H))

    # top overlay: series badge + section counter
    frame.alpha_composite(st["badge"], (MARGIN, SAFE_TOP))
    show_captions = True
    for sc in st["scenes"]:
        if sc.start - 0.001 <= t < sc.end:
            fade_out = min(1.0, max(0.0, (sc.end - t) / SCENE_FADE)) if t < st["narration_end"] else 1.0
            for el in sc.elements:
                paste(frame, el, t, fade_out)
            show_captions = show_captions and sc.captions
            if sc.counter:
                frame.alpha_composite(st["counters"][sc.counter], (CONTENT_R - st["counters"][sc.counter].width, SAFE_TOP))

    # caption box
    for start, end, img in st["captions"] if show_captions else ():
        if start <= t < end:
            a = min(1.0, (t - start) / 0.12)
            box_w, box_h = CONTENT_W, img.height + 40
            box = st["caption_boxes"].setdefault(box_h, _caption_box(box_w, box_h))
            y = CAPTION_CY - box_h // 2
            frame.alpha_composite(with_alpha(box, a), (MARGIN, y))
            frame.alpha_composite(with_alpha(img, a), (CX - img.width // 2, y + 20))
            break
    return frame.convert("RGB").tobytes()


def _caption_box(w: int, h: int) -> Image.Image:
    box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle((0, 0, w - 1, h - 1), radius=28, fill=(8, 10, 20, 200))
    return box


# ---------------------------------------------------------------- sound effects

def synth_sfx(kind: str, sr: int) -> np.ndarray:
    rng = np.random.default_rng(len(kind))
    if kind == "pop":
        n = int(0.09 * sr)
        t = np.arange(n) / sr
        f = 1100 * np.exp(-t * 18) + 500
        return 0.35 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 45)
    if kind == "ding":
        n = int(1.2 * sr)
        t = np.arange(n) / sr
        tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((1318.5, 1), (2637, .35), (1975.5, .25)))
        return 0.22 * tone * np.exp(-t * 4) * np.minimum(1, t * 400)
    # whoosh / swish: noise through a sweeping one-pole low-pass, with a rise-and-fall envelope
    dur = 0.55 if kind == "whoosh" else 0.3
    n = int(dur * sr)
    noise = rng.standard_normal(n)
    cutoff = np.linspace(300, 5000, n) if kind == "whoosh" else np.linspace(1500, 7000, n)
    alpha = 1 - np.exp(-2 * np.pi * cutoff / sr)
    out, y = np.empty(n), 0.0
    for i in range(n):
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return (0.5 if kind == "whoosh" else 0.25) * out * env / (np.abs(out).max() + 1e-9)


def mix_audio(narration: Path, sfx: list, total: float, out: Path, sfx_gain_db: float = -10,
              music: dict | None = None) -> None:
    voice, sr = sf.read(str(narration), dtype="float32")
    mix = np.zeros(int(total * sr) + sr, dtype=np.float32)
    voice = voice[: len(mix)]  # --seconds previews cut the narration short
    mix[: len(voice)] += voice
    if music:  # storyboard "music": {"synth": "ambient"} or {"file": ...}; see music.music_bed
        from .music import music_bed
        mix += music_bed(music, voice, sr, len(mix), PROJECT_ROOT)
    gain = 10 ** (sfx_gain_db / 20)
    cache: dict = {}
    for t, kind in sfx:
        s = cache.setdefault(kind, synth_sfx(kind, sr).astype(np.float32))
        i = int(t * sr)
        seg = s[: max(0, len(mix) - i)]
        mix[i: i + len(seg)] += gain * seg
    mix = mix[: int(total * sr)]
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    sf.write(str(out), mix, sr)


# ---------------------------------------------------------------- main

def default_workers(per_worker_gb: float = 0.7) -> int:
    """Render workers: one per CPU (minus two), but no more than free memory allows
    (each worker holds its own frames; too many get the render killed)."""
    n = max(1, mp.cpu_count() - 2)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                n = min(n, max(1, math.floor(int(line.split()[1]) / 1024 / 1024 / per_worker_gb)))
    except OSError:  # macOS: no /proc
        pass
    return n


def load_storyboard(path: Path, narration: str | None = None) -> dict:
    """Read a storyboard; `narration` swaps in another `khmer_tts read` output (e.g. a different voice)."""
    sb = json.loads(path.read_text(encoding="utf-8"))
    if narration:
        sb["narration"] = narration
    return sb


def render(storyboard_path: Path, out: Path | None = None, workers: int | None = None, seconds: float | None = None,
           narration: str | None = None):
    root = PROJECT_ROOT
    sb = load_storyboard(storyboard_path, narration)
    if sb.get("style") == "news":
        from .news_video import render as render_news
        return render_news(storyboard_path, out, workers, seconds, narration)
    if sb.get("style") == "story":
        from .story_video import render as render_story
        return render_story(storyboard_path, out, workers, seconds)
    out = out or root / sb["output"]
    out.parent.mkdir(parents=True, exist_ok=True)

    scenes, captions, sfx, narration_end, total = build_timeline(sb, root)
    if seconds:
        total = min(total, seconds)
    badge = pill(sb["title"], 30, TEXT, (255, 255, 255, 26), outline=(255, 255, 255, 60))
    counters = {sc.counter: pill(sc.counter, 30, ACCENT_INK, HIGHLIGHT + (255,), weight="Bold")
                for sc in scenes if sc.counter}
    _STATE.update(bg=background(sb), badge=badge, counters=counters, scenes=scenes, captions=captions,
                  caption_boxes={}, narration_end=narration_end, total=total)

    audio = out.with_suffix(".audio.wav")
    mix_audio(root / sb["narration"] / "article.wav", sfx, total, audio, music=sb.get("music"))

    n_frames = int(total * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(audio),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-af", LOUDNORM, "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    ctx = mp.get_context("fork")  # workers inherit the prebuilt layers in _STATE
    with ctx.Pool(workers or default_workers()) as pool:
        for i, frame in enumerate(pool.imap(render_frame, range(n_frames), chunksize=8)):
            ff.stdin.write(frame)
            if i % (FPS * 10) == 0:
                print(f"  frame {i}/{n_frames} ({i / FPS:.0f}s)", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    audio.unlink()
    return out, total, len(sfx)


def render_still(storyboard_path: Path, t: float, out: Path, narration: str | None = None) -> None:
    """One frame at time t, for checking layout without rendering the whole video."""
    sb = load_storyboard(storyboard_path, narration)
    if sb.get("style") == "news":
        from .news_video import render_still as still_news
        return still_news(storyboard_path, t, out, narration)
    if sb.get("style") == "story":
        from .story_video import render_still as still_story
        return still_story(storyboard_path, t, out)
    scenes, captions, sfx, narration_end, total = build_timeline(sb, PROJECT_ROOT)
    badge = pill(sb["title"], 30, TEXT, (255, 255, 255, 26), outline=(255, 255, 255, 60))
    counters = {sc.counter: pill(sc.counter, 30, ACCENT_INK, HIGHLIGHT + (255,), weight="Bold")
                for sc in scenes if sc.counter}
    _STATE.update(bg=background(sb), badge=badge, counters=counters, scenes=scenes, captions=captions,
                  caption_boxes={}, narration_end=narration_end, total=total)
    Image.frombytes("RGB", (W, H), render_frame(int(t * FPS))).save(out)
