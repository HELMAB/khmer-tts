"""Render a vertical (TikTok/Shorts) explainer video from a storyboard JSON and a narrated article.

The narration comes from `khmer_tts read`, whose parts/NN.wav give exact per-line timing, so captions
and on-screen items are synced to the voice without any alignment model. Sound effects are synthesized.

Storyboard (see videos/laravel_spring_boot.json):
    tts_text   prepared text that was read (one line per part)
    narration  read output folder (article.wav + parts/)
    captions   one caption per TTS line, in display spelling (English terms in Latin script)
    brands     a/b: name, color, tag
    scenes     "title" or "compare"; every element names the 1-based line at which it appears
    outro      seconds + text shown after the narration ends
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .article import SENTENCE_ENDINGS, read_lines
from .voices import PROJECT_ROOT

W, H, FPS = 1080, 1920, 30
MARGIN = 70
SHORT_PAUSE, LONG_PAUSE = 0.25, 0.6  # must match article.read_article defaults

# TikTok covers roughly the top 150 px, bottom 400 px and right 130 px with its UI.
HEADER_Y, CARD_A_Y, CARD_B_Y, CARD_H = 250, 470, 850, 350
CAPTION_CY, PROGRESS_Y = 1340, 1480

BG_TOP, BG_BOTTOM = (11, 14, 26), (20, 24, 48)
CARD_BG = (24, 28, 48, 235)
TEXT, MUTED, HIGHLIGHT = (245, 246, 250), (160, 166, 190), (255, 210, 70)
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
    """Split a long unspaced Khmer run into equal-width pieces at cluster boundaries (no 1-letter orphans)."""
    clusters = khmer_clusters(word)
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


def text_image(text: str, size: int, weight="Regular", color=TEXT, max_w=W - 2 * MARGIN,
               highlight=None, align="center", line_gap=1.55) -> Image.Image:
    """Render (wrapped) text to a tight RGBA image. Latin tokens get `highlight` colour if given."""
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


def build_card(brand: dict, items: list, y: int, at, sfx: list, height: int = CARD_H) -> list[Element]:
    color = hex_rgb(brand["color"])
    card = Image.new("RGBA", (W - 2 * MARGIN, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, card.width - 1, height - 1), radius=36, fill=CARD_BG,
                        outline=color + (110,), width=3)
    d.rounded_rectangle((0, 0, 14, height - 1), radius=7, fill=color)
    name = text_image(brand["name"], 58, "Bold", color, line_gap=1.2)
    card.alpha_composite(name, (48, 18))
    tag = pill(brand["tag"], 30, TEXT, color + (60,), outline=color)
    card.alpha_composite(tag, (card.width - tag.width - 34, 34))

    t_card = at(items[0][1]) if items else 0
    elements = [Element(card, MARGIN, y, t_card)]
    sfx.append((t_card, "swish"))

    # chips flow left-to-right, wrapping inside the card; shrink the text until every chip fits
    x0, max_x, top, bottom = MARGIN + 48, MARGIN + card.width - 36, y + 125, y + height - 22
    for size in (48, 44, 40, 36, 33, 30, 27, 24) if height > CARD_H else (36, 33, 30, 27, 24):
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
        t = text_image(num, 52, "Bold", (20, 22, 35), line_gap=1.2)
        badge.alpha_composite(t, ((110 - t.width) // 2, (110 - t.height) // 2))
    title = text_image(scene["title"], 54, "Bold", TEXT, max_w=W - 2 * MARGIN - 140, align="left", line_gap=1.4)
    sub = text_image(scene["subtitle"], 32, "Medium", MUTED, max_w=W - 2 * MARGIN - 140, align="left", line_gap=1.3)
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
    vt = text_image("VS", 64, "Bold", (20, 22, 35), line_gap=1.2)
    vs.alpha_composite(vt, ((150 - vt.width) // 2, (150 - vt.height) // 2))
    b = text_image(brands["b"]["name"], 130, "Bold", b_col, line_gap=1.2)
    els = [
        Element(a, (W - a.width) // 2, 330, t0),
        Element(vs, (W - 150) // 2, 560, t0 + 0.35, dy=0, scale_in=True),
        Element(b, (W - b.width) // 2, 730, t0 + 0.6),
    ]
    sfx += [(t0, "whoosh"), (t0 + 0.35, "pop"), (t0 + 0.6, "swish")]
    if sub := scene.get("subtitle"):
        s = text_image(sub["text"], 50, "Medium", TEXT, highlight=HIGHLIGHT)
        els.append(Element(s, (W - s.width) // 2, 960, at(sub["line"])))
    if topics := scene.get("topics"):
        t = at(topics["line"])
        chips = [pill(f"{i}. {x}", 34, TEXT, (255, 255, 255, 22), outline=HIGHLIGHT + (160,))
                 for i, x in enumerate(topics["items"], 1)]
        rows, row = [], []
        for c in chips:  # two rows max, centred
            if row and sum(r.width + 16 for r in row) + c.width > W - 2 * MARGIN:
                rows.append(row)
                row = []
            row.append(c)
        rows.append(row)
        for r_i, r in enumerate(rows):
            x = (W - (sum(c.width for c in r) + 16 * (len(r) - 1))) // 2
            for c in r:
                els.append(Element(c, x, 1070 + r_i * 86, t, dy=20, scale_in=True))
                sfx.append((t, "pop"))
                x += c.width + 16
                t += 0.12
    return els


def build_headline(scene: dict, brands: dict, at, sfx: list) -> list[Element]:
    """News-style intro: kicker label, big headline, subtitle, then numbered key points."""
    t0 = at(scene["line"])
    accent = hex_rgb(brands["b"]["color"])
    kicker = pill(scene["kicker"], 34, (255, 255, 255), accent + (255,), weight="Bold")
    head = text_image(scene["headline"], 84, "Bold", TEXT, align="left", highlight=HIGHLIGHT, line_gap=1.45)
    els = [Element(kicker, MARGIN, 300, t0, dy=0, scale_in=True), Element(head, MARGIN, 300 + kicker.height + 24, t0 + 0.15)]
    sfx += [(t0, "whoosh"), (t0 + 0.15, "swish")]
    y = 300 + kicker.height + 24 + head.height + 16
    if sub := scene.get("subtitle"):
        s = text_image(sub["text"], 48, "Medium", HIGHLIGHT, align="left")
        els.append(Element(s, MARGIN, y, at(sub["line"])))
        sfx.append((at(sub["line"]), "pop"))
        y += s.height + 26
    color = hex_rgb(brands["a"]["color"])
    for i, (text, line) in enumerate(scene.get("points", []), 1):
        num = pill(str(i), 34, (255, 255, 255), color + (255,), weight="Bold", pad_x=22)
        txt = text_image(text, 42, "Medium", TEXT, align="left", max_w=W - 2 * MARGIN - num.width - 24, line_gap=1.35)
        row = Image.new("RGBA", (num.width + 20 + txt.width, max(num.height, txt.height)), (0, 0, 0, 0))
        row.alpha_composite(num, (0, (row.height - num.height) // 2))
        row.alpha_composite(txt, (num.width + 20, (row.height - txt.height) // 2))
        els.append(Element(row, MARGIN, y, at(line) + 0.1 * (i - 1), dy=24))
        sfx.append((at(line) + 0.1 * (i - 1), "pop"))
        y += row.height + 14
    return els


def build_quote(scene: dict, brands: dict, at, sfx: list) -> list[Element]:
    """A direct quote revealed phrase by phrase, then the attribution."""
    t0 = at(scene["line"])
    color = hex_rgb(brands["b"]["color"])
    mark = text_image("\u201c", 220, "Bold", color, line_gap=1.0)
    els = [Element(mark, MARGIN - 6, 230, t0, dy=0, scale_in=True)]
    sfx.append((t0, "whoosh"))
    bar_top, y = 470, 470
    for text, line in scene["phrases"]:
        ph = text_image(text, 52, "Bold", TEXT, align="left", max_w=W - 2 * MARGIN - 50, line_gap=1.45)
        els.append(Element(ph, MARGIN + 44, y, at(line), dy=20))
        y += ph.height + 4
    bar = Image.new("RGBA", (10, y - bar_top), color + (255,))
    els.insert(1, Element(bar, MARGIN, bar_top, t0, dy=0))
    if who := scene.get("by"):
        name = text_image(who["name"], 44, "Bold", color, align="left", line_gap=1.3)
        role = text_image(who["role"], 34, "Medium", MUTED, align="left", line_gap=1.3)
        t = at(who["line"])
        els += [Element(name, MARGIN + 44, y + 30, t), Element(role, MARGIN + 44, y + 30 + name.height, t + 0.1)]
        sfx.append((t, "pop"))
    return els


def build_timeline(sb: dict, root: Path):
    spans, narration_end = line_timings(sb, root)
    at = lambda line: spans[line - 1][0]  # noqa: E731
    sfx: list[tuple[float, str]] = []
    scenes: list[Scene] = []
    compare_total = sum(1 for s in sb["scenes"] if s["type"] in ("compare", "points") and not s.get("verdict"))
    starts = [at(s["line"]) for s in sb["scenes"]] + [narration_end]
    for i, s in enumerate(sb["scenes"]):
        t0, t1 = starts[i], starts[i + 1]
        kind, counter = s["type"], ""
        if kind == "title":
            els = build_title(s, sb["brands"], at, sfx)
        elif kind == "headline":
            els = build_headline(s, sb["brands"], at, sfx)
        elif kind == "quote":
            els = build_quote(s, sb["brands"], at, sfx)
        elif kind in ("compare", "points"):
            verdict = s.get("verdict", False)
            els = build_header(s, t0, verdict)
            if kind == "compare":
                els += build_card(sb["brands"]["a"], s["a"]["items"], CARD_A_Y, at, sfx)
                els += build_card(sb["brands"]["b"], s["b"]["items"], CARD_B_Y, at, sfx)
            else:  # one tall card filling both card slots
                brand = {"color": sb["brands"]["a"]["color"], **s["card"]}
                els += build_card(brand, s["card"]["items"], CARD_A_Y, at, sfx, height=CARD_B_Y + CARD_H - CARD_A_Y)
            sfx.append((t0, "whoosh"))
            counter = "" if verdict else f"{s['number']} / {compare_total:02d}"
        else:
            raise SystemExit(f"unknown scene type {kind!r} (title, headline, compare, points, quote)")
        scenes.append(Scene(t0, t1, els, counter, captions=s.get("captions", kind != "quote")))

    outro = sb.get("outro")
    total = narration_end + (outro["seconds"] if outro else 0.5)
    if outro:
        t0 = narration_end
        big = text_image(outro["text"], 70, "Bold", TEXT)
        sub = text_image(outro["subtitle"], 40, "Medium", HIGHLIGHT)
        heart = check_icon(150, HIGHLIGHT + (255,))
        scenes.append(Scene(t0, total + 1, [
            Element(heart, (W - 150) // 2, 560, t0, dy=0, scale_in=True),
            Element(big, (W - big.width) // 2, 760, t0 + 0.15),
            Element(sub, (W - sub.width) // 2, 760 + big.height + 10, t0 + 0.3),
        ]))
        sfx += [(t0, "whoosh"), (t0 + 0.15, "ding")]

    captions = [
        (s, (spans[i + 1][0] if i + 1 < len(spans) else narration_end),
         text_image(c, 50, "Bold", TEXT, max_w=W - 2 * MARGIN - 60, highlight=HIGHLIGHT, line_gap=1.5))
        for i, (c, (s, _)) in enumerate(zip(sb["captions"], spans))
    ]
    return scenes, captions, sfx, narration_end, total


# ---------------------------------------------------------------- frames

def make_background(glow_a=(255, 45, 32), glow_b=(109, 179, 63)) -> Image.Image:
    """Larger-than-frame backdrop (gradient, glows, dot grid); frames crop a slowly drifting window."""
    bw, bh = W + 160, H + 160
    grad = np.linspace(0, 1, bh)[:, None, None]
    arr = (np.array(BG_TOP) * (1 - grad) + np.array(BG_BOTTOM) * grad).astype(np.uint8)
    bg = Image.fromarray(np.repeat(arr, bw, axis=1), "RGB").convert("RGBA")
    glow = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    d.ellipse((-250, -150, 650, 750), fill=glow_a + (70,))
    d.ellipse((bw - 650, bh - 950, bw + 250, bh - 50), fill=glow_b + (70,))
    bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(160)))
    dots = ImageDraw.Draw(bg)
    for y in range(0, bh, 54):
        for x in range(0, bw, 54):
            dots.ellipse((x, y, x + 3, y + 3), fill=(255, 255, 255, 18))
    return bg


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
    frame = st["bg"].crop((ox, oy, ox + W, oy + H))

    # top overlay: series badge + section counter
    frame.alpha_composite(st["badge"], (MARGIN, 150))
    show_captions = True
    for sc in st["scenes"]:
        if sc.start - 0.001 <= t < sc.end:
            fade_out = min(1.0, max(0.0, (sc.end - t) / SCENE_FADE)) if t < st["narration_end"] else 1.0
            for el in sc.elements:
                paste(frame, el, t, fade_out)
            show_captions = show_captions and sc.captions
            if sc.counter:
                frame.alpha_composite(st["counters"][sc.counter], (W - MARGIN - st["counters"][sc.counter].width, 150))

    # caption box
    for start, end, img in st["captions"] if show_captions else ():
        if start <= t < end:
            a = min(1.0, (t - start) / 0.12)
            box_w, box_h = W - 2 * MARGIN, img.height + 40
            box = st["caption_boxes"].setdefault(box_h, _caption_box(box_w, box_h))
            y = CAPTION_CY - box_h // 2
            frame.alpha_composite(with_alpha(box, a), (MARGIN, y))
            frame.alpha_composite(with_alpha(img, a), ((W - img.width) // 2, y + 20))
            break

    # progress bar
    d = ImageDraw.Draw(frame)
    p = min(1.0, t / st["total"])
    d.rounded_rectangle((MARGIN, PROGRESS_Y, W - MARGIN, PROGRESS_Y + 8), radius=4, fill=(255, 255, 255, 40))
    if p > 0.005:
        d.rounded_rectangle((MARGIN, PROGRESS_Y, MARGIN + int((W - 2 * MARGIN) * p), PROGRESS_Y + 8),
                            radius=4, fill=HIGHLIGHT)
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


def mix_audio(narration: Path, sfx: list, total: float, out: Path, sfx_gain_db: float = -10) -> None:
    voice, sr = sf.read(str(narration), dtype="float32")
    mix = np.zeros(int(total * sr) + sr, dtype=np.float32)
    mix[: len(voice)] += voice
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
    out = out or root / sb["output"]
    out.parent.mkdir(parents=True, exist_ok=True)

    scenes, captions, sfx, narration_end, total = build_timeline(sb, root)
    if seconds:
        total = min(total, seconds)
    badge = pill(sb["title"], 30, TEXT, (255, 255, 255, 26), outline=(255, 255, 255, 60))
    counters = {sc.counter: pill(sc.counter, 30, (20, 22, 35), HIGHLIGHT + (255,), weight="Bold")
                for sc in scenes if sc.counter}
    _STATE.update(bg=make_background(hex_rgb(sb["brands"]["a"]["color"]), hex_rgb(sb["brands"]["b"]["color"])), badge=badge, counters=counters, scenes=scenes, captions=captions,
                  caption_boxes={}, narration_end=narration_end, total=total)

    audio = out.with_suffix(".audio.wav")
    mix_audio(root / sb["narration"] / "article.wav", sfx, total, audio)

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
    with ctx.Pool(workers or max(1, mp.cpu_count() - 2)) as pool:
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
    scenes, captions, sfx, narration_end, total = build_timeline(sb, PROJECT_ROOT)
    badge = pill(sb["title"], 30, TEXT, (255, 255, 255, 26), outline=(255, 255, 255, 60))
    counters = {sc.counter: pill(sc.counter, 30, (20, 22, 35), HIGHLIGHT + (255,), weight="Bold")
                for sc in scenes if sc.counter}
    _STATE.update(bg=make_background(hex_rgb(sb["brands"]["a"]["color"]), hex_rgb(sb["brands"]["b"]["color"])), badge=badge, counters=counters, scenes=scenes, captions=captions,
                  caption_boxes={}, narration_end=narration_end, total=total)
    Image.frombytes("RGB", (W, H), render_frame(int(t * FPS))).save(out)
