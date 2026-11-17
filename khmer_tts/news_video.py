"""Broadcast-news style vertical video: storyboards with "style": "news" (see videos/mercedes_rma_news.json).

Layout (1080x1920, kept clear of TikTok's UI and side crop on tall phones, see SAFE_TOP etc. in video.py):
    full-bleed article photos with slow Ken Burns zoom/pan and cross-fades, over a blurred copy of the photo
    top-left channel bug + date, optional disclosure tag top-right
    key-fact chips stacked above a lower third (red tag, white headline bar, navy subtitle strip)
    captions under the lower third, scrolling ticker with a progress line
    an animated intro stinger before the narration and an outro card with credits after it

Scene fields: line, tag, title, subtitle, images [[file, line], ...], facts [[text, line], ...], numbered.
Landscape photos are shown whole (fit) over the blurred fill; square/portrait ones fill the photo area.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

from .video import (CROP_X, FPS, H, HIGHLIGHT, LOUDNORM, RAIL_X, SAFE_TOP, W, back_out, ease_out, line_timings, load_storyboard,
                    synth_sfx, text_image)
from .voices import PROJECT_ROOT

L, R = CROP_X + 45, RAIL_X        # inside the side crop and left of TikTok's right-hand buttons
INTRO, OUTRO, OUTRO_GAP = 3.2, 4.5, 0.3
PHOTO_H = 1250                     # sharp photo area; fades into the blurred fill below ~880
FACTS_BOTTOM, TAG_Y, CAPTION_Y = 975, 1000, 1300
TICKER_Y, TICKER_H, LABEL_W, TICKER_SPEED = 1452, 62, 230, 110

RED, NAVY, INK, WHITE, SOFT = (206, 28, 38), (9, 17, 40), (14, 17, 26), (255, 255, 255), (196, 203, 222)
ANIM, FADE, XFADE = 0.35, 0.3, 0.5


# ---------------------------------------------------------------- drawing helpers

def block(text: str, size: int, fg, bg, weight="Bold", pad_x=22, pad_y=8) -> Image.Image:
    """Square-cornered label, the broadcast look."""
    t = text_image(text, size, weight, fg, max_w=10_000, line_gap=1.3)
    img = Image.new("RGBA", (t.width + 2 * pad_x, t.height + 2 * pad_y), bg + (255,))
    img.alpha_composite(t, (pad_x, pad_y))
    return img


def shadowed(img: Image.Image, r: int = 7, strength: int = 210) -> Image.Image:
    """Soft drop shadow; the result is padded by 2*r on every side."""
    pad = 2 * r
    out = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 255))
    sh.putalpha(img.getchannel("A").point(lambda v: v * strength // 255))
    out.alpha_composite(sh, (pad, pad + 3))
    out = out.filter(ImageFilter.GaussianBlur(r))
    out.alpha_composite(img, (pad, pad))
    return out


def comp(frame: Image.Image, img: Image.Image, x: int, y: int) -> None:
    """alpha_composite that clips to the frame."""
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + img.width), min(H, y + img.height)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (x, y, x + img.width, y + img.height):
        img = img.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    frame.alpha_composite(img, (x0, y0))


def fade(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    return out


def logo_bug(channel: str, size: int = 30) -> Image.Image:
    a = block("NEWS", size, WHITE, RED)
    b = block(channel, size, INK, WHITE)
    img = Image.new("RGBA", (a.width + b.width, max(a.height, b.height)), (0, 0, 0, 0))
    img.alpha_composite(a, (0, (img.height - a.height) // 2))
    img.alpha_composite(b, (a.width, (img.height - b.height) // 2))
    return img


# ---------------------------------------------------------------- timeline

@dataclass
class El:
    img: Image.Image
    x: int
    y: int
    t_in: float
    mode: str = "left"      # left | up | wipe | pop | fade
    t_out: float = 1e9


@dataclass
class Shot:
    t0: float
    t1: float
    src: Image.Image
    bg: Image.Image
    fit: bool
    idx: int


def draw_el(frame: Image.Image, el: El, t: float) -> None:
    if t < el.t_in or t >= el.t_out:
        return
    p = min(1.0, (t - el.t_in) / ANIM)
    e = ease_out(p)
    img, x, y, a = el.img, el.x, el.y, min(1.0, p * 1.8)
    if el.mode == "wipe":
        img, a = img.crop((0, 0, max(1, int(img.width * e)), img.height)), 1.0
    elif el.mode == "pop" and p < 1:
        s = back_out(p) * 0.35 + 0.65
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        x, y = x + (el.img.width - img.width) // 2, y + (el.img.height - img.height) // 2
    elif el.mode == "left":
        x -= int(70 * (1 - e))
    elif el.mode == "up":
        y += int(40 * (1 - e))
    elif el.mode == "fade":
        a = e
    a *= min(1.0, max(0.0, (el.t_out - t) / FADE))
    if a > 0.01:
        comp(frame, fade(img, a), x, y)


def lower_third(sc: dict, t0: float, t_out: float, sfx: list) -> list[El]:
    tag = block(sc["tag"], 30, WHITE, RED)
    title = text_image(sc["title"], 50, "Bold", INK, max_w=R - L - 64, align="left", line_gap=1.4, highlight=RED)
    bar = Image.new("RGBA", (R - L, title.height + 24), WHITE + (255,))
    ImageDraw.Draw(bar).rectangle((0, 0, 11, bar.height), fill=RED)
    bar.alpha_composite(title, (34, 12))
    els = [El(tag, L, TAG_Y, t0, "left", t_out), El(bar, L, TAG_Y + tag.height, t0 + 0.12, "wipe", t_out)]
    bottom = TAG_Y + tag.height + bar.height
    if sub := sc.get("subtitle"):
        s = text_image(sub, 30, "Medium", SOFT, max_w=R - L - 64, align="left", line_gap=1.35)
        strip = Image.new("RGBA", (R - L, s.height + 14), NAVY + (240,))
        strip.alpha_composite(s, (34, 7))
        els.append(El(strip, L, bottom, t0 + 0.3, "fade", t_out))
        bottom += strip.height
    if bottom > CAPTION_Y - 8:
        print(f"  warning: lower third of scene at line {sc['line']} reaches y={bottom}, over the captions")
    sfx += [(t0, "whoosh"), (t0 + 0.12, "swish")]
    return els


def fact_chips(sc: dict, at, t_out: float, sfx: list) -> list[El]:
    """Key facts as dark glass chips, flowing in rows that sit just above the lower third."""
    chips, times, seen = [], [], {}
    for i, (text, line) in enumerate(sc.get("facts", []), 1):
        t = text_image(text, 38, "Medium", WHITE, max_w=R - L - 110, align="left", highlight=HIGHLIGHT, line_gap=1.35)
        lead = 0
        chip = Image.new("RGBA", (t.width + 44 + (62 if sc.get("numbered") else 0), t.height + 14), (0, 0, 0, 0))
        d = ImageDraw.Draw(chip)
        d.rectangle((0, 0, chip.width, chip.height), fill=(8, 12, 26, 215))
        if sc.get("numbered"):
            d.rectangle((0, 0, 50, chip.height), fill=RED)
            n = text_image(str(i), 32, "Bold", WHITE, line_gap=1.2)
            chip.alpha_composite(n, ((50 - n.width) // 2, (chip.height - n.height) // 2))
            lead = 62
        else:
            d.rectangle((0, 0, 7, chip.height), fill=RED)
            lead = 8
        chip.alpha_composite(t, (lead + 14, 7))
        k = seen[line] = seen.get(line, -1) + 1
        chips.append(chip)
        times.append(at(line) + 0.15 * k)
    rows, row, w = [], [], 0
    for c in chips:
        if row and w + 14 + c.width > R - L:
            rows.append(row)
            row, w = [], 0
        w += (14 if row else 0) + c.width
        row.append(c)
    if row:
        rows.append(row)
    y = FACTS_BOTTOM - sum(max(c.height for c in r) for r in rows) - 14 * (len(rows) - 1)
    els, i = [], 0
    for r in rows:
        x = L
        for c in r:
            els.append(El(c, x, y, times[i], "left", t_out))
            sfx.append((times[i], "pop"))
            x += c.width + 14
            i += 1
        y += max(c.height for c in r) + 14
    return els


def band(text: str, size: int, fg, bg, height: int) -> Image.Image:
    img = Image.new("RGBA", (W, height), bg + (255,))
    t = text_image(text, size, "Bold", fg, max_w=W - 2 * L, line_gap=1.35)
    img.alpha_composite(t, ((W - t.width) // 2, (height - t.height) // 2))
    return img


def build_intro(sb: dict, sfx: list) -> list[El]:
    first = sb["scenes"][0]
    red = band(sb.get("channel", "ព័ត៌មាន"), 84, WHITE, RED, 190)
    white = band(" · ".join(x for x in (first.get("tag"), sb.get("date")) if x), 38, INK, WHITE, 96)
    head = shadowed(text_image(first["title"], 66, "Bold", WHITE, max_w=R - L, line_gap=1.4, highlight=HIGHLIGHT))
    sfx += [(0.0, "riser"), (0.5, "impact"), (0.85, "swish")]
    return [El(red, 0, 640, 0.15, "wipe", INTRO), El(white, 0, 830, 0.45, "wipe", INTRO),
            El(head, (L + R - head.width) // 2, 970, 0.85, "up", INTRO)]  # centred left of the button rail


def build_outro(sb: dict, t0: float, total: float, sfx: list) -> list[El]:
    o = sb.get("outro", {})
    bug = logo_bug(sb.get("channel", "ព័ត៌មាន"), 40)
    red = band(o.get("text", "តាមដានសម្រាប់ព័ត៌មានបន្ថែម"), 60, WHITE, RED, 190)
    white = band(o.get("subtitle", "Follow for more news"), 38, INK, WHITE, 96)
    els = [El(bug, (W - bug.width) // 2, 520, t0, "pop", total + 1), El(red, 0, 640, t0 + 0.15, "wipe", total + 1),
           El(white, 0, 830, t0 + 0.4, "wipe", total + 1)]
    y = 980
    for i, line in enumerate(sb.get("credits", [])):
        c = shadowed(text_image(line, 30, "Medium", SOFT, max_w=R - L, line_gap=1.35))
        els.append(El(c, (L + R - c.width) // 2, y, t0 + 0.8 + 0.1 * i, "fade", total + 1))
        y += c.height - 20
    sfx += [(t0, "whoosh"), (t0 + 0.15, "impact"), (t0 + 0.5, "ding")]
    return els


def load_shot(path: Path, t0: float, idx: int) -> Shot:
    src = Image.open(path).convert("RGB")
    fit = src.width / src.height > 1.2
    s = max(W / src.width, H / src.height)
    small = src.resize((int(src.width * s / 4) + 1, int(src.height * s / 4) + 1), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(10)).resize((int(src.width * s), int(src.height * s)), Image.BILINEAR)
    bg = small.crop(((small.width - W) // 2, (small.height - H) // 2, (small.width - W) // 2 + W, (small.height - H) // 2 + H))
    bg = Image.blend(bg, Image.new("RGB", (W, H), NAVY), 0.55).convert("RGBA")
    return Shot(t0, 0, src, bg, fit, idx)


def build_timeline(sb: dict, root: Path):
    spans, narration_end = line_timings(sb, root)
    at = lambda line: INTRO + spans[line - 1][0]  # noqa: E731
    t_outro = INTRO + narration_end + OUTRO_GAP
    total = t_outro + OUTRO
    sfx: list = []
    els = build_intro(sb, sfx)

    img_dir = root / sb.get("images_dir", ".")
    plan = [(0.0, sb["intro_image"])]
    starts = [at(s["line"]) for s in sb["scenes"]] + [t_outro]
    for i, sc in enumerate(sb["scenes"]):
        t0, t1 = starts[i], starts[i + 1]
        els += lower_third(sc, t0, t1, sfx) + fact_chips(sc, at, t1, sfx)
        plan += [(at(line), f) for f, line in sc.get("images", [])]
    plan.append((t_outro, sb.get("outro_image", sb["intro_image"])))
    els += build_outro(sb, t_outro, total, sfx)

    plan.sort(key=lambda p: p[0])
    cache: dict = {}
    shots = []
    for i, (t0, f) in enumerate(plan):
        base = cache.setdefault(f, load_shot(img_dir / f, 0, 0))
        shots.append(Shot(t0, plan[i + 1][0] if i + 1 < len(plan) else total, base.src, base.bg, base.fit, i))

    captions = []
    for i, (c, (s, _)) in enumerate(zip(sb["captions"], spans)):
        end = spans[i + 1][0] if i + 1 < len(spans) else narration_end
        img = text_image(c, 42, "Bold", WHITE, max_w=R - L, align="left", highlight=HIGHLIGHT, line_gap=1.45)
        captions.append((INTRO + s, INTRO + end, shadowed(img)))
    return els, shots, captions, sfx, t_outro, total


# ---------------------------------------------------------------- frames

_ST: dict = {}


def photo_mask(w: int, h: int, feather_top: int, solid_to: int) -> Image.Image:
    col = np.full(h, 255.0)
    if feather_top:
        col[:feather_top] = np.linspace(0, 255, feather_top)
    col[solid_to:] = np.linspace(255, 0, h - solid_to)
    return Image.fromarray(np.repeat(col[:, None], w, axis=1).astype(np.uint8), "L")


def render_shot(sh: Shot, t: float) -> Image.Image:
    frame = sh.bg.copy()
    u = min(1.0, max(0.0, (t - sh.t0) / max(0.1, sh.t1 - sh.t0)))
    z = 1.0 + 0.09 * (u if sh.idx % 2 == 0 else 1 - u)
    sw, sh_ = sh.src.size
    if sh.fit:
        tw, th, y = W, int(W * sh_ / sw), 0
        y = 560 - th // 2
        mask = _ST["masks"].setdefault(th, photo_mask(W, th, 40, th - 40))
    else:
        tw, th, y, mask = W, PHOTO_H, 0, _ST["masks"]["full"]
    s0 = max(tw / sw, th / sh_)
    cw, ch = tw / (s0 * z), th / (s0 * z)
    pan = (u - 0.5) * 0.05 * (1 if sh.idx % 2 else -1)
    cx = min(max(sw * (0.5 + pan), cw / 2), sw - cw / 2)
    cy = sh_ / 2
    crop = sh.src.resize((tw, th), Image.BILINEAR, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
    frame.paste(crop, (0, y), mask)
    return frame


def render_frame(n: int) -> bytes:
    st, t = _ST, n / FPS
    shots = st["shots"]
    i = max(k for k, s in enumerate(shots) if s.t0 <= t + 1e-6)
    frame = render_shot(shots[i], t)
    if i > 0 and t - shots[i].t0 < XFADE:
        frame = Image.blend(render_shot(shots[i - 1], t), frame, (t - shots[i].t0) / XFADE)
    frame.alpha_composite(st["overlay"])
    main = INTRO <= t < st["t_outro"]
    if not main:
        frame.alpha_composite(st["dim"])

    if main:  # persistent chrome: channel bug, date, disclosure
        a = min(1.0, (t - INTRO) / 0.4, (st["t_outro"] - t) / FADE)
        comp(frame, fade(st["bug"], a), L, SAFE_TOP)
        comp(frame, fade(st["date"], a), L - 14, SAFE_TOP + st["bug"].height - 6)
        if st["disclosure"] is not None:
            comp(frame, fade(st["disclosure"], a), R - st["disclosure"].width, SAFE_TOP + 2)

    for el in st["els"]:
        draw_el(frame, el, t)

    if main:
        for start, end, img in st["captions"]:
            if start <= t < end:
                comp(frame, fade(img, min(1.0, (t - start) / 0.12)), L - 14, CAPTION_Y - 14)
                break
        # ticker: navy strip with scrolling headlines, red label, progress line on top
        frame.alpha_composite(st["ticker_bg"], (0, TICKER_Y))
        loop = st["ticker_loop"]
        off = int((t - INTRO) * TICKER_SPEED) % (loop.width // 2)
        frame.alpha_composite(loop.crop((off, 0, off + W - CROP_X - LABEL_W, loop.height)),
                              (CROP_X + LABEL_W, TICKER_Y + (TICKER_H - loop.height) // 2))
        frame.alpha_composite(st["ticker_label"], (0, TICKER_Y))
        d = ImageDraw.Draw(frame)
        p = (t - INTRO) / (st["t_outro"] - INTRO)
        d.rectangle((0, TICKER_Y - 6, W, TICKER_Y - 1), fill=(255, 255, 255, 50))
        d.rectangle((0, TICKER_Y - 6, int(W * p), TICKER_Y - 1), fill=HIGHLIGHT)
    return frame.convert("RGB").tobytes()


def prepare(sb: dict, root: Path):
    els, shots, captions, sfx, t_outro, total = build_timeline(sb, root)
    ov = np.zeros((H, 1, 4), np.float32)
    ov[:, :, :3] = NAVY
    y = np.arange(H)
    ov[:, 0, 3] = np.clip(np.where(y < 380, 150 * (1 - y / 380), 0)
                          + np.clip((y - 820) / 360, 0, 1) * 215, 0, 235)
    overlay = Image.fromarray(np.repeat(ov, W, axis=1).astype(np.uint8), "RGBA")

    label = Image.new("RGBA", (CROP_X + LABEL_W, TICKER_H), RED + (255,))  # text inside the side crop
    lt = text_image(sb.get("ticker_label", "ព័ត៌មានថ្មី"), 30, "Bold", WHITE, line_gap=1.3)
    label.alpha_composite(lt, (CROP_X + (LABEL_W - lt.width) // 2, (TICKER_H - lt.height) // 2))
    tt = text_image("     •     ".join(sb.get("ticker", [])) + "     •     ", 30, "Medium", WHITE,
                    max_w=100_000, align="left", line_gap=1.3)
    reps = math.ceil(W / tt.width) + 1
    loop = Image.new("RGBA", (tt.width * reps * 2, tt.height), (0, 0, 0, 0))
    for k in range(reps * 2):
        loop.alpha_composite(tt, (k * tt.width, 0))
    # the crop window must fit: loop.width // 2 + (W - CROP_X - LABEL_W) <= loop.width
    disclosure = sb.get("disclosure")
    _ST.update(
        els=els, shots=shots, captions=captions, t_outro=t_outro, total=total, overlay=overlay,
        dim=Image.new("RGBA", (W, H), (0, 0, 0, 80)),
        masks={"full": photo_mask(W, PHOTO_H, 0, 880)},
        bug=logo_bug(sb.get("channel", "ព័ត៌មាន")),
        date=shadowed(text_image(sb.get("date", ""), 26, "Medium", WHITE, align="left", line_gap=1.3)),
        disclosure=block(disclosure, 24, WHITE, (60, 66, 84), weight="Medium", pad_x=14, pad_y=6) if disclosure else None,
        ticker_bg=Image.new("RGBA", (W, TICKER_H), NAVY + (245,)), ticker_label=label, ticker_loop=loop,
    )
    return sfx, total


# ---------------------------------------------------------------- audio

def synth(kind: str, sr: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    if kind == "impact":  # low boom + short noise hit
        n = int(1.4 * sr)
        t = np.arange(n) / sr
        boom = np.sin(2 * np.pi * (55 * t + 40 * (1 - np.exp(-t * 8)) / 8)) * np.exp(-t * 3.2)
        hit = rng.standard_normal(n) * np.exp(-t * 30) * 0.35
        return 0.6 * (boom + hit) * np.minimum(1, t * 300)
    if kind == "riser":  # noise swelling and brightening into the impact
        n = int(0.55 * sr)
        noise = rng.standard_normal(n)
        cutoff = np.geomspace(400, 9000, n)
        alpha = 1 - np.exp(-2 * np.pi * cutoff / sr)
        out, y = np.empty(n), 0.0
        for i in range(n):
            y += alpha[i] * (noise[i] - y)
            out[i] = y
        return 0.45 * out / (np.abs(out).max() + 1e-9) * np.linspace(0, 1, n) ** 2
    return synth_sfx(kind, sr)


def mix_audio(narration: Path, sfx: list, total: float, out: Path, sfx_gain_db: float = -11) -> None:
    voice, sr = sf.read(str(narration), dtype="float32")
    mix = np.zeros(int(total * sr) + sr, dtype=np.float32)
    start = int(INTRO * sr)
    mix[start: start + len(voice)] += voice
    gain = 10 ** (sfx_gain_db / 20)
    cache: dict = {}
    for t, kind in sfx:
        s = cache.setdefault(kind, synth(kind, sr).astype(np.float32))
        i = int(t * sr)
        seg = s[: max(0, len(mix) - i)]
        mix[i: i + len(seg)] += gain * seg
    mix = mix[: int(total * sr)]
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    sf.write(str(out), mix, sr)


# ---------------------------------------------------------------- main

def render(storyboard_path: Path, out: Path | None = None, workers: int | None = None, seconds: float | None = None,
           narration: str | None = None):
    root = PROJECT_ROOT
    sb = load_storyboard(storyboard_path, narration)
    out = out or root / sb["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    sfx, total = prepare(sb, root)
    if seconds:
        total = min(total, seconds)
    audio = out.with_suffix(".audio.wav")
    mix_audio(root / sb["narration"] / "article.wav", sfx, total, audio)

    n_frames = int(total * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(audio),
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-af", LOUDNORM + ",alimiter=limit=0.84:level=disabled", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    ctx = mp.get_context("fork")
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
    sb = load_storyboard(storyboard_path, narration)
    prepare(sb, PROJECT_ROOT)
    Image.frombytes("RGB", (W, H), render_frame(int(t * FPS))).save(out)
