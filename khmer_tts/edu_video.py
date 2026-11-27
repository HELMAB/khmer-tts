"""Paper "quote card" video (after the "edu" Facebook reels, e.g. facebook.com/reel/1190247251551392): a short photo
intro with a big outlined title, then the advice itself as whole paragraphs on light crumpled paper above one still
monochrome picture (a silhouette works best), each point blurring into the next, a thin audio waveform, a small
hashtag, and a credits card on a darkened photo at the end. Calm, quiet, no sound effects; the music bed is ducked.

Storyboard ("style": "edu"; example videos/builders/mother_merit.py -> videos/mother_merit_edu.json):
    tts_text, narration, output, music, tail (seconds after the narration, default 4)
    aspect     "feed" (default, 1080x1350, Facebook/Instagram 4:5), "vertical" (1080x1920, TikTok/Reels) or
               "landscape" (1920x1080, YouTube: a wide text band on top, the picture centred below)
    accent     hex colour of the title, hashtag and logo tint (default deep blue #2B3A8C)
    tag        small text bottom-right on the pages, e.g. "#រឿងអប់រំ" (optional)
    logo       image file shown top-right on title/outro scenes (optional)
    images     {key: file}
    art        image key of the default page picture (each page may set its own "art")
    captions   one per TTS line; pages show them as text, title scenes as a caption; an .srt is always written
    read_along true: sentences not yet read are light gray (default false: the whole page in one ink colour)
    scenes     {"type": "title", "line", "image", "title"?, "focus"?: [x, y] (image px), "zoom"?: [from, to]}
               {"type": "page",  "line", "art"?}     text = the captions from this line to the next scene's line - 1
               {"type": "outro", "line", "image", "credits": ["...", ...]}
    font       page-text font: a font file (path from the project root, .ttf/.otf) or an installed font's family name
               (resolved with fc-match; a silent fallback to another font is refused). Default Google Sans. Titles,
               captions and credits stay Google Sans. The user asked for a handwriting font here (like the reel's).
    font_gap   line spacing for that font (default 1.62; handwriting fonts with tall stacks may need ~1.75)
Otherwise text is Google Sans (the user's rule).
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .article import read_lines
from .video import default_workers, hex_rgb, line_timings, load_storyboard, mix_audio, text_image, with_alpha
from .voices import PROJECT_ROOT
from .youtube_video import LOUDNORM, analyse, ease, write_srt

FPS = 30
SIZES = {"feed": (1080, 1350), "vertical": (1080, 1920), "landscape": (1920, 1080)}
W, H = SIZES["feed"]
PAPER = (236, 235, 231)
INK = (52, 52, 58)
UNREAD = (165, 165, 170)
XFADE = 0.6          # cross-fade between scenes of different kinds (photo <-> paper)
BLUR_OUT, BLUR_IN = 0.35, 0.5   # page -> page: the old text blurs away, the new one comes into focus
VIS_BARS = 72


def layout_for(aspect: str) -> dict:
    """Page geometry per aspect: the text above the picture, the waveform in the gap between them (landscape: a
    wide text band across the top and the picture centred below, so a 16:9 picture shows ~800 px wide)."""
    w, h = SIZES[aspect]
    if aspect == "landscape":
        return dict(
            W=w, H=h, TOP_Y=60, TEXT_MAX=50,
            TEXT_TOP=80, TEXT_BOTTOM=505, TEXT_W=1320, TEXT_CX=w // 2,      # wide band across the top
            ART_X0=260, ART_X1=1660, ART_TOP=585, ART_BOTTOM=h - 40,        # picture centred below
            WAVE=(w // 2, 545, 1000),                                       # centre x, centre y, width
            CAPTION_Y=h - 110, TAG_XY=(w - 60, h - 30),
        )
    tall = aspect == "vertical"
    art_top = int(h * (0.52 if tall else 0.56))
    return dict(
        W=w, H=h, TOP_Y=330 if tall else 70, TEXT_MAX=46,
        TEXT_TOP=330 if tall else 150,                  # page text area (vertical: below TikTok's top bar)
        TEXT_BOTTOM=int(h * (0.50 if tall else 0.54)),
        TEXT_W=880, TEXT_CX=w // 2,
        ART_X0=int(w * 0.07), ART_X1=int(w * 0.93),
        ART_TOP=art_top,                                # page picture area, down to ART_BOTTOM
        ART_BOTTOM=h - (500 if tall else 70),          # vertical: above the tag and TikTok's caption block
        WAVE=(w // 2, art_top - 34, int(w * 0.62)),    # in the gap between the text and the picture
        CAPTION_Y=h - (520 if tall else 150),          # title scenes: caption centre
        TAG_XY=(w - 60, h - (430 if tall else 46)),    # hashtag, right-aligned, baseline (vertical: under the picture)
    )


globals().update(layout_for("feed"))


# ---------------------------------------------------------------- paper and pictures

def paper(w: int, h: int, seed: int = 7) -> Image.Image:
    """Light crumpled paper: flat facets (Voronoi cells, each tilted to the light by its own small gradient, so the
    folds show as straight light/dark edges), a softer second layer of smaller facets, fine grain, darker edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def facets(n: int, depth: float) -> np.ndarray:
        s = 4  # work at quarter size, then upscale smoothly
        gy, gx = np.mgrid[0:h // s, 0:w // s].astype(np.float32) * s
        pts = rng.random((n, 2)) * [w, h]
        d = (gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2
        cell = d.argmin(-1)
        tilt = rng.normal(0, 1, (n, 2)) * depth / 180  # brightness change per px inside each facet
        base = rng.normal(0, depth * 0.35, n)
        f = base[cell] + tilt[cell, 0] * (gx - pts[cell, 0]) + tilt[cell, 1] * (gy - pts[cell, 1])
        return np.asarray(Image.fromarray(f.astype(np.float32), "F").resize((w, h), Image.BILINEAR), np.float32)

    shade = facets(max(40, w * h // 20000), 4.5) + facets(max(110, w * h // 6000), 2.0)
    as_l = Image.fromarray(np.clip(shade * 2 + 128, 0, 255).astype(np.uint8), "L")  # PIL can't blur float images
    shade = (np.asarray(as_l.filter(ImageFilter.GaussianBlur(2.2)), np.float32) - 128) / 2
    vign = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 26
    img = np.array(PAPER, np.float32)[None, None, :] + (shade - vign + rng.normal(0, 1.8, (h, w)))[..., None]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.6))
    return out.convert("RGBA")


def page_back(base: Image.Image, art: Image.Image | None) -> Image.Image:
    """The paper with the page picture in the lower area: toned to monochrome, multiplied into the paper so its
    white background disappears, edges feathered so a full painting also sits softly on the page."""
    frame = base.copy()
    if art is None:
        return frame
    box_w, box_h = ART_X1 - ART_X0, ART_BOTTOM - ART_TOP
    a = ImageOps.contain(ImageOps.grayscale(art.convert("RGB")), (box_w, box_h), Image.LANCZOS)
    a = ImageOps.autocontrast(a, cutoff=1)
    ax = ART_X0 + (box_w - a.width) // 2
    ay = ART_BOTTOM - a.height
    region = np.asarray(frame.crop((ax, ay, ax + a.width, ay + a.height)).convert("RGB"), np.float32)
    mult = region * (np.asarray(a, np.float32)[..., None] / 255)
    yy, xx = np.mgrid[0:a.height, 0:a.width].astype(np.float32)
    fx = np.clip(np.minimum(xx, a.width - 1 - xx) / (a.width * 0.12), 0, 1)
    fy = np.clip(np.minimum(yy, a.height - 1 - yy) / (a.height * 0.12), 0, 1)
    m = (fx * fy)[..., None]
    out = region * (1 - m) + mult * m
    frame.paste(Image.fromarray(out.astype(np.uint8), "RGB"), (ax, ay))
    return frame


def cover(img: Image.Image, t: float, focus, zoom) -> Image.Image:
    """Photo filling the frame (cover crop) at push-in progress t (0..1), centred near `focus`."""
    iw, ih = img.size
    z = zoom[0] + (zoom[1] - zoom[0]) * ease(t)
    scale = max(W / iw, H / ih)
    vw, vh = W / scale / z, H / scale / z
    fx, fy = focus or (iw / 2, ih / 2)
    x0 = min(max(fx - vw / 2, 0), iw - vw)
    y0 = min(max(fy - vh / 2, 0), ih - vh)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + vw, y0 + vh)).convert("RGBA")


def outlined(img: Image.Image, color=(255, 255, 255), r: int = 4, shadow: bool = True) -> Image.Image:
    """Text image with a solid outline (the reel's white-edged title) and a soft drop shadow."""
    pad = r + 14
    out = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    a = img.getchannel("A")
    if shadow:
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 150), (pad + 3, pad + 5), a)
        out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(7)))
    sil = Image.new("RGBA", img.size, color + (255,))
    sil.putalpha(a)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out.alpha_composite(sil, (pad + dx, pad + dy))
    out.alpha_composite(img, (pad, pad))
    return out


def resolve_font(spec: str | None, root: Path) -> str | None:
    """storyboard "font" -> a font file path (None = Google Sans)."""
    if not spec:
        return None
    if spec.lower().endswith((".ttf", ".otf", ".ttc")):
        path = (root / spec).expanduser()
        if not path.exists():
            raise SystemExit(f'edu style: font file not found: {path}')
        return str(path)
    out = subprocess.run(["fc-match", "-f", "%{family}\t%{file}", spec], capture_output=True, text=True).stdout
    family, _, path = out.partition("\t")
    if spec.lower() not in family.lower():
        raise SystemExit(f'edu style: font "{spec}" is not installed (fc-match offered "{family}"); '
                         f'install it or give the .ttf path')
    return path


def fit_text(sentences: list[str], colors: list, max_h: int) -> Image.Image:
    """Sentences stacked as centred paragraphs, the largest size (TEXT_MAX -> 30 px) whose block fits max_h."""
    face, gap_ratio = _ST.get("face"), _ST.get("font_gap", 1.62)
    for size in range(TEXT_MAX, 29, -2):
        parts = [text_image(s, size, "Regular", c, max_w=TEXT_W, line_gap=gap_ratio, face=face)
                 for s, c in zip(sentences, colors)]
        gap = int(size * 0.55)
        h = sum(p.height for p in parts) + gap * (len(parts) - 1)
        if h <= max_h:
            break
    img = Image.new("RGBA", (max(p.width for p in parts), h), (0, 0, 0, 0))
    y = 0
    for p in parts:
        img.alpha_composite(p, ((img.width - p.width) // 2, y))
        y += p.height + gap
    return img


# ---------------------------------------------------------------- state and frames

_ST: dict = {}


def scene_at(t: float) -> int:
    sc = _ST["scenes"]
    return max(k for k, s in enumerate(sc) if s["t0"] <= t + 1e-6) if t >= sc[0]["t0"] else 0


def page_text(sc: dict, t: float) -> Image.Image:
    if not _ST["read_along"]:
        return sc["text"][-1]
    k = max([i for i, s0 in enumerate(sc["sent_starts"]) if s0 <= t + 1e-6], default=0)
    return sc["text"][k]


def base_frame(i: int, t: float, text_alpha: float = 1.0, blur: float = 0.0) -> Image.Image:
    sc = _ST["scenes"][i]
    if sc["type"] == "page":
        frame = _ST["backs"][sc["art"]].copy()
        txt = page_text(sc, t)
        if blur > 0.05:
            pad = int(blur * 3) + 2
            big = Image.new("RGBA", (txt.width + 2 * pad, txt.height + 2 * pad), (0, 0, 0, 0))
            big.alpha_composite(txt, (pad, pad))
            txt, off = big.filter(ImageFilter.GaussianBlur(blur)), pad
        else:
            off = 0
        y = TEXT_TOP + (TEXT_BOTTOM - TEXT_TOP - txt.height + 2 * off) // 2 - off
        frame.alpha_composite(with_alpha(txt, text_alpha), (TEXT_CX - txt.width // 2, y))
        return frame
    prog = (t - sc["t0"]) / max(0.1, sc["t1"] - sc["t0"])
    frame = cover(_ST["photos"][sc["image"]], prog, sc.get("focus"), sc["zoom"])
    frame.alpha_composite(_ST["shade_title"] if sc["type"] == "title" else _ST["shade_outro"])
    for img, x, y in sc["els"]:
        frame.alpha_composite(img, (x, y))
    if sc["type"] == "title":
        for start, end, img in _ST["captions"]:
            if start <= t < end:
                frame.alpha_composite(with_alpha(img, ease(min(1, (t - start) / 0.25))),
                                      ((W - img.width) // 2, CAPTION_Y - img.height // 2))
                break
    if _ST["logo"] is not None:
        frame.alpha_composite(_ST["logo"], (W - _ST["logo"].width - 48, TOP_Y - 22))
    return frame


def draw_wave(frame: Image.Image, levels: np.ndarray, on_paper: bool) -> None:
    """Thin mirrored waveform with a dotted baseline (the reel's faint line across the picture)."""
    cx, cy, width = WAVE if on_paper else (W // 2, CAPTION_Y - 110, int(min(W, H) * 0.62))
    x0 = cx - width // 2
    col = INK + (90,) if on_paper else (255, 255, 255, 170)
    layer = Image.new("RGBA", (W, 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x in range(x0, x0 + width, 9):
        d.ellipse((x - 1, 39, x + 1, 41), fill=col[:3] + (col[3] // 2,))
    step = width / VIS_BARS
    taper = np.sin(np.linspace(0.15, np.pi - 0.15, VIS_BARS)) ** 0.7  # quiet at the ends, like a waveform
    for k, v in enumerate(levels):
        h = 1 + v * taper[k] * 30
        x = x0 + k * step + step / 2
        d.line((x, 40 - h, x, 40 + h), fill=col, width=2)
    frame.alpha_composite(layer, (0, cy - 40))


def render_frame(n: int) -> bytes:
    st = _ST
    t = n / FPS
    scenes = st["scenes"]
    i = scene_at(t)
    nxt = i + 1 if i + 1 < len(scenes) else None
    cur, nx = scenes[i], scenes[nxt] if nxt is not None else None
    if nx and cur["type"] == nx["type"] == "page" and t > nx["t0"] - BLUR_OUT:  # old text blurs away
        p = ease((t - (nx["t0"] - BLUR_OUT)) / BLUR_OUT)
        frame = base_frame(i, t, 1 - p, 8 * p)
        if cur["art"] != nx["art"]:
            frame = Image.blend(frame, base_frame(nxt, nx["t0"], 0.0), p)
    elif nx and not (cur["type"] == nx["type"] == "page") and t > nx["t0"] - XFADE:  # photo <-> paper
        frame = Image.blend(base_frame(i, t), base_frame(nxt, nx["t0"]), ease((t - (nx["t0"] - XFADE)) / XFADE))
    elif cur["type"] == "page" and i and scenes[i - 1]["type"] == "page" and t < cur["t0"] + BLUR_IN:
        p = ease((t - cur["t0"]) / BLUR_IN)  # new text comes into focus
        frame = base_frame(i, t, p, 8 * (1 - p))
    else:
        frame = base_frame(i, t)
    on_paper = cur["type"] == "page"
    if cur["type"] != "outro":
        draw_wave(frame, st["levels"][min(n, len(st["levels"]) - 1)], on_paper)
    if on_paper and st["tag"] is not None:
        frame.alpha_composite(st["tag"], (TAG_XY[0] - st["tag"].width, TAG_XY[1] - st["tag"].height))
    fade = min(1.0, (st["total"] - t) / 1.5)
    if fade < 1:
        frame = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), frame, max(0.0, fade))
    return frame.convert("RGB").tobytes()


# ---------------------------------------------------------------- build

def prepare(sb: dict, root: Path, seconds: float | None = None):
    aspect = sb.get("aspect", "feed")
    if aspect not in SIZES:
        raise SystemExit(f'edu style: "aspect" must be one of {list(SIZES)}, not {aspect!r}')
    globals().update(layout_for(aspect))  # workers are forked after this, so they see it
    spans, narration_end = line_timings(sb, root)
    at = lambda line: spans[line - 1][0]  # noqa: E731
    accent = hex_rgb(sb.get("accent", "#2B3A8C"))
    total = narration_end + sb.get("tail", 4)
    if seconds:
        total = min(total, seconds)
    _ST.update(face=resolve_font(sb.get("font"), root), font_gap=sb.get("font_gap", 1.62))  # before the pages
    ends = [l.endswith(("។", "?", "!", "៕")) for l in read_lines(root / sb["tts_text"])]
    images = {k: Image.open(root / f) for k, f in sb.get("images", {}).items()}
    base = paper(W, H, sb.get("paper_seed", 7))
    starts = [at(s["line"]) for s in sb["scenes"]] + [total]
    scenes, arts = [], set()
    for k, s in enumerate(sb["scenes"]):
        kind = s["type"]
        sc = {"type": kind, "t0": starts[k] if k else 0.0, "t1": starts[k + 1]}
        if kind == "page":
            first, last = s["line"], (sb["scenes"][k + 1]["line"] - 1 if k + 1 < len(sb["scenes"]) else len(spans))
            sents, cur, sent_starts = [], [], []
            for ln in range(first, last + 1):
                if not cur:
                    sent_starts.append(at(ln))
                cur.append(sb["captions"][ln - 1])
                if ends[ln - 1] or ln == last:
                    sents.append(" ".join(cur))
                    cur = []
            # one image per reading state (read_along) or just the finished page
            states = range(len(sents)) if sb.get("read_along") else [len(sents) - 1]
            sc["text"] = [fit_text(sents, [INK if j <= r else UNREAD for j in range(len(sents))],
                                   TEXT_BOTTOM - TEXT_TOP) for r in states]
            sc["sent_starts"] = sent_starts
            sc["art"] = s.get("art", sb.get("art"))
            arts.add(sc["art"])
        elif kind in ("title", "outro"):
            sc.update(image=s["image"], focus=s.get("focus"), zoom=s.get("zoom", [1.0, 1.08]), els=[])
            if kind == "title" and s.get("title"):
                for size in range(96, 55, -4):
                    t_img = text_image(s["title"], size, "Bold", accent, max_w=W - 120, line_gap=1.3)
                    if t_img.width <= W - 120:
                        break
                t_img = outlined(t_img)
                sc["els"].append((t_img, (W - t_img.width) // 2, TOP_Y))
            if kind == "outro":
                rows = [outlined(text_image(c, 46, "Bold", (255, 255, 255), max_w=W - 160, line_gap=1.3),
                                 color=(0, 0, 0), r=0) for c in s.get("credits", [])]
                gap = 18
                y = (H - sum(r.height for r in rows) - gap * (len(rows) - 1)) // 2
                for r in rows:
                    sc["els"].append((r, (W - r.width) // 2, y))
                    y += r.height + gap
        else:
            raise SystemExit(f"edu style supports title, page and outro scenes, not {kind!r}")
        scenes.append(sc)
    backs = {a: page_back(base, images[a] if a else None) for a in arts}
    ramp = np.clip((np.arange(H, dtype=np.float32) - H * 0.55) / (H * 0.4), 0, 1) ** 1.2 * 200
    shade_title = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shade_title.putalpha(Image.fromarray(np.repeat(ramp[:, None], W, 1).astype(np.uint8), "L"))
    shade_outro = Image.new("RGBA", (W, H), (0, 0, 0, 175))
    captions = []
    for idx, (c, (s0, _)) in enumerate(zip(sb["captions"], spans)):
        end = spans[idx + 1][0] if idx + 1 < len(spans) else narration_end
        captions.append((s0, end, outlined(text_image(c, 46, "Bold", (255, 255, 255), max_w=W - 140, line_gap=1.4),
                                           color=(0, 0, 0), r=0)))
    tag = text_image(sb["tag"], 30, "Medium", accent, max_w=10_000, line_gap=1.3) if sb.get("tag") else None
    logo = None
    if sb.get("logo"):
        logo = Image.open(root / sb["logo"]).convert("RGBA")
        logo = ImageOps.contain(logo, (150, 90), Image.LANCZOS)
    photos = {k: ImageEnhance.Color(v.convert("RGB")).enhance(0.9) for k, v in images.items()}
    _ST.update(scenes=scenes, backs=backs, photos=photos, shade_title=shade_title, shade_outro=shade_outro,
               captions=captions, tag=tag, logo=logo, read_along=bool(sb.get("read_along")), total=total)
    return total, spans, narration_end


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
    _ST["levels"] = analyse(mixed if mixed.ndim == 1 else mixed.mean(1), sr, n_frames, VIS_BARS)
    write_srt(sb, spans, narration_end, out.with_suffix(".srt"))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(audio),
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-af", LOUDNORM, "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    with mp.get_context("fork").Pool(workers or default_workers(1.2)) as pool:
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
    wave = np.abs(np.sin(np.arange(VIS_BARS) / 2.3)) * np.linspace(0.4, 0.9, VIS_BARS)
    _ST["levels"] = np.tile(wave, (n + 1, 1))
    Image.frombytes("RGB", (W, H), render_frame(n)).save(out)
