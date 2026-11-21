"""`ktts make`: raw text → prepared TTS text → narration → storyboard → video, in one command.

Every step writes a file you can edit, and a re-run reuses what is there:
    texts/<stem>_tts.txt   what the voice reads (edit a line, re-run: only that line is re-voiced)
    videos/<stem>.json     the storyboard (scenes, captions, on-screen facts); kept unless --redo
    outputs/<stem>_<voice>/  narration (article.wav + parts/)
    outputs/video_<stem>/tiktok_<voice>.mp4
With --images DIR (article photos) the video uses the broadcast news style, otherwise the explainer style.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from .prepare import CONNECTORS, LEADS_IN, URL, Prepared, clean_line, pieces, prepare_text, to_caption

KH_NUM = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
MONTHS_SHORT = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
HELPERS = {"បាន", "នឹង", "ត្រូវ", "គឺ", "ជា", "មាន", "ក៏", "នូវ", "ដល់", "ពី"}  # a chip shouldn't end on these

# spoken greeting and sign-off per kind of video: (tts line, caption)
OPENINGS = {
    "news": [("សួស្តីបងប្អូនទាំងអស់គ្នា។", "សួស្តីបងប្អូនទាំងអស់គ្នា"), ("សូមជូនព័ត៌មានថ្មីថ្មី។", "សូមជូនព័ត៌មានថ្មីៗ")],
    "tutorial": [("សួស្តីបងប្អូនទាំងអស់គ្នា។", "សួស្តីបងប្អូនទាំងអស់គ្នា"), ("ថ្ងៃនេះ យើងនឹងរៀនជាមួយគ្នា។", "ថ្ងៃនេះ យើងនឹងរៀនជាមួយគ្នា")],
    "promo": [("សួស្តីបងប្អូនទាំងអស់គ្នា។", "សួស្តីបងប្អូនទាំងអស់គ្នា")],
    "advice": [("សួស្តីបងប្អូនទាំងអស់គ្នា។", "សួស្តីបងប្អូនទាំងអស់គ្នា")],
}
CLOSINGS = {
    "news": [("សូមអរគុណបងប្អូន ដែលបានតាមដាន។", "សូមអរគុណបងប្អូន ដែលបានតាមដាន"),
             ("កុំភ្លេចចុចតាមដាន ដើម្បីទទួលបានព័ត៌មានថ្មីថ្មី។", "កុំភ្លេចចុចតាមដាន ដើម្បីទទួលបានព័ត៌មានថ្មីៗ")],
    "tutorial": [("សង្ឃឹមថាមេរៀននេះមានប្រយោជន៍។", "សង្ឃឹមថាមេរៀននេះមានប្រយោជន៍"),
                 ("កុំភ្លេចចុចតាមដាន សម្រាប់មេរៀនបន្ទាប់។", "កុំភ្លេចចុចតាមដាន សម្រាប់មេរៀនបន្ទាប់")],
    "promo": [("សាកល្បងថ្ងៃនេះ។", "សាកល្បងថ្ងៃនេះ"), ("កុំភ្លេចចុចតាមដាន។", "កុំភ្លេចចុចតាមដាន")],
    "advice": [("សូមអរគុណដែលបានស្តាប់។", "សូមអរគុណដែលបានស្តាប់"), ("សូមឱ្យបងប្អូនមានសុភមង្គល។", "សូមឱ្យបងប្អូនមានសុភមង្គល")],
}
OUTRO_TEXT = {"news": ("តាមដានសម្រាប់ព័ត៌មានបន្ថែម", "Follow for more news"),
              "tutorial": ("តាមដានសម្រាប់មេរៀនបន្ថែម", "Follow for more"),
              "promo": ("តាមដានសម្រាប់ព័ត៌មានបន្ថែម", "Follow for more"),
              "advice": ("សូមអរគុណ", "Follow for more")}
KICKER = {"news": "ព័ត៌មានថ្មី", "tutorial": "មេរៀន", "promo": "ថ្មី", "advice": "ពាក្យល្អ"}
BADGE = {"news": "ព័ត៌មាន", "tutorial": "មេរៀនខ្លី", "promo": "ណែនាំ", "advice": "គំនិតល្អ"}  # top-left series badge


# ---------------------------------------------------------------- header (title, date)

DATE_LINE = re.compile(r"^\s*(?:ថ្ងៃ\S*\s+)?ទី\s*([0-9០-៩]{1,2})\s*ខែ\s*(\S+?)\s*ឆ្នាំ\s*([0-9០-៩]{4})")


def split_header(raw: str) -> tuple[str, str, str, list[str]]:
    """Title (first line, if it reads like a headline), display date, body text, source lines."""
    paras = [p.strip() for p in raw.splitlines() if p.strip()]
    title = date = ""
    sources = [p for p in paras if URL.search(p)]
    if paras and len(paras) > 2 and len(paras[0]) < 140 and not paras[0].endswith(("។", "៕")):
        title = clean_line(paras.pop(0))
    if paras and (m := DATE_LINE.match(paras[0])):
        d, mo, y = m.groups()
        date = f"{d} {mo} {y}".translate(KH_NUM)
        paras.pop(0)
    return title, date, "\n".join(paras), sources


def today_kh() -> str:
    t = dt.date.today()
    return f"{t.day} {MONTHS_SHORT[t.month - 1]} {t.year}".translate(KH_NUM)


# ---------------------------------------------------------------- short on-screen phrases

def _join(ws) -> str:
    return "".join((" " if sp and k else "") + x for k, (x, sp) in enumerate(ws))


def snippet(text: str, limit: int, around_number: bool = False) -> str:
    """A short phrase from a caption, cut at word boundaries to at most ~limit characters: its start, or
    (around_number) the words around its first number. Never ends on a word that needs the next one
    (ទី, លោក…) or on a linking word, never starts with one."""
    ws = pieces(to_caption(text))
    first = 0
    if around_number:
        idx = next((i for i, (w, _) in enumerate(ws) if has_number(w)), 0)
        first = max(0, idx - 2)
        while first < idx and ws[first][0].startswith(CONNECTORS):
            first += 1
    out = []
    for w in ws[first:]:
        if out and len(_join(out + [w])) > limit:
            break
        out.append(w)
    while len(out) > 1 and (out[-1][0].endswith(LEADS_IN) or out[-1][0] in CONNECTORS or out[-1][0] in HELPERS):
        out.pop()
    return _join(out).strip(" ,")


def has_number(s: str) -> bool:
    return bool(re.search(r"[0-9០-៩]", s))


def pick_facts(captions: list[tuple[int, str]], n: int, limit: int, starts: set[int]) -> list[list]:
    """Up to n short key phrases in spoken order: phrases around numbers first, then sentence openings.
    Lines that start mid-sentence or with a linking word make poor chips and are skipped."""
    cands = []
    for line, cap in captions:
        if has_number(cap):
            cands.append((0, line, snippet(cap, limit, around_number=True)))
        elif line in starts and not cap.startswith(CONNECTORS):
            cands.append((1, line, snippet(cap, limit)))
    chosen, seen = [], set()
    for _, line, text in sorted(cands):
        if text and len(text) >= 6 and text not in seen and len(chosen) < n:
            chosen.append([text, line])
            seen.add(text)
    return sorted(chosen, key=lambda f: f[1])


# ---------------------------------------------------------------- sections

def sections(prep: Prepared, offset: int, max_sections: int = 4) -> list[list[int]]:
    """Group body lines (1-based TTS line numbers) into up to 4 sections, cut at paragraph starts."""
    n = len(prep.lines)
    k = max(1, min(max_sections, round(n / 7)))
    target = n / k
    out, cur = [], []
    for i in range(n):
        new_para = i > 0 and prep.paragraphs[i] != prep.paragraphs[i - 1]
        if cur and new_para and len(out) < k - 1 and len(cur) >= target * 0.7:
            out.append(cur)
            cur = []
        cur.append(i + 1 + offset)
    out.append(cur)
    return out


# ---------------------------------------------------------------- storyboard

def build_storyboard(stem: str, voice: str, kind: str, title: str, date: str, sources: list[str],
                     prep: Prepared, lines: list[str], captions: list[str], images: list[str] | None,
                     images_dir: str | None, accent: str | None) -> dict:
    n_open = len(OPENINGS[kind])
    secs = sections(prep, n_open)
    secs[0] = list(range(1, n_open + 1)) + secs[0]
    close_line = len(lines) - len(CLOSINGS[kind]) + 1
    cap = lambda ln: captions[ln - 1]  # noqa: E731
    starts = {i for i in range(1, len(lines) + 1) if i == 1 or lines[i - 2].endswith(("។", "៕", "!", "?"))}
    head = title or snippet(cap(n_open + 1), 40)
    outro_text, outro_sub = OUTRO_TEXT[kind]
    sb: dict = {
        "tts_text": f"texts/{stem}_tts.txt",
        "narration": f"outputs/{stem}_{voice}",
        "output": f"outputs/video_{stem}/tiktok_{voice}.mp4",
        "captions": captions,
    }
    if accent:
        sb["accent"] = accent

    if images:
        sb.update({
            "style": "news", "channel": "ព័ត៌មាន", "date": date or today_kh(), "disclosure": "សំឡេង AI",
            "images_dir": images_dir, "intro_image": images[0], "outro_image": images[-1],
            "ticker_label": KICKER[kind],
        })
        img = iter(images * (2 * len(secs) + 2))
        scenes = []
        for k, sec in enumerate(secs):
            body = [(ln, cap(ln)) for ln in sec if ln > n_open]
            shots = [[next(img), sec[0]]]
            if len(sec) > 5:
                shots.append([next(img), sec[len(sec) // 2]])
            scenes.append({
                "line": sec[0],
                "tag": KICKER[kind] if k == 0 else f"{k:02d}",
                "title": snippet(head, 28) if k == 0 else snippet(body[0][1], 28),
                "subtitle": date if k == 0 and date else "",
                "images": shots,
                "facts": pick_facts(body[1:] if k else body, 4, 26, starts),
            })
        sb["scenes"] = scenes
        ticker = [snippet(f[0], 40) for s in scenes for f in s["facts"] if has_number(f[0])][:3]
        sb["ticker"] = ticker or [s["title"] for s in scenes[1:4]] or [head]
        sb["outro"] = {"line": close_line, "text": outro_text, "subtitle": outro_sub}
        domains = sorted({m.group(0).split("/")[2] if "//" in m.group(0) else m.group(0).split("/")[0]
                          for s in sources for m in [URL.search(s)] if m})
        if domains:
            sb["credits"] = ["ប្រភព៖ " + ", ".join(domains)]
        if m := next((URL.search(s) for s in sources if URL.search(s)), None):
            sb["source"] = m.group(0)  # picks the news site's colours (news_video.SITE_THEMES)
        return sb

    # explainer style: a headline card, then one "points" card per section
    sb.update({
        "title": BADGE[kind],
        "brands": {"a": {"name": "", "color": accent or "#3D6FE0", "tag": ""},
                   "b": {"name": "", "color": "#E0393E", "tag": ""}},
    })
    first_body = [(ln, cap(ln)) for ln in secs[0] if ln > n_open]
    scenes = [{
        "type": "headline", "line": 1, "kicker": KICKER[kind], "headline": snippet(head, 48),
        "points": pick_facts(first_body, 3, 30, starts),
    }]
    for k, sec in enumerate(secs[1:], 1):
        body = [(ln, cap(ln)) for ln in sec]
        items = pick_facts(body, 5, 22, starts) or [[snippet(body[0][1], 22), sec[0]]]
        scenes.append({
            "type": "points", "line": sec[0], "number": f"{k:02d}", "title": snippet(body[0][1], 24),
            "subtitle": snippet(body[1][1], 34) if len(body) > 1 else KICKER[kind],
            "card": {"name": snippet(body[0][1], 18), "tag": f"{len(items)} ចំណុច".translate(KH_NUM), "items": items},
        })
    sb["scenes"] = scenes
    sb["outro"] = {"line": close_line, "seconds": 3.0, "text": outro_text, "subtitle": outro_sub}
    return sb


# ---------------------------------------------------------------- the command

def make(source: Path, voice_name: str, kind: str = "news", images_dir: Path | None = None,
         accent: str | None = None, redo: bool = False, render_video: bool = True,
         workers: int | None = None, seconds: float | None = None) -> dict:
    from .voices import PROJECT_ROOT, Voice

    root = PROJECT_ROOT
    stem = source.stem.removesuffix("_raw")
    voice = Voice.load(voice_name)  # fail early on a wrong voice name
    tts_file, sb_file = root / f"texts/{stem}_tts.txt", root / f"videos/{stem}.json"
    report: dict = {"tts_text": tts_file, "storyboard": sb_file}

    # 1-2. text + storyboard (kept if they exist, so hand edits survive a re-run)
    if not redo and tts_file.exists() != sb_file.exists():
        have, missing = (tts_file, sb_file) if tts_file.exists() else (sb_file, tts_file)
        raise SystemExit(f"{have.relative_to(root)} exists but {missing.relative_to(root)} doesn't; "
                         "--redo rebuilds both from the source (overwriting it)")
    if redo or not tts_file.exists():
        title, date, body, sources = split_header(source.read_text(encoding="utf-8"))
        prep = prepare_text(body)
        if not prep.lines:
            raise SystemExit(f"{source}: no text to read")
        lines = [t for t, _ in OPENINGS[kind]] + prep.lines + [t for t, _ in CLOSINGS[kind]]
        captions = [c for _, c in OPENINGS[kind]] + prep.captions + [c for _, c in CLOSINGS[kind]]
        images = None
        if images_dir:
            images = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not images:
                raise SystemExit(f"no photos (.jpg/.png/.webp) in {images_dir}")
        rel_images = str(images_dir.resolve().relative_to(root)) if images_dir and images_dir.resolve().is_relative_to(root) \
            else (str(images_dir.resolve()) if images_dir else None)
        sb = build_storyboard(stem, voice.name, kind, title, date, sources, prep, lines, captions,
                              images, rel_images, accent)
        tts_file.parent.mkdir(parents=True, exist_ok=True)
        sb_file.parent.mkdir(parents=True, exist_ok=True)
        tts_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sb_file.write_text(json.dumps(sb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report.update(english=prep.english, wrote_text=True)
        print(f"Wrote {tts_file.relative_to(root)} ({len(lines)} lines) and {sb_file.relative_to(root)}")
    else:
        sb = json.loads(sb_file.read_text(encoding="utf-8"))
        n_lines = sum(1 for ln in tts_file.read_text(encoding="utf-8").splitlines() if ln.strip())
        if len(sb["captions"]) != n_lines:
            raise SystemExit(f"{tts_file.name} has {n_lines} lines but {sb_file.name} has {len(sb['captions'])} "
                             "captions: edit both to match, or re-run with --redo to rebuild them from the source")
        print(f"Using existing {tts_file.relative_to(root)} and {sb_file.relative_to(root)}")

    # 3. narration (only new or changed lines are generated)
    from .article import read_article
    from .synth import VoxPocket
    out_dir = root / sb["narration"]
    rep = read_article(lambda: VoxPocket(), voice, tts_file, out_dir)
    report.update(narration=rep.output, duration=rep.duration, flagged=rep.flagged_lines, reused=rep.reused)
    print(f"Narration {rep.output.relative_to(root)}: {rep.duration:.1f}s ({rep.reused} lines reused)")

    # 4. video
    if render_video:
        # a fresh process: this one still holds the voice model, and each render worker would inherit it
        import subprocess
        import sys
        cmd = [sys.executable, "-m", "khmer_tts", "video", str(sb_file)]
        cmd += ["--workers", str(workers)] if workers else []
        cmd += ["--seconds", str(seconds)] if seconds else []
        if subprocess.run(cmd, cwd=root).returncode != 0:
            raise SystemExit("video render failed (see above)")
        out = root / sb["output"]
        report.update(video=out, video_seconds=float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True).stdout or 0))
    return report

