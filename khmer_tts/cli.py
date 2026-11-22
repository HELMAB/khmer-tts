"""Command-line interface: python -m khmer_tts <command> ...

    voices                                   list cloned voices
    clips SOURCE                             suggest reference clips (speech between pauses)
    add-voice NAME SOURCE --text "..."       register a voice from a recording
    speak  --voice NAME "text" ["text" ...]  say a few sentences in a cloned voice
    prepare texts/raw.txt                    turn raw text into short TTS lines (-> texts/raw_tts.txt)
    read   --voice NAME texts/article.txt    read a whole prepared article
    vits   "text"                            baseline single-voice VITS model
    video  videos/STORYBOARD.json            vertical explainer video from a narrated article
    make   texts/article.txt --voice NAME    all of it: prepare, narrate, storyboard, video
    song   URL|FILE --out DIR                song for a Facebook Story: download, vocals, sung phrases
    song   DIR --align LYRICS.txt --from A --to B   lyric line timings for the story storyboard
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .voices import PROJECT_ROOT, Voice, add_voice, list_voices, probe_duration, suggest_clips

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SENTENCES = [
    "សួស្តីបងប្អូនទាំងអស់គ្នា អរគុណច្រើនសម្រាប់ការគាំទ្រ។",
    "ថ្ងៃនេះអាកាសធាតុល្អណាស់ ខ្ញុំចង់ទៅដើរលេងនៅមាត់ទន្លេ។",
    "(happy, smiling tone)ខ្ញុំសប្បាយចិត្តខ្លាំងណាស់ដែលបានជួបអ្នកទាំងអស់គ្នាថ្ងៃនេះ!",
]


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voice", required=True, help="voice name (see: python -m khmer_tts voices)")
    p.add_argument("--cfg", type=float, default=2.0, help="guidance strength (default 2.0)")
    p.add_argument("--steps", type=int, default=10, help="diffusion steps (default 10)")


def cmd_voices(args) -> None:
    names = list_voices()
    if not names:
        print("No voices yet. Add one with: python -m khmer_tts add-voice NAME SOURCE --text '...'")
    width = max(map(len, names), default=0)
    for name in names:
        v = Voice.load(name)
        print(f"{name:<{width}}  {v.duration:4.1f}s  {v.text}")


def cmd_clips(args) -> None:
    source = Path(args.source)
    clips = suggest_clips(source)
    print(f"{source}: {probe_duration(source):.1f}s. Candidate reference clips (speech between pauses, 4-9 s):")
    for start, dur in clips:
        print(f"  --start {start:.2f} --duration {dur:.2f}")
    if not clips:
        print("  none found; pick a stretch by ear and pass --start/--duration yourself")


def cmd_add_voice(args) -> None:
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    v = add_voice(
        args.name, Path(args.source), text, start=args.start, duration=args.duration,
        transcript=Path(args.transcript) if args.transcript else None, overwrite=args.overwrite,
        clean=args.clean,
    )
    print(f"Added voice '{v.name}': {v.wav} ({v.duration:.1f}s)\n  text: {v.text}")


def cmd_speak(args) -> None:
    from .synth import VoxPocket, save

    voice = Voice.load(args.voice)
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS_DIR / f"samples_{voice.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = VoxPocket(cfg=args.cfg, steps=args.steps)
    for i, text in enumerate(args.text or DEFAULT_SENTENCES, 1):
        result = tts.generate(text, voice, seed=args.seed)
        out = out_dir / f"{args.prefix}_{i}.wav"
        save(result, out)
        print(f"{out}: {result.duration:.2f}s in {result.seconds_taken:.1f}s  <- {text}")


def cmd_read(args) -> None:
    from .article import read_article
    from .synth import VoxPocket

    voice = Voice.load(args.voice)
    text_file = Path(args.text_file)
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS_DIR / f"{text_file.stem}_{voice.name}"
    report = read_article(lambda: VoxPocket(cfg=args.cfg, steps=args.steps), voice, text_file, out_dir,
                          tries=args.tries, short_pause=args.short_pause, long_pause=args.long_pause,
                          reuse=not args.fresh)
    print(f"Wrote {report.output}: {report.duration:.1f}s audio in {report.seconds_taken:.0f}s "
          f"({report.reused} unchanged lines reused); parts in {report.output.parent / 'parts'}/")
    if report.flagged_lines:
        print(f"Lines that may be cut short: {report.flagged_lines}")


def cmd_vits(args) -> None:
    from .synth import Vits, save

    out = Path(args.output) if args.output else OUTPUTS_DIR / "vits.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = Vits().generate(args.text, seed=args.seed)
    save(result, out)
    print(f"Wrote {out}: {result.duration:.2f}s @ {result.sample_rate} Hz in {result.seconds_taken:.2f}s")


def cmd_prepare(args) -> None:
    from .prepare import prepare_text

    src = Path(args.text_file)
    out = Path(args.output) if args.output else src.with_name(src.stem + "_tts.txt")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    result = prepare_text(src.read_text(encoding="utf-8"), max_len=args.max_len)
    out.write_text("\n".join(result.lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {len(result.lines)} lines")
    if not result.segmented:
        print("  note: install khmer-nltk for better line breaks and ៗ handling")
    if result.dropped:
        print(f"  skipped {len(result.dropped)} source/link line(s): " + " | ".join(d[:50] for d in result.dropped))
    if result.english:
        print("  English words kept (listen to these lines after `read`): " + ", ".join(result.english))


def cmd_video(args) -> None:
    from .video import render, render_still

    sb = Path(args.storyboard)
    if args.still is not None:
        for t in args.still:
            out = OUTPUTS_DIR / "stills" / f"{sb.stem}_{t:06.2f}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            render_still(sb, t, out, narration=args.narration)
            print(out)
        return
    out, total, n_sfx = render(sb, Path(args.output) if args.output else None, workers=args.workers,
                               seconds=args.seconds, narration=args.narration)
    print(f"Wrote {out}: {total:.1f}s, {n_sfx} sound effects")


def cmd_make(args) -> None:
    from .make import make

    r = make(Path(args.source), args.voice, kind=args.kind, images_dir=Path(args.images) if args.images else None,
             accent=args.accent, redo=args.redo, render_video=not args.no_video, workers=args.workers,
             seconds=args.seconds)
    print()
    if r.get("video"):
        print(f"Video:      {r['video']}" + (f" ({r['video_seconds']:.0f}s)" if r.get("video_seconds") else ""))
    print(f"Narration:  {r['narration']}")
    print(f"Edit:       {r['tts_text']} (what is read) and {r['storyboard']} (scenes, captions)")
    print("            then run the same command again: only changed lines are re-voiced")
    if r.get("flagged"):
        print(f"Listen to:  lines {r['flagged']} (may be cut short)")
    if r.get("english"):
        print("English:    " + ", ".join(r["english"]) + " (kept as written; check how they sound)")


def cmd_song(args) -> None:
    import json
    import shutil
    import subprocess

    from .song import align_lyrics, fetch_song, separate_vocals, vocal_phrases

    if args.align:
        folder = Path(args.source)
        lines = [[p.strip() for p in ln.split("/")] for ln in Path(args.align).read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        times = align_lyrics(folder / "vocals.wav", lines, args.t_from, args.t_to)
        print(",\n".join(json.dumps({"t": t, "text": parts}, ensure_ascii=False) for t, parts in zip(times, lines)))
        return
    if not args.out:
        raise SystemExit("song: pass --out videos/audio/<song> (or --align LYRICS.txt to time lyrics)")
    out = Path(args.out)
    if args.source.startswith(("http://", "https://")):
        song, desc = fetch_song(args.source, out)
        print(f"Song: {song}\nDescription (look for lyrics): {desc}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        song = out / "song.wav"
        if args.source.endswith(".wav"):
            shutil.copy(args.source, song)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.source, str(song)], check=True)
    vocals = separate_vocals(song)
    print(f"Vocals: {vocals}\nSung phrases (song seconds):")
    for a, b in vocal_phrases(vocals):
        if b - a >= 1.0:
            print(f"  {a:7.2f} - {b:7.2f}  ({b - a:4.1f}s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="khmer_tts", description="Khmer TTS with voice cloning")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("voices", help="list cloned voices")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("clips", help="suggest 4-9 s clips between pauses for add-voice")
    p.add_argument("source", help="audio file (mp3/wav/...)")
    p.set_defaults(func=cmd_clips)

    p = sub.add_parser("add-voice", help="register a new voice from a recording")
    p.add_argument("name")
    p.add_argument("source", help="audio file (mp3/wav/...)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="exact words spoken in the cut clip")
    g.add_argument("--text-file", help="file holding the exact words spoken in the cut clip")
    p.add_argument("--start", type=float, default=0.0, help="clip start in seconds (default 0)")
    p.add_argument("--duration", type=float, default=6.0, help="clip length, 4-9 s works best (default 6)")
    p.add_argument("--transcript", help="full transcript of SOURCE, stored alongside for reference")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--clean", action="store_true",
                   help="gentle cleanup: 70 Hz high-pass, light denoise, loudness normalise to -20 LUFS")
    p.set_defaults(func=cmd_add_voice)

    p = sub.add_parser("speak", help="say sentences in a cloned voice")
    _add_model_args(p)
    p.add_argument("text", nargs="*", help="sentences to speak (default: three demo sentences)")
    p.add_argument("--out-dir", help="default: outputs/samples_<voice>/")
    p.add_argument("--prefix", default="sample", help="output file prefix (default: sample)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("read", help="read a prepared article, one chunk per line")
    _add_model_args(p)
    p.add_argument("text_file")
    p.add_argument("--out-dir", help="default: outputs/<text-file-name>_<voice>/")
    p.add_argument("--tries", type=int, default=3, help="seeds to try when a line comes out too short")
    p.add_argument("--short-pause", type=float, default=0.25)
    p.add_argument("--long-pause", type=float, default=0.6)
    p.add_argument("--fresh", action="store_true", help="regenerate every line, even unchanged ones")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("prepare", help="turn raw text into short TTS-ready lines")
    p.add_argument("text_file")
    p.add_argument("-o", "--output", help="default: <text_file>_tts.txt next to it")
    p.add_argument("--max-len", type=int, default=60, help="longest line in characters (default 60)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("vits", help="baseline single-voice VITS model (no cloning)")
    p.add_argument("text")
    p.add_argument("-o", "--output", help="default: outputs/vits.wav")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_vits)

    p = sub.add_parser("video", help="render a vertical explainer video from a storyboard")
    p.add_argument("storyboard", help="storyboard JSON (see videos/)")
    p.add_argument("-o", "--output", help="default: the storyboard's output path")
    p.add_argument("--still", type=float, nargs="+", metavar="SEC", help="only render frames at these times (PNG)")
    p.add_argument("--seconds", type=float, help="render only the first N seconds (quick preview)")
    p.add_argument("--narration", help="use another `read` output folder instead of the storyboard's (e.g. another voice)")
    p.add_argument("--workers", type=int, help="render processes (default: by CPU count and free memory)")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("song", help="prepare a song for a Facebook Story video (style: story)")
    p.add_argument("source", help="song URL (YouTube…) or audio file; with --align: the song folder")
    p.add_argument("--out", help="song folder, e.g. videos/audio/<stem> (required without --align)")
    p.add_argument("--align", metavar="LYRICS.txt", help="one lyric line per row, halves split by ' / '")
    p.add_argument("--from", dest="t_from", type=float, help="align window start (song seconds)")
    p.add_argument("--to", dest="t_to", type=float, help="align window end (song seconds)")
    p.set_defaults(func=cmd_song)

    p = sub.add_parser("make", help="text → narration → storyboard → video, in one command")
    p.add_argument("source", help="raw text file (article, script)")
    p.add_argument("--voice", required=True)
    p.add_argument("--kind", choices=["news", "tutorial", "promo", "advice"], default="news",
                   help="sets the spoken greeting/sign-off and on-screen labels (default news)")
    p.add_argument("--images", help="folder of photos: broadcast news style with these photos")
    p.add_argument("--accent", help="theme colour, e.g. #F05032")
    p.add_argument("--redo", action="store_true", help="rebuild the TTS text and storyboard from the source")
    p.add_argument("--no-video", action="store_true", help="stop after the narration")
    p.add_argument("--seconds", type=float, help="render only the first N seconds (quick preview)")
    p.add_argument("--workers", type=int, help="render processes (default: by CPU count and free memory)")
    p.set_defaults(func=cmd_make)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
