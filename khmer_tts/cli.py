"""Command-line interface: python -m khmer_tts <command> ...

    voices                                   list cloned voices
    clips SOURCE                             suggest reference clips (speech between pauses)
    add-voice NAME SOURCE --text "..."       register a voice from a recording
    speak  --voice NAME "text" ["text" ...]  say a few sentences in a cloned voice
    read   --voice NAME texts/article.txt    read a whole prepared article
    vits   "text"                            baseline single-voice VITS model
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
    tts = VoxPocket(cfg=args.cfg, steps=args.steps)
    report = read_article(tts, voice, text_file, out_dir, tries=args.tries,
                          short_pause=args.short_pause, long_pause=args.long_pause)
    print(f"Wrote {report.output}: {report.duration:.1f}s audio in {report.seconds_taken:.0f}s; "
          f"parts in {report.output.parent / 'parts'}/")
    if report.flagged_lines:
        print(f"Lines that may be cut short: {report.flagged_lines}")


def cmd_vits(args) -> None:
    from .synth import Vits, save

    out = Path(args.output) if args.output else OUTPUTS_DIR / "vits.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = Vits().generate(args.text, seed=args.seed)
    save(result, out)
    print(f"Wrote {out}: {result.duration:.2f}s @ {result.sample_rate} Hz in {result.seconds_taken:.2f}s")


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
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("vits", help="baseline single-voice VITS model (no cloning)")
    p.add_argument("text")
    p.add_argument("-o", "--output", help="default: outputs/vits.wav")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_vits)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
