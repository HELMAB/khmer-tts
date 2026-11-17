---
name: khmer-voice
description: Generate Khmer speech in a cloned voice, add (clone) a new voice from a recording, or turn a narrated text into a TikTok/Reels/Shorts explainer video, in the khmer-tts project. Use when the user asks to read, speak, voice, narrate or generate audio for Khmer text or a text file "using/with <voice>", to clone a voice from an mp3/wav, to add or list voices, to regenerate a bad line, or to make a vertical/TikTok video (captions, overlays, sound effects) from a text or narration.
---

# Khmer voice: generate audio, clone voices, make TikTok videos

Everything runs through the project CLI, from the project root:

```sh
KT=".venv-voxpocket/bin/python -m khmer_tts"
$KT voices                      # list voices: name, clip length, transcript
$KT clips SOURCE                # suggest 4-9 s clips between pauses
$KT add-voice NAME SOURCE --start S --duration D --text "..." [--transcript FILE] [--clean] [--overwrite]
$KT speak --voice NAME "line" ["line" ...] [--out-dir DIR] [--prefix P] [--seed N]
$KT read  --voice NAME FILE.txt [--out-dir DIR]
$KT video videos/STORY.json [--narration DIR] [-o OUT.mp4] [--still SEC ...] [--seconds N]
```

Model loading takes ~10 s; generation runs at roughly 2 s of compute per 1 s of audio on the
4 GB GTX 1650. Use a 600000 ms timeout for `read`, and expect ~2 min per minute of audio.
Filter the noisy logs: `2>&1 | grep -E "^\[|Wrote|Lines that|Error|Traceback|outputs/"`.

## A. Generate audio for a text

1. Check the voice exists (`$KT voices`). If the user names an unknown voice, show the list and ask.
2. **Short input (a sentence or two):** `$KT speak --voice NAME "..."` → `outputs/samples_NAME/`.
   Pass a distinct `--prefix` so earlier samples are not overwritten.
3. **A text file or anything longer than ~2 lines:** never feed raw paragraphs to the model.
   Write a prepared copy next to it as `texts/<stem>_tts.txt` (keep the user's original untouched),
   then `$KT read --voice NAME texts/<stem>_tts.txt --out-dir outputs/<stem>_NAME`.
4. Report: output path, duration, any lines listed under "Lines that may be cut short", and every
   editorial change you made while preparing the text. The user must listen; you cannot.

### Preparing text (the step that decides quality)

One phrase per line, about 25-90 characters, split at natural pauses (spaces between clauses,
before និង / ប៉ុន្តែ / ដែល / ដោយ / គឺ). Lines ending in `។ ! ? ៕` get a sentence pause, other lines a short one,
so end each sentence's last line with `។`.

- Numbers → Khmer words: ២០ → ម្ភៃ, ១៦ → ដប់ប្រាំមួយ, ២៩ → ម្ភៃប្រាំបួន, ២០២៦ → ពីរពាន់ម្ភៃប្រាំមួយ,
  ១២៧ → មួយរយម្ភៃប្រាំពីរ, ទី៣ → ទីបី. Same for Latin digits.
- Expand the repeat sign: ថ្មីៗ → ថ្មីថ្មី, សំខាន់ៗ → សំខាន់សំខាន់.
- Remove `« » " ( )`, bullets `•`, `៖` (turn a heading with `៖` into its own line ending in `។`),
  zero-width spaces (U+200B), and Latin glosses in brackets such as `(Emmanuel Macron)`.
- Abbreviations and foreign words: write them as they should be pronounced in Khmer.
- Fix obvious typos only if the meaning is clear, and tell the user about each one.

### Redoing a bad line

Lines are saved as `<out-dir>/parts/NN.wav`. To redo line N, regenerate just that line with another seed
and rebuild the article. Use the API so the voice and settings match:

```sh
.venv-voxpocket/bin/python - <<'EOF'
from pathlib import Path
import numpy as np, soundfile as sf
from khmer_tts.synth import VoxPocket
from khmer_tts.voices import Voice
from khmer_tts.article import read_lines, SENTENCE_ENDINGS
N, SEED, TXT, OUT = 7, 5, Path("texts/news_tts.txt"), Path("outputs/news_rfi")
lines = read_lines(TXT); tts = VoxPocket()
r = tts.generate(lines[N - 1], Voice.load("rfi"), seed=SEED)
sf.write(OUT / f"parts/{N:02d}.wav", r.audio, r.sample_rate)
pieces = []
for i, line in enumerate(lines, 1):
    a, sr = sf.read(OUT / f"parts/{i:02d}.wav", dtype="float32")
    pieces += [a, np.zeros(int((0.6 if line.endswith(SENTENCE_ENDINGS) else 0.25) * sr), np.float32)]
sf.write(OUT / "article.wav", np.concatenate(pieces), sr)
EOF
```

## B. Add (clone) a voice

Needed: a recording of one speaker, and the **exact** words spoken in the chosen clip.
A wrong transcript is the main cause of a bad clone.

1. Name: short, lowercase, no spaces (the user's choice; default to the source file's stem).
   If `voices/NAME/` exists, ask before passing `--overwrite`.
2. `$KT clips SOURCE` lists speech stretches between pauses. Prefer the earliest 5-8 s one
   (avoid intros with music or jingles; a clip that starts at 0 on news audio is usually fine).
3. Get the clip's words:
   - If the user gave a full transcript, estimate where the clip ends: Khmer speech runs about
     13-15 characters/second (count without spaces/U+200B; `len(text) / duration_seconds` on the full
     recording gives this speaker's rate). Take the transcript text up to the phrase boundary
     closest to `start + duration`. Prefer a clip starting at 0 so the text starts at the transcript's start.
   - If there is no transcript, ask the user for the words in that clip (give them the start/end
     times). Do not guess. Only run an ASR model (e.g. `seanghay/whisper-medium-khmer`) with the user's OK;
     it is a large download and they may decline.
4. Decide on `--clean` (70 Hz high-pass, light denoise, loudness normalise to -20 LUFS, peaks capped at -2 dB).
   Measure the raw clip first:
   `ffmpeg -i SOURCE -ss S -t D -af "astats=metadata=0,ebur128=framelog=quiet" -f null - 2>&1 | grep -E "Noise floor dB|^ +I:"`
   - Noise floor above about -55 dB, or loudness below about -25 LUFS → use `--clean`
     (e.g. `preapsovath`: -48 dB / -29 LUFS gave quieter-sounding, inconsistent clones; cleaned output came out 6-12 dB louder and steadier).
   - Clean broadcast audio (e.g. `rfi`: -69 dB / -15 LUFS) → skip it; cleaning only risks artefacts.
   - Background music under speech: `--clean` won't remove it; tell the user it needs vocal separation (e.g. Demucs) first.
   Never pitch-shift, EQ, compress or "enhance" a reference: the model clones those artefacts as part of the voice.
   When unsure, add both `NAME` and `NAME_clean` and let the user compare by ear.
5. `$KT add-voice NAME SOURCE --start S --duration D --text "WORDS" --transcript FULL.txt [--clean]`
   (omit `--transcript` if there is none). This writes `voices/NAME/reference.wav` (mono 44.1 kHz),
   `reference.txt`, and copies the source.
6. Test: `$KT speak --voice NAME` (three demo sentences → `outputs/samples_NAME/`).
   Durations around 0.06-0.09 s per character are normal.
7. Tell the user how the clip's words were matched, and that if the clip doesn't end exactly
   on the last word, they should fix `voices/NAME/reference.txt` (or ask you to try another clip).
   Add the voice to the Voices table in README.md.

## C. Make a TikTok explainer video

`$KT video` renders a 1080×1920, 30 fps MP4 from a narration (a `read` output folder: `article.wav` +
`parts/NN.wav`) and a storyboard JSON. It already handles the following:
- **Timing:** taken from `parts/`, so captions and on-screen items are in sync with no alignment step.
- **Animation:** cards and labels slide and pop in, and scenes fade into each other.
- **Captions:** English terms are highlighted, short runs of Latin words never break across lines, and Khmer is wrapped without splitting letter stacks.
- **Overlays:** a series badge, a section counter (`01 / 04`) and a progress bar, kept clear of TikTok's UI. TikTok covers about the top 150 px, bottom 400 px and right 130 px.
- **Sound effects:** synthesized, so there are no licensing issues: a whoosh at each scene, a swish for each card, a pop for each label, and a ding for each verdict and the outro.
- **Loudness:** normalized for social platforms. Font: Google Sans (it covers Khmer).

Inputs the user gives: the original text, the TTS text, the narration (or a voice to narrate with).
If any is missing, produce it first with section A. The narration's voice is a separate choice:
the same storyboard works with any `read` output via `--narration`.

1. **Storyboard** `videos/<stem>.json` (copy the structure of `videos/laravel_spring_boot.json`):
   - `tts_text`, `narration`, `output` (`outputs/video_<stem>/tiktok.mp4`), `title` (top badge).
   - `captions`: **exactly one per line of the TTS file, same order** (the renderer refuses a mismatch).
     Write them in the original's display spelling: English terms back in Latin script, digits as in
     the original, no trailing `។`. The TTS file is for the ear; captions are for the eye.
   - `brands.a` / `brands.b`: `name`, `color` (hex), `tag` (short label shown on the card).
     For a non-comparison topic, still use two cards (e.g. "Pros"/"Cons", "Before"/"After").
   - `scenes`, each starting at a 1-based TTS `line` (a scene lasts until the next one starts):
     - `title`: big a VS b intro; optional `subtitle {line, text}` and `topics {line, items[]}` (≤ 4 short items).
     - `compare`: header `number` ("01"…), `title` (Khmer), `subtitle` (English); cards
       `a`/`b` = `{line, items: [[text, line], ...]}`. Each item pops in when its line is spoken.
       Keep items to short key phrases (≤ ~22 chars), ≤ 6 per card; chips shrink to fit two-three rows.
     - Conclusion: a `compare` with `"number": "✓", "verdict": true`; mark the winning item
       `[text, line, "strong"]` (filled chip + check, plays a ding).
   - `outro`: `{seconds, text, subtitle}` shown after the narration (e.g. "តាមដានសម្រាប់មេរៀនបន្ថែម" / "Follow for more").
   Put scene starts on the line that introduces the section (e.g. "ចំណុចទីមួយ…"), and each card's
   first item on the line that names that brand, so visuals land on the words.
2. **Preview before rendering:** `$KT video videos/<stem>.json --still 2 20 45 …` (one time per scene,
   plus a crowded moment near each scene's end) → `outputs/stills/`. Join them and look:
   `ffmpeg -i a.png -i b.png … -filter_complex "hstack=N,scale=2400:-1" contact.png`, then Read it.
   Check: nothing overflows a card or overlaps the caption box, no names split across caption lines,
   Khmer renders with correct stacks, check marks visible.
3. **Render:** `$KT video videos/<stem>.json` (about 1.5-2 min for 2.5 min of video; use a 600000 ms timeout).
   For another voice: `--narration outputs/<stem>_<voice> -o outputs/video_<stem>/tiktok_<voice>.mp4`.
4. **Verify the real file:** grab 2-4 frames mid-animation from the MP4
   (`ffmpeg -ss T -i out.mp4 -frames:v 1 f.png`) and Read them. Measure loudness
   (`ffmpeg -i out.mp4 -af ebur128=framelog=quiet:peak=true -f null - 2>&1 | grep -E "^ +(I|Peak):"`):
   about -14 to -17 LUFS is fine.
5. **Report:** path, duration, size, what each scene shows, how captions differ from the TTS text, and
   that the user should watch it before posting (you cannot hear or watch it). Offer background music
   only from a track the user has rights to; never add music yourself.

Layout constants (card positions, caption height, colours, SFX levels) live at the top of
`khmer_tts/video.py`. Change them there, not per video.
