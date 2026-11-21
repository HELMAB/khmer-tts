---
name: khmer-voice
description: Generate Khmer speech in a cloned voice, add (clone) a new voice from a recording, or turn a narrated text into a TikTok/Reels/Shorts explainer video, in the khmer-tts project. Use when the user asks to read, speak, voice, narrate or generate audio for Khmer text or a text file "using/with <voice>", to clone a voice from an mp3/wav, to add or list voices, to regenerate a bad line, or to make a vertical/TikTok video (captions, overlays, sound effects) from a text, a narration, a news article URL (e.g. thmeythmey.com) or a step-by-step tutorial (with browser screenshots, code and terminal windows).
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
4 GB GTX 1650, and ~1.35 s per 1 s on an Apple Silicon Mac (MPS, picked automatically). Run `read`
in the background (or with a 600000 ms timeout); a 2.5 min article takes ~3.5 min on the Mac.
Filter the noisy logs: `2>&1 | tr '\r' '\n' | grep -E "^\[|Wrote|Lines that|Error|Traceback|outputs/"`.

**macOS rendering:** video text needs Pillow's Raqm shaping, which loads Homebrew's fribidi. If `video`
stops with "Pillow has no Raqm text shaping", the user hasn't run
`sudo ln -s /opt/homebrew/lib/libfribidi.dylib /usr/local/lib/`; until then prefix render commands with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. Never render Khmer without Raqm (letter stacks break).

**Picking a voice** when the user doesn't name one: `rfi` (news anchor) for news, `general` (casual
tech-TikTok) for coding tutorials. Say which you picked; any narration can be swapped in later.

## A. Generate audio for a text

1. Check the voice exists (`$KT voices`). If the user names an unknown voice, show the list and ask.
2. **Short input (a sentence or two):** `$KT speak --voice NAME "..."` → `outputs/samples_NAME/`.
   Pass a distinct `--prefix` so earlier samples are not overwritten.
3. **A text file or anything longer than ~2 lines:** never feed raw paragraphs to the model.
   Write a prepared copy next to it as `texts/<stem>_tts.txt` (keep the user's original untouched):
   start with `$KT prepare texts/<stem>.txt` (numbers, dates, %, $, ៗ, URLs, bullets, brackets and short
   lines are done for you), then read the result and edit by hand what it can't judge: drop the title/byline
   lines, add the greeting and sign-off, fix typos in the source, and split any line that still reads long.
   Then `$KT read --voice NAME texts/<stem>_tts.txt --out-dir outputs/<stem>_NAME`.
4. Report: output path, duration, any lines listed under "Lines that may be cut short", and every
   editorial change you made while preparing the text. The user must listen; you cannot.

### Preparing text (the step that decides quality)

One phrase per line, about 25-60 characters (never over ~70), split at natural pauses (spaces between clauses,
before និង / ប៉ុន្តែ / ដែល / ដោយ / គឺ). Lines ending in `។ ! ? ៕` get a sentence pause, other lines a short one,
so end each sentence's last line with `។`.

**Clear, unhurried speech comes first** (the user asked: every word must be clear, not rushed or slurred).
Long lines are what make the voice hurry, so split a long phrase rather than squeeze it, and give a list
of commands or terms one or two items per line. After generating, flag every line that came out rushed
(well under ~0.07 s per Khmer character, or a line whose English word sounds swallowed) and regenerate it
with another seed (snippet under "Redoing a bad line"); tell the user which lines you redid and why.

- **Open every video narration with a spoken greeting** suited to the video type (the user asked for this),
  as its own 1-2 lines at the top of the TTS file, each ending in `។`, with matching captions:
  - news: `សួស្តីបងប្អូនទាំងអស់គ្នា។` + `សូមជូនព័ត៌មាន…` (name the place/topic without repeating the
    next line, e.g. `សូមជូនព័ត៌មានថ្មីថ្មី ពីតំបន់អង្គរ។`);
  - tutorial / how-to: `សួស្តីបងប្អូនទាំងអស់គ្នា។` + `ថ្ងៃនេះ យើងនឹងរៀនពី…។`;
  - comparison: `សួស្តីបងប្អូនទាំងអស់គ្នា។` + `ថ្ងៃនេះ យើងនឹងប្រៀបធៀប … និង …។`;
  - advice / quotes: `សួស្តីបងប្អូនទាំងអស់គ្នា។` + `ថ្ងៃនេះ ខ្ញុំសូមចែករំលែកពាក្យពេចន៍ល្អៗ…។`
- **End every video narration with a spoken sign-off** (1-2 lines, each ending in `។`, with captions) and set
  the storyboard's `outro.line` to its first line, so the outro card comes up as the goodbye starts:
  - news: `សូមអរគុណបងប្អូន ដែលបានតាមដាន។` + `កុំភ្លេចចុចតាមដាន ដើម្បីទទួលបានព័ត៌មានថ្មីថ្មី។`;
  - tutorial: `សង្ឃឹមថាមេរៀននេះមានប្រយោជន៍។` + `កុំភ្លេចចុចតាមដាន សម្រាប់មេរៀនបន្ទាប់។`;
  - comparison: `តើបងប្អូនចូលចិត្តមួយណាជាង? សរសេរប្រាប់នៅខាងក្រោម។` + `កុំភ្លេចចុចតាមដាន។`;
  - advice / quotes: `សូមអរគុណដែលបានស្តាប់។` + `សូមឱ្យបងប្អូនមានសុភមង្គល។`
  Plain audio-only reads (section A) need neither greeting nor sign-off unless asked. Keep the user's original text file untouched.
  The first scene starts at line 1, so the greeting plays over the intro visuals.
  Adding a greeting to an existing narration: rename `parts/NN.wav` up by the number of new lines, generate
  only the new lines (API snippet below), rebuild `article.wav`, and shift every `line` in the storyboard.
- Numbers → Khmer words: ២០ → ម្ភៃ, ១៦ → ដប់ប្រាំមួយ, ២៩ → ម្ភៃប្រាំបួន, ២០២៦ → ពីរពាន់ម្ភៃប្រាំមួយ,
  ១២៧ → មួយរយម្ភៃប្រាំពីរ, ទី៣ → ទីបី. Same for Latin digits.
- Expand the repeat sign: ថ្មីៗ → ថ្មីថ្មី, សំខាន់ៗ → សំខាន់សំខាន់.
- Remove `« » " ( )`, bullets `•`, `៖` (turn a heading with `៖` into its own line ending in `។`),
  zero-width spaces (U+200B), and Latin glosses in brackets such as `(Emmanuel Macron)`.
- **English words stay English: never translate them and never transliterate them into Khmer script**
  (the user asked for this). Keep every English word, command, product and brand name in Latin script exactly
  as in the original: "Git", "commit", "branch", "Branching", "GitHub", "repository", "SSH key", "Vercel"
  stay as they are, never "ហ្គីត" / "ប្រេន" / "ហ្គីតហាប់" and never a Khmer equivalent such as
  "ការរក្សាទុក" for "commit". Put a space before and after each English word or command so it is read on its
  own, and don't run two long English terms together on one short line. VoxPocket reads Latin words; listen
  for swallowed or mumbled ones in the durations check above and regenerate those lines.
  Abbreviations that are not English (Khmer abbreviations, symbols) are still written out as spoken in Khmer.
- **Tech keywords are spoken in English too, even when the text gives a Khmer translation** (the user asked:
  "don't translate keyword like folder, snapshot, history"). When the original writes a Khmer word with an
  English gloss, e.g. `ថត (Folder)`, `ម៉ាស៊ីនថតរូប (Snapshot)`, `ប្រវត្តិសាស្រ្តកូដ (Version Control)`,
  `ពពក (Cloud)`, `ពាក្យសម្ងាត់ (Password)`, say the English keyword in its place (`Folder`, `Snapshot`,
  `Version Control … History`, `Cloud`, `Password`) and use the same wording in the caption, without the
  bracket. Same for un-glossed Khmer translations of the topic's keywords (Git: ឯកសារ → File,
  កាតាបត្រៀម → Staging Area, សោ → Key, គណនី → Account, ពាក្យបញ្ជា → Command, បញ្ចូលគ្នា → Merge,
  មុខងារថ្មី → Feature). Everyday Khmer words that were never English in the text stay Khmer (កុំព្យូទ័រ,
  អ៊ីមែល, កូដ, ដំឡើង as a plain verb). When unsure whether a word counts as a keyword, ask the user.
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
     it is a large download (~6 GB in the HF cache) and they may decline. Known pitfalls with that model:
     - its generation config predates `language=`; load `GenerationConfig.from_pretrained("openai/whisper-medium")`
       into `pipe.model.generation_config`, then pass `generate_kwargs={"language": "km", "task": "transcribe"}`;
     - on a Mac run it on `device="cpu"` (MPS hangs; CPU takes ~35 s for a 7 s clip);
     - if a Hugging Face download stops growing, restart it with `HF_HUB_DISABLE_XET=1` (plain HTTP resumes).
     ASR output is rough, especially for English names in Khmer speech (React → "រៀអែក"). Clean it up,
     show the user your reading and let them confirm before `add-voice`. In the reference text, write foreign
     names as they are pronounced in Khmer (e.g. `general`: "រីអែក … វ្យូ"), matching what is said.
   - `clips` finding nothing usually means the recording has gated (digital) silence or no 4-9 s stretch
     between its pauses; find pauses yourself with
     `ffmpeg -i SOURCE -t 40 -af silencedetect=noise=-35dB:d=0.18 -f null - 2>&1 | grep -oE "silence_(start|end): [0-9.]+"`
     and pick a clip that ends inside a pause.
4. Decide on `--clean` (70 Hz high-pass, light denoise, loudness normalise to -20 LUFS, peaks capped at -2 dB).
   Measure the raw clip first:
   `ffmpeg -i SOURCE -ss S -t D -af "astats=metadata=0,ebur128=framelog=quiet" -f null - 2>&1 | grep -E "Noise floor dB|^ +I:"`
   - Noise floor above about -55 dB, or loudness below about -25 LUFS → use `--clean`
     (e.g. a quiet interview clip at -48 dB / -29 LUFS gave quieter-sounding, inconsistent clones; cleaned output came out 6-12 dB louder and steadier).
   - Clean broadcast audio (e.g. `rfi`: -69 dB / -15 LUFS) → skip it; cleaning only risks artefacts.
   - Background music under speech: `--clean` won't remove it. Signs: few pauses found, and the 100 ms RMS
     between words stays within ~10 dB of the speech (e.g. `advice`: -33..-44 dB floor vs -29 LUFS speech).
     Ask the user, then separate the voice with Demucs (installed in the venv; ~80 MB model on first use):
     `.venv-voxpocket/bin/python -m demucs --two-stems=vocals -n htdemucs -d cpu -o <scratchpad>/demucs SOURCE`
     (~8 s for 40 s of audio; copy the source to an ASCII file name first). Check `no_vocals.wav` is loud
     (confirms the music) and `vocals.wav` drops to about -80 dB between words, then `add-voice` from
     `vocals.wav` (usually with `--clean`, separated vocals come out quiet) and keep the original as
     `voices/NAME/original_with_music.<ext>`.
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
- **Overlays:** a series badge and a section counter (`01 / 04`), kept clear of TikTok's UI. On tall phones (iPhone 14 Pro Max) TikTok crops ~55 px off each side, covers the top ~225 px, the right-hand buttons cover x > ~905 for y ~900–1610 and the caption block covers below ~1620 (`CROP_X`, `SAFE_TOP`, `RAIL_X`, `SAFE_BOTTOM` in `video.py`).
- **Sound effects:** synthesized, so there are no licensing issues: a whoosh at each scene, a swish for each card, a pop for each label, and a ding for each verdict and the outro.
- **Loudness:** normalized for social platforms. Font: Google Sans (it covers Khmer).

Inputs the user gives: the original text, the TTS text, the narration (or a voice to narrate with).
If any is missing, produce it first with section A. The narration's voice is a separate choice:
the same storyboard works with any `read` output via `--narration`.

1. **Storyboard** `videos/<stem>.json` (copy the structure of `videos/laravel_spring_boot.json`):
   - `tts_text`, `narration`, `output` (`outputs/video_<stem>/tiktok.mp4`), `title` (top badge).
   - `accent`: theme the video in the subject's brand colour (e.g. Git `#F05032`, Vercel `#0070F3`): it replaces
     the yellow on badges, counters, kicker, caption highlights and focus rings (badge text turns white on dark
     accents). Set `brands.a`/`b` colours to the same family so chips and background glows match.
   - `captions`: **exactly one per line of the TTS file, same order** (the renderer refuses a mismatch).
     Write them in the original's display spelling: English terms back in Latin script, digits as in
     the original, no trailing `។`. The TTS file is for the ear; captions are for the eye.
   - `brands.a` / `brands.b`: `name`, `color` (hex), `tag` (short label shown on the card).
     For a one-sided topic use `points` scenes (below) instead of forcing two cards.
   - `scenes`, each starting at a 1-based TTS `line` (a scene lasts until the next one starts):
     - `title`: big a VS b intro; optional `subtitle {line, text}` and `topics {line, items[]}` (≤ 4 short items).
     - `compare`: header `number` ("01"…), `title` (Khmer), `subtitle` (English); cards
       `a`/`b` = `{line, items: [[text, line], ...]}`. Each item pops in when its line is spoken.
       Keep items to short key phrases (≤ ~22 chars), ≤ 6 per card; chips shrink to fit two-three rows.
     - Conclusion: a `compare` with `"number": "✓", "verdict": true`; mark the winning item
       `[text, line, "strong"]` (filled chip + check, plays a ding).
     - `headline` (news intro): `kicker`, `headline`, optional `subtitle {line, text}`,
       `points: [[text, line], ...]` (numbered rows, ≤ 3-4).
     - `points` (one-sided topics, e.g. news): header like `compare` plus one tall card
       `card: {name, tag, items: [[text, line], ...]}` (fits ~11 short items).
     - `quote`: `phrases: [[text, line], ...]` revealed as spoken, `by: {name, role, line}`.
       Hides the caption box (the quote is already on screen). Optional `photo: {file, crop}` puts the speaker's
       portrait (square, rounded, accent ring) with name/role as a header and the quote below; `size` sets the
       quote text (default 52; 44-46 for 4 long phrases or verse). Follow each quote with a short `headline`
       "lesson" scene (kicker "មេរៀនទី N", 1-2 `points`) for lines that explain it, so those lines get captions.
       Example: `videos/khmer_advice.json` (portraits cropped from the user's image).
     - `step` (tutorials / how-to): header like `compare` (`number`, `title`, `subtitle`), a large window
       `media` (one object, or a list where each later window covers the previous one at its `line`), and a row
       of chips `items: [[text, line(, "strong")], ...]`. Keep chips to **one row (~3 short ones)**; a second
       row runs into the caption box. Optional `media_h` (default 600). Media kinds:
       - `{"kind": "browser", "file", "url", "crop": [x, y, w, h], "line", "focus": [x, y, w, h], "focus_line"}`:
         a screenshot region in a browser frame with the address `url`; `focus` (screenshot pixels) draws a
         glowing ring that pops in at `focus_line`. Use it to point at the button being described.
       - `{"kind": "code"|"terminal", "title", "line", "size", "lines": [[text, line], ...]}`: a code editor or
         terminal; each code line appears when its narration line is spoken (HTML, `$` prompts, `#` comments
         and `✓` output are coloured). Size 30 for ~9 short lines, 26 for ~7 lines up to ~45 chars.
         Font: Dank Mono (user-installed, commercial; comments in its cursive italic), then Fira Code
         (`brew install --cask font-fira-code`), then Menlo — first installed in `MONO_FONTS` wins, with
         ligatures. Glyphs Dank Mono lacks (`✓`, listed in `SYMBOLS`) are drawn with Menlo; if a still shows an
         empty box in a code line, add that character to `SYMBOLS`. Show the first command when the scene
         starts so the window is never empty for long.
       A `headline` can also take `media` (placed at `media_y`, default 720, filling the space down to the captions).
     Pick by content: two-sided comparison → `title` + `compare`; news/report → `headline` +
     `points` + `quote` for direct speech (see `videos/news.json`); step-by-step tutorial → `headline` with a
     screenshot + one `step` per step (see `videos/vercel_deploy.json`). `brands.a`/`b` colours tint the
     background glows (news uses blue `#3D6FE0` / red `#E0393E`) and `brands.a` colours the chips.
     Khmer has no spaces between words, so a long phrase may wrap mid-word: check stills and add a
     space at a natural phrase break in the display text (never in the TTS text).
   - `outro`: `{seconds, text, subtitle, line}`; `line` = first TTS line of the spoken sign-off (the card starts there; without it, after the narration) (e.g. "តាមដានសម្រាប់មេរៀនបន្ថែម" / "Follow for more").
   Put scene starts on the line that introduces the section (e.g. "ចំណុចទីមួយ…"), and each card's
   first item on the line that names that brand, so visuals land on the words.

   **Screenshots for tutorials.** Capture public pages with a throwaway headless Chrome (no user profile,
   so no accounts or cookies are involved), into `videos/images/<stem>/`:
   ```sh
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
     --user-data-dir=<scratchpad>/chrome-prof --window-size=1440,900 --virtual-time-budget=8000 \
     --screenshot=videos/images/<stem>/NAME.png "https://example.com/page"
   ```
   Look at every capture: pages behind a login (e.g. github.com/new, dashboards) come back as a sign-in page.
   Never use the user's own logged-in browser for this without asking. For logged-in steps, show the
   equivalent `terminal`/`code` window instead, or a small local demo page captured the same way
   (`file://…`, e.g. a "live site" for `your-project.vercel.app`). To find `crop`/`focus` coordinates, draw a
   100 px grid with labels over each capture (PIL), Read it, and read the button's box off the grid.
   Keep crops near 1.56:1 so a window fills the 780 px column at ~550 px tall. Tell the user which pages
   were captured; using a product's screenshots in a tutorial is their call.
2. **Preview before rendering:** `$KT video videos/<stem>.json --still 2 20 45 …` (one time per scene,
   plus a crowded moment near each scene's end) → `outputs/stills/`. Join them and look:
   `ffmpeg -i a.png -i b.png … -filter_complex "hstack=N,scale=2400:-1" contact.png`, then Read it.
   Check: nothing overflows a card or overlaps the caption box, no names split across caption lines,
   Khmer renders with correct stacks, check marks visible, focus rings sit on the right button,
   a long card `name` doesn't run into its `tag`, and no caption leaves one short word alone on a line.
   Compute the times from the narration instead of guessing (end of each scene = next scene's start - 0.4 s):
   `line_timings(json.load(open(sb)), Path("."))` in `khmer_tts.video` gives `(spans, narration_end)`
   (news style: add the 3.2 s intro).
3. **Render:** `$KT video videos/<stem>.json` (~1 min for 2.5 min of video on an M-series Mac, ~2 min on the
   GTX 1650 box; use a 600000 ms timeout).
   For another voice: `--narration outputs/<stem>_<voice> -o outputs/video_<stem>/tiktok_<voice>.mp4`.
4. **Verify the real file:** grab 2-4 frames mid-animation from the MP4
   (`ffmpeg -ss T -i out.mp4 -frames:v 1 f.png`) and Read them. Measure loudness
   (`ffmpeg -i out.mp4 -af ebur128=framelog=quiet:peak=true -f null - 2>&1 | grep -E "^ +(I|Peak):"`):
   about -14 to -17 LUFS is fine.
5. **Report:** path, duration, size, what each scene shows, how captions differ from the TTS text, and
   that the user should watch it before posting (you cannot hear or watch it).

**Background music** (storyboard `music`, mixed in `khmer_tts/music.py`, ducked under the voice):
- `{"synth": "ambient", "seed": 3, "level_db": -26, "duck_db": 7}`: an original synthesized bed (A minor
  pentatonic pads, drone, roneat-like mallet motif, reverb), so there are no licensing issues; change `seed` for
  a different motif. Sits ~13 dB under the voice while it speaks and swells in pauses and the outro.
- `{"file": "path.mp3", "level_db": -26, "duck_db": 7}`: a track the user has rights to (looped if short).
Ask which the user wants; never pick copyrighted music, a TikTok sound, or music separated from someone
else's recording on your own. Explainer style only (the news renderer has no music bed yet).
Example: `videos/khmer_advice.json`.

Layout constants (card positions, caption height, colours, SFX levels) live at the top of
`khmer_tts/video.py`. Change them there, not per video. The look: near-black background with a fading
grid and two brand glows (`make_background`), macOS-style windows with drop shadows, no progress bar
(the user asked for it to be removed; don't bring it back).

### News style with article photos (`"style": "news"`)

For a news article, prefer the broadcast style in `khmer_tts/news_video.py` (example: `videos/mercedes_rma_news.json`):
full-bleed photos with Ken Burns motion and cross-fades, a channel bug, date and optional `disclosure` tag,
key-fact chips, a lower third (red `tag`, white `title` bar, `subtitle` strip), captions, a scrolling `ticker`,
a 3.2 s intro stinger before the narration and an outro card with `credits`. Scene fields: `line`, `tag`, `title`,
`subtitle`, `images [[file, line], ...]` (from `images_dir`), `facts [[text, line], ...]`, `numbered`.
Landscape photos are shown whole over a blurred fill; square ones fill the photo area.
An article with a single photo is fine: list it in every scene's `images` and the zoom/pan alternates.
Keep each lower-third `title` to one line (~22 Khmer characters); the renderer warns
"lower third … reaches y=…, over the captions" when it wraps. For a cloned voice, set
`"disclosure": "សំឡេង AI"` and remind the user to tick TikTok's AI-generated label (they may remove the tag).
Example: `videos/kirirom_land.json`.

Getting a URL's text and photos: Thmey Thmey (and similar sites) sit behind Cloudflare, so curl gets a 403.
Read the article through Chrome (claude-in-chrome). `get_page_text` often returns a sidebar story instead
of the article; find the body with JS instead (`document.body.innerText`, sliced from the dateline such as
"ភ្នំពេញ៖" to the author box "អ្នកសរសេរអត្ថបទ"). The article photos are the full-size
`image.thmeythmey.com/pictures/<date>/image_*.jpg` files (also `og:image`); `thumb2/` images are other stories.
The image host blocks curl too (403). Ask before downloading (file name, host, size), then navigate the tab to
the image host so fetches are same-origin: for one photo, `fetch` it as a blob and click a download link;
for several, bundle them as base64 into ONE JSON download (Chrome blocks multiple automatic downloads).
Move/decode from `~/Downloads` into `videos/images/<stem>/`, delete the download, and close the tab.
Credit the source, author and photographer (if named) in `credits`. `អត្ថបទពាណិជ្ជកម្ម` (advertorial) is also
a menu item on every Thmey Thmey page, so check for it near the article's title/tags, not in the whole page
text; if the article really is one, set `disclosure` and tell the user.
