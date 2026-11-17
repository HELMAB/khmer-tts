# Khmer TTS

Khmer text-to-speech with voice cloning, built on
[VoxPocket](https://huggingface.co/muyleanging/VoxPocket-TTS-Khmer) (a Khmer fine-tune of VoxCPM 1.5).
A 4–9 second clip of a speaker plus its exact transcript is enough to speak new Khmer text in that voice.

## Layout

```
khmer_tts/          Python package and CLI (python -m khmer_tts)
  cli.py            commands: voices, add-voice, speak, read, vits
  voices.py         voice library (voices/<name>/)
  synth.py          model wrappers: VoxPocket (cloning), Vits (baseline)
  article.py        read a long text line by line and join the audio
voices/<name>/      reference.wav + reference.txt (+ original source.* / source.txt)
texts/              input texts (article_raw.txt = original, article.txt = prepared for TTS)
outputs/            generated audio (not tracked)
```

## Setup

Python 3.12, an NVIDIA GPU (tested on a 4 GB GTX 1650) and `ffmpeg` on `PATH`.

```sh
python3.12 -m venv .venv-voxpocket
.venv-voxpocket/bin/pip install -r requirements.txt
```

The model (~2 GB) downloads from Hugging Face on first use.

## Usage

```sh
alias ktts='.venv-voxpocket/bin/python -m khmer_tts'

ktts voices                                   # list voices
ktts speak --voice rfi "សួស្តីបងប្អូនទាំងអស់គ្នា។"  # -> outputs/samples_rfi/
ktts read  --voice rfi texts/article.txt      # -> outputs/article_rfi/article.wav (+ parts/)
ktts vits  "សួស្តី"                             # baseline single-voice model, no cloning
```

### Adding a voice

Pick a clean 4–9 s stretch (no music, ends on a pause) and write down exactly what is said in it:

```sh
ktts add-voice rfi recordings/rfi.mp3 --start 0 --duration 5 \
    --text "ជាមួយនឹងការរីកលូតលាស់យ៉ាងលឿនស្លេវនៃវិស័យបញ្ញាសិប្បនិម្មិត" \
    --transcript recordings/rfi.txt
```

A wrong transcript is the most common cause of a poor clone.
To find pauses: `ffmpeg -i in.mp3 -af silencedetect=noise=-30dB:d=0.18 -f null -`.

### Preparing text for `read`

VoxPocket reads long paragraphs badly, so prepare the text first (see `texts/article.txt`):

- one short phrase per line; lines ending in `។ ! ? ៕` get a longer pause
- write numbers as Khmer words (`២៩` → `ម្ភៃប្រាំបួន`) and expand `ៗ` (`សំខាន់ៗ` → `សំខាន់សំខាន់`)
- drop Latin text in brackets and bullet symbols

Lines that come out suspiciously short are retried with other seeds and listed at the end.
Each line is also saved to `parts/NN.wav`, so a bad line can be regenerated alone.

## Voices

| name | source | reference clip |
|------|--------|----------------|
| `my`  | `voices/my/source.mp3` | first 6.9 s |
| `rfi` | RFI Khmer news report, `voices/rfi/source.mp3` (full transcript in `source.txt`) | 0–5 s |
