<p align="center">
  <img src="docs/banner.png" alt="Khmer TTS: speak Khmer in any voice" width="100%">
</p>

# Khmer TTS

Turn Khmer text into natural speech in **any voice**, and turn that speech into a ready-to-post
TikTok / Reels / Shorts video.

- **Clone a voice** from a short 4–9 second recording
- **Read Khmer text aloud**: a sentence, or a whole article
- **Make vertical videos** with captions, cards and sound effects

Built on [VoxPocket](https://huggingface.co/muyleanging/VoxPocket-TTS-Khmer), a Khmer model based on VoxCPM 1.5.

## Example

A news video made from one article, voiced with the `rfi` voice:

<a href="docs/example.mp4"><img src="docs/example.jpg" alt="Example TikTok news video" width="300"></a>

▶️ [Watch the video](docs/example.mp4)

## Requirements

- Python 3.12
- `ffmpeg`
- An NVIDIA GPU (4 GB is enough) or an Apple Silicon Mac
- The [Google Sans](https://fonts.google.com/specimen/Google+Sans) font, for videos

## Install

```sh
python3.12 -m venv .venv-voxpocket
.venv-voxpocket/bin/pip install -r requirements.txt
alias ktts='.venv-voxpocket/bin/python -m khmer_tts'
```

The model (~2 GB) downloads on first use.

<details>
<summary>Mac only: fix Khmer text in videos</summary>

```sh
brew install fribidi
sudo ln -s /opt/homebrew/lib/libfribidi.dylib /usr/local/lib/
```
</details>

## Quick start

```sh
ktts voices                                      # list available voices
ktts speak --voice rfi "សួស្តីបងប្អូនទាំងអស់គ្នា។"     # say one sentence
ktts read  --voice rfi texts/article.txt         # read a whole text file
ktts video videos/my_video.json                  # make a TikTok video
```

Results are saved in `outputs/`.

## Make videos with Claude Code

The easiest way to make a video is to open this folder in [Claude Code](https://claude.com/claude-code) and
describe the video you want. It writes the Khmer script, records the voice and renders the video for you.

**News video** from an article link:

```text
Make a TikTok news video from https://www.thmeythmey.com/detail/123456 using the rfi voice
```

**Comparison video** between two things:

```text
Make a TikTok video comparing Laravel and Spring Boot for Khmer developers, using the general voice
```

**Tutorial video** with step-by-step screens:

```text
Make a step-by-step TikTok tutorial in Khmer on how to deploy a website to Vercel, using the general voice
```

**Promotion video** for a product or project:

```text
Make a 60-second promotion video in Khmer for my app Sevakam, a marketplace for booking trusted home-service workers in Cambodia.
Use the advice voice and end with a call to sign up.
```

## Add your own voice

1. Find a clean 4–9 second part of a recording (no music, ends on a pause).
2. Write down **exactly** what is said in it.
3. Add it:

```sh
ktts add-voice myvoice recording.mp3 --start 0 --duration 5 --text "what is said in the clip"
```

Tips:
- Not sure where to cut? `ktts clips recording.mp3` suggests good clips.
- Noisy or quiet recording? Add `--clean`.
- A wrong transcript is the most common cause of a bad voice.

## Tips for long text

The model reads short phrases best. Before using `ktts read`:

- Put **one short phrase per line**
- Write numbers as words (`២៩` → `ម្ភៃប្រាំបួន`)
- Write `ៗ` out in full (`សំខាន់ៗ` → `សំខាន់សំខាន់`)
- Remove English text in brackets and bullet symbols

Each line is saved separately in `parts/`, so you can redo a single bad line.

## Included voices

| Voice | Description |
|-------|-------------|
| `rfi` | RFI Khmer news reader |
| `preapsovath` | Preap Sovath, interview clip (`preapsovath_clean` = cleaned audio) |
| `general` | Tech TikTok narrator |
| `advice` | Calm life-advice narrator |
