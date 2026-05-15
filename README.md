# voicepack

`voicepack` is a small library of portable voice IDs for local TTS agents.

It does **not** ship full TTS models. It ships saved voice presets, previews,
metadata, and a runner that knows how to use them.

## The Simple Model

```text
voice.pt
  The saved voice identity. Small. This is the voice ID.

voice.json
  Metadata agents can read: name, tags, model, engine, style, and safety notes.

runners/qwen/speak.py
  The runner. It loads a voice ID and asks Qwen to generate audio from text.

runners/qwen/serve.py
  Optional warm localhost server. It keeps Qwen loaded and renders requests
  without starting a fresh Python process every time.

Qwen Base model
  The actual TTS engine. Downloaded from Hugging Face on the user's machine.
```

So the flow is:

```text
voice ID + text -> runner -> local Qwen engine -> mp3/wav audio
```

## What's Included

```text
library/
  index.json
  cortana/
    voice.pt
    voice.json
    preview.mp3
    README.md

runners/
  qwen/
    speak.py
    serve.py
    requirements.txt

schemas/
  voice.schema.json
```

## Prerequisites

You need:

- Python 3.10+
- `ffmpeg` for MP3 output
- enough disk space for the Qwen model download
- the Python packages in `runners/qwen/requirements.txt`

On macOS:

```bash
brew install ffmpeg
```

Install Python deps:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r runners/qwen/requirements.txt
```

The first run downloads the Qwen base model from Hugging Face:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

That model is not stored in this repo.

## Use A Voice

```bash
python runners/qwen/speak.py \
  --voice library/cortana \
  --text "Hello. I have the system summary ready." \
  --output /tmp/voicepack-demo.mp3
```

Play it:

```bash
afplay /tmp/voicepack-demo.mp3
```

For non-macOS systems, use any audio player that can play MP3 files.

## Keep Qwen Warm

For agent hooks, a one-shot `speak.py` call is simple but slow because it starts
Python and loads Qwen for every line. Run the warm server when you want lower
latency:

```bash
python runners/qwen/serve.py \
  --voice library/cortana \
  --host 127.0.0.1 \
  --port 8765 \
  --device mps \
  --dtype float16
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

Render through the server:

```bash
curl -sS http://127.0.0.1:8765/speak \
  -H "Content-Type: application/json" \
  --data '{"text":"System ready.","voice":"library/cortana","output":"/tmp/voicepack-server.wav"}'
```

## Search The Library

Agents and scripts should inspect `library/index.json`.

Find a holographic voice:

```bash
jq '.[] | select(.keywords[]? == "holographic")' library/index.json
```

Find a sci-fi assistant:

```bash
jq '.[] | select((.keywords | index("sci-fi")) and (.keywords | index("ai-assistant")))' library/index.json
```

## Add A New Voice

Create a new folder:

```text
library/my-voice/
  voice.pt
  voice.json
  preview.mp3
  README.md
```

Then add it to:

```text
library/index.json
```

Keep each voice folder self-contained. The library should stay flat by voice ID:

```text
library/cortana
library/atlas
library/nova
```

Do not group voices by model unless the same voice has multiple engine-specific
implementations.

## What Is Not Included

The voice factory is not part of the public library. Local experiments,
generated batches, raw references, research notes, and prompt exploration are
ignored by Git:

```text
experiments/
references/
samples/
scripts/
voicepacks/
research/
```

Those folders are useful for creating voices, but they are not the product.

## Naming And Safety

Some voice IDs use familiar codenames for local ergonomics. Voices in this
library are original generated presets. They are not affiliated with, endorsed
by, or extracted from any named character, actor, company, or product.

Do not market a voice as official, extracted, cloned from copyrighted media, or
endorsed by the owner of a character or brand.
