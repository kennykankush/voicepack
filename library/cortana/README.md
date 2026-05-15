# Cortana

Original holographic female AI assistant voice preset for Qwen3 TTS Base.

This folder is the distributable voice ID:

- `voice.pt` - saved Qwen voice prompt
- `voice.json` - agent-readable metadata
- `preview.mp3` - short audition clip

## Use

```bash
python runners/qwen/speak.py \
  --voice library/cortana \
  --text "Hello. I have the system summary ready." \
  --output /tmp/cortana-demo.mp3
```

## Notes

This is an unofficial original generated preset. It is not affiliated with or
extracted from Microsoft, Halo, 343 Industries, Xbox, Jen Taylor, or any named
character owner.
