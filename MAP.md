# MAP.md — Qwen TTS warm-server stack (devour atlas)

_Studied 2026-06-18. Scope: the local Qwen TTS server powering the agent-announcer voice, its model library, and its shell consumer. Claims are verifiable — file:line given. Future readers: verify, don't trust._

## Repo identity
`~/dev/voicepack` — distributable voice library + runners. Python 3.12, PyTorch/MPS, the `qwen_tts` package (Qwen3-TTS-12Hz-1.7B). Venvs: `.venv`, **`.venv-qwen` (active)**, `.venv-f5`. Voices in `library/<id>/` (currently only `cortana`, female). Git repo.

## Terrain
- `runners/qwen/serve.py` — warm HTTP server (the always-on path). Loads ONE voice + model at startup; serves `/speak` + `/health`; self-exits on idle.
- `runners/qwen/speak.py` — one-shot CLI + shared helpers (`resolve_voice`, `load_prompt_items`, `set_seed`, `convert_to_mp3`) imported by serve.py.
- `library/cortana/{voice.json,voice.pt,preview.mp3}` — metadata + speaker embedding (ref_code (89,16) + 2048-d x-vector, ICL mode, female).
- `.venv-qwen/.../qwen_tts/` — model library (`Qwen3TTSModel`, `generate_voice_clone`).
- CONSUMER (separate repo): `~/dev/skillpack/plugins/agent-announcer-when-agent-finishes/scripts/announce.sh` + `~/.config/tab-tts/env`.

## Runtime route (one announcement)
Stop hook → announce.sh → summary via ollama (localhost:11434) → `play_qwen_tts` → `start_qwen_server_if_needed` (health-check :8765, cold-start if down) → POST `/speak` → serve.py `do_POST` (serve.py:255) → `render()` under one `self.lock` (serve.py:264) → `tts.generate_voice_clone` (qwen3_tts_model.py:469, `@torch.no_grad`) → `talker.generate` (modeling_qwen3_tts.py:2272) → wav → write file → `afplay`. `release_runtime_memory()` (gc + mps.empty_cache) in render finally (serve.py:221-223).

## Memory lifecycle (the investigated question)
- Model resident **~4.9GB MPS, STABLE** — measured flat across renders and lengths (NO per-render leak, NO high-water pinning). Wrapper holds no growing Python state.
- `modeling_qwen3_tts.py:2064` forces `output_hidden_states=True` (retains per-layer/per-step hidden states *during* a call) — a per-call high-water mark, but released between calls in practice (measured).
- `qwen_tts` never calls `empty_cache`; the MPS allocator doesn't return freed blocks to the OS → reserved memory sits at high-water but does NOT ratchet for fixed workloads.
- **Real failure mode = SYSTEM memory pressure**, not a leak: TTS 4.9GB + ollama summary 4.8GB + user workload on a 24GB box → swap → render exceeds the `/speak` timeout → `say` fallback.

## Lifecycle / knobs
- serve.py `--idle-timeout` watchdog daemon (serve.py:273-295) → `server.shutdown()` when idle. Single render lock serializes everything; if a render hangs, `active_renders` can stick >0 and suppress idle-exit (serve.py fragility, medium).
- Key env knobs (full list in AUDIT.md): `TAB_TTS_SUMMARY_MODEL`, `TAB_TTS_QWEN_SERVER_TIMEOUT_SEC`, `TAB_TTS_QWEN_SERVER_IDLE_TIMEOUT_SEC`, `TAB_TTS_QWEN_SKIP_PRESSURE_LEVEL`, `TAB_TTS_SAY_VOICE`, `TAB_TTS_PROVIDER`.

## Blast radius
- serve.py is the only long-running process and it is **clean — NO changes were needed there.**
- All fixes live in the announcer + config (footprint, timeouts, pressure guard, female say). Concurrency safety rides on the `mkdir` start-lock + pid file (NOT the 2s debounce) — any new kill/restart path must reuse them.

## Unknowns / next probes
- Slow PyTorch MPS backend leak (#164299) over many hours not measured (8-render test too short); idle-timeout recycling mitigates.
- Real-summary render-time distribution under the new gemma3:1b path (expected ~3-6s) — confirm from /tmp/tab-tts.log over a work session.
