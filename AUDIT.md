# AUDIT.md — agent-announcer Qwen TTS (bedrock ledger)

## 2026-06-18 — "recurring male voice / RAM blowup"   [report-then-fix]

### Symptom
Announcements intermittently switch to a male voice; coincides with RAM ~92% and heavy swap. User had to manually kill the Qwen python server to recover; it kept recurring.

### Claim under test
"The warm Qwen TTS server leaks GPU/MPS memory per render and ratchets until swap."

### Verdict: **FALSE** — disproven empirically
Runnable repros: `/tmp/qwen_leak_test.py` (constant workload), `/tmp/qwen_leak_test2.py` (variable/long).
- 6 constant renders: torch MPS `current` flat at 4004MB; `driver` +74MB on render 1, then **+0MB** renders 2-6.
- Interleaved short / 28s-audio-long renders: peak driver 4928MB, **returns to 4916MB** after a long render → no high-water pinning. Model is a stable ~4.9GB.

### Actual root cause
Whole-machine memory **pressure**, not a leak. The announcer loaded **~9.7GB of models per announcement** (4.9GB cortana TTS + 4.8GB `qwen2.5:7b` summary) on a 24GB Mac already running heavy work (REAPER, iOS sim, multiple agents). Under swap, a normally ~6s render exceeds the 120s `/speak` timeout and falls to macOS `say` (default voice = male). Killing the server frees ~5GB → temporary fix; recurs on the next announcement. (Original trigger also included a 2-day orphaned Prisma `schema-engine` pegging a core, ~36GB footprint.)

### Fixes applied
| # | Change | Where | Effect |
|---|--------|-------|--------|
| 1 | summary model `qwen2.5:7b` → `gemma3:1b` | `~/.config/tab-tts/env` | −~4GB footprint (9.7→5.7GB) |
| 2 | `/speak` timeout 120s → 60s | env | fail fast to fallback under thrash |
| 3 | TTS idle-timeout 300s → 120s | env | unload the 4.9GB model sooner |
| 4 | memory-pressure guard: skip cold-start at pressure ≥ warn | announce.sh `system_memory_critical` + env `TAB_TTS_QWEN_SKIP_PRESSURE_LEVEL=2` | don't pile 5GB onto a thrashing box |
| 5 | `say` last-resort → female Samantha | announce.sh + env `TAB_TTS_SAY_VOICE` | worst case no longer male |
| 6 | `reclaim_wedged_qwen_server` (kill port squatter before rebind) | announce.sh | recover from a genuine port wedge |

serve.py itself was **not** modified — it is clean (no leak).

### Residual risks / fragility (not yet fixed)
- **serve.py single render lock** (serve.py:264): a hung render keeps `active_renders` > 0 and suppresses idle-exit, so the server never self-unloads. Recommend a server-side wall-clock render cap. _(medium)_
- Slow PyTorch MPS backend leak (#164299) over many hours unmeasured; idle-timeout recycling mitigates. _(low)_
- gemma3:1b summary quality slightly below 7b — acceptable for a spoken line; bump to `gemma3:4b` (+~1.5GB) if desired.
- ElevenLabs primary parked (`TAB_TTS_PROVIDER=qwen`); restore after topping up credits.

### Verification
`bash -n` clean; config loads (gemma3:1b, 60s/120s, guard=2); gemma3:1b summarizes in 1.7s; guard logic correct at current pressure (level 1 → start normally). Live proof on the next Stop-hook announcement.
