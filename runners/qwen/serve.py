from __future__ import annotations

import argparse
import gc
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from speak import (
    convert_to_mp3,
    dtype_from_name,
    load_prompt_items,
    resolve_voice,
    set_seed,
)


def release_runtime_memory() -> None:
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm localhost Qwen TTS server.")
    parser.add_argument("--voice", required=True, help="Default library voice directory or voice.json.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0,
        help="Exit after this many seconds without /speak requests. Use 0 to disable.",
    )
    parser.add_argument(
        "--idle-check-interval",
        type=float,
        default=5,
        help="Seconds between idle-timeout checks.",
    )
    args = parser.parse_args()
    if args.idle_timeout < 0:
        parser.error("--idle-timeout must be 0 or greater.")
    if args.idle_check_interval <= 0:
        parser.error("--idle-check-interval must be greater than 0.")
    return args


class QwenRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.voice_key = ""
        self.voice_dir = Path()
        self.voice: dict[str, Any] = {}
        self.prompt_items = []
        self.model_name = ""
        self.language = ""
        self.device = ""
        self.dtype = ""
        self.seed = 0
        self.temperature = 0.0
        self.top_k = 0
        self.top_p = 0.0
        self.repetition_penalty = 0.0
        self.tts: Qwen3TTSModel | None = None
        self.lock = threading.Lock()
        self.activity_lock = threading.Lock()
        self.last_speak_at = time.monotonic()
        self.active_renders = 0
        self.load_voice(args.voice)
        self.load_model()

    def reset_idle_clock(self) -> None:
        with self.activity_lock:
            self.last_speak_at = time.monotonic()

    def mark_render_start(self) -> None:
        with self.activity_lock:
            self.active_renders += 1
            self.last_speak_at = time.monotonic()

    def mark_render_end(self) -> None:
        with self.activity_lock:
            self.active_renders = max(0, self.active_renders - 1)
            self.last_speak_at = time.monotonic()

    def idle_status(self) -> tuple[bool, float]:
        with self.activity_lock:
            return self.active_renders > 0, time.monotonic() - self.last_speak_at

    def load_voice(self, voice_path: str) -> None:
        voice_dir, voice = resolve_voice(voice_path)
        prompt_file = (voice_dir / voice.get("voice_file", "voice.pt")).resolve()
        if not prompt_file.exists():
            raise FileNotFoundError(f"Voice prompt file not found: {prompt_file}")

        defaults = voice.get("generation_defaults", {})
        payload, prompt_items = load_prompt_items(prompt_file)
        language = self.args.language or voice.get("language", "English")
        if language == "en-US":
            language = "English"

        self.voice_key = str(voice_dir)
        self.voice_dir = voice_dir
        self.voice = voice
        self.prompt_items = prompt_items
        self.model_name = self.args.model or voice.get("model") or payload.get("model")
        if not self.model_name:
            raise ValueError("No model found in voice metadata or prompt file.")
        self.language = language
        self.device = self.args.device or defaults.get("device", "mps")
        self.dtype = self.args.dtype or defaults.get("dtype", "float16")
        self.seed = self.args.seed if self.args.seed is not None else int(defaults.get("seed", 16010))
        self.temperature = (
            self.args.temperature
            if self.args.temperature is not None
            else float(defaults.get("temperature", 0.82))
        )
        self.top_k = self.args.top_k if self.args.top_k is not None else int(defaults.get("top_k", 50))
        self.top_p = self.args.top_p if self.args.top_p is not None else float(defaults.get("top_p", 0.95))
        self.repetition_penalty = (
            self.args.repetition_penalty
            if self.args.repetition_penalty is not None
            else float(defaults.get("repetition_penalty", 1.05))
        )

    def load_model(self) -> None:
        if self.tts is not None:
            self.tts = None
            release_runtime_memory()
        print(f"Loading {self.model_name} on {self.device} ({self.dtype})", flush=True)
        self.tts = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=dtype_from_name(self.dtype),
            attn_implementation=None,
        )

    def maybe_reload_voice(self, voice_path: str | None) -> None:
        if not voice_path:
            return
        requested = str(Path(voice_path).expanduser().resolve())
        requested_dir = requested if Path(requested).is_dir() else str(Path(requested).parent)
        if requested_dir == self.voice_key:
            return
        old_model = self.model_name
        self.load_voice(requested)
        if self.model_name != old_model:
            self.load_model()

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Missing text.")
        output_text = str(payload.get("output", "")).strip()
        output = Path(output_text).expanduser() if output_text else None

        self.maybe_reload_voice(payload.get("voice"))
        seed = int(payload.get("seed", self.seed))
        temperature = float(payload.get("temperature", self.temperature))
        top_k = int(payload.get("top_k", self.top_k))
        top_p = float(payload.get("top_p", self.top_p))
        repetition_penalty = float(payload.get("repetition_penalty", self.repetition_penalty))

        set_seed(seed)
        assert self.tts is not None
        started = time.perf_counter()
        wavs: Any = None
        try:
            wavs, sample_rate = self.tts.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self.prompt_items,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Render WAV bytes in-memory so audio can be returned over HTTP to a
            # remote client (Mac plays magi's render) as well as / instead of a file.
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sample_rate, format="WAV")
            audio_bytes = buffer.getvalue()

            if output is not None:
                output = output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.suffix.lower() == ".wav":
                    output.write_bytes(audio_bytes)
                elif output.suffix.lower() == ".mp3":
                    wav_path = output.with_suffix(".wav")
                    wav_path.write_bytes(audio_bytes)
                    convert_to_mp3(wav_path, output)
                    wav_path.unlink(missing_ok=True)
                else:
                    raise ValueError("Output path must end in .mp3 or .wav.")

            return {
                "ok": True,
                "output": str(output) if output is not None else "",
                "voice": self.voice_key,
                "model": self.model_name,
                "sample_rate": sample_rate,
                "render_ms": round((time.perf_counter() - started) * 1000),
                "_audio": audio_bytes,
            }
        finally:
            wavs = None
            release_runtime_memory()


class Handler(BaseHTTPRequestHandler):
    runtime: QwenRuntime

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "voice": self.runtime.voice_key,
                    "model": self.runtime.model_name,
                    "device": self.runtime.device,
                    "dtype": self.runtime.dtype,
                },
            )
            return
        self.send_json(404, {"ok": False, "error": "not found"})

    def send_audio(self, code: int, audio: bytes, meta: dict[str, Any]) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("X-Model", str(meta.get("model", "")))
        self.send_header("X-Sample-Rate", str(meta.get("sample_rate", "")))
        self.send_header("X-Render-Ms", str(meta.get("render_ms", "")))
        self.end_headers()
        self.wfile.write(audio)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/speak":
            self.send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            # Remote clients ask for raw audio via {"return":"audio"} or Accept: audio/*
            want_audio = (
                str(payload.get("return", "")).lower() == "audio"
                or "audio/" in (self.headers.get("Accept", "") or "")
            )
            self.runtime.mark_render_start()
            try:
                with self.runtime.lock:
                    response = self.runtime.render(payload)
            finally:
                self.runtime.mark_render_end()
            audio = response.pop("_audio", None)
            if want_audio and audio is not None:
                self.send_audio(200, audio, response)
            else:
                self.send_json(200, response)
        except Exception as exc:  # noqa: BLE001 - surface local server failures as JSON.
            self.send_json(500, {"ok": False, "error": str(exc)})


def start_idle_watchdog(
    server: ThreadingHTTPServer,
    runtime: QwenRuntime,
    timeout_seconds: float,
    check_interval_seconds: float,
) -> threading.Event:
    stop_event = threading.Event()

    def watch_idle() -> None:
        while not stop_event.wait(check_interval_seconds):
            render_active, idle_for = runtime.idle_status()
            if render_active or idle_for < timeout_seconds:
                continue
            print(
                f"Idle timeout reached after {idle_for:.1f}s without /speak requests; shutting down.",
                flush=True,
            )
            server.shutdown()
            break

    thread = threading.Thread(target=watch_idle, name="qwen-tts-idle-watchdog", daemon=True)
    thread.start()
    return stop_event


def main() -> int:
    args = parse_args()
    runtime = QwenRuntime(args)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Qwen TTS server listening on http://{args.host}:{args.port}", flush=True)
    runtime.reset_idle_clock()
    idle_stop_event: threading.Event | None = None
    if args.idle_timeout > 0:
        idle_stop_event = start_idle_watchdog(
            server,
            runtime,
            args.idle_timeout,
            args.idle_check_interval,
        )
        print(
            f"Idle timeout enabled: {args.idle_timeout:g}s without /speak requests.",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if idle_stop_event is not None:
            idle_stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
