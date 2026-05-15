from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speak text with a voicepack Qwen voice ID.")
    parser.add_argument("--voice", required=True, help="Path to a library voice directory or voice.json.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--output", required=True, help="Output .mp3 or .wav path.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--keep-wav", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def convert_to_mp3(wav_path: Path, mp3_path: Path, quality: int = 2) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        str(quality),
        str(mp3_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg failed for {wav_path}")


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()
    if not text:
        raise ValueError("Provide --text or --text-file.")
    return text


def resolve_voice(path_text: str) -> tuple[Path, dict]:
    path = Path(path_text).expanduser().resolve()
    metadata_path = path if path.is_file() else path / "voice.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Voice metadata not found: {metadata_path}")
    return metadata_path.parent, json.loads(metadata_path.read_text(encoding="utf-8"))


def load_prompt_items(prompt_file: Path) -> tuple[dict, list[VoiceClonePromptItem]]:
    payload = torch.load(prompt_file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"Invalid Qwen voice prompt file: {prompt_file}")

    items = []
    for raw in payload["items"]:
        ref_code = raw.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = raw.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Voice prompt item is missing ref_spk_embedding.")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(raw.get("x_vector_only_mode", False)),
                icl_mode=bool(raw.get("icl_mode", not bool(raw.get("x_vector_only_mode", False)))),
                ref_text=raw.get("ref_text", None),
            )
        )
    return payload, items


def main() -> int:
    args = parse_args()
    voice_dir, voice = resolve_voice(args.voice)
    text = read_text(args)

    prompt_file = (voice_dir / voice.get("voice_file", "voice.pt")).resolve()
    if not prompt_file.exists():
        raise FileNotFoundError(f"Voice prompt file not found: {prompt_file}")

    defaults = voice.get("generation_defaults", {})
    payload, prompt_items = load_prompt_items(prompt_file)

    model = args.model or voice.get("model") or payload.get("model")
    if not model:
        raise ValueError("No model found in voice metadata or prompt file.")
    language = args.language or voice.get("language", "English")
    if language == "en-US":
        language = "English"
    device = args.device or defaults.get("device", "mps")
    dtype = args.dtype or defaults.get("dtype", "float16")
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 16010))
    temperature = args.temperature if args.temperature is not None else defaults.get("temperature", 0.82)
    top_k = args.top_k if args.top_k is not None else defaults.get("top_k", 50)
    top_p = args.top_p if args.top_p is not None else defaults.get("top_p", 0.95)
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else defaults.get("repetition_penalty", 1.05)
    )

    set_seed(seed)
    print(f"Loading {model} on {device} ({dtype})")
    tts = Qwen3TTSModel.from_pretrained(
        model,
        device_map=device,
        dtype=dtype_from_name(dtype),
        attn_implementation=None,
    )

    wavs, sample_rate = tts.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt_items,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".wav":
        sf.write(output_path, wavs[0], sample_rate)
    elif output_path.suffix.lower() == ".mp3":
        wav_path = output_path.with_suffix(".wav")
        sf.write(wav_path, wavs[0], sample_rate)
        convert_to_mp3(wav_path, output_path)
        if not args.keep_wav:
            wav_path.unlink(missing_ok=True)
    else:
        raise ValueError("Output path must end in .mp3 or .wav.")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
