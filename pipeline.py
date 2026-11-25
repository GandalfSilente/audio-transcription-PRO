from __future__ import annotations

import gc
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ASR_MODEL = "large-v3"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def transcribe_audio(
    audio_path: Path, language: str | None = "it"
) -> str:
    import ctranslate2
    from faster_whisper import WhisperModel

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    use_cuda = ctranslate2.get_cuda_device_count() > 0
    model = WhisperModel(
        ASR_MODEL,
        device="cuda" if use_cuda else "cpu",
        compute_type="float16" if use_cuda else "int8",
    )
    segments, _ = model.transcribe(
        str(audio_path), language=language, vad_filter=True
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    del model
    gc.collect()
    if not transcript:
        raise RuntimeError("No speech detected in the audio.")
    return transcript


def split_text(text: str, max_chars: int = 5000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in text.split():
        extra_length = len(word) + (1 if current else 0)
        if current and current_length + extra_length > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def correct_transcript(
    transcript: str,
    model: str = OLLAMA_MODEL,
    progress: Callable[[str], None] | None = None,
) -> str:
    chunks = split_text(transcript)
    if not chunks:
        raise ValueError("Transcript is empty.")

    corrected: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress(f"Local correction {index}/{len(chunks)}")
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Proofread this speech-recognition transcript in its "
                        "original language. Correct spelling, punctuation, "
                        "grammar, and obvious recognition errors. Improve "
                        "readability with natural paragraph breaks and clear "
                        "phrasing, removing only redundant filler or stutters. "
                        "Preserve the speaker's meaning, facts, names, order, "
                        "and level of detail. Do not summarize, translate, add "
                        "information, or explain edits. Return only the "
                        "corrected transcript."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
        }
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Ollama API error ({error.code}): {detail or error.reason}"
            ) from error
        corrected_chunk = result.get("message", {}).get("content", "").strip()
        if not corrected_chunk:
            raise RuntimeError(f"Ollama returned an empty correction for chunk {index}.")
        corrected.append(corrected_chunk)

    return "\n\n".join(corrected)


def transcript_paths(project_dir: Path, audio_path: Path) -> tuple[Path, Path]:
    output_dir = project_dir / "Trascrizioni"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = audio_path.stem
    revision = 1
    raw_path = output_dir / f"{name}.raw.txt"
    corrected_path = output_dir / f"{name}.corrected.txt"
    while raw_path.exists() or corrected_path.exists():
        name = f"{audio_path.stem} ({revision})"
        revision += 1
        raw_path = output_dir / f"{name}.raw.txt"
        corrected_path = output_dir / f"{name}.corrected.txt"
    return raw_path, corrected_path


def next_corrected_path(raw_path: Path) -> Path:
    name = raw_path.name.removesuffix(".raw.txt")
    candidate = raw_path.parent / f"{name}.corrected.txt"
    revision = 1
    while candidate.exists():
        candidate = raw_path.parent / f"{name} ({revision}).corrected.txt"
        revision += 1
    return candidate
