from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

MODEL = "qwen3:8b"
OLLAMA_URL = "http://127.0.0.1:11434"
GUIDE_URL = (
    "https://github.com/GandalfSilente/audio-transcription-PRO"
    "#installazione-modello-pro"
)


class ModelUnavailable(RuntimeError):
    pass


class ProCancelled(RuntimeError):
    pass


def ensure_model_installed(model: str = MODEL) -> None:
    request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise ModelUnavailable(
            "Ollama locale non è in esecuzione o non risponde."
        ) from error

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ModelUnavailable("Ollama ha restituito un elenco modelli non valido.")
    installed = {
        entry.get("name") or entry.get("model")
        for entry in data.get("models", [])
        if isinstance(entry, dict)
    }
    if model not in installed:
        raise ModelUnavailable(f"Il modello locale {model} non è installato.")


def split_text(text: str, max_chars: int = 4500) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in text.split():
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def correct_transcript(
    text: str,
    progress: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    model: str = MODEL,
) -> str:
    chunks = split_text(text)
    if not chunks:
        raise ValueError("La trascrizione è vuota.")

    corrected: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(index, len(chunks))
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Correggi questa trascrizione automatica nella lingua "
                        "originale. Correggi errori di riconoscimento evidenti, "
                        "ortografia, grammatica e punteggiatura; migliora la "
                        "leggibilità con paragrafi naturali. Conserva significato, "
                        "fatti, nomi, ordine, dettagli, timestamp e tag degli "
                        "speaker. Non riassumere, tradurre o aggiungere "
                        "informazioni. Restituisci solo il testo corretto."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
        }
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Errore API Ollama ({error.code}): {detail or error.reason}"
            ) from error
        corrected_chunk = result.get("message", {}).get("content", "").strip()
        if not corrected_chunk:
            raise RuntimeError(f"Ollama non ha corretto il blocco {index}.")
        corrected.append(corrected_chunk)
        if check_cancelled:
            check_cancelled()
    return "\n\n".join(corrected)


def corrected_path_for(raw_path: Path) -> Path:
    candidate = raw_path.with_name(f"{raw_path.stem}.pro.txt")
    revision = 1
    while candidate.exists():
        candidate = raw_path.with_name(f"{raw_path.stem}.pro-{revision}.txt")
        revision += 1
    return candidate
