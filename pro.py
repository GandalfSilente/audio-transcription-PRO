from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

OLLAMA_URL = "http://127.0.0.1:11434"
LM_STUDIO_URL = "http://127.0.0.1:1234"
OLLAMA_MODEL = "qwen3:8b"
GUIDE_URL = (
    "https://github.com/GandalfSilente/audio-transcription-PRO"
    "#installazione-modelli-locali-pro"
)


@dataclass(frozen=True)
class LocalBackend:
    provider: str
    model: str
    label: str


class ModelUnavailable(RuntimeError):
    pass


class ProCancelled(RuntimeError):
    pass


def _read_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _lmstudio_disk_models() -> list[dict]:
    executable = shutil.which("lms")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        models = json.loads(result.stdout) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    if not isinstance(models, list):
        return []
    return [
        model
        for model in models
        if isinstance(model, dict) and model.get("type") == "llm"
    ]


def _model_key(model: dict) -> str:
    return str(model.get("id") or model.get("modelKey") or "")


def _is_text_model(model: dict) -> bool:
    return model.get("type") in ("llm", "vlm")


def _preferred(models: list[dict]) -> dict:
    def rank(model: dict) -> tuple[int, int, int]:
        name = f"{_model_key(model)} {model.get('displayName', '')}".lower()
        is_qwen7 = "qwen" in name and "7b" in name
        is_qwen = "qwen" in name
        size = int(model.get("sizeBytes") or 0)
        return (0 if is_qwen7 else 1, 0 if is_qwen else 1, size)

    return min(models, key=rank)


def _ollama_backend() -> LocalBackend | None:
    data = _read_json(f"{OLLAMA_URL}/api/tags")
    models = data.get("models") if data else None
    if not isinstance(models, list):
        return None
    candidates = [
        model
        for model in models
        if isinstance(model, dict)
        and (model.get("name") or model.get("model"))
        and "embed" not in str(model.get("name", "")).lower()
    ]
    if not candidates:
        return None
    exact = next(
        (
            model
            for model in candidates
            if (model.get("name") or model.get("model")) == OLLAMA_MODEL
        ),
        None,
    )
    selected = exact or _preferred(
        [
            {
                "id": model.get("name") or model.get("model"),
                "displayName": model.get("name") or model.get("model"),
                "sizeBytes": model.get("size", 0),
            }
            for model in candidates
        ]
    )
    model_id = str(
        selected.get("id") or selected.get("modelKey") or selected.get("name") or ""
    )
    return LocalBackend("ollama", model_id, f"Ollama • {model_id}")


def detect_local_backend() -> LocalBackend:
    lmstudio = _read_json(f"{LM_STUDIO_URL}/api/v0/models")
    if lmstudio is None:
        lmstudio = _read_json(f"{LM_STUDIO_URL}/v1/models")
        if lmstudio and isinstance(lmstudio.get("data"), list):
            lmstudio = {
                **lmstudio,
                "data": [
                    {
                        **model,
                        "type": model.get("type") or "llm",
                        "state": model.get("state") or "loaded",
                    }
                    for model in lmstudio["data"]
                    if isinstance(model, dict)
                ],
            }
    if lmstudio is not None and isinstance(lmstudio.get("data"), list):
        models = [model for model in lmstudio["data"] if isinstance(model, dict)]
        loaded = [
            model
            for model in models
            if _is_text_model(model)
            and model.get("state", "loaded") == "loaded"
        ]
        if loaded:
            model = _preferred(loaded)
            model_id = _model_key(model)
            return LocalBackend("lmstudio", model_id, f"LM Studio • {model_id}")

    ollama_backend = _ollama_backend()
    if ollama_backend:
        return ollama_backend

    installed = (
        [model for model in lmstudio["data"] if _is_text_model(model)]
        if lmstudio and isinstance(lmstudio.get("data"), list)
        else []
    )
    if not installed:
        installed = _lmstudio_disk_models()
    if installed:
        model = _preferred(installed)
        model_id = _model_key(model)
        raise ModelUnavailable(
            f"LM Studio ha trovato {model_id}, ma il modello non è caricato. "
            "Caricalo in LM Studio e avvia il Local Server, poi riattiva PRO."
        )

    raise ModelUnavailable(
        "Nessun modello locale trovato. Installa un modello con Ollama oppure "
        "scaricane uno in LM Studio, caricalo e avvia il Local Server."
    )


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
    provider: str = "ollama",
    model: str = OLLAMA_MODEL,
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
        messages = [
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
        ]
        if provider == "ollama":
            url = f"{OLLAMA_URL}/api/chat"
            payload = {
                "model": model,
                "stream": False,
                "think": False,
                "messages": messages,
            }
        elif provider == "lmstudio":
            url = f"{LM_STUDIO_URL}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
        else:
            raise ValueError(f"Provider LLM locale non supportato: {provider}")

        request = urllib.request.Request(
            url,
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
                f"Errore API {provider} ({error.code}): {detail or error.reason}"
            ) from error

        if provider == "ollama":
            content = result.get("message", {}).get("content", "")
        else:
            choices = result.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "")
                if choices and isinstance(choices[0], dict)
                else ""
            )
        corrected_chunk = content.strip() if isinstance(content, str) else ""
        if not corrected_chunk:
            raise RuntimeError(f"Il modello locale non ha corretto il blocco {index}.")
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
