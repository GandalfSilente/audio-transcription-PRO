from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
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
    size_bytes: int = 0


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
        if isinstance(model, dict) and model.get("type") in ("llm", "vlm")
    ]


def _run_lms(*args: str, timeout: int) -> str:
    executable = shutil.which("lms")
    if not executable:
        raise ModelUnavailable(
            "LM Studio CLI `lms` non trovato. Abilita il comando CLI nelle "
            "impostazioni di LM Studio oppure installa llmster headless."
        )
    try:
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ModelUnavailable(
            f"Comando LM Studio non completato (`lms {' '.join(args)}`): {error}"
        ) from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ModelUnavailable(
            f"Comando LM Studio fallito (`lms {' '.join(args)}`): "
            f"{detail or f'codice {result.returncode}'}"
        )
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _lmstudio_server_available() -> bool:
    return (
        _read_json(f"{LM_STUDIO_URL}/api/v0/models", timeout=0.5) is not None
        or _read_json(f"{LM_STUDIO_URL}/v1/models", timeout=0.5) is not None
    )


def _model_key(model: dict) -> str:
    return str(model.get("id") or model.get("modelKey") or "")


def _is_text_model(model: dict) -> bool:
    return model.get("type") in ("llm", "vlm")


def _display_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "dimensione sconosciuta"
    return f"{size_bytes / (1024 ** 3):.1f} GiB"


def _local_api_post(url: str, payload: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise ModelUnavailable(
            f"Errore API locale ({error.code}): {detail or error.reason}"
        ) from error
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise ModelUnavailable(f"Errore API locale: {error}") from error
    if not isinstance(data, dict):
        raise ModelUnavailable("Il server locale ha restituito una risposta non valida.")
    return data


class LocalModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ollama_process: subprocess.Popen[bytes] | None = None
        self._started_lmstudio_daemon = False
        self._selected_backend: LocalBackend | None = None

    def _ollama_server(self) -> dict | None:
        data = _read_json(f"{OLLAMA_URL}/api/tags", timeout=0.5)
        if data is not None:
            return data
        executable = shutil.which("ollama")
        if not executable:
            return None
        try:
            self._ollama_process = subprocess.Popen(
                [executable, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise ModelUnavailable(f"Impossibile avviare Ollama in background: {error}") from error
        for _ in range(40):
            data = _read_json(f"{OLLAMA_URL}/api/tags", timeout=0.5)
            if data is not None:
                return data
            if self._ollama_process.poll() is not None:
                break
            time.sleep(0.25)
        self._stop_ollama_process()
        raise ModelUnavailable(
            "Ollama è installato ma il servizio locale non si è avviato."
        )

    def discover_models(self) -> list[LocalBackend]:
        with self._lock:
            models = [
                LocalBackend(
                    "lmstudio",
                    _model_key(model),
                    f"LM Studio — {model.get('displayName') or _model_key(model)} "
                    f"({_display_size(int(model.get('sizeBytes') or 0))})",
                    int(model.get("sizeBytes") or 0),
                )
                for model in _lmstudio_disk_models()
                if _is_text_model(model)
                if _model_key(model)
            ]
            lmstudio_data = _read_json(
                f"{LM_STUDIO_URL}/api/v0/models", timeout=0.5
            )
            if lmstudio_data is None:
                lmstudio_data = _read_json(f"{LM_STUDIO_URL}/v1/models", timeout=0.5)
            if not models and lmstudio_data and isinstance(lmstudio_data.get("data"), list):
                models.extend(
                    LocalBackend(
                        "lmstudio",
                        _model_key(model),
                        f"LM Studio — {model.get('displayName') or _model_key(model)}",
                        int(model.get("sizeBytes") or 0),
                    )
                    for model in lmstudio_data["data"]
                    if isinstance(model, dict)
                    and (
                        _is_text_model(model)
                        or (
                            model.get("object") == "model"
                            and "embed" not in _model_key(model).lower()
                        )
                    )
                    and _model_key(model)
                )

            ollama_error = ""
            try:
                data = self._ollama_server()
                ollama_models = data.get("models", []) if data else []
                if isinstance(ollama_models, list):
                    models.extend(
                        LocalBackend(
                            "ollama",
                            str(model.get("name") or model.get("model")),
                            f"Ollama — {model.get('name') or model.get('model')} "
                            f"({_display_size(int(model.get('size') or 0))})",
                            int(model.get("size") or 0),
                        )
                        for model in ollama_models
                        if isinstance(model, dict)
                        and (model.get("name") or model.get("model"))
                        and "embed"
                        not in str(model.get("name") or model.get("model")).lower()
                    )
            except ModelUnavailable as error:
                ollama_error = str(error)

            if not models:
                self._stop_ollama_process()
                raise ModelUnavailable(
                    ollama_error
                    or "Nessun modello chat installato in LM Studio o Ollama."
                )
            return sorted(models, key=lambda model: (model.size_bytes or 2**63, model.label))

    def _ensure_lmstudio_server(self) -> None:
        if _lmstudio_server_available():
            return
        status = _run_lms("daemon", "status", timeout=10).lower()
        if "not running" in status:
            _run_lms("daemon", "up", timeout=60)
            self._started_lmstudio_daemon = True
        _run_lms("server", "start", timeout=30)
        for _ in range(40):
            if _lmstudio_server_available():
                return
            time.sleep(0.25)
        raise ModelUnavailable("Il server headless di LM Studio non risponde.")

    def _ensure_ollama_server(self) -> None:
        if _read_json(f"{OLLAMA_URL}/api/tags", timeout=0.5) is None:
            data = self._ollama_server()
            if data is None:
                raise ModelUnavailable("Il server locale di Ollama non è disponibile.")

    def _backend_is_loaded(self, backend: LocalBackend) -> bool | None:
        if backend.provider == "lmstudio":
            data = _read_json(f"{LM_STUDIO_URL}/api/v0/models", timeout=2)
            if data is None:
                return None
            models = data.get("data", []) if data else []
            return any(
                isinstance(model, dict)
                and _model_key(model) == backend.model
                and model.get("state") == "loaded"
                for model in models
            )
        if backend.provider == "ollama":
            data = _read_json(f"{OLLAMA_URL}/api/ps", timeout=2)
            if data is None:
                return None
            models = data.get("models", []) if data else []
            return any(
                isinstance(model, dict)
                and (model.get("name") or model.get("model")) == backend.model
                for model in models
            )
        return False

    def load_backend(
        self,
        backend: LocalBackend,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            if backend.provider not in ("lmstudio", "ollama"):
                raise ModelUnavailable(f"Provider locale non supportato: {backend.provider}")
            self._selected_backend = backend
            if backend.provider == "lmstudio":
                self._stop_ollama_process()
                self._ensure_lmstudio_server()
                if not self._backend_is_loaded(backend):
                    if progress:
                        progress(f"Caricamento {backend.label}...")
                    _run_lms("load", backend.model, "--yes", timeout=600)
                    for _ in range(40):
                        if self._backend_is_loaded(backend):
                            break
                        time.sleep(0.25)
                    else:
                        raise ModelUnavailable(
                            f"LM Studio non ha confermato il caricamento di {backend.model}."
                        )
            elif backend.provider == "ollama":
                self._ensure_ollama_server()
                if not self._backend_is_loaded(backend):
                    if progress:
                        progress(f"Caricamento {backend.label}...")
                    _local_api_post(
                        f"{OLLAMA_URL}/api/generate",
                        {
                            "model": backend.model,
                            "prompt": " ",
                            "stream": False,
                            "keep_alive": -1,
                            "options": {"num_predict": 1},
                        },
                    )
    def _stop_ollama_process(self) -> None:
        process = self._ollama_process
        if process is not None:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                if process.poll() is None:
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            self._ollama_process = None

    def release(self) -> None:
        errors: list[str] = []
        with self._lock:
            backend = self._selected_backend
            if backend is not None:
                try:
                    loaded = self._backend_is_loaded(backend)
                    if loaded is None:
                        raise ModelUnavailable(
                            f"Impossibile verificare se il modello {backend.model} "
                            "è ancora caricato."
                        )
                    if loaded:
                        if backend.provider == "lmstudio":
                            _run_lms("unload", backend.model, timeout=60)
                        elif backend.provider == "ollama":
                            executable = shutil.which("ollama")
                            if not executable:
                                raise ModelUnavailable(
                                    "Comando Ollama `ollama` non trovato; "
                                    "impossibile scaricare il modello."
                                )
                            result = subprocess.run(
                                [executable, "stop", backend.model],
                                capture_output=True,
                                text=True,
                                timeout=60,
                                check=False,
                            )
                            if result.returncode:
                                raise ModelUnavailable(
                                    result.stderr.strip() or "Ollama non ha scaricato il modello."
                                )
                except (OSError, subprocess.TimeoutExpired, ModelUnavailable) as error:
                    errors.append(str(error))
                else:
                    self._selected_backend = None

            if self._started_lmstudio_daemon:
                if _lmstudio_server_available():
                    try:
                        _run_lms("server", "stop", timeout=30)
                    except ModelUnavailable as error:
                        errors.append(str(error))
                try:
                    _run_lms("daemon", "down", timeout=30)
                except ModelUnavailable as error:
                    errors.append(str(error))
                else:
                    self._started_lmstudio_daemon = False
            try:
                self._stop_ollama_process()
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))
        if errors:
            raise ModelUnavailable("Impossibile liberare tutte le risorse PRO: " + "; ".join(errors))


_LOCAL_MODELS = LocalModelManager()


def discover_local_models() -> list[LocalBackend]:
    return _LOCAL_MODELS.discover_models()


def load_local_model(
    backend: LocalBackend,
    progress: Callable[[str], None] | None = None,
) -> None:
    _LOCAL_MODELS.load_backend(backend, progress)


def ensure_local_model(
    provider: str,
    model: str,
    progress: Callable[[str], None] | None = None,
) -> None:
    load_local_model(LocalBackend(provider, model, f"{provider} — {model}"), progress)


def release_local_model() -> None:
    _LOCAL_MODELS.release()


def split_text(text: str, max_chars: int = 4500) -> list[str]:
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            candidate = f"{current}\n\n{para}" if current else para
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = para
            continue
        if current:
            chunks.append(current)
            current = ""
        for word in para.split():
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word
    if current:
        chunks.append(current)
    return chunks


def correct_transcript(
    text: str,
    progress: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    provider: str = "ollama",
    model: str = OLLAMA_MODEL,
    context: str = "",
) -> str:
    chunks = split_text(text)
    if not chunks:
        raise ValueError("La trascrizione è vuota.")

    context = context.strip()
    corrected: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(index - 1, len(chunks))
        messages = [
            {
                "role": "system",
                "content": (
                    "Correggi questa trascrizione automatica nella lingua "
                    "originale. Correggi errori di riconoscimento evidenti, "
                    "ortografia, grammatica e punteggiatura; migliora la "
                    "leggibilità con paragrafi naturali. Conserva significato, "
                    "fatti, nomi, ordine, dettagli, timestamp e tag degli "
                    "speaker. Usa il contesto specialistico solo come indizio: "
                    "correggi un termine ambiguo solo se frase e contesto lo "
                    "supportano. Se non sei sicuro, mantieni il testo originale. "
                    "Se è già corretto, non modificarlo. Non riassumere, tradurre "
                    "o aggiungere informazioni. Restituisci solo il testo corretto."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Contesto specialistico (solo come guida terminologica):\n"
                    f"{context}\n\n"
                    f"Trascrizione:\n{chunk}"
                    if context
                    else chunk
                ),
            },
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
        if progress:
            progress(index, len(chunks))
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
