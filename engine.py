#!/usr/bin/env python3
"""Lazy-loaded transcription engine for faster-whisper (NVIDIA CUDA / CPU + VAD + Ensemble)."""

from __future__ import annotations

import ctypes
import gc
import glob
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Trascrizioni"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    / "trascrizione-audio"
)
CONFIG_PATH = CONFIG_DIR / "config.json"

MEDICAL_INITIAL_PROMPT = (
    "Lezione universitaria di medicina, chirurgia e radiologia diagnostica. "
    "Terminologia medica, anatomia, fisiopatologia, patologia, clinica, semeiotica, "
    "farmacologia, TC, RM, risonanza magnetica, tomografia computerizzata, ecografia, "
    "radiografia, PET, scintigrafia, mezzo di contrasto, parenchima, lesione, "
    "neoplasia, metastasi, ischemia, stenosi, ipertensione, diagnosi differenziale."
)

PHYSICS_EN_PROMPT = (
    "University lecture in physics, nuclear physics, and quantum mechanics. "
    "Technical terminology: Hamiltonian, quadrupole, octupole, dipole, multipole, "
    "phonon, Weisskopf units, matrix element, creation and annihilation operators, "
    "harmonic oscillator, angular momentum, spin-orbit coupling, Coriolis, prolate, "
    "oblate, deformed nucleus, rotational bands, magic numbers, Grodzins relation, "
    "MeV, keV, eV, fm, Aage Bohr, Ben Mottelson, Rainwater."
)

PHYSICS_IT_PROMPT = (
    "Lezione universitaria di fisica, meccanica quantistica, fisica nucleare e ingegneria. "
    "Terminologia tecnica: Hamiltoniana, quadrupolo, ottupolo, dipolo, multipolo, fonone, "
    "unità di Weisskopf, elemento di matrice, operatori di creazione e distruzione, "
    "oscillatore armonico, momento angolare, accoppiamento spin-orbita, Coriolis, "
    "nucleo deformato, bande rotazionali, numeri magici, elettronvolt, MeV, keV."
)

CS_EN_PROMPT = (
    "University lecture in computer science, software engineering, and machine learning. "
    "Technical terminology: algorithm, complexity, neural network, transformer, GPU, "
    "CUDA, latency, API, thread, concurrency, database, compiler, Linux, Python, kernel."
)

CS_IT_PROMPT = (
    "Lezione universitaria di informatica, ingegneria del software e intelligenza artificiale. "
    "Terminologia tecnica: algoritmo, complessità, rete neurale, transformer, GPU, "
    "CUDA, latenza, API, thread, concorrenza, database, compilatore, Linux, Python, kernel."
)

# Tuned Silero VAD defaults: more sensitive to distant classroom microphones
# while padding speech boundaries so opening words are not clipped.
VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.35,
    "min_silence_duration_ms": 600,
    "speech_pad_ms": 400,
}

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("large-v3", "large-v3 (Massima precisione - Predefinito)"),
    ("ensemble", "Combinato: large-v3 + turbo (Doppio controllo Ensemble)"),
    ("large-v3-turbo", "large-v3-turbo (Ultra veloce)"),
    ("parakeet-v3", "Parakeet TDT 0.6B v3 (Multilingua • Alta precisione)"),
]

LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Automatica (IT / EN)"),
    ("it", "Italiano"),
    ("en", "Inglese"),
]

DIARIZATION_OPTIONS: list[tuple[str, str]] = [
    ("off", "NO (Disattivato)"),
    ("on", "SI (Più voci)"),
    ("auto", "Auto (Rileva se più voci)"),
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".mkv",
        ".ogg",
        ".flac",
        ".webm",
        ".aac",
        ".opus",
        ".mov",
        ".avi",
        ".wma",
        ".mpeg",
        ".mpg",
    }
)


@dataclass(frozen=True)
class ContextPreset:
    id: str
    name: str
    language: str
    prompt: str
    builtin: bool = True


BUILTIN_PRESETS: tuple[ContextPreset, ...] = (
    ContextPreset(
        id="none",
        name="Generale (Nessun contesto)",
        language="auto",
        prompt="",
        builtin=True,
    ),
    ContextPreset(
        id="medical_it",
        name="Medicina e Radiologia (IT)",
        language="it",
        prompt=MEDICAL_INITIAL_PROMPT,
        builtin=True,
    ),
    ContextPreset(
        id="physics_en",
        name="Fisica / Fisica Nucleare (EN)",
        language="en",
        prompt=PHYSICS_EN_PROMPT,
        builtin=True,
    ),
    ContextPreset(
        id="physics_it",
        name="Fisica e Ingegneria (IT)",
        language="it",
        prompt=PHYSICS_IT_PROMPT,
        builtin=True,
    ),
    ContextPreset(
        id="cs_en",
        name="Informatica e AI (EN)",
        language="en",
        prompt=CS_EN_PROMPT,
        builtin=True,
    ),
    ContextPreset(
        id="cs_it",
        name="Informatica e AI (IT)",
        language="it",
        prompt=CS_IT_PROMPT,
        builtin=True,
    ),
)

_DEFAULT_CONFIG: dict[str, Any] = {
    "model": "large-v3",
    "language": "auto",
    "diarization": "off",
    "pro_enabled": False,
    "preset_id": "none",
    "custom_prompt": "",
    "show_timestamps": False,
    "user_presets": [],
}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load persisted UI settings and custom user presets from XDG config."""
    path = config_path or CONFIG_PATH
    cfg = dict(_DEFAULT_CONFIG)
    cfg["user_presets"] = []
    if not path.is_file():
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return cfg
    except Exception:
        return cfg

    valid_models = {k for k, _ in MODEL_OPTIONS}
    valid_langs = {k for k, _ in LANGUAGE_OPTIONS}
    valid_diar = {k for k, _ in DIARIZATION_OPTIONS}

    if raw.get("model") in valid_models:
        cfg["model"] = raw["model"]
    if raw.get("language") in valid_langs:
        cfg["language"] = raw["language"]
    if raw.get("diarization") in valid_diar:
        cfg["diarization"] = raw["diarization"]
    if isinstance(raw.get("preset_id"), str) and raw["preset_id"].strip():
        cfg["preset_id"] = raw["preset_id"].strip()
    if isinstance(raw.get("custom_prompt"), str):
        cfg["custom_prompt"] = raw["custom_prompt"]
    if isinstance(raw.get("show_timestamps"), bool):
        cfg["show_timestamps"] = raw["show_timestamps"]
    if isinstance(raw.get("pro_enabled"), bool):
        cfg["pro_enabled"] = raw["pro_enabled"]

    raw_presets = raw.get("user_presets")
    if isinstance(raw_presets, list):
        cleaned: list[dict[str, str]] = []
        seen_ids: set[str] = {p.id for p in BUILTIN_PRESETS}
        for item in raw_presets:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            lang = str(item.get("language", "auto")).strip()
            prompt = str(item.get("prompt", ""))
            if not pid or not name or pid in seen_ids:
                continue
            if lang not in valid_langs:
                lang = "auto"
            seen_ids.add(pid)
            cleaned.append(
                {
                    "id": pid,
                    "name": name,
                    "language": lang,
                    "prompt": prompt,
                }
            )
        cfg["user_presets"] = cleaned

    all_ids = {p.id for p in get_all_presets(cfg)}
    if cfg["preset_id"] not in all_ids:
        cfg["preset_id"] = "none"
    return cfg


def save_config(cfg: dict[str, Any], config_path: Path | None = None) -> None:
    """Persist settings and custom presets atomically to XDG config."""
    path = config_path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def get_all_presets(cfg: dict[str, Any]) -> list[ContextPreset]:
    presets = list(BUILTIN_PRESETS)
    for item in cfg.get("user_presets", []):
        if isinstance(item, dict) and item.get("id") and item.get("name"):
            presets.append(
                ContextPreset(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    language=str(item.get("language", "auto")),
                    prompt=str(item.get("prompt", "")),
                    builtin=False,
                )
            )
    return presets


def _slugify_preset_id(name: str, existing_ids: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "custom"
    base = f"user_{slug}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def upsert_user_preset(
    cfg: dict[str, Any], name: str, language: str, prompt: str
) -> tuple[dict[str, Any], str]:
    """Create or update a user preset by name, returning (updated_cfg, preset_id)."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Inserisci un nome per il profilo.")

    valid_langs = {k for k, _ in LANGUAGE_OPTIONS}
    clean_lang = language if language in valid_langs else "auto"
    user_presets = [dict(p) for p in cfg.get("user_presets", [])]

    for item in user_presets:
        if item["name"].lower() == clean_name.lower():
            item["name"] = clean_name
            item["language"] = clean_lang
            item["prompt"] = prompt.strip()
            updated = dict(cfg)
            updated["user_presets"] = user_presets
            updated["preset_id"] = item["id"]
            updated["language"] = clean_lang
            updated["custom_prompt"] = prompt.strip()
            return updated, item["id"]

    existing_ids = {p.id for p in BUILTIN_PRESETS} | {p["id"] for p in user_presets}
    new_id = _slugify_preset_id(clean_name, existing_ids)
    user_presets.append(
        {
            "id": new_id,
            "name": clean_name,
            "language": clean_lang,
            "prompt": prompt.strip(),
        }
    )
    updated = dict(cfg)
    updated["user_presets"] = user_presets
    updated["preset_id"] = new_id
    updated["language"] = clean_lang
    updated["custom_prompt"] = prompt.strip()
    return updated, new_id


def delete_user_preset(cfg: dict[str, Any], preset_id: str) -> dict[str, Any]:
    """Delete a user-created preset by id (built-in presets cannot be deleted)."""
    user_presets = [
        dict(p) for p in cfg.get("user_presets", []) if p.get("id") != preset_id
    ]
    updated = dict(cfg)
    updated["user_presets"] = user_presets
    if updated.get("preset_id") == preset_id:
        updated["preset_id"] = "none"
        updated["custom_prompt"] = ""
    return updated


# Lazy backend state — never loaded synchronously on module import.
_backend_lock = threading.Lock()
_backend_done = threading.Event()
_backend_started = False
_whisper_model_cls: Any = None
_backend_error: Exception | None = None


def _preload_cuda_libs() -> None:
    """Preload bundled cuBLAS and cuDNN shared libraries from the project .venv."""
    venv_lib = BASE_DIR / ".venv" / "lib"
    for pattern in (
        "*/site-packages/nvidia/cublas/lib/lib*.so*",
        "*/site-packages/nvidia/cudnn/lib/lib*.so*",
    ):
        for so_path in sorted(glob.glob(str(venv_lib / pattern))):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _load_backend_impl() -> None:
    global _whisper_model_cls, _backend_error
    try:
        _preload_cuda_libs()
        from faster_whisper import WhisperModel

        _whisper_model_cls = WhisperModel
        _backend_error = None
    except Exception as exc:
        _backend_error = exc
    finally:
        _backend_done.set()


def preload_backend_async(
    on_ready: Callable[[], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Warm up CUDA shared libraries and faster_whisper in a background thread (0 VRAM)."""
    global _backend_started
    with _backend_lock:
        if _backend_started:
            already_done = _backend_done.is_set()
            err = _backend_error
        else:
            _backend_started = True
            already_done = False
            err = None

            def _runner() -> None:
                _load_backend_impl()
                if _backend_error is not None:
                    if on_error is not None:
                        on_error(_backend_error)
                elif on_ready is not None:
                    on_ready()

            threading.Thread(
                target=_runner, name="cuda-backend-preload", daemon=True
            ).start()
            return

    if already_done:
        if err is not None and on_error is not None:
            on_error(err)
        elif err is None and on_ready is not None:
            on_ready()


def ensure_backend_loaded() -> Any:
    """Block until CUDA libraries and WhisperModel are ready, then return WhisperModel."""
    if not _backend_done.is_set():
        preload_backend_async()
        _backend_done.wait()
    if _backend_error is not None:
        raise RuntimeError(
            f"Impossibile caricare faster-whisper/CUDA: {_backend_error}"
        ) from _backend_error
    return _whisper_model_cls


def format_short(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def build_speaker_blocks(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive segments by speaker (and readable paragraph breaks)."""
    blocks: list[dict[str, Any]] = []
    cur_speaker = ""
    cur_texts: list[str] = []
    cur_start: float | None = None
    last_end: float | None = None

    for seg in raw_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = str(seg.get("speaker", "") or "")
        start = float(seg["start"])
        end = float(seg["end"])

        if cur_start is None:
            cur_speaker = speaker
            cur_start = start

        gap = (start - last_end) if last_end is not None else 0.0
        joined_len = sum(len(t) for t in cur_texts) + len(cur_texts)
        prev_ends_sentence = bool(cur_texts and cur_texts[-1][-1:] in ".!?")
        speaker_changed = speaker != cur_speaker
        para_break = (
            not speaker and (gap >= 1.5 or (joined_len >= 420 and prev_ends_sentence))
        ) or (bool(speaker) and gap >= 3.0 and joined_len >= 420 and prev_ends_sentence)

        if cur_texts and (speaker_changed or para_break):
            blocks.append(
                {
                    "speaker": cur_speaker,
                    "start": cur_start,
                    "text": " ".join(cur_texts),
                }
            )
            cur_speaker = speaker
            cur_texts = [text]
            cur_start = start
        else:
            cur_texts.append(text)
        last_end = end

    if cur_texts and cur_start is not None:
        blocks.append(
            {
                "speaker": cur_speaker,
                "start": cur_start,
                "text": " ".join(cur_texts),
            }
        )
    return blocks


def format_blocks(
    blocks: list[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
    show_timestamps: bool = False,
) -> list[str]:
    """Render transcript paragraphs with optional speaker headers and timestamps."""
    names = speaker_names or {}
    rendered: list[str] = []
    prev_speaker: str | None = None

    for blk in blocks:
        spk_id = blk.get("speaker", "")
        text = blk.get("text", "")
        ts = format_short(float(blk.get("start", 0.0)))

        if spk_id:
            display_name = names.get(spk_id, spk_id).strip() or spk_id
            if spk_id != prev_speaker:
                header = (
                    f"[{display_name}] ({ts})" if show_timestamps else f"[{display_name}]"
                )
                rendered.append(f"{header}\n{text}")
            else:
                rendered.append(f"({ts}) {text}" if show_timestamps else text)
            prev_speaker = spk_id
        else:
            rendered.append(f"({ts}) {text}" if show_timestamps else text)
            prev_speaker = None
    return rendered


def save_blocks_to_txt(
    txt_path: Path,
    blocks: list[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> str:
    """Write formatted transcript blocks to a .txt file and return the written text."""
    lines = format_blocks(blocks, speaker_names=speaker_names, show_timestamps=False)
    content = ("\n\n".join(lines) + "\n") if lines else ""
    txt_path.write_text(content, encoding="utf-8")
    return content


def load_whisper(model_name: str) -> tuple[Any, str]:
    """Load WhisperModel on CUDA float16, falling back to CPU int8 if CUDA fails."""
    whisper_cls = ensure_backend_loaded()
    try:
        return whisper_cls(model_name, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return whisper_cls(model_name, device="cpu", compute_type="int8"), "cpu"


def unload_whisper(model: Any | None) -> None:
    """Release WhisperModel weights from GPU VRAM and trim native heap memory."""
    if model is None:
        return
    try:
        model.model.unload_model()
    except Exception:
        pass
    del model
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def pick_best_text(seg_v3: Any, turbo_segs: list[dict[str, Any]]) -> str:
    v3_text = seg_v3.text.strip()
    if not turbo_segs or not v3_text:
        return v3_text
    mid = 0.5 * (seg_v3.start + seg_v3.end)
    best_match = None
    for tseg in turbo_segs:
        if tseg["start"] - 0.8 <= mid <= tseg["end"] + 0.8:
            best_match = tseg
            break
    if not best_match:
        return v3_text
    t_text = best_match["text"]
    len_ratio = len(t_text) / max(1, len(v3_text))
    if (
        0.75 <= len_ratio <= 1.35
        and best_match["avg_logprob"] > seg_v3.avg_logprob + 0.12
    ):
        return t_text
    return v3_text


class _CancelledError(Exception):
    """Internal signal raised when the user cancels a running transcription."""


@dataclass(frozen=True)
class TranscriptionJob:
    file_path: Path
    language: str = "auto"
    model_mode: str = "large-v3"
    initial_prompt: str = ""
    diarization_mode: str = "off"
    pro_enabled: bool = False


class TranscriptionWorker:
    """Single-worker sequential FIFO queue with live updates, cancellation and GPU cleanup."""

    def __init__(self, on_update: Callable[[dict[str, Any]], None]) -> None:
        self._on_update = on_update
        self._lock = threading.Lock()
        self._queue: deque[TranscriptionJob] = deque()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "progress": 0.0,
            "message": "Pronto. Trascina uno o più file audio o video.",
            "filename": "",
            "language": "",
            "duration": 0.0,
            "elapsed": 0.0,
            "job_started_at": 0.0,
            "lines": [],
            "ts_lines": [],
            "speaker_blocks": [],
            "speakers": [],
            "speaker_names": {},
            "txt_path": "",
            "queue_size": 0,
            "error": "",
            "pro_error": False,
        }

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def is_busy(self) -> bool:
        with self._lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def enqueue(self, jobs: list[TranscriptionJob]) -> int:
        valid_jobs = [job for job in jobs if job.file_path.is_file()]
        if not valid_jobs:
            return 0

        with self._lock:
            self._queue.extend(valid_jobs)
            should_start = (
                self._worker_thread is None or not self._worker_thread.is_alive()
            )
            if should_start:
                self._cancel_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._run_loop, name="whisper-worker", daemon=True
                )
                self._worker_thread.start()
            else:
                self._state["queue_size"] = len(self._queue)
                snap = self._snapshot_locked()
                self._emit_unlocked(snap)
        return len(valid_jobs)

    def cancel(self) -> None:
        with self._lock:
            self._cancel_event.set()
            self._queue.clear()
            self._state["queue_size"] = 0
            if self._state["status"] in ("loading", "transcribing", "pro"):
                self._state["message"] = (
                    "Annullamento della correzione PRO in corso..."
                    if self._state["status"] == "pro"
                    else "Annullamento in corso e rilascio GPU..."
                )
                snap = self._snapshot_locked()
            else:
                snap = None
        if snap is not None:
            self._emit_unlocked(snap)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.cancel()
        with self._lock:
            worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def _snapshot_locked(self) -> dict[str, Any]:
        snap = dict(self._state)
        snap["lines"] = list(self._state["lines"])
        snap["ts_lines"] = list(self._state["ts_lines"])
        snap["speaker_blocks"] = [dict(b) for b in self._state["speaker_blocks"]]
        snap["speakers"] = list(self._state["speakers"])
        snap["speaker_names"] = dict(self._state["speaker_names"])
        snap["queue_size"] = len(self._queue)
        return snap

    def _update_state(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            snap = self._snapshot_locked()
        self._emit_unlocked(snap)

    def _emit_unlocked(self, snap: dict[str, Any]) -> None:
        try:
            self._on_update(snap)
        except Exception:
            pass

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise _CancelledError()

    def _run_loop(self) -> None:
        cached_model: Any = None
        cached_model_name: str | None = None
        cached_device: str = "cuda"

        try:
            while True:
                with self._lock:
                    if self._cancel_event.is_set() or not self._queue:
                        break
                    job = self._queue.popleft()
                    remaining = len(self._queue)

                cached_model, cached_model_name, cached_device = self._process_job(
                    job, remaining, cached_model, cached_model_name, cached_device
                )
                if job.pro_enabled:
                    snap = self.get_snapshot()
                    if snap["status"] == "done" and snap["lines"]:
                        unload_whisper(cached_model)
                        cached_model = None
                        cached_model_name = None
                        self._process_pro_job(job)
        finally:
            unload_whisper(cached_model)
            cached_model = None
            cached_model_name = None
            emit_cancelled = False
            with self._lock:
                self._worker_thread = None
                if (
                    self._cancel_event.is_set()
                    and self._state.get("status") != "cancelled"
                ):
                    emit_cancelled = True
            if emit_cancelled:
                self._update_state(
                    status="cancelled",
                    progress=0.0,
                    job_started_at=0.0,
                    message="Trascrizione annullata (GPU liberata).",
                )

    def _process_job(
        self,
        job: TranscriptionJob,
        remaining_after: int,
        cached_model: Any,
        cached_model_name: str | None,
        cached_device: str,
    ) -> tuple[Any, str | None, str]:
        t0 = time.time()
        orig_name = job.file_path.name
        model_mode = (
            job.model_mode
            if job.model_mode in ("large-v3", "ensemble", "large-v3-turbo", "parakeet-v3")
            else "large-v3"
        )
        is_ensemble = model_mode == "ensemble"
        transient_model: Any = None

        try:
            self._check_cancelled()
            self._update_state(
                status="loading",
                progress=0.0,
                filename=orig_name,
                language="",
                duration=0.0,
                elapsed=0.0,
                job_started_at=t0,
                lines=[],
                ts_lines=[],
                speaker_blocks=[],
                speakers=[],
                speaker_names={},
                txt_path="",
                error="",
                pro_error=False,
                message=f"Preparazione modello per {orig_name}...",
            )

            diar_mode = (
                job.diarization_mode
                if job.diarization_mode in ("off", "on", "auto")
                else "off"
            )
            max_transcribe_pct = 92.0 if diar_mode != "off" else 99.0

            if model_mode == "parakeet-v3":
                import av
                import numpy as np
                import onnx_asr

                self._update_state(
                    status="loading",
                    message=f"Caricamento Parakeet TDT 0.6B v3 (ONNX) per {orig_name}...",
                )
                parakeet_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
                self._check_cancelled()

                resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                chunks: list[np.ndarray] = []
                with av.open(str(job.file_path)) as container:
                    if not container.streams.audio:
                        raise ValueError("Nessuna traccia audio trovata nel file.")
                    for frame in container.decode(container.streams.audio[0]):
                        self._check_cancelled()
                        for rf in resampler.resample(frame):
                            arr = rf.to_ndarray()
                            if arr.ndim == 2:
                                arr = arr[0]
                            chunks.append(arr.astype(np.float32, copy=False))
                audio_16k = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
                total_dur = len(audio_16k) / 16000.0
                self._update_state(
                    status="transcribing",
                    language="auto (Parakeet v3 • 25 lingue)",
                    duration=total_dur,
                    message=f"Trascrizione (Parakeet TDT 0.6B v3 • ONNX): {orig_name}",
                )

                raw_segments = []
                txt_paragraphs = []
                ts_paragraphs = []
                chunk_samples = 25 * 16000
                for start_idx in range(0, len(audio_16k), chunk_samples):
                    self._check_cancelled()
                    end_idx = min(len(audio_16k), start_idx + chunk_samples)
                    if end_idx - start_idx < 1600:
                        continue
                    t_start = start_idx / 16000.0
                    t_end = end_idx / 16000.0
                    text = str(parakeet_model.recognize(audio_16k[start_idx:end_idx], sample_rate=16000)).strip()
                    if text:
                        raw_segments.append({"start": t_start, "end": t_end, "text": text})
                        txt_paragraphs.append(text)
                        ts_paragraphs.append(f"({format_short(t_start)}) {text}")
                    pct = min(max_transcribe_pct, (t_end / total_dur) * max_transcribe_pct) if total_dur > 0 else 0.0
                    self._update_state(
                        lines=list(txt_paragraphs),
                        ts_lines=list(ts_paragraphs),
                        progress=round(pct, 1),
                        elapsed=round(time.time() - t0, 1),
                    )
                return self._finalize_job(
                    job,
                    orig_name,
                    raw_segments,
                    diar_mode,
                    t0,
                    cached_model,
                    cached_model_name,
                    cached_device,
                )

            ensure_backend_loaded()
            self._check_cancelled()

            kwargs: dict[str, Any] = {
                "beam_size": 5,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": dict(VAD_PARAMETERS),
            }
            clean_prompt = job.initial_prompt.strip()
            if clean_prompt:
                # With condition_on_previous_text=False, faster-whisper's `hotwords`
                # injects the context prompt into every 30s window (unlike initial_prompt
                # which is only kept on window 1) without autoregressive punctuation drift.
                kwargs["hotwords"] = clean_prompt
            if job.language and job.language != "auto":
                kwargs["language"] = job.language

            turbo_segs: list[dict[str, Any]] = []

            if is_ensemble:
                # Ensemble alternates models to keep peak GPU VRAM usage low.
                if cached_model is not None:
                    unload_whisper(cached_model)
                    cached_model = None
                    cached_model_name = None

                self._update_state(
                    status="loading",
                    message=f"Passaggio 1/2: caricamento large-v3-turbo per {orig_name}...",
                )
                transient_model, dev1 = load_whisper("large-v3-turbo")
                self._check_cancelled()

                dev_label = "GPU CUDA" if dev1 == "cuda" else "CPU"
                self._update_state(
                    status="transcribing",
                    message=f"Passaggio 1/2 (large-v3-turbo • {dev_label}): {orig_name}",
                )

                segs_1, info_1 = transient_model.transcribe(str(job.file_path), **kwargs)
                self._update_state(
                    language=f"{info_1.language} ({info_1.language_probability * 100:.0f}%)",
                    duration=float(info_1.duration or 0.0),
                )

                for seg in segs_1:
                    self._check_cancelled()
                    txt = seg.text.strip()
                    if txt:
                        turbo_segs.append(
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": txt,
                                "avg_logprob": seg.avg_logprob,
                            }
                        )
                    pct = (
                        min(38.0, (seg.end / info_1.duration * 38.0))
                        if info_1.duration > 0
                        else 0.0
                    )
                    self._update_state(
                        progress=round(pct, 1),
                        elapsed=round(time.time() - t0, 1),
                    )

                unload_whisper(transient_model)
                transient_model = None

            target_model = (
                "large-v3" if model_mode in ("large-v3", "ensemble") else "large-v3-turbo"
            )

            if cached_model is not None and cached_model_name == target_model:
                model = cached_model
                device_used = cached_device
            else:
                if cached_model is not None:
                    unload_whisper(cached_model)
                    cached_model = None
                    cached_model_name = None

                self._update_state(
                    status="loading",
                    message=(
                        f"Passaggio 2/2: caricamento {target_model} (raffinamento)..."
                        if is_ensemble
                        else f"Caricamento {target_model}..."
                    ),
                )
                model, device_used = load_whisper(target_model)
                cached_model = model
                cached_model_name = target_model
                cached_device = device_used

            self._check_cancelled()
            dev_label = "GPU CUDA" if device_used == "cuda" else "CPU"
            self._update_state(
                status="transcribing",
                message=(
                    f"Passaggio 2/2 ({target_model} + Ensemble • {dev_label}): {orig_name}"
                    if is_ensemble
                    else f"Trascrizione ({target_model} • {dev_label}): {orig_name}"
                ),
            )

            segments, info = model.transcribe(str(job.file_path), **kwargs)
            self._update_state(
                language=f"{info.language} ({info.language_probability * 100:.0f}%)",
                duration=float(info.duration or 0.0),
            )

            diar_mode = (
                job.diarization_mode
                if job.diarization_mode in ("off", "on", "auto")
                else "off"
            )
            max_transcribe_pct = 92.0 if diar_mode != "off" else 99.0

            raw_segments: list[dict[str, Any]] = []
            txt_paragraphs: list[str] = []
            ts_paragraphs: list[str] = []
            current_texts: list[str] = []
            current_start: float | None = None
            last_end: float | None = None

            for seg in segments:
                self._check_cancelled()
                text = (
                    pick_best_text(seg, turbo_segs)
                    if is_ensemble
                    else seg.text.strip()
                )
                if not text:
                    continue

                raw_segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": text,
                    }
                )

                if current_start is None:
                    current_start = seg.start

                gap = (seg.start - last_end) if last_end is not None else 0.0
                joined_len = sum(len(t) for t in current_texts) + len(current_texts)
                prev_ends_sentence = bool(
                    current_texts and current_texts[-1][-1:] in ".!?"
                )

                if current_texts and (
                    gap >= 1.5 or (joined_len >= 420 and prev_ends_sentence)
                ):
                    para = " ".join(current_texts)
                    txt_paragraphs.append(para)
                    ts_paragraphs.append(f"({format_short(current_start)}) {para}")
                    current_texts = [text]
                    current_start = seg.start
                else:
                    current_texts.append(text)

                last_end = seg.end

                if info.duration > 0:
                    raw_pct = seg.end / info.duration
                    if is_ensemble:
                        pct = 38.0 + raw_pct * (max_transcribe_pct - 38.0)
                    else:
                        pct = raw_pct * max_transcribe_pct
                    pct = min(max_transcribe_pct, pct)
                else:
                    pct = 0.0

                live_clean = txt_paragraphs + (
                    [" ".join(current_texts)] if current_texts else []
                )
                live_ts = ts_paragraphs + (
                    [f"({format_short(current_start)}) {' '.join(current_texts)}"]
                    if current_texts
                    else []
                )
                self._update_state(
                    lines=live_clean,
                    ts_lines=live_ts,
                    progress=round(pct, 1),
                    elapsed=round(time.time() - t0, 1),
                )

            self._check_cancelled()
            return self._finalize_job(
                job,
                orig_name,
                raw_segments,
                diar_mode,
                t0,
                cached_model,
                cached_model_name,
                cached_device,
            )

        except _CancelledError:
            unload_whisper(transient_model)
            unload_whisper(cached_model)
            self._update_state(
                status="cancelled",
                progress=0.0,
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                message="Trascrizione annullata (GPU liberata).",
            )
            return None, None, cached_device

        except Exception as exc:
            unload_whisper(transient_model)
            unload_whisper(cached_model)
            self._update_state(
                status="error",
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                error=str(exc),
                message=f"Errore: {exc}",
            )
            return None, None, cached_device

    def _process_pro_job(self, job: TranscriptionJob) -> None:
        started_at = time.time()
        snapshot = self.get_snapshot()
        raw_path = Path(str(snapshot.get("txt_path", "")))
        self._update_state(
            status="pro",
            progress=0.0,
            job_started_at=started_at,
            elapsed=0.0,
            message=f"PRO: correzione locale di {job.file_path.name}...",
            pro_error=False,
        )
        try:
            if not raw_path.is_file():
                raise FileNotFoundError("Trascrizione originale non trovata.")
            raw_text = raw_path.read_text(encoding="utf-8")
            import pro

            def report_progress(index: int, total: int) -> None:
                self._check_cancelled()
                self._update_state(
                    status="pro",
                    progress=round((index - 1) / total * 100, 1),
                    message=f"PRO: correzione blocco {index}/{total}...",
                )

            corrected = pro.correct_transcript(
                raw_text,
                progress=report_progress,
                check_cancelled=self._check_cancelled,
            )
            self._check_cancelled()
            corrected_path = pro.corrected_path_for(raw_path)
            corrected_path.write_text(corrected.rstrip() + "\n", encoding="utf-8")
            paragraphs = [
                paragraph for paragraph in corrected.split("\n\n") if paragraph.strip()
            ]
            self._update_state(
                status="done",
                progress=100.0,
                elapsed=round(time.time() - started_at, 1),
                job_started_at=0.0,
                message=(
                    f"PRO completato. Originale: {raw_path.name}; "
                    f"versione corretta: {corrected_path.name}"
                ),
                lines=paragraphs,
                ts_lines=paragraphs,
                speaker_blocks=[],
                speakers=[],
                speaker_names={},
                txt_path=str(corrected_path),
                error="",
                pro_error=False,
            )
        except _CancelledError:
            self._update_state(
                status="cancelled",
                progress=0.0,
                elapsed=round(time.time() - started_at, 1),
                job_started_at=0.0,
                message="Correzione PRO annullata. Trascrizione originale conservata.",
            )
        except Exception as exc:
            self._update_state(
                status="error",
                elapsed=round(time.time() - started_at, 1),
                job_started_at=0.0,
                error=str(exc),
                pro_error=True,
                txt_path=str(raw_path),
                message=f"Errore PRO: {exc}. Originale conservata.",
            )

    def _finalize_job(
        self,
        job: TranscriptionJob,
        orig_name: str,
        raw_segments: list[dict[str, Any]],
        diar_mode: str,
        t0: float,
        cached_model: Any,
        cached_model_name: str | None,
        cached_device: str,
    ) -> tuple[Any, str | None, str]:
        ordered_speakers: list[str] = []
        if diar_mode != "off" and raw_segments:
            self._update_state(
                progress=94.0,
                message=f"Riconoscimento voci in corso per {orig_name}...",
                elapsed=round(time.time() - t0, 1),
            )
            import diarizer

            raw_segments, ordered_speakers = diarizer.assign_speakers(
                job.file_path, raw_segments, mode=diar_mode
            )
            self._check_cancelled()

        speaker_blocks = build_speaker_blocks(raw_segments)
        speaker_names = {spk: spk for spk in ordered_speakers}
        txt_paragraphs = format_blocks(
            speaker_blocks, speaker_names=speaker_names, show_timestamps=False
        )
        ts_paragraphs = format_blocks(
            speaker_blocks, speaker_names=speaker_names, show_timestamps=True
        )

        OUTPUT_DIR.mkdir(exist_ok=True)
        stem = job.file_path.stem
        out_txt = OUTPUT_DIR / f"{stem}.txt"
        save_blocks_to_txt(out_txt, speaker_blocks, speaker_names=speaker_names)

        with self._lock:
            has_more = len(self._queue) > 0 and not self._cancel_event.is_set()

        if not has_more:
            unload_whisper(cached_model)
            cached_model = None
            cached_model_name = None

        done_msg = (
            f"Completato {orig_name} — passaggio al file successivo..."
            if has_more
            else f"Completato! Salvato in Trascrizioni/{out_txt.name} (GPU liberata)"
        )
        self._update_state(
            status="done",
            progress=100.0,
            elapsed=round(time.time() - t0, 1),
            job_started_at=0.0,
            message=done_msg,
            lines=txt_paragraphs,
            ts_lines=ts_paragraphs,
            speaker_blocks=speaker_blocks,
            speakers=ordered_speakers,
            speaker_names=speaker_names,
            txt_path=str(out_txt),
        )
        return cached_model, cached_model_name, cached_device
