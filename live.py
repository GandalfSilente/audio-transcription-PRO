#!/usr/bin/env python3
"""Real-time microphone LIVE transcription (Nemotron 3.5 when NeMo is installed, faster-whisper CUDA fallback)."""

from __future__ import annotations

import gc
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

import engine

_LANG_TAG_RE = re.compile(r"<[a-z]{2}(?:-[A-Z]{2})?>")


def _find_audio_capture_cmd() -> list[str]:
    """Return a native Linux 16 kHz mono s16le stdout capture command."""
    if shutil.which("pw-record"):
        return [
            "pw-record",
            "--format=s16",
            "--rate=16000",
            "--channels=1",
            "-",
        ]
    if shutil.which("parec"):
        return [
            "parec",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--raw",
        ]
    if shutil.which("arecord"):
        return [
            "arecord",
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "raw",
            "-q",
            "-",
        ]
    raise RuntimeError(
        "Nessun tool di cattura microfono trovato (pw-record, parec o arecord)."
    )


def _try_load_nemotron() -> tuple[Any | None, str]:
    """Try loading NVIDIA Nemotron 3.5 ASR Streaming if NeMo toolkit is installed."""
    try:
        import torch  # type: ignore
        import nemo.collections.asr as nemo_asr  # type: ignore
        from huggingface_hub import hf_hub_download

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            nemo_path = hf_hub_download(
                "nvidia/nemotron-3.5-asr-streaming-0.6b",
                "nemotron-3.5-asr-streaming-0.6b.nemo",
                local_files_only=True,
            )
            model = nemo_asr.models.ASRModel.restore_from(
                nemo_path, map_location=dev
            ).eval()
        except Exception:
            model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/nemotron-3.5-asr-streaming-0.6b"
            ).eval()
            if dev == "cuda":
                model = model.cuda()
        return model, dev
    except Exception:
        return None, ""


def _unload_nemotron(model: Any | None) -> None:
    """Release Nemotron model and free GPU VRAM if PyTorch CUDA is active."""
    if model is None:
        return
    del model
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _map_nemotron_lang(language: str) -> str:
    """Map UI language code ('auto', 'it', 'en') to Nemotron prompt key."""
    if language == "it":
        return "it-IT"
    if language == "en":
        return "en-US"
    return "auto"


def _clean_nemotron_text(text: str) -> str:
    """Strip inline language tags like <en-US> and collapse whitespace."""
    cleaned = _LANG_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _text_to_blocks(text: str, total_elapsed: float) -> list[dict[str, Any]]:
    """Split cumulative streaming text into sentence blocks for UI/export."""
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return [{"speaker": "", "start": 0.0, "text": text}]
    n = len(sentences)
    return [
        {
            "speaker": "",
            "start": round((i / n) * total_elapsed, 1) if n > 1 else 0.0,
            "text": sent,
        }
        for i, sent in enumerate(sentences)
    ]


class LiveTranscriber:
    """Streams microphone audio, transcribes in real time, and saves .txt on stop."""

    def __init__(self, on_update: Callable[[dict[str, Any]], None]) -> None:
        self._on_update = on_update
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, language: str = "auto", initial_prompt: str = "") -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(language, initial_prompt),
                name="live-transcriber",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    @staticmethod
    def _read_exact(stream: Any, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = stream.read(n - len(buf))
            if not part:
                break
            buf += part
        return buf

    def _emit(self, **state: Any) -> None:
        try:
            self._on_update(state)
        except Exception:
            pass

    def _run_nemotron_stream(
        self,
        model: Any,
        stdout: Any,
        pending: bytes,
        t0: float,
        backend_label: str,
    ) -> list[dict[str, Any]]:
        import torch  # type: ignore

        device = next(model.parameters()).device
        step_frames = model.encoder.streaming_cfg.chunk_size[1]
        pre_cache = model.encoder.streaming_cfg.pre_encode_cache_size[1]
        drop_extra = model.encoder.streaming_cfg.drop_extra_pre_encoded
        # 1.28s chunks (20480 samples * 2 bytes = 40960 bytes)
        chunk_bytes = 16000 * 2 * 128 // 100

        blocks: list[dict[str, Any]] = []
        last_text = ""

        with torch.inference_mode():
            cache_ch, cache_tm, cache_len = model.encoder.get_initial_cache_state(
                batch_size=1
            )
            prev_hyp = None
            prev_pred = None
            tail_frames = None
            step = 0

            while not self._stop_event.is_set():
                raw = pending + self._read_exact(stdout, chunk_bytes - len(pending))
                pending = b""
                raw = raw[: len(raw) // 2 * 2]
                if not raw:
                    break
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio) < 1600:
                    continue

                sig = torch.from_numpy(audio).unsqueeze(0).to(device)
                sig_len = torch.tensor([sig.shape[1]], dtype=torch.long, device=device)
                mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)
                mel = mel[:, :, : mel_len[0]]
                if tail_frames is not None:
                    combined = torch.cat([tail_frames, mel], dim=2)
                    offset = tail_frames.shape[2]
                else:
                    combined = mel
                    offset = 0

                idx = offset
                total = combined.shape[2]
                texts = None
                while idx < total:
                    end = min(idx + step_frames, total)
                    start = max(0, idx - (pre_cache if step > 0 else 0))
                    sub = combined[:, :, start:end]
                    slen = torch.tensor([sub.shape[2]], dtype=torch.long, device=device)
                    res = model.conformer_stream_step(
                        processed_signal=sub,
                        processed_signal_length=slen,
                        cache_last_channel=cache_ch,
                        cache_last_time=cache_tm,
                        cache_last_channel_len=cache_len,
                        keep_all_outputs=False,
                        previous_hypotheses=prev_hyp,
                        previous_pred_out=prev_pred,
                        drop_extra_pre_encoded=(drop_extra if step != 0 else 0),
                        return_transcription=True,
                    )
                    prev_pred, texts, cache_ch, cache_tm, cache_len, prev_hyp = res[:6]
                    idx = end
                    step += 1

                tail_frames = (
                    combined[:, :, -pre_cache:]
                    if combined.shape[2] >= pre_cache
                    else combined
                )
                elapsed_now = time.time() - t0
                raw_t = texts[0].text if texts and hasattr(texts[0], "text") else ""
                clean_t = _clean_nemotron_text(raw_t)
                if clean_t and clean_t != last_text:
                    last_text = clean_t
                    blocks = _text_to_blocks(clean_t, elapsed_now)
                    lines = engine.format_blocks(blocks, show_timestamps=False)
                    ts_lines = engine.format_blocks(blocks, show_timestamps=True)
                    self._emit(
                        status="transcribing",
                        lines=lines,
                        ts_lines=ts_lines,
                        speaker_blocks=blocks,
                        duration=round(elapsed_now, 1),
                        elapsed=round(elapsed_now, 1),
                        message=f"{backend_label} attivo — parla nel microfono...",
                    )
                else:
                    self._emit(elapsed=round(elapsed_now, 1))

        return blocks

    def _run_whisper_stream(
        self,
        whisper_model: Any,
        stdout: Any,
        pending: bytes,
        t0: float,
        language: str,
        initial_prompt: str,
        backend_label: str,
    ) -> list[dict[str, Any]]:
        chunk_bytes = 16000 * 2 * 3
        blocks: list[dict[str, Any]] = []
        while not self._stop_event.is_set():
            raw = pending + self._read_exact(stdout, chunk_bytes - len(pending))
            pending = b""
            raw = raw[: len(raw) // 2 * 2]
            if not raw:
                break
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 3200:
                continue
            if float(np.sqrt(np.mean(audio**2))) < 0.004:
                self._emit(elapsed=round(time.time() - t0, 1))
                continue

            elapsed_now = time.time() - t0
            kw: dict[str, Any] = {
                "beam_size": 3,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": dict(engine.VAD_PARAMETERS),
            }
            if initial_prompt.strip():
                kw["hotwords"] = initial_prompt.strip()
            if language and language != "auto":
                kw["language"] = language
            segs, _ = whisper_model.transcribe(audio, **kw)
            text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
            if text:
                blocks.append(
                    {
                        "speaker": "",
                        "start": max(0.0, elapsed_now - 3.0),
                        "text": text,
                    }
                )
                lines = engine.format_blocks(blocks, show_timestamps=False)
                ts_lines = engine.format_blocks(blocks, show_timestamps=True)
                self._emit(
                    status="transcribing",
                    lines=lines,
                    ts_lines=ts_lines,
                    speaker_blocks=blocks,
                    duration=round(elapsed_now, 1),
                    elapsed=round(elapsed_now, 1),
                    message=f"{backend_label} attivo — parla nel microfono...",
                )
        return blocks

    def _run(self, language: str, initial_prompt: str) -> None:
        t0 = time.time()
        whisper_model: Any = None
        nemotron_model: Any = None
        blocks: list[dict[str, Any]] = []

        try:
            cmd = _find_audio_capture_cmd()
            self._emit(
                status="loading",
                progress=0.0,
                filename="LIVE (Microfono)",
                language=language,
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
                message="LIVE: caricamento Nemotron 3.5 / CUDA (~1-2 min)...",
            )

            nemotron_model, nem_dev = _try_load_nemotron()
            if nemotron_model is not None:
                nemotron_model.set_inference_prompt(_map_nemotron_lang(language))
                backend_label = (
                    "Nemotron 3.5 Streaming (CUDA)"
                    if nem_dev == "cuda"
                    else "Nemotron 3.5 Streaming (CPU)"
                )
            else:
                whisper_model, dev = engine.load_whisper("large-v3-turbo")
                backend_label = (
                    "LIVE Streaming (CUDA)" if dev == "cuda" else "LIVE Streaming (CPU)"
                )

            if self._stop_event.is_set():
                self._emit(status="idle", message="")
                return

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            with self._lock:
                self._proc = proc

            self._emit(
                status="transcribing",
                progress=0.0,
                message=f"{backend_label} attivo — parla nel microfono...",
            )

            assert proc.stdout is not None
            head = self._read_exact(proc.stdout, 4)
            if head == b"RIFF":  # pw-record writes a WAV header on stdout
                self._read_exact(proc.stdout, 40)
                head = b""

            if nemotron_model is not None:
                blocks = self._run_nemotron_stream(
                    nemotron_model, proc.stdout, head, t0, backend_label
                )
            else:
                blocks = self._run_whisper_stream(
                    whisper_model,
                    proc.stdout,
                    head,
                    t0,
                    language,
                    initial_prompt,
                    backend_label,
                )

            engine.OUTPUT_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_txt = engine.OUTPUT_DIR / f"LIVE_{stamp}.txt"
            engine.save_blocks_to_txt(out_txt, blocks)

            lines = engine.format_blocks(blocks, show_timestamps=False)
            ts_lines = engine.format_blocks(blocks, show_timestamps=True)
            self._emit(
                status="done",
                progress=100.0,
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                message=f"LIVE terminato! Salvato in Trascrizioni/{out_txt.name}",
                lines=lines,
                ts_lines=ts_lines,
                speaker_blocks=blocks,
                speakers=[],
                speaker_names={},
                txt_path=str(out_txt),
            )
        except Exception as exc:
            self._emit(
                status="error",
                elapsed=round(time.time() - t0, 1),
                job_started_at=0.0,
                error=str(exc),
                message=f"Errore LIVE: {exc}",
            )
        finally:
            with self._lock:
                proc = self._proc
                self._proc = None
                self._thread = None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            _unload_nemotron(nemotron_model)
            engine.unload_whisper(whisper_model)
