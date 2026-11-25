#!/usr/bin/env python3
"""Local audio/video transcriber using faster-whisper large-v3-turbo (NVIDIA CUDA / CPU + VAD)."""

import os
import sys
import subprocess
import time
from pathlib import Path
from faster_whisper import WhisperModel

MODEL_NAME = "large-v3-turbo"


def format_timestamp_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_short(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def pick_file() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    try:
        res = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Seleziona il file audio o video da trascrivere",
                "--file-filter=Audio/Video | *.mp3 *.wav *.m4a *.mp4 *.mkv *.ogg *.flac *.webm *.aac *.opus",
                "--file-filter=Tutti i file | *",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip()).expanduser().resolve()
    except FileNotFoundError:
        pass

    raw = input("Trascina qui il file audio/video e premi Invio: ").strip().strip("'\"")
    if not raw:
        sys.exit(0)
    return Path(raw).expanduser().resolve()


def load_best_model() -> WhisperModel:
    try:
        print(f"Caricamento modello '{MODEL_NAME}' su GPU NVIDIA (CUDA float16)...")
        return WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    except Exception as exc:
        print(f"CUDA non disponibile ({exc}), uso CPU (int8)...")
        return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def main() -> None:
    audio_path = pick_file()
    if not audio_path.is_file():
        print(f"Errore: file non trovato -> {audio_path}")
        sys.exit(1)

    t0 = time.time()
    model = load_best_model()

    print(f"\nTrascrizione avviata: {audio_path.name}")
    print("(La prima volta scarica il modello large-v3-turbo ~1.6 GB, poi resta in cache.)\n")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    print(
        f"Lingua rilevata: {info.language} ({info.language_probability * 100:.1f}%) | "
        f"Durata: {format_timestamp_short(info.duration)}\n"
    )

    txt_lines = []
    srt_blocks = []

    for idx, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        start_s = format_timestamp_short(seg.start)
        end_s = format_timestamp_short(seg.end)
        print(f"[{start_s} -> {end_s}] {text}")
        txt_lines.append(text)
        srt_blocks.append(
            f"{idx}\n{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n{text}\n"
        )

    out_txt = audio_path.with_suffix(".txt")
    out_srt = audio_path.with_suffix(".srt")

    out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    out_srt.write_text("\n".join(srt_blocks), encoding="utf-8")

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print(f"Completato in {elapsed:.1f}s!")
    print(f"Testo:       {out_txt}")
    print(f"Sottotitoli: {out_srt}")
    print("=" * 50)


if __name__ == "__main__":
    main()
