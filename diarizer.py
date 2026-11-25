#!/usr/bin/env python3
"""Lightweight ONNX speaker diarization using WeSpeaker ResNet34 + NumPy clustering."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

SPEAKER_REPO_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SPEAKER_MODEL_FILE = "voxceleb_resnet34_LM.onnx"

_session_lock = threading.Lock()
_ort_session: Any = None
_mel_basis: np.ndarray | None = None


def _get_session() -> Any:
    """Lazy-load and cache the ONNX Runtime CPU session for speaker embeddings."""
    global _ort_session
    with _session_lock:
        if _ort_session is not None:
            return _ort_session
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=SPEAKER_REPO_ID, filename=SPEAKER_MODEL_FILE
        )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        _ort_session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        return _ort_session


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _get_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: float = 7600.0,
) -> np.ndarray:
    global _mel_basis
    if _mel_basis is not None:
        return _mel_basis

    mel_min = float(_hz_to_mel(fmin))
    mel_max = float(_hz_to_mel(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fbank[m - 1, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / (center - left)
        if right > center:
            fbank[m - 1, center:right] = (
                right - np.arange(center, right, dtype=np.float32)
            ) / (right - center)

    _mel_basis = fbank
    return fbank


def _compute_fbank(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute 80-dim log-Mel filterbank features (25ms win, 10ms hop, CMVN)."""
    win_len = int(0.025 * sr)  # 400
    hop_len = int(0.010 * sr)  # 160
    n_fft = 512

    if len(audio) < win_len:
        audio = np.pad(audio, (0, win_len - len(audio)))

    # Pre-emphasis (standard Kaldi/WeSpeaker pipeline)
    pre_emph = np.empty_like(audio)
    pre_emph[0] = audio[0]
    pre_emph[1:] = audio[1:] - 0.97 * audio[:-1]

    n_frames = 1 + (len(pre_emph) - win_len) // hop_len
    stride = pre_emph.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        pre_emph,
        shape=(n_frames, win_len),
        strides=(hop_len * stride, stride),
    ).copy()

    frames -= frames.mean(axis=1, keepdims=True)
    window = np.hamming(win_len).astype(np.float32)
    frames *= window

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)

    fbank = _get_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=80)
    mel_spec = np.maximum(power @ fbank.T, 1e-10)
    log_mel = np.log(mel_spec).astype(np.float32)

    # Utterance-level CMVN (mean subtraction)
    log_mel -= log_mel.mean(axis=0, keepdims=True)
    return log_mel


def _load_audio_16k(file_path: Path) -> np.ndarray:
    """Decode audio/video file to 16 kHz mono float32 waveform using PyAV."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []
    with av.open(str(file_path)) as container:
        if not container.streams.audio:
            raise ValueError("Nessuna traccia audio trovata nel file.")
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype(np.float32, copy=False))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _extract_embedding(session: Any, audio_slice: np.ndarray) -> np.ndarray:
    feats = _compute_fbank(audio_slice)
    inp_name = session.get_inputs()[0].name
    out = session.run(None, {inp_name: feats[np.newaxis, :, :]})[0][0]
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out = out / norm
    return out.astype(np.float32)


def _agglomerative_cosine_cluster(
    embeddings: np.ndarray, threshold: float
) -> np.ndarray:
    """Cluster L2-normalized embeddings with average-linkage cosine similarity."""
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    sim_matrix = embeddings @ embeddings.T
    clusters: list[list[int]] = [[i] for i in range(n)]

    while len(clusters) > 1:
        best_sim = -2.0
        best_pair = (-1, -1)
        m = len(clusters)
        for i in range(m):
            ci = clusters[i]
            for j in range(i + 1, m):
                cj = clusters[j]
                avg_sim = float(sim_matrix[np.ix_(ci, cj)].mean())
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (i, j)

        if best_sim < threshold:
            break

        i, j = best_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


def _smooth_short_islands(
    labels: list[int], durations: list[float], min_island_sec: float = 1.2
) -> list[int]:
    """Absorb isolated single-segment speaker flips shorter than min_island_sec."""
    if len(labels) < 3:
        return labels
    smoothed = list(labels)
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i - 1] == smoothed[i + 1]
            and smoothed[i] != smoothed[i - 1]
            and durations[i] < min_island_sec
        ):
            smoothed[i] = smoothed[i - 1]
    return smoothed


def assign_speakers(
    file_path: Path,
    segments: list[dict[str, Any]],
    mode: str = "auto",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assign speaker labels ('Voce 1', 'Voce 2', ...) in chronological order.

    Returns (updated_segments, ordered_speaker_ids).
    In 'auto' mode, if only 1 speaker is detected, speaker tags are omitted and
    ordered_speaker_ids is empty.
    """
    if mode == "off" or not segments:
        return segments, []

    audio = _load_audio_16k(file_path)
    total_samples = len(audio)
    if total_samples == 0:
        return segments, []

    session = _get_session()
    sr = 16000
    min_window_samples = int(1.4 * sr)

    embeddings: list[np.ndarray] = []
    valid_indices: list[int] = []
    durations: list[float] = []

    for idx, seg in enumerate(segments):
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        durations.append(max(0.1, end_s - start_s))

        c_start = int(start_s * sr)
        c_end = int(end_s * sr)
        if c_end - c_start < min_window_samples:
            mid = (c_start + c_end) // 2
            c_start = max(0, mid - min_window_samples // 2)
            c_end = min(total_samples, c_start + min_window_samples)

        c_start = max(0, min(c_start, total_samples))
        c_end = max(c_start, min(c_end, total_samples))
        if c_end - c_start < int(0.4 * sr):
            continue

        emb = _extract_embedding(session, audio[c_start:c_end])
        embeddings.append(emb)
        valid_indices.append(idx)

    if not embeddings:
        if mode == "on":
            for seg in segments:
                seg["speaker"] = "Voce 1"
            return segments, ["Voce 1"]
        return segments, []

    emb_matrix = np.vstack(embeddings)
    # 'auto' uses a stricter split threshold (0.50) to avoid false splits on single speakers;
    # 'on' uses 0.56 so multi-speaker conversations separate cleanly.
    threshold = 0.50 if mode == "auto" else 0.56
    raw_cluster_ids = _agglomerative_cosine_cluster(emb_matrix, threshold=threshold)

    # Map valid segments back to all segments (nearest valid neighbor for tiny segments)
    seg_clusters = [-1] * len(segments)
    for pos, seg_idx in enumerate(valid_indices):
        seg_clusters[seg_idx] = int(raw_cluster_ids[pos])

    for i in range(len(seg_clusters)):
        if seg_clusters[i] == -1:
            # Find nearest valid segment
            best_dist = len(seg_clusters) + 1
            best_cid = 0
            for j in valid_indices:
                if abs(j - i) < best_dist:
                    best_dist = abs(j - i)
                    best_cid = seg_clusters[j]
            seg_clusters[i] = best_cid

    seg_clusters = _smooth_short_islands(seg_clusters, durations)

    # Filter out spurious tiny clusters in 'auto' mode (e.g. < 2 segments and < 2.5s total)
    unique_clusters = set(seg_clusters)
    if mode == "auto" and len(unique_clusters) > 1:
        cluster_stats: dict[int, tuple[int, float]] = {}
        for cid, dur in zip(seg_clusters, durations):
            cnt, tot_d = cluster_stats.get(cid, (0, 0.0))
            cluster_stats[cid] = (cnt + 1, tot_d + dur)
        major_clusters = [
            cid
            for cid, (cnt, tot_d) in cluster_stats.items()
            if cnt >= 2 or tot_d >= 2.5
        ]
        if len(major_clusters) <= 1:
            return segments, []

    if mode == "auto" and len(set(seg_clusters)) <= 1:
        return segments, []

    # Assign "Voce 1", "Voce 2", ... strictly in order of first appearance
    cluster_to_name: dict[int, str] = {}
    ordered_speakers: list[str] = []
    for i, cid in enumerate(seg_clusters):
        if cid not in cluster_to_name:
            name = f"Voce {len(ordered_speakers) + 1}"
            cluster_to_name[cid] = name
            ordered_speakers.append(name)
        segments[i]["speaker"] = cluster_to_name[cid]

    return segments, ordered_speakers
