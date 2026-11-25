# Trascrizioni Audio PRO

Desktop app for a local, two-stage audio workflow:

1. `faster-whisper` (`large-v3`) transcribes Italian audio on the GPU when CUDA
   is available, otherwise on CPU.
2. A local Ollama model (`qwen3:8b` by default) proofreads the transcript,
   preserving its meaning and detail.

The original transcript is saved before correction as
`Trascrizioni/<audio-name>.raw.txt`; the corrected version is saved as
`Trascrizioni/<audio-name>.corrected.txt`. Audio and text stay on this machine.

## Linux setup

Install GTK 4 / PyGObject system packages, then:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3:8b
.venv/bin/python app.py
```

Keep the Ollama service running. The app talks only to its local API at
`http://127.0.0.1:11434`; choose another installed model by editing the model
field. Whisper's model downloads to its normal local cache the first time.

On Ubuntu/Debian, GTK bindings can be installed with:

```bash
sudo apt install python3-gi gir1.2-gtk-4.0
```

## Notes

- Requires Python 3.10+, GTK 4, Ollama, and `faster-whisper`.
- NVIDIA CUDA is used when detected; CPU `int8` is the fallback.
- Correction runs in bounded text chunks, suitable for long recordings.
- Generated transcripts and uploaded audio are excluded from Git.
