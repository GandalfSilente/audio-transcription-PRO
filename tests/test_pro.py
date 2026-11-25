from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pro
import engine


class ProTests(unittest.TestCase):
    @patch("pro._ollama_backend", return_value=None)
    @patch(
        "pro._read_json",
        return_value={
            "data": [
                {
                    "id": "qwen2.5-coder-7b-instruct-abliterated",
                    "type": "llm",
                    "state": "loaded",
                }
            ]
        },
    )
    def test_detects_loaded_lmstudio_model(self, _read_json, _ollama) -> None:
        backend = pro.detect_local_backend()
        self.assertEqual(backend.provider, "lmstudio")
        self.assertEqual(backend.model, "qwen2.5-coder-7b-instruct-abliterated")

    @patch("pro._ollama_backend", return_value=None)
    @patch(
        "pro._read_json",
        side_effect=[
            None,
            {"data": [{"id": "lmstudio-chat-model", "object": "model"}]},
        ],
    )
    def test_detects_loaded_lmstudio_openai_api_model(
        self, _read_json, _ollama
    ) -> None:
        backend = pro.detect_local_backend()
        self.assertEqual(backend.provider, "lmstudio")
        self.assertEqual(backend.model, "lmstudio-chat-model")

    @patch("pro._ollama_backend", return_value=None)
    @patch("pro._lmstudio_disk_models")
    @patch("pro._read_json", return_value=None)
    def test_installed_lmstudio_model_requires_load_and_server(
        self, _read_json, disk_models, _ollama
    ) -> None:
        disk_models.return_value = [
            {
                "modelKey": "qwen2.5-coder-7b-instruct-abliterated",
                "displayName": "Qwen 7B",
                "type": "llm",
            }
        ]
        with self.assertRaisesRegex(pro.ModelUnavailable, "qwen2.5-coder-7b"):
            pro.detect_local_backend()

    @patch("pro._read_json")
    def test_uses_installed_ollama_model_when_available(self, read_json) -> None:
        read_json.side_effect = [
            None,
            None,
            {"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]},
        ]
        backend = pro.detect_local_backend()
        self.assertEqual(backend.provider, "ollama")
        self.assertEqual(backend.model, "qwen3:8b")

    @patch("pro.urllib.request.urlopen")
    def test_correction_uses_local_chat_and_preserves_output(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {"message": {"content": "Testo corretto e leggibile."}}
                ).encode()

        urlopen.return_value = Response()
        self.assertEqual(
            pro.correct_transcript("Testo grezzo.", provider="ollama"),
            "Testo corretto e leggibile.",
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, f"{pro.OLLAMA_URL}/api/chat")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], pro.OLLAMA_MODEL)
        self.assertFalse(payload["think"])

    @patch("pro.urllib.request.urlopen")
    def test_correction_uses_lmstudio_openai_api(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "Testo corretto."}}]}
                ).encode()

        urlopen.return_value = Response()
        self.assertEqual(
            pro.correct_transcript(
                "Testo grezzo.",
                provider="lmstudio",
                model="qwen2.5-coder-7b-instruct-abliterated",
            ),
            "Testo corretto.",
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url, f"{pro.LM_STUDIO_URL}/v1/chat/completions"
        )
        self.assertEqual(
            json.loads(request.data)["model"],
            "qwen2.5-coder-7b-instruct-abliterated",
        )

    def test_corrected_output_does_not_overwrite_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "lecture.txt"
            previous = pro.corrected_path_for(raw)
            previous.write_text("Old correction", encoding="utf-8")
            self.assertEqual(
                pro.corrected_path_for(raw),
                Path(directory) / "lecture.pro-1.txt",
            )

    def test_pro_switch_setting_survives_config_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config = engine.load_config(config_path)
            self.assertFalse(config["pro_enabled"])
            config["pro_enabled"] = True
            engine.save_config(config, config_path)
            self.assertTrue(engine.load_config(config_path)["pro_enabled"])

    def test_chunks_preserve_words(self) -> None:
        text = "one two three four five"
        chunks = pro.split_text(text, max_chars=9)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 9 for chunk in chunks))

    @patch("pro.correct_transcript", return_value="Polished transcript.")
    def test_engine_pro_stage_keeps_original_and_saves_new_output(
        self, _correct
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "lecture.txt"
            raw_path.write_text("Rough transcript.", encoding="utf-8")
            worker = engine.TranscriptionWorker(lambda _state: None)
            worker._state["txt_path"] = str(raw_path)
            job = engine.TranscriptionJob(
                Path(directory) / "lecture.wav",
                pro_provider="lmstudio",
                pro_model="qwen2.5-coder-7b-instruct-abliterated",
            )

            worker._process_pro_job(job)

            snapshot = worker.get_snapshot()
            corrected_path = Path(snapshot["txt_path"])
            self.assertEqual(snapshot["status"], "done")
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "Rough transcript.")
            self.assertEqual(
                corrected_path.read_text(encoding="utf-8"), "Polished transcript.\n"
            )
            _correct.assert_called_once()
            self.assertEqual(_correct.call_args.kwargs["provider"], "lmstudio")
            self.assertEqual(
                _correct.call_args.kwargs["model"],
                "qwen2.5-coder-7b-instruct-abliterated",
            )

    @patch(
        "pro.correct_transcript",
        side_effect=pro.ModelUnavailable("No local model is loaded"),
    )
    def test_pro_failure_keeps_original_available(self, _correct) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "lecture.txt"
            raw_path.write_text("Rough transcript.", encoding="utf-8")
            worker = engine.TranscriptionWorker(lambda _state: None)
            worker._state["txt_path"] = str(raw_path)
            job = engine.TranscriptionJob(Path(directory) / "lecture.wav")

            worker._process_pro_job(job)

            snapshot = worker.get_snapshot()
            self.assertEqual(snapshot["status"], "error")
            self.assertTrue(snapshot["pro_error"])
            self.assertEqual(snapshot["txt_path"], str(raw_path))
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "Rough transcript.")


if __name__ == "__main__":
    unittest.main()
