from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pro
import engine


class ProTests(unittest.TestCase):
    @patch("pro.urllib.request.urlopen")
    def test_requires_installed_local_model(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"models":[{"name":"qwen3:8b"}]}'

        urlopen.return_value = Response()
        pro.ensure_model_installed()
        self.assertIn("/api/tags", urlopen.call_args.args[0].full_url)

    @patch("pro.urllib.request.urlopen")
    def test_missing_model_is_reported(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"models":[]}'

        urlopen.return_value = Response()
        with self.assertRaisesRegex(pro.ModelUnavailable, pro.MODEL):
            pro.ensure_model_installed()

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
            pro.correct_transcript("Testo grezzo."),
            "Testo corretto e leggibile.",
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, f"{pro.OLLAMA_URL}/api/chat")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], pro.MODEL)
        self.assertFalse(payload["think"])

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
            job = engine.TranscriptionJob(Path(directory) / "lecture.wav")

            worker._process_pro_job(job)

            snapshot = worker.get_snapshot()
            corrected_path = Path(snapshot["txt_path"])
            self.assertEqual(snapshot["status"], "done")
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "Rough transcript.")
            self.assertEqual(
                corrected_path.read_text(encoding="utf-8"), "Polished transcript.\n"
            )

    @patch(
        "pro.correct_transcript",
        side_effect=pro.ModelUnavailable("qwen3:8b is not installed"),
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
