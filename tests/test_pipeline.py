from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import (
    correct_transcript,
    next_corrected_path,
    split_text,
    transcript_paths,
)


class PipelineTests(unittest.TestCase):
    def test_split_text_preserves_words_and_caps_chunks(self) -> None:
        text = "one two three four five"
        chunks = split_text(text, max_chars=9)
        self.assertEqual(chunks, ["one two", "three", "four five"])
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 9 for chunk in chunks))

    @patch("pipeline.urllib.request.urlopen")
    def test_correct_transcript_uses_local_chat_api(self, urlopen) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {"message": {"content": "Testo corretto."}}
                ).encode()

        urlopen.return_value = Response()
        self.assertEqual(correct_transcript("Testo originale"), "Testo corretto.")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertFalse(payload["think"])

    def test_transcript_paths_preserve_previous_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "lezione.wav"
            raw_path, corrected_path = transcript_paths(Path(directory), audio)
            raw_path.write_text("Old raw", encoding="utf-8")
            corrected_path.write_text("Old corrected", encoding="utf-8")
            next_raw, next_corrected = transcript_paths(Path(directory), audio)
            self.assertNotEqual(next_raw, raw_path)
            self.assertNotEqual(next_corrected, corrected_path)

    def test_repeat_correction_does_not_overwrite_previous_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "lezione.raw.txt"
            previous = next_corrected_path(raw_path)
            previous.write_text("Old correction", encoding="utf-8")
            self.assertEqual(
                next_corrected_path(raw_path),
                Path(directory) / "lezione (1).corrected.txt",
            )


if __name__ == "__main__":
    unittest.main()
