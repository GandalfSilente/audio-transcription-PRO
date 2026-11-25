from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import pro
import engine


class ProTests(unittest.TestCase):
    @patch("pro._run_lms")
    @patch("pro._lmstudio_server_available", side_effect=[False, True])
    def test_load_starts_headless_lmstudio_and_loads_model(
        self, _server, run_lms
    ) -> None:
        manager = pro.LocalModelManager()
        backend = pro.LocalBackend("lmstudio", "qwen7b", "LM Studio — Qwen 7B")
        run_lms.side_effect = ["LM Studio is not running", "", "", ""]
        with patch.object(manager, "_backend_is_loaded", side_effect=[False, True]):
            manager.load_backend(backend)
        self.assertEqual(
            [call.args for call in run_lms.call_args_list],
            [
                ("daemon", "status"),
                ("daemon", "up"),
                ("server", "start"),
                ("load", "qwen7b", "--yes"),
            ],
        )

    @patch("pro._lmstudio_disk_models")
    def test_discovery_lists_all_chat_models_and_excludes_embeddings(
        self, disk_models
    ) -> None:
        manager = pro.LocalModelManager()
        disk_models.return_value = [
            {"modelKey": "qwen7b", "displayName": "Qwen 7B", "type": "llm", "sizeBytes": 5},
            {"modelKey": "embedding", "type": "embedding", "sizeBytes": 1},
            {"modelKey": "qwen30b", "displayName": "Qwen 30B", "type": "llm", "sizeBytes": 30},
        ]
        with patch.object(manager, "_ollama_server", return_value={"models": []}):
            models = manager.discover_models()
        self.assertEqual([model.model for model in models], ["qwen7b", "qwen30b"])

    @patch("pro._lmstudio_disk_models", return_value=[])
    @patch("pro._read_json")
    def test_discovery_uses_lmstudio_api_when_cli_list_is_empty(
        self, read_json, _disk_models
    ) -> None:
        read_json.side_effect = [
            None,
            {
                "data": [
                    {"id": "chat-model", "object": "model"},
                    {"id": "text-embedding-model", "object": "model"},
                ]
            },
        ]
        manager = pro.LocalModelManager()
        with patch.object(manager, "_ollama_server", return_value={"models": []}):
            models = manager.discover_models()
        self.assertEqual([model.model for model in models], ["chat-model"])

    @patch("pro._lmstudio_disk_models", return_value=[])
    @patch("pro._read_json", return_value=None)
    def test_discovery_includes_ollama_models_and_filters_embeddings(
        self, _read_json, _disk_models
    ) -> None:
        manager = pro.LocalModelManager()
        with patch.object(
            manager,
            "_ollama_server",
            return_value={
                "models": [
                    {"name": "qwen3:8b", "size": 8},
                    {"name": "nomic-embed-text", "size": 1},
                ]
            },
        ):
            models = manager.discover_models()
        self.assertEqual([model.model for model in models], ["qwen3:8b"])

    @patch("pro._local_api_post")
    @patch("pro._read_json")
    def test_load_warms_ollama_model_in_background(self, read_json, post) -> None:
        read_json.side_effect = lambda url, timeout=3.0: (
            {"models": []} if url.endswith("/api/tags") else {"models": []}
        )
        manager = pro.LocalModelManager()
        backend = pro.LocalBackend("ollama", "qwen3:8b", "Ollama — qwen3:8b")
        with patch.object(manager, "_backend_is_loaded", return_value=False):
            manager.load_backend(backend)
        self.assertEqual(post.call_args.args[0], f"{pro.OLLAMA_URL}/api/generate")
        self.assertEqual(post.call_args.args[1]["keep_alive"], -1)

    @patch("pro._run_lms")
    def test_release_unloads_selected_model_and_stops_owned_daemon(
        self, run_lms
    ) -> None:
        manager = pro.LocalModelManager()
        manager._selected_backend = pro.LocalBackend(
            "lmstudio", "qwen7b", "LM Studio — Qwen 7B"
        )
        manager._started_lmstudio_daemon = True
        with (
            patch.object(manager, "_backend_is_loaded", return_value=True),
            patch("pro._lmstudio_server_available", return_value=True),
        ):
            manager.release()
        self.assertEqual(
            [call.args for call in run_lms.call_args_list],
            [
                ("unload", "qwen7b"),
                ("server", "stop"),
                ("daemon", "down"),
            ],
        )

    @patch("pro.subprocess.run")
    @patch("pro.shutil.which", return_value="/usr/bin/ollama")
    def test_release_unloads_selected_ollama_model(
        self, _which, run_process
    ) -> None:
        manager = pro.LocalModelManager()
        manager._selected_backend = pro.LocalBackend(
            "ollama", "qwen3:8b", "Ollama — qwen3:8b"
        )
        run_process.return_value.returncode = 0
        with patch.object(manager, "_backend_is_loaded", return_value=True):
            manager.release()
        self.assertEqual(
            run_process.call_args.args[0],
            ["/usr/bin/ollama", "stop", "qwen3:8b"],
        )

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
            pro.correct_transcript(
                "Testo grezzo.",
                provider="ollama",
                context="Lezione di radiologia: risonanza magnetica, parenchima.",
            ),
            "Testo corretto e leggibile.",
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, f"{pro.OLLAMA_URL}/api/chat")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], pro.OLLAMA_MODEL)
        self.assertFalse(payload["think"])
        self.assertIn("Contesto specialistico", payload["messages"][1]["content"])
        self.assertIn("parenchima", payload["messages"][1]["content"])
        self.assertIn("Testo grezzo.", payload["messages"][1]["content"])

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

    def test_rename_speaker_tags_in_corrected_text(self) -> None:
        text = "[Voce 1]\nCiao.\n\n[Voce 2]\nBuongiorno."
        renamed = engine.rename_speaker_tags(
            text,
            {"Voce 1": "Voce 1", "Voce 2": "Voce 2"},
            {"Voce 1": "Anna", "Voce 2": "Luca"},
        )
        self.assertEqual(renamed, "[Anna]\nCiao.\n\n[Luca]\nBuongiorno.")

    @patch("pro.ensure_local_model")
    @patch(
        "pro.correct_transcript",
        return_value="[Voce 1]\nPolished transcript.",
    )
    def test_engine_pro_stage_keeps_original_and_saves_new_output(
        self, _correct, _ensure
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "lecture.txt"
            raw_path.write_text("[Voce 1]\nRough transcript.", encoding="utf-8")
            worker = engine.TranscriptionWorker(lambda _state: None)
            worker._state["txt_path"] = str(raw_path)
            worker._state["speakers"] = ["Voce 1"]
            worker._state["speaker_names"] = {"Voce 1": "Voce 1"}
            job = engine.TranscriptionJob(
                Path(directory) / "lecture.wav",
                initial_prompt="Lezione di fisica: Hamiltoniana, quadrupolo.",
                pro_provider="lmstudio",
                pro_model="qwen2.5-coder-7b-instruct-abliterated",
            )

            worker._process_pro_job(job)

            snapshot = worker.get_snapshot()
            corrected_path = Path(snapshot["txt_path"])
            self.assertEqual(snapshot["status"], "done")
            self.assertEqual(snapshot["speakers"], ["Voce 1"])
            self.assertEqual(snapshot["speaker_blocks"], [])
            self.assertEqual(
                raw_path.read_text(encoding="utf-8"),
                "[Voce 1]\nRough transcript.",
            )
            self.assertEqual(
                corrected_path.read_text(encoding="utf-8"),
                "[Voce 1]\nPolished transcript.\n",
            )
            _correct.assert_called_once()
            self.assertEqual(_correct.call_args.kwargs["provider"], "lmstudio")
            self.assertEqual(
                _correct.call_args.kwargs["model"],
                "qwen2.5-coder-7b-instruct-abliterated",
            )
            _ensure.assert_called_once_with(
                "lmstudio",
                "qwen2.5-coder-7b-instruct-abliterated",
                progress=ANY,
            )
            self.assertEqual(
                _correct.call_args.kwargs["context"],
                "Lezione di fisica: Hamiltoniana, quadrupolo.",
            )

    @patch("pro.ensure_local_model")
    @patch(
        "pro.correct_transcript",
        side_effect=pro.ModelUnavailable("No local model is loaded"),
    )
    def test_pro_failure_keeps_original_available(self, _correct, _ensure) -> None:
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
