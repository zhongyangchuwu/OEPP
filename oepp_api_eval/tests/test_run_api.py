import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_api
from run_api import _validate_manifest


class ExpectedSplitTests(unittest.TestCase):
    def test_rejects_manifest_for_other_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected base split"):
            _validate_manifest([{"sample_id": "novel-sample", "split": "novel"}], "base")

    def test_accepts_manifest_for_expected_split(self) -> None:
        _validate_manifest([{"sample_id": "base-sample", "split": "base"}], "base")


class RunBudgetTests(unittest.TestCase):
    def test_new_run_does_not_consume_another_runs_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "prompt.txt").write_text("ignored", encoding="utf-8")
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "base-1",
                        "split": "base",
                        "T": 1,
                        "vid": "video",
                        "event": "event",
                        "candidate_actions": [{"id": 0, "text": "action"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "runs" / "api_call_ledger.jsonl").write_text(
                json.dumps({"run_name": "completed-qwen-run"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "experiment": {"api_enabled": True, "max_calls": 1, "expected_split": "base"},
                "model": {
                    "api_key_env": "TEST_API_KEY",
                    "base_url_env": "TEST_BASE_URL",
                    "model_env": "TEST_MODEL",
                },
                "request": {
                    "concurrency": 1,
                    "max_retries": 0,
                    "backoff_seconds": 1,
                    "image_detail": "auto",
                    "timeout_seconds": 1,
                    "temperature": 0,
                    "max_tokens": 16,
                },
                "prompt": {"version": "test", "mode": "candidate_ids_json", "path": "prompt.txt"},
            }
            requests: list[dict[str, object]] = []
            test_case = self

            class Completion:
                id = "completion-1"
                usage = None
                choices = [SimpleNamespace(message=SimpleNamespace(content='{"actions": [0]}'))]

                def model_dump(self, mode: str) -> dict[str, str]:
                    test_case.assertEqual(mode, "json")
                    return {"id": self.id}

            completion = Completion()

            def create(**request: object) -> Completion:
                requests.append(request)
                return completion

            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            modules = {
                "dotenv": SimpleNamespace(load_dotenv=lambda _: None),
                "openai": SimpleNamespace(OpenAI=lambda **_: client),
            }

            with (
                patch.object(run_api, "EXPERIMENT_ROOT", root),
                patch.object(run_api, "load_yaml", return_value=config),
                patch.object(
                    run_api,
                    "_prepare_request",
                    return_value=([], {"sample_id": "base-1"}),
                ),
                patch.object(run_api.importlib, "import_module", side_effect=modules.__getitem__),
                patch.dict(
                    os.environ,
                    {
                        "TEST_API_KEY": "test-key",
                        "TEST_BASE_URL": "https://example.test/v1",
                        "TEST_MODEL": "google/gemini-3.1-flash-lite",
                    },
                    clear=False,
                ),
            ):
                result = run_api.run(
                    root / "config.yaml", root / "manifest.jsonl", "gemini-pilot", None
                )

            self.assertEqual(result["api_attempts"], 1)
            self.assertEqual(result["completed_samples"], 1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["model"], "google/gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main()
