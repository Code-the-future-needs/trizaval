"""
Tests for trizaval.cli -- the command-line entrypoint for running
eval suites.
"""

import json
from unittest.mock import patch

from trizaval.cli import main
from trizaval.harness.providers.base import ProviderResponse
from trizaval.suite.loader import load_suite


def _always_correct_generate(self, prompt, *, temperature, max_tokens):
    suite = load_suite("suites/example_suite.yaml")
    for tc in suite.test_cases:
        if tc.prompt == prompt:
            return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
    return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})


class TestCliRun:
    def test_text_format_exits_zero_and_prints_report(self, capsys):
        with patch(
            "trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate
        ):
            exit_code = main(["run", "suites/example_suite.yaml", "--format", "text"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "arithmetic-sanity-check" in captured.out
        assert "Bootstrap:" in captured.out

    def test_json_format_produces_valid_parseable_json(self, capsys):
        with patch(
            "trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate
        ):
            exit_code = main(["run", "suites/example_suite.yaml", "--format", "json"])

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)  # must not raise
        assert data["suite_name"] == "arithmetic-sanity-check"
        assert data["candidate_reports"][0]["statistic_result"]["method"] == "bootstrap"

    def test_missing_suite_file_exits_nonzero(self, capsys):
        exit_code = main(["run", "does_not_exist.yaml"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_malformed_suite_file_exits_nonzero(self, tmp_path, capsys):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("name: [invalid: yaml: here")
        exit_code = main(["run", str(bad_file)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()


class TestCliStorageIntegration:
    def test_storage_dir_persists_run(self, tmp_path, capsys):
        from trizaval.storage.arrow_store import read_history

        with patch(
            "trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate
        ):
            exit_code = main(["run", "suites/example_suite.yaml", "--storage-dir", str(tmp_path)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Saved run to" in captured.err

        table = read_history(tmp_path, "arithmetic-sanity-check")
        assert table.num_rows == 1

    def test_multiple_runs_with_storage_dir_accumulate(self, tmp_path):
        from trizaval.storage.arrow_store import read_history

        with patch(
            "trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate
        ):
            main(["run", "suites/example_suite.yaml", "--storage-dir", str(tmp_path)])
            main(["run", "suites/example_suite.yaml", "--storage-dir", str(tmp_path)])

        table = read_history(tmp_path, "arithmetic-sanity-check")
        assert table.num_rows == 2

    def test_without_storage_dir_nothing_is_persisted(self, tmp_path, capsys):
        with patch(
            "trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate
        ):
            exit_code = main(["run", "suites/example_suite.yaml"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Saved run to" not in captured.err
        # No file should exist anywhere under tmp_path since we never
        # pointed storage at it.
        assert list(tmp_path.iterdir()) == []