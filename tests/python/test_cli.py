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