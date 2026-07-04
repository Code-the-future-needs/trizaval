"""
Tests for trizaval.storage -- Parquet-backed run history persistence
and DuckDB-backed querying over it.
"""

import time

import pytest

from trizaval.harness.providers.base import ProviderResponse
from trizaval.harness.runner import run_suite
from trizaval.storage.arrow_store import StorageError, append_run, read_history
from trizaval.storage.duckdb_store import latest_run, query, score_trend


def _always_correct_generate(self, prompt, *, temperature, max_tokens):
    from trizaval.suite.loader import load_suite

    suite = load_suite("suites/example_suite.yaml")
    for tc in suite.test_cases:
        if tc.prompt == prompt:
            return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
    return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})


@pytest.fixture
def suite(example_suite_path):
    from trizaval.suite.loader import load_suite

    return load_suite(example_suite_path)


class TestArrowStore:
    def test_append_and_read_single_run(self, suite, tmp_path):
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            report = run_suite(suite)

        path = append_run(report, tmp_path, run_id="run-1")
        assert path.exists()

        table = read_history(tmp_path, suite.name)
        assert table.num_rows == 1
        assert table.column("run_id").to_pylist() == ["run-1"]

    def test_multiple_runs_accumulate_not_overwrite(self, suite, tmp_path):
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            report1 = run_suite(suite)
            report2 = run_suite(suite)

        append_run(report1, tmp_path, run_id="run-1")
        append_run(report2, tmp_path, run_id="run-2")

        table = read_history(tmp_path, suite.name)
        assert table.num_rows == 2
        assert set(table.column("run_id").to_pylist()) == {"run-1", "run-2"}

    def test_read_missing_suite_raises_storage_error(self, tmp_path):
        with pytest.raises(StorageError, match="no history found"):
            read_history(tmp_path, "does-not-exist")

    def test_statistic_is_stored_as_valid_json(self, suite, tmp_path):
        import json
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            report = run_suite(suite)

        append_run(report, tmp_path, run_id="run-1")
        table = read_history(tmp_path, suite.name)

        parsed = json.loads(table.column("statistic_json")[0].as_py())
        assert parsed["method"] == "bootstrap"
        assert parsed["point_estimate"] == 1.0

    def test_suite_name_sanitized_for_filesystem(self, tmp_path):
        """Suite names come from user YAML and could contain
        characters unsafe for filenames; confirm this doesn't crash
        or escape the storage directory."""
        from trizaval.harness.runner import CandidateReport, SuiteReport

        report = SuiteReport(
            suite_name="weird/name with spaces!.yaml",
            candidate_reports=[
                CandidateReport(
                    candidate_name="c",
                    baseline_results=[],
                    candidate_results=[],
                    baseline_scores=[1.0],
                    candidate_scores=[1.0],
                )
            ],
        )
        path = append_run(report, tmp_path, run_id="run-1")
        assert path.parent == tmp_path  # did not escape the storage dir
        assert path.exists()


class TestDuckdbStore:
    def test_score_trend_reflects_improvement_over_runs(self, suite, tmp_path):
        from unittest.mock import patch

        run_correctness = {"n_correct": 1}

        def routed(self, prompt, *, temperature, max_tokens):
            for i, tc in enumerate(suite.test_cases):
                if tc.prompt == prompt:
                    if self.model == "gpt-4o-mini":
                        return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
                    if i < run_correctness["n_correct"]:
                        return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
                    return ProviderResponse(text="wrong", latency_seconds=0.01, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", routed):
            for n, run_id in [(1, "run-1"), (2, "run-2"), (4, "run-3")]:
                run_correctness["n_correct"] = n
                report = run_suite(suite)
                append_run(report, tmp_path, run_id=run_id)
                time.sleep(0.01)

        trend = score_trend(tmp_path, suite.name, "candidate-gpt4o")
        assert [row["run_id"] for row in trend] == ["run-1", "run-2", "run-3"]
        assert trend[0]["mean_score"] == 0.25
        assert trend[1]["mean_score"] == 0.5
        assert trend[2]["mean_score"] == 1.0

    def test_score_trend_respects_limit(self, suite, tmp_path):
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            for run_id in ["run-1", "run-2", "run-3"]:
                report = run_suite(suite)
                append_run(report, tmp_path, run_id=run_id)
                time.sleep(0.01)

        trend = score_trend(tmp_path, suite.name, "candidate-gpt4o", limit=2)
        assert len(trend) == 2
        assert [row["run_id"] for row in trend] == ["run-2", "run-3"]

    def test_latest_run_returns_only_most_recent(self, suite, tmp_path):
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            for run_id in ["run-1", "run-2", "run-3"]:
                report = run_suite(suite)
                append_run(report, tmp_path, run_id=run_id)
                time.sleep(0.01)

        latest = latest_run(tmp_path, suite.name)
        assert len(latest) == 1
        assert latest[0]["run_id"] == "run-3"

    def test_arbitrary_sql_escape_hatch(self, suite, tmp_path):
        from unittest.mock import patch

        with patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", _always_correct_generate):
            report = run_suite(suite)
            append_run(report, tmp_path, run_id="run-1")

        result = query(tmp_path, suite.name, "SELECT COUNT(*) AS total_runs FROM {table}")
        assert result == [{"total_runs": 1}]

    def test_query_on_missing_suite_raises_storage_error(self, tmp_path):
        with pytest.raises(StorageError, match="no history found"):
            score_trend(tmp_path, "does-not-exist", "some-candidate")