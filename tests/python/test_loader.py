"""
Tests for trizaval.suite.loader -- loading and validating eval suite
YAML files.
"""

import pytest

from trizaval.suite.loader import SuiteLoadError, load_suite


class TestLoadSuite:
    def test_loads_real_example_suite(self, example_suite_path):
        suite = load_suite(example_suite_path)
        assert suite.name == "arithmetic-sanity-check"
        assert len(suite.test_cases) == 4
        assert suite.baseline.model == "gpt-4o-mini"
        assert [c.model for c in suite.candidates] == ["gpt-4o"]
        assert suite.statistics.method == "bootstrap"

    def test_missing_file_raises_suite_load_error(self):
        with pytest.raises(SuiteLoadError, match="not found"):
            load_suite("does_not_exist_anywhere.yaml")

    def test_malformed_yaml_raises_suite_load_error(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("name: [this is not valid: yaml: at all")
        with pytest.raises(SuiteLoadError, match="YAML"):
            load_suite(bad_file)

    def test_incomplete_suite_raises_suite_load_error(self, tmp_path):
        incomplete_file = tmp_path / "incomplete.yaml"
        incomplete_file.write_text("name: incomplete-suite\n")
        with pytest.raises(SuiteLoadError, match="validation"):
            load_suite(incomplete_file)

    def test_empty_file_raises_suite_load_error(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        with pytest.raises(SuiteLoadError, match="empty"):
            load_suite(empty_file)

    def test_non_mapping_yaml_raises_suite_load_error(self, tmp_path):
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n")
        with pytest.raises(SuiteLoadError, match="mapping"):
            load_suite(list_file)