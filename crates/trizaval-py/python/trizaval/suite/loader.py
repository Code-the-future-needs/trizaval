"""
Loads and validates a trizaval eval-suite config file (YAML) into an
`EvalSuite` Pydantic model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from trizaval.suite.schema import EvalSuite


class SuiteLoadError(Exception):
    """Raised when a suite file cannot be read, parsed, or validated."""


def load_suite(path: Union[str, Path]) -> EvalSuite:
    """Loads an eval suite from a YAML file at `path`.

    Raises `SuiteLoadError` with a clear message on any failure: file
    not found, malformed YAML, or schema validation failure. Callers
    should not need to catch YAML-library or Pydantic-specific
    exceptions directly.
    """
    path = Path(path)

    if not path.exists():
        raise SuiteLoadError(f"suite file not found: {path}")

    try:
        raw_text = path.read_text()
    except OSError as e:
        raise SuiteLoadError(f"failed to read {path}: {e}") from e

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise SuiteLoadError(f"failed to parse {path} as YAML: {e}") from e

    if data is None:
        raise SuiteLoadError(f"{path} is empty")

    if not isinstance(data, dict):
        raise SuiteLoadError(
            f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )

    try:
        return EvalSuite.model_validate(data)
    except ValidationError as e:
        raise SuiteLoadError(f"{path} failed schema validation:\n{e}") from e