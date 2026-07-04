"""
Arrow/Parquet-backed persistence for eval suite run history.

Each suite gets its own Parquet file, named after the suite. Every
call to `append_run` adds new rows for that run without overwriting
prior history, so a suite's file accumulates a queryable record of
every run over time.

A CandidateReport's statistic_result can be one of three
different shapes (BootstrapResult, EffectSizeResult,
SequentialStatResult) depending on the suite's configured method.
Rather than a wide schema with a nullable column per possible field
across all three types (which would need a migration every time a new
statistic type is added), the statistic is stored as a single JSON
string column. DuckDB and most Parquet tooling can query JSON columns
directly, so this is a schema-stability tradeoff, not a loss of
queryability.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from trizaval.cli import _statistic_to_dict
from trizaval.harness.runner import SuiteReport


_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("run_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("suite_name", pa.string()),
        pa.field("candidate_name", pa.string()),
        pa.field("baseline_scores", pa.list_(pa.float64())),
        pa.field("candidate_scores", pa.list_(pa.float64())),
        pa.field("statistic_json", pa.string()),
        pa.field("length_bias_applied", pa.bool_()),
        pa.field("errors_json", pa.string()),
    ]
)


class StorageError(Exception):
    """Raised for failures reading or writing suite run history."""


def _report_to_rows(report: SuiteReport, run_id: str, run_timestamp: datetime) -> list[dict]:
    rows = []
    for cr in report.candidate_reports:
        rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "suite_name": report.suite_name,
                "candidate_name": cr.candidate_name,
                "baseline_scores": cr.baseline_scores,
                "candidate_scores": cr.candidate_scores,
                "statistic_json": json.dumps(_statistic_to_dict(cr.statistic_result)),
                "length_bias_applied": cr.length_bias_applied,
                "errors_json": json.dumps(cr.errors),
            }
        )
    return rows


def _store_path(storage_dir: Union[str, Path], suite_name: str) -> Path:
    storage_dir = Path(storage_dir)
    # Suite names are free-form strings from user YAML; sanitize to a
    # safe filename rather than trusting them directly as a path
    # component.
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in suite_name)
    return storage_dir / f"{safe_name}.parquet"


def append_run(
    report: SuiteReport,
    storage_dir: Union[str, Path],
    run_id: Optional[str] = None,
    run_timestamp: Optional[datetime] = None,
) -> Path:
    """Appends one suite run's results to that suite's Parquet file,
    creating the file and storage directory if they don't exist yet.

    Returns the path written to. Raises StorageError if the existing
    file can't be read (e.g. corrupted or schema-incompatible).
    """
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or str(uuid.uuid4())
    run_timestamp = run_timestamp or datetime.now(timezone.utc)

    rows = _report_to_rows(report, run_id, run_timestamp)
    if not rows:
        # A suite with zero candidates is a degenerate but not
        # invalid case (schema allows it); nothing to append.
        return _store_path(storage_dir, report.suite_name)

    new_table = pa.Table.from_pylist(rows, schema=_SCHEMA)

    path = _store_path(storage_dir, report.suite_name)

    if path.exists():
        try:
            existing_table = pq.read_table(path)
        except Exception as e:
            raise StorageError(f"failed to read existing history file {path}: {e}") from e

        if existing_table.schema != _SCHEMA:
            raise StorageError(
                f"{path} has an incompatible schema (likely from an older version of trizaval); "
                "cannot safely append. Back up and remove the file to start fresh, or migrate it "
                "manually."
            )

        combined_table = pa.concat_tables([existing_table, new_table])
    else:
        combined_table = new_table

    try:
        pq.write_table(combined_table, path)
    except Exception as e:
        raise StorageError(f"failed to write {path}: {e}") from e

    return path


def read_history(storage_dir: Union[str, Path], suite_name: str) -> pa.Table:
    """Reads all recorded history for `suite_name` as a PyArrow Table.

    Raises StorageError if no history file exists yet for this suite.
    """
    path = _store_path(storage_dir, suite_name)
    if not path.exists():
        raise StorageError(f"no history found for suite '{suite_name}' at {path}")

    try:
        return pq.read_table(path)
    except Exception as e:
        raise StorageError(f"failed to read {path}: {e}") from e