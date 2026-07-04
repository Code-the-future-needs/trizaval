"""
DuckDB-backed querying over trizaval's Parquet eval-history files.

DuckDB queries Parquet files directly on disk with no import or
loading step, so this module is a thin SQL layer over the files
arrow_store.py writes -- not a separate database that needs syncing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import duckdb

from trizaval.storage.arrow_store import StorageError, _store_path


def _connect_to_suite_file(storage_dir: Union[str, Path], suite_name: str) -> tuple[duckdb.DuckDBPyConnection, Path]:
    path = _store_path(storage_dir, suite_name)
    if not path.exists():
        raise StorageError(f"no history found for suite '{suite_name}' at {path}")
    conn = duckdb.connect(database=":memory:")
    return conn, path


def score_trend(
    storage_dir: Union[str, Path],
    suite_name: str,
    candidate_name: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """Returns run_timestamp, run_id, and the candidate's mean score
    for every recorded run of `candidate_name` within `suite_name`,
    ordered oldest to newest. `limit`, if given, returns only the
    most recent N runs (still in oldest-to-newest order).

    "Mean score" here is computed from candidate_scores at query time
    (list_avg over the stored score array), not read from
    statistic_json, since not every configured statistic (e.g.
    sequential testing) has a directly comparable point estimate --
    the raw scores are always present and always comparable.
    """
    conn, path = _connect_to_suite_file(storage_dir, suite_name)

    query = """
        SELECT run_timestamp, run_id, list_avg(candidate_scores) AS mean_score
        FROM read_parquet(?)
        WHERE candidate_name = ?
        ORDER BY run_timestamp ASC
    """
    if limit is not None:
        query = f"""
            SELECT * FROM (
                {query}
            )
            ORDER BY run_timestamp DESC
            LIMIT ?
        """
        rows = conn.execute(query, [str(path), candidate_name, limit]).fetchall()
        rows = list(reversed(rows))  # back to oldest-to-newest after the DESC/LIMIT trick
    else:
        rows = conn.execute(query, [str(path), candidate_name]).fetchall()

    return [
        {"run_timestamp": r[0], "run_id": r[1], "mean_score": r[2]}
        for r in rows
    ]


def latest_run(storage_dir: Union[str, Path], suite_name: str) -> list[dict]:
    """Returns every candidate's row from the most recent run of
    `suite_name` (there can be multiple candidates per run)."""
    conn, path = _connect_to_suite_file(storage_dir, suite_name)

    query = """
        SELECT run_id, run_timestamp, candidate_name, candidate_scores, statistic_json, length_bias_applied, errors_json
        FROM read_parquet(?)
        WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM read_parquet(?))
    """
    rows = conn.execute(query, [str(path), str(path)]).fetchall()
    columns = ["run_id", "run_timestamp", "candidate_name", "candidate_scores", "statistic_json", "length_bias_applied", "errors_json"]
    return [dict(zip(columns, r)) for r in rows]


def query(storage_dir: Union[str, Path], suite_name: str, sql: str, params: Optional[list] = None) -> list[dict]:
    """Escape hatch: runs arbitrary SQL against a suite's history
    file. Use `{table}` in `sql` as a placeholder for the Parquet
    file reference, e.g.:

        query(dir, "my-suite", "SELECT DISTINCT candidate_name FROM {table}")

    This exists because score_trend and latest_run cover the common
    cases, but DuckDB is the actual underlying engine here -- hiding
    arbitrary SQL access behind only two fixed functions would be a
    real, avoidable limitation for anyone with a more specific
    question about their eval history.
    """
    conn, path = _connect_to_suite_file(storage_dir, suite_name)
    resolved_sql = sql.replace("{table}", f"read_parquet('{path}')")
    rows = conn.execute(resolved_sql, params or []).fetchall()
    columns = [desc[0] for desc in conn.description]
    return [dict(zip(columns, r)) for r in rows]