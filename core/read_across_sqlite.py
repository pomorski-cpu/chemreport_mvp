from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_CATEGORY_TABLE = "category_dataset"
DEFAULT_TARGET_TABLES = {
    "logp": "logp_dataset",
    "pesticide_class": "pesticide_class_dataset",
    "toxicity": "toxicity_dataset",
}


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _normalize_columns(fieldnames: Iterable[str] | None) -> list[str]:
    columns = [str(name).strip() for name in (fieldnames or []) if str(name).strip()]
    if not columns:
        raise ValueError("CSV file must contain a header row.")
    return columns


def import_csv_to_table(conn: sqlite3.Connection, csv_path: Path, table_name: str) -> int:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = _normalize_columns(reader.fieldnames)
        table_sql = ", ".join(f"{quote_identifier(column)} TEXT" for column in columns)
        conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
        conn.execute(f"CREATE TABLE {quote_identifier(table_name)} ({table_sql})")

        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            f"INSERT INTO {quote_identifier(table_name)} "
            f"({', '.join(quote_identifier(column) for column in columns)}) "
            f"VALUES ({placeholders})"
        )

        row_count = 0
        for row in reader:
            values = [row.get(column, "") for column in columns]
            conn.execute(insert_sql, values)
            row_count += 1
    return row_count


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def fetch_table_rows(db_path: Path, table_name: str) -> list[dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {quote_identifier(table_name)}").fetchall()
    return [dict(row) for row in rows]


def build_read_across_sqlite(
    db_path: Path,
    *,
    category_csv: Path,
    logp_csv: Path | None = None,
    pesticide_csv: Path | None = None,
    toxicity_csv: Path | None = None,
    category_table: str = DEFAULT_CATEGORY_TABLE,
    logp_table: str = DEFAULT_TARGET_TABLES["logp"],
    pesticide_table: str = DEFAULT_TARGET_TABLES["pesticide_class"],
    toxicity_table: str = DEFAULT_TARGET_TABLES["toxicity"],
) -> dict[str, object]:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("category_dataset", category_table, category_csv),
        ("logp", logp_table, logp_csv),
        ("pesticide_class", pesticide_table, pesticide_csv),
        ("toxicity", toxicity_table, toxicity_csv),
    ]

    summary: dict[str, object] = {
        "db_path": str(db_path),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {},
    }

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_manifest (
                dataset_key TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                built_at_utc TEXT NOT NULL
            )
            """
        )

        conn.execute("BEGIN")
        try:
            for dataset_key, table_name, source_path in datasets:
                if source_path is None:
                    conn.execute(
                        "DELETE FROM dataset_manifest WHERE dataset_key = ?",
                        (dataset_key,),
                    )
                    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
                    continue

                row_count = import_csv_to_table(conn, source_path, table_name)
                built_at = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT INTO dataset_manifest (
                        dataset_key,
                        table_name,
                        source_path,
                        row_count,
                        built_at_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_key) DO UPDATE SET
                        table_name = excluded.table_name,
                        source_path = excluded.source_path,
                        row_count = excluded.row_count,
                        built_at_utc = excluded.built_at_utc
                    """,
                    (dataset_key, table_name, str(source_path), row_count, built_at),
                )
                summary["tables"][dataset_key] = {
                    "table_name": table_name,
                    "source_path": str(source_path),
                    "row_count": row_count,
                }
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    return summary
