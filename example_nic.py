"""
FFIEC NIC Data Loader - loads XML files (FULL_P + INCR_P) from S3 to PostgreSQL.
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import psycopg
from psycopg import sql

from table_functions import add_files_to_metadata_table, update_table, get_s3_client, s3_glob


TYPE_MAP = {
    "int": "BIGINT",
    "string": "TEXT",
    "float": "DOUBLE PRECISION",
    "boolean": "BOOLEAN",
    "datetime": "TIMESTAMP",
}


def parse_date(path: str) -> datetime.date | None:
    """Extract YYYYMMDDHHmmss timestamp from filename, return as date."""
    if match := re.search(r"(\d{14})", path.rsplit("/", 1)[-1]):
        return datetime.strptime(match[1], "%Y%m%d%H%M%S").date()
    return None


def ensure_tables_exist(conn, patterns: list[str] | None = None):
    """Create nic schema and tables if they don't exist."""
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS nic")
    for pattern in patterns or FFIEC_TABLES:
        cfg = FFIEC_TABLES[pattern]
        table_name = cfg["table_name"]
        cols = sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(TYPE_MAP[typ]))
            for col, (_, typ) in cfg["column_mapping"].items()
            if col != "default"
        )
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS nic.{} ({})").format(
                sql.Identifier(table_name), cols
            )
        )
    cur.close()
    conn.commit()


S3_SOURCE_DIR = "s3://your-bucket/ffiec/nic"

FFIEC_TABLES = {
    "ATTRIBUTES_ACTIVE": {
        "table_name": "attributes_active",
        "pk": ["id_rssd"],
        "column_mapping": {"id_rssd": ([], "int"), "transtype": ([], "string")},
    },
    "ATTRIBUTES_BRANCH": {
        "table_name": "attributes_branch",
        "pk": ["id_rssd"],
        "column_mapping": {"id_rssd": ([], "int"), "transtype": ([], "string")},
    },
    # Add remaining tables...
}


class ParsedFile(NamedTuple):
    path: str
    date: datetime.date
    is_full: bool  # True for FULL_1_P, FULL_2_P, etc.


def is_full_file(path: str) -> bool:
    """Check if path is a full load file (FULL_P, FULL_1_P, FULL_2_P, etc.)."""
    return bool(re.search(r"FULL_(\d+_)?P", path))


def load_ffiec_nic(
    conninfo: str,
    source_dir: str = S3_SOURCE_DIR,
    patterns: list[str] | None = None,
):
    print(f"Listing {source_dir}...")
    client = get_s3_client()
    all_files = s3_glob(client, source_dir, "**/*.xml")
    print(f"Found {len(all_files)} files")

    # Parse and group files by pattern
    by_pattern = defaultdict(list)
    for path in all_files:
        if not (date := parse_date(path)):
            continue
        parsed = ParsedFile(path, date, is_full_file(path))
        for pattern in FFIEC_TABLES:
            if pattern in path:
                by_pattern[pattern].append(parsed)

    with psycopg.connect(conninfo) as conn:
        ensure_tables_exist(conn, patterns)
        for pattern in patterns or FFIEC_TABLES:
            cfg = FFIEC_TABLES[pattern]
            table_name = cfg["table_name"]
            files = by_pattern[pattern]
            tbl = sql.Identifier(table_name)

            # Get current state: all FULL_*_P files from latest date + any INCR_P after
            if not (fulls := [f for f in files if f.is_full]):
                raise ValueError(f"No FULL_*_P file found for {pattern}")
            latest_full_date = max(f.date for f in fulls)
            # Include ALL full file parts from the latest date (FULL_1_P, FULL_2_P, etc.)
            current = {f.path for f in fulls if f.date == latest_full_date} | {
                f.path for f in files if not f.is_full and f.date > latest_full_date
            }

            cur = conn.cursor()
            try:
                cur.execute(
                    sql.SQL(
                        "SELECT source_path FROM staging.metadata WHERE source_path LIKE {}"
                    ).format(sql.Literal(f"%{pattern}%"))
                )
                existing = {r[0] for r in cur}
            except psycopg.errors.UndefinedTable:
                conn.rollback()
                existing = set()

            # Reset if newer full available
            old_full_paths = [p for p in existing if is_full_file(p)]
            old_full_dates = [d for p in old_full_paths if (d := parse_date(p))]
            old_full_date = max(old_full_dates) if old_full_dates else None
            if old_full_date and old_full_date < latest_full_date:
                print(
                    f"New full detected ({old_full_date} -> {latest_full_date}), resetting {table_name}"
                )
                cur.execute(sql.SQL("TRUNCATE staging.{}").format(tbl))
                cur.execute(
                    sql.SQL(
                        "DELETE FROM staging.metadata WHERE source_path LIKE {}"
                    ).format(sql.Literal(f"%{pattern}%"))
                )
                conn.commit()
                existing = set()

            if to_add := current - existing:
                print(f"Adding {len(to_add)} new files for {table_name}")
                add_files_to_metadata_table(
                    conninfo=conninfo,
                    schema="staging",
                    source_dir=source_dir,
                    filetype="xml",
                    file_list_filter_fn=lambda paths, ta=to_add: [
                        p for p in paths if p in ta
                    ],
                )

            update_table(
                conninfo=conninfo,
                schema="staging",
                source_dir=source_dir,
                output_table=table_name,
                filetype="xml",
                column_mapping=cfg["column_mapping"],
                sql_glob=f"%{pattern}%",
                resume=True,
            )
            # Dedupe by PK (latest wins), exclude deletes
            cols = sql.SQL(", ").join(
                sql.Identifier(c)
                for c in cfg["column_mapping"]
                if c not in ("transtype", "default")
            )
            pk = sql.SQL(", ").join(sql.Identifier(c) for c in cfg["pk"])
            cur.execute(sql.SQL("TRUNCATE nic.{}").format(tbl))
            cur.execute(
                sql.SQL("""
                INSERT INTO nic.{}
                SELECT {} FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY {} ORDER BY substring(source_path FROM '\\d{{8}}')::date DESC
                    ) AS rn
                    FROM staging.{}
                ) ranked
                WHERE rn = 1 AND transtype != 'D'
            """).format(tbl, cols, pk, tbl)
            )
            cur.execute(sql.SQL("SELECT count(*) FROM nic.{}").format(tbl))
            print(f"nic.{table_name}: {cur.fetchone()[0]} rows")  # type: ignore[index]
            cur.close()
            conn.commit()

    print("\nDone!")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--conninfo", default="postgresql://localhost/ffiec")
    p.add_argument("--source-dir", default=S3_SOURCE_DIR)
    p.add_argument(
        "--patterns", nargs="+", help="Patterns to process (e.g., ATTRIBUTES_ACTIVE)"
    )
    a = p.parse_args()
    load_ffiec_nic(a.conninfo, a.source_dir, a.patterns)
