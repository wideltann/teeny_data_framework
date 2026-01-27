"""
FFIEC NIC Data Loader - loads XML files (FULL_P + INCR_P) from S3 to PostgreSQL.
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import psycopg
from psycopg import sql
import s3fs

from table_functions import add_files_to_metadata_table, update_table


TYPE_MAP = {
    "int": "BIGINT",
    "string": "TEXT",
    "float": "DOUBLE PRECISION",
    "boolean": "BOOLEAN",
    "datetime": "TIMESTAMP",
}


def parse_date(path: str) -> datetime.date | None:
    """Extract YYYYMMDD date from filename."""
    if match := re.search(r"(\d{8})", path.rsplit("/", 1)[-1]):
        return datetime.strptime(match[1], "%Y%m%d").date()
    return None


def ensure_tables_exist(conn, tables: list[str] | None = None):
    """Create nic schema and tables if they don't exist."""
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS nic")
    for name in tables or FFIEC_TABLES:
        cfg = FFIEC_TABLES[name]
        cols = sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(TYPE_MAP[typ]))
            for col, (_, typ) in cfg["column_mapping"].items()
            if col != "default"
        )
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS nic.{} ({})").format(
                sql.Identifier(name), cols
            )
        )
    cur.close()
    conn.commit()


S3_SOURCE_DIR = "s3://your-bucket/ffiec/nic"

FFIEC_TABLES = {
    "attributes_active": {
        "pk": ["id_rssd"],
        "pattern": "ATTRIBUTES_ACTIVE",
        "column_mapping": {"id_rssd": ([], "int"), "transtype": ([], "string")},
    },
    "attributes_branch": {
        "pk": ["id_rssd"],
        "pattern": "ATTRIBUTES_BRANCH",
        "column_mapping": {"id_rssd": ([], "int"), "transtype": ([], "string")},
    },
    # Add remaining tables...
}


class ParsedFile(NamedTuple):
    path: str
    date: datetime.date
    is_full: bool


def load_ffiec_nic(
    conninfo: str,
    source_dir: str = S3_SOURCE_DIR,
    tables: list[str] | None = None,
):
    print(f"Listing {source_dir}...")
    paths = s3fs.S3FileSystem().glob(f"{source_dir}/**/*.xml")
    all_files = [f"s3://{p}" for p in paths]
    print(f"Found {len(all_files)} files")

    # Parse and group files by table name
    by_table = defaultdict(list)
    for path in all_files:
        if not (date := parse_date(path)):
            continue
        parsed = ParsedFile(path, date, "FULL_P" in path)
        for name, cfg in FFIEC_TABLES.items():
            if cfg["pattern"] in path:
                by_table[name].append(parsed)

    with psycopg.connect(conninfo) as conn:
        ensure_tables_exist(conn, tables)
        for name in tables or FFIEC_TABLES:
            cfg = FFIEC_TABLES[name]
            files = by_table[name]
            tbl = sql.Identifier(name)
            pattern = cfg["pattern"]

            # Get current state: latest FULL_P + any INCR_P after it
            if not (fulls := [f for f in files if f.is_full]):
                raise ValueError("No FULL_P file found")
            latest_full = max(fulls, key=lambda f: f.date)
            current = {latest_full.path} | {
                f.path for f in files if not f.is_full and f.date > latest_full.date
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
            old_full_path = next((p for p in existing if "FULL_P" in p), None)
            if (
                old_full_path
                and (old_date := parse_date(old_full_path))
                and old_date < latest_full.date
            ):
                print(f"New full detected, resetting {name}")
                cur.execute(sql.SQL("TRUNCATE staging.{}").format(tbl))
                cur.execute(
                    sql.SQL(
                        "DELETE FROM staging.metadata WHERE source_path LIKE {}"
                    ).format(sql.Literal(f"%{pattern}%"))
                )
                conn.commit()
                existing = set()

            if to_add := current - existing:
                print(f"Adding {len(to_add)} new files")
                add_files_to_metadata_table(
                    conninfo=conninfo,
                    schema="staging",
                    source_dir=source_dir,
                    filetype="xml",
                    file_list_filter_fn=lambda paths: [p for p in paths if p in to_add],
                )

            update_table(
                conninfo=conninfo,
                schema="staging",
                source_dir=source_dir,
                output_table=name,
                filetype="xml",
                column_mapping=cfg["column_mapping"],
                sql_glob=f"%{cfg['pattern']}%",
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
            print(f"nic.{name}: {cur.fetchone()[0]} rows")  # type: ignore[index]
            cur.close()
            conn.commit()

    print("\nDone!")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--conninfo", default="postgresql://localhost/ffiec")
    p.add_argument("--source-dir", default=S3_SOURCE_DIR)
    p.add_argument("--tables", nargs="+")
    a = p.parse_args()
    load_ffiec_nic(a.conninfo, a.source_dir, a.tables)
