import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import psycopg
    from src.table_functions_postgres import add_files_to_metadata_table, update_table
    return Path, add_files_to_metadata_table, mo, psycopg, update_table


@app.cell
def _(Path, mo, psycopg):
    # Connect to PostgreSQL with psycopg3
    conn = psycopg.connect("postgresql://tanner@localhost:5432/postgres", autocommit=False)

    # Setup paths
    base_dir = Path(__file__).parent.parent  # Go up to project root
    search_dir = base_dir / "data" / "raw"
    landing_dir = base_dir / "data" / "landing"

    # Create landing directory
    landing_dir.mkdir(parents=True, exist_ok=True)

    mo.md(f"""
    # UCI Iris Dataset Ingestion

    **Search dir:** `{search_dir}`
    **Landing dir:** `{landing_dir}`
    **Backend:** PostgreSQL (pure psycopg)
    **Compression:** ZIP extraction enabled
    """)

    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.commit()
    return conn, landing_dir, mo, search_dir


@app.cell
def _():
    """Define column mapping for Iris dataset"""
    # Iris dataset columns: sepal_length, sepal_width, petal_length, petal_width, class
    # The .data files are CSV without headers

    column_mapping = {
        "sepal_length": ([], "float64"),
        "sepal_width": ([], "float64"),
        "petal_length": ([], "float64"),
        "petal_width": ([], "float64"),
        "class": ([], "string"),
    }
    return (column_mapping,)


@app.cell
def _(add_files_to_metadata_table, conn, landing_dir, search_dir):
    # Extract .data files from ZIP and add to metadata
    # filetype="csv" because .data files are CSV format
    # archive_glob="*.data" specifies which files to extract from the ZIP
    add_files_to_metadata_table(
        conn=conn,
        schema="raw",
        search_dir=str(search_dir),
        landing_dir=str(landing_dir),
        filetype="csv",  # The actual file format (CSV)
        compression_type="zip",
        archive_glob="*.data",  # Extract only .data files from ZIP
        has_header=False,
        encoding="utf-8",
        resume=True,
    )
    return


@app.cell
def _(column_mapping, conn, landing_dir, update_table):
    # Ingest the extracted .data files
    # Derive header from column mapping keys
    def iris_header_fn(file):
        return list(column_mapping.keys())

    update_table(
        conn=conn,
        schema="raw",
        output_table="iris",
        filetype="csv",  # Read as CSV format
        sql_glob="%.data",  # Match .data files from landing directory
        column_mapping=column_mapping,
        header_fn=iris_header_fn,
        landing_dir=str(landing_dir),
        resume=True,
    )
    return




@app.cell
def _(conn, mo):
    _df = mo.sql(
        """
        SELECT * FROM raw.iris LIMIT 20
        """,
        engine=conn
    )
    return


if __name__ == "__main__":
    app.run()
