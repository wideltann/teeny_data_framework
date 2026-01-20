import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


with app.setup:
    import sys
    from pathlib import Path

    # Add project root to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import psycopg
    from src.table_functions_postgres import add_files_to_metadata_table, update_table


@app.cell
def _(mo, psycopg):
    # Connection string for all database operations
    conninfo = "postgresql://tanner@localhost:5432/postgres"

    # Also create a connection for mo.sql() queries
    conn = psycopg.connect(conninfo, autocommit=False)

    # Setup paths
    base_dir = mo.notebook_dir().parent  # Go up to project root
    source_dir = base_dir / "data" / "raw"

    mo.md(f"""
    # UCI Iris Dataset Ingestion

    **Source dir:** `{source_dir}`
    **Backend:** PostgreSQL (pure psycopg)
    **Compression:** ZIP extraction enabled
    **Cache:** Files extracted to `temp/` directory
    """)

    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.commit()
    return conn, conninfo, source_dir


@app.cell
def _(mo):
    mo.md("""
    ## Schema Inference with Snake Case Conversion

    The CLI automatically converts column names to snake_case and tracks original names.
    This is useful when working with files that have mixed naming conventions.

    **Example Usage:**
    ```bash
    # For files with headers like "Sepal Length", "Petal Width", "Class Name"
    python src/table_functions_postgres.py data/your_file.csv --pretty
    ```

    **Output format:**
    ```json
    {
      "sepal_length": [["Sepal Length"], "float64"],
      "petal_width": [["Petal Width"], "float64"],
      "class_name": [["Class Name"], "string"],
      "already_snake": [[], "string"]
    }
    ```

    - Converts: `"Sepal Length"` → `"sepal_length"`
    - Original name stored in array: `[["Sepal Length"], "float64"]`
    - Already snake_case gets empty array: `[[], "string"]`
    - Framework matches either snake_case or original name when reading files
    """)
    return


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
def _(add_files_to_metadata_table, conninfo, source_dir):
    # Extract .data files from ZIP and add to metadata
    # filetype="csv" because .data files are CSV format
    # archive_glob="*.data" specifies which files to extract from the ZIP
    add_files_to_metadata_table(
        conninfo=conninfo,
        schema="raw",
        source_dir=str(source_dir),
        filetype="csv",  # The actual file format (CSV)
        compression_type="zip",
        archive_glob="*.data",  # Extract only .data files from ZIP
        has_header=False,
        encoding="utf-8",
        resume=True,
    )
    return


@app.cell
def _(column_mapping, conninfo, source_dir, update_table):
    # Ingest the extracted .data files
    # Derive header from column mapping keys
    def iris_header_fn(file):
        return list(column_mapping.keys())

    update_table(
        conninfo=conninfo,
        schema="raw",
        output_table="iris",
        filetype="csv",  # Read as CSV format
        sql_glob="%.data",  # Match .data files
        column_mapping=column_mapping,
        header_fn=iris_header_fn,
        source_dir=str(source_dir),
        resume=False,
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM raw.iris LIMIT 20
        """,
        engine=conn
    )
    return


if __name__ == "__main__":
    app.run()
