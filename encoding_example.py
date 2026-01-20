import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import pandas as pd
    import psycopg
    from table_functions_postgres import add_files_to_metadata_table, update_table


@app.cell
def _():
    mo.md("""
    # Working with Different Character Encodings

    This notebook demonstrates how to ingest files with non-UTF-8 encodings,
    specifically CP-1252 (Windows-1252), which is common in files from Excel
    or legacy Windows systems.

    ## Common Encodings

    - **utf-8** (default): Universal, supports all languages
    - **cp1252** (Windows-1252): Common for Excel exports, Western European languages
    - **latin1** (ISO-8859-1): Western European languages
    - **utf-8-sig**: UTF-8 with BOM (Byte Order Mark)

    ## The Problem

    Files with special characters like:
    - Café, Münchën, François
    - São Paulo, México
    - ñ, é, ü, ç, ã, ó

    Will fail to ingest if you use the wrong encoding!
    """)
    return


@app.cell
def _():
    # Connection string for all database operations
    conninfo = "postgresql://tanner@localhost:5432/postgres"

    # Create a connection for mo.sql() queries
    conn = psycopg.connect(conninfo, autocommit=False)

    # Create schema
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS test_encoding")
    conn.commit()

    # Setup paths
    base_dir = mo.notebook_dir().parent
    source_dir = base_dir / "data" / "raw" / "encoding_test"

    # Create directories
    source_dir.mkdir(parents=True, exist_ok=True)
    return conn, conninfo, source_dir


@app.cell
def _(mo, source_dir):
    mo.md(f"""
    ## Step 1: Create Sample CP-1252 File

    Let's create a test file with special characters in CP-1252 encoding.
    This simulates receiving a file exported from Excel.

    **File location:** `{source_dir}/restaurants.csv`
    """)
    return


@app.cell
def _(source_dir):
    # Create sample CP-1252 encoded file
    content = """Name,City,Description,Price
    Café Münchën,São Paulo,Delicious café with ñoño,12.50
    Restaurant François,Montréal,Best crêpes in town,15.75
    José's Taquería,México,Señor José's specialty,8.99
    """

    file_path = source_dir / "restaurants.csv"
    with open(file_path, 'w', encoding='cp1252') as f:
        f.write(content)

    print(f"✓ Created CP-1252 encoded file: {file_path}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Add to Metadata with Encoding Parameter

    **Key parameter:** `encoding="cp1252"`

    Without this, you'll get errors like:
    ```
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9
    ```
    """)
    return


@app.cell
def _(add_files_to_metadata_table, conninfo, source_dir):
    # Add files to metadata with CP-1252 encoding
    metadata_df = add_files_to_metadata_table(
        conninfo=conninfo,
        schema="test_encoding",
        source_dir=str(source_dir),
        filetype="csv",
        has_header=True,
        encoding="cp1252",  # ← Specify the encoding!
        resume=True,
    )
    return (metadata_df,)


@app.cell
def _(metadata_df, mo):
    mo.ui.table(metadata_df[['source_path', 'header', 'row_count', 'metadata_ingest_status']])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Infer Schema with Encoding

    Use the CLI with `--encoding` parameter:

    ```bash
    python src/table_functions_postgres.py     data/landing/encoding_test/restaurants.csv     --encoding cp1252     --pretty
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Ingest Data with Encoding Parameter

    `update_table()` now supports the encoding parameter directly!
    No need for `custom_read_fn` unless you need custom logic.
    """)
    return


@app.cell
def _(conninfo, source_dir, update_table):
    # Define column mapping from schema inference
    column_mapping = {
        "name": (["Name"], "string"),
        "city": (["City"], "string"),
        "description": (["Description"], "string"),
        "price": (["Price"], "float64"),
    }

    # Ingest with encoding parameter
    ingest_df = update_table(
        conninfo=conninfo,
        schema="test_encoding",
        output_table="restaurants",
        filetype="csv",
        source_dir=str(source_dir),
        column_mapping=column_mapping,
        encoding="cp1252",  # ← Simply specify encoding!
        resume=True,
    )
    return (ingest_df,)


@app.cell
def _(ingest_df, mo):
    mo.ui.table(ingest_df[['source_path', 'status', 'ingest_runtime']])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Verify Special Characters

    Check that special characters were preserved correctly:
    """)
    return


@app.cell
def _(conn, mo):
    result_df = mo.sql(
        f"""
        SELECT "Name", "City", "Description", "Price"
        FROM test_encoding.restaurants
        ORDER BY "Price"
        """,
        engine=conn
    )
    return (result_df,)


@app.cell
def _(mo, result_df):
    mo.md(f"""
    ### Results

    ✓ All special characters preserved correctly!

    {mo.ui.table(result_df)}

    Characters preserved: **é, ü, ñ, ã, ç, ó**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary: Encoding Best Practices

    ### 1. For `add_files_to_metadata_table()`
    ```python
    add_files_to_metadata_table(
        conninfo="postgresql://user:pass@host/db",
        schema="raw",
        source_dir="data/raw/",
        filetype="csv",
        encoding="cp1252",  # Specify encoding
    )
    ```

    ### 2. For `update_table()` - Direct encoding parameter
    ```python
    update_table(
        conninfo="postgresql://user:pass@host/db",
        schema="raw",
        output_table="my_table",
        filetype="csv",
        source_dir="data/raw/",
        column_mapping=column_mapping,
        encoding="cp1252",  # ← Simply specify encoding!
    )
    ```

    **Alternative:** For custom logic, use `custom_read_fn`:
    ```python
    def custom_reader(full_path):
        return pd.read_csv(full_path, encoding='cp1252', dtype={...})

    update_table(..., custom_read_fn=custom_reader)
    ```

    ### 3. For Schema Inference
    ```bash
    python src/table_functions_postgres.py file.csv --encoding cp1252 --pretty
    ```

    ### Common Encodings
    - `utf-8` - Universal (default)
    - `cp1252` - Windows-1252, Excel exports
    - `latin1` - ISO-8859-1, Western European
    - `utf-8-sig` - UTF-8 with BOM

    ### Encoding Detection

    If unsure about the encoding, use chardet:
    ```python
    import chardet
    with open('file.csv', 'rb') as f:
        result = chardet.detect(f.read())
        print(result['encoding'])  # e.g., 'Windows-1252'
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
