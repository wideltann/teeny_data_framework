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
    # Connection string for all database operations
    conninfo = "postgresql://tanner@localhost:5432/postgres"

    # Create a connection for mo.sql() queries
    conn = psycopg.connect(conninfo, autocommit=False)

    # Setup paths
    base_dir = Path(__file__).parent.parent  # Go up to project root
    source_dir = base_dir / "data" / "raw" / "census"

    mo.md(f"""
    # 2020 Decennial Census DHC Data Ingestion

    **Source dir:** `{source_dir}`
    **Backend:** PostgreSQL (pure psycopg)
    **Compression:** ZIP extraction enabled
    **Cache:** Files extracted to `temp/` directory
    """)

    with conn.cursor() as _cur:
        _cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.commit()
    return conn, conninfo, mo, source_dir


@app.cell
def _():
    """Define column mapping for 2020 DHC Census data"""
    # DHC geo files contain geographic identifiers
    # Note: The actual demographic data is in separate segment files (00001-00044)
    # This mapping handles the geography file (degeo2020.dhc)

    column_mapping = {
        # File/Summary identifiers
        "FILEID": ([], "string"),
        "STUSAB": ([], "string"),
        "SUMLEV": ([], "string"),
        "GEOVAR": ([], "string"),
        "GEOCOMP": ([], "string"),
        "CHARITER": ([], "string"),
        "CIFSN": ([], "string"),
        "LOGRECNO": ([], "string"),
        # Geographic identifiers
        "GEOID": ([], "string"),
        "GEOCODE": ([], "string"),
        "REGION": ([], "string"),
        "DIVISION": ([], "string"),
        "STATE": ([], "string"),
        "COUNTY": ([], "string"),
        "TRACT": ([], "string"),
        "BLKGRP": ([], "string"),
        "BLOCK": ([], "string"),
        "NAME": ([], "string"),
        "BASENAME": ([], "string"),
        # Geographic codes
        "CBSA": ([], "string"),
        "CSA": ([], "string"),
        "PLACE": ([], "string"),
        "ZCTA": ([], "string"),
        "PUMA": ([], "string"),
        # Population/Housing (100% counts)
        "POP100": ([], "Int64"),
        "HU100": ([], "Int64"),
        # Area
        "AREALAND": ([], "Int64"),
        "AREAWATR": ([], "Int64"),
        # Coordinates
        "INTPTLAT": ([], "string"),
        "INTPTLON": ([], "string"),
        # Status
        "FUNCSTAT": ([], "string"),
        # Default: all other columns as string
        "default": ([], "string"),
    }
    return (column_mapping,)


@app.cell
def _(add_files_to_metadata_table, conninfo, source_dir):
    # Extract DHC files from ZIP and add to metadata
    # DHC files are pipe-delimited (.dhc extension)
    add_files_to_metadata_table(
        conninfo=conninfo,
        schema="raw",
        source_dir=str(source_dir),
        filetype="psv",  # Pipe-separated values
        compression_type="zip",
        archive_glob="*.dhc",  # Extract .dhc files from ZIP
        has_header=False,  # DHC files do NOT have headers
        encoding="utf-8",
        resume=True,
    )
    return


@app.cell
def _(column_mapping, conninfo, source_dir, update_table):
    # Ingest the extracted DHC files
    # Derive header from column mapping keys (excluding 'default')
    def dhc_header_fn(file):
        # Return column names in order from the mapping (excluding 'default')
        return [k for k in column_mapping.keys() if k != "default"]


    update_table(
        conninfo=conninfo,
        schema="raw",
        output_table="dhc_2020",
        filetype="psv",  # Pipe-separated values
        sql_glob="%.dhc",  # Match .dhc files
        column_mapping=column_mapping,
        header_fn=dhc_header_fn,
        source_dir=str(source_dir),
        resume=True,
    )
    return




@app.cell
def _(conn, mo):
    # Query metadata to see what files were processed
    metadata_df = mo.sql(
        f"""
        SELECT
            *
        FROM raw.metadata
        WHERE source_path LIKE '%%census%%'
        ORDER BY source_path
        """,
        engine=conn,
    )
    return


@app.cell
def _(conn, mo):
    # Show summary statistics of the DHC data
    summary_df = mo.sql(
        """
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT "GEOID") as unique_geoids,
            SUM("POP100"::bigint) as total_population,
            SUM("HU100"::bigint) as total_housing_units,
            AVG("POP100"::bigint) as avg_population_per_record
        FROM raw.dhc_2020
        """,
        engine=conn,
    )
    return


@app.cell
def _(conn, mo):
    # Preview the data
    preview_df = mo.sql(
        """
        SELECT *
        FROM raw.dhc_2020
        LIMIT 20
        """,
        engine=conn,
    )
    return


@app.cell
def _(conn, mo):
    # Show population by race/ethnicity summary
    race_df = mo.sql(
        """
        SELECT
            "SUMLEV" as summary_level,
            COUNT(*) as count,
            SUM("POP100"::bigint) as population,
            SUM("HU100"::bigint) as housing_units
        FROM raw.dhc_2020
        GROUP BY "SUMLEV"
        ORDER BY "SUMLEV"
        LIMIT 20
        """,
        engine=conn,
    )
    return


if __name__ == "__main__":
    app.run()
