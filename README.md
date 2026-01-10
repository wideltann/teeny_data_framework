# Teeny Data Framework

A lightweight data ingestion framework for loading CSV/TSV/PSV files (including from ZIP archives) into PostgreSQL with metadata tracking and row count validation.

## Overview

Teeny Data Framework provides a simple, reliable way to:
- Extract files from ZIP archives
- Track file metadata (row counts, hashes, processing status)
- Load data into PostgreSQL with automatic schema creation
- Validate row counts between source files and database
- Resume interrupted ingestion jobs
- Handle headerless files with custom column mappings

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│  Raw Files  │ ───> │   Metadata   │ ───> │  PostgreSQL  │
│ (ZIP/CSV)   │      │    Table     │      │    Tables    │
└─────────────┘      └──────────────┘      └──────────────┘
     │                      │                      │
     │                      │                      │
  Phase 1:             Extraction              Phase 2:
  Extract &             Tracking              Read & Load
  Catalog
```

**Tech Stack:**
- **Database**: PostgreSQL (direct psycopg3 connection)
- **Processing**: pandas DataFrames
- **File Formats**: CSV, TSV, PSV, XLSX, Parquet, Fixed-width
- **Compression**: ZIP archives

## Installation

```bash
# Core dependencies
pip install pandas psycopg numpy

# Optional dependencies
pip install openpyxl          # For Excel files
pip install pyarrow           # For Parquet files
pip install zipfile-deflate64 # For advanced ZIP compression
pip install marimo            # For running notebooks
```

## Quick Start

### 1. Setup PostgreSQL Schema

```sql
CREATE SCHEMA raw;
```

### 2. Create a Data Ingestion Script

```python
import psycopg
from pathlib import Path
from src.table_functions_postgres import add_files_to_metadata_table, update_table

# Connect to PostgreSQL
conn = psycopg.connect("postgresql://user@localhost:5432/mydb", autocommit=False)

# Setup paths
search_dir = Path("data/raw")
landing_dir = Path("data/landing")
landing_dir.mkdir(parents=True, exist_ok=True)

# Define column mapping
column_mapping = {
    "sepal_length": ([], "float64"),
    "sepal_width": ([], "float64"),
    "petal_length": ([], "float64"),
    "petal_width": ([], "float64"),
    "class": ([], "string"),
}

# Extract files from ZIP and add to metadata
add_files_to_metadata_table(
    conn=conn,
    schema="raw",
    search_dir=str(search_dir),
    landing_dir=str(landing_dir),
    filetype="csv",
    compression_type="zip",
    archive_glob="*.csv",
    has_header=False,
    encoding="utf-8",
    resume=True,
)

# Ingest extracted files into PostgreSQL
def header_fn(file):
    return list(column_mapping.keys())

update_table(
    conn=conn,
    schema="raw",
    output_table="my_table",
    filetype="csv",
    sql_glob="%.csv",
    column_mapping=column_mapping,
    header_fn=header_fn,
    landing_dir=str(landing_dir),
    resume=True,
)
```

## Key Concepts

### Two-Phase Ingestion

**Phase 1: `add_files_to_metadata_table`** - Extract & Catalog
- Searches for files (or ZIPs) in `search_dir`
- Extracts compressed files to `landing_dir`
- Records metadata: file path, row count, hash, header

**Phase 2: `update_table`** - Read & Load
- Reads files from `landing_dir` based on metadata
- Applies column mappings and transformations
- Bulk loads into PostgreSQL using COPY
- Validates row counts match metadata

### Column Mapping Pattern

Column mappings handle variations, renaming, and default types:

```python
column_mapping = {
    # Simple column (no variations)
    "id": ([], "string"),

    # Column with possible name variations
    "county": (["COUNTY", "County", "county_name"], "string"),

    # Default type for all unmapped columns
    "default": ([], "float64"),
}
```

**Format**: `{alias: ([possible_names], dtype)}`

### Header Function Pattern (CRITICAL)

For headerless files, **ALWAYS derive headers from column_mapping** to maintain single source of truth:

```python
column_mapping = {
    "col1": ([], "string"),
    "col2": ([], "float64"),
    "default": ([], "string"),
}

# Derive header from mapping (exclude 'default')
def my_header_fn(file):
    return [k for k in column_mapping.keys() if k != "default"]

update_table(
    column_mapping=column_mapping,
    header_fn=my_header_fn,
    has_header=False,  # File has no header row
    ...
)
```

### Compression Type Flow

```python
# Phase 1: Extract from ZIP
add_files_to_metadata_table(
    compression_type="zip",    # Look for *.zip files
    filetype="csv",           # Files inside are CSV
    archive_glob="*.csv",     # Extract only .csv files
    search_dir="data/raw",    # Find: *.zip
    landing_dir="data/landing"
)
# Result: Extracts CSV files to data/landing/{zipname}/*.csv

# Phase 2: Ingest CSV files
update_table(
    filetype="csv",           # Read as CSV
    sql_glob="%.csv",        # Match .csv files in metadata
    landing_dir="data/landing"
)
```

**Without compression:**
```python
# Phase 1: Copy files directly
add_files_to_metadata_table(
    compression_type=None,    # No compression
    filetype="csv",          # Look for *.csv files
    search_dir="data/raw"
)
```

## Data Types

Use pandas-compatible dtype strings:

| Type | String |
|------|--------|
| Text | `"string"` or `"object"` |
| Float | `"float64"` |
| Integer | `"int64"` |
| Nullable Integer | `"Int64"` (capital I) |
| Boolean | `"bool"` |

PostgreSQL types are auto-inferred:
- `"string"` → `TEXT`
- `"float64"` → `DOUBLE PRECISION`
- `"int64"` / `"Int64"` → `BIGINT`
- `"bool"` → `BOOLEAN`

## File Type Support

| Format | filetype | has_header | header_fn needed? | Notes |
|--------|----------|------------|-------------------|-------|
| CSV with headers | `"csv"` | `True` | No | Standard CSV |
| CSV without headers | `"csv"` | `False` | Yes | Use header function pattern |
| Pipe-delimited (.psv) | `"psv"` | `False` | Yes | Census DHC files |
| Tab-delimited | `"tsv"` | `True/False` | If False | TSV files |
| Excel | `"xlsx"` | `True` | No | Requires openpyxl |
| Parquet | `"parquet"` | N/A | No | Requires pyarrow |
| Fixed-width | `"fixed_width"` | `False` | Special | Custom format |

## Example Notebooks

The `notebooks/` directory contains working examples:

### Iris Dataset (`notebooks/iris_dataset.py`)
- Loads UCI Iris dataset from ZIP
- Demonstrates headerless CSV ingestion
- Simple 5-column dataset

**Run it:**
```bash
python notebooks/iris_dataset.py
```

### Census DHC 2020 (`notebooks/census_dhc_2020.py`)
- Loads 2020 Census DHC data from ZIP archives
- Pipe-delimited files (`.dhc`)
- Complex column mapping with default types

**Run it:**
```bash
python notebooks/census_dhc_2020.py
```

Both notebooks use `resume=True` to skip already-processed files.

## Advanced Features

### Resume Capability

Set `resume=True` to skip already-processed files:

```python
add_files_to_metadata_table(
    resume=True,  # Skip files already in metadata
    ...
)

update_table(
    resume=True,  # Skip files already ingested
    ...
)
```

### Retry Failed Files

```python
update_table(
    resume=True,
    retry_failed=True,  # Re-process failed files
    ...
)
```

### Custom Transformations

```python
def transform_fn(df):
    df['new_column'] = df['existing_column'] * 2
    return df

update_table(
    transform_fn=transform_fn,
    ...
)
```

### Additional Columns from Filename

```python
def additional_cols_fn(file):
    # Extract year from filename: "data_2020.csv" → 2020
    year = int(file.stem.split('_')[1])
    return {"year": year}

update_table(
    additional_cols_fn=additional_cols_fn,
    ...
)
```

### Unpivot/Melt Operations

```python
pivot_mapping = {
    "id_vars": ["id", "date"],
    "variable_column_name": "metric",
    "value_column_name": "value",
}

update_table(
    pivot_mapping=pivot_mapping,
    ...
)
```

### Excel Files with Label Rows

```python
update_table(
    filetype="xlsx",
    excel_skiprows=2,  # Skip first 2 rows of labels
    ...
)
```

## Metadata Table Schema

The framework automatically creates a metadata table:

```sql
CREATE TABLE {schema}.metadata (
    search_dir TEXT,
    landing_dir TEXT,
    full_path TEXT PRIMARY KEY,
    filesize BIGINT,
    header TEXT[],
    row_count BIGINT,
    archive_full_path TEXT,
    file_hash TEXT,
    metadata_ingest_datetime TIMESTAMP,
    metadata_ingest_status TEXT,
    ingest_datetime TIMESTAMP,
    ingest_runtime INTEGER,
    status TEXT,
    error_message TEXT,
    unpivot_row_multiplier INTEGER
)
```

**Query metadata:**
```sql
SELECT
    full_path,
    row_count,
    status,
    error_message
FROM raw.metadata
WHERE status = 'Failure';
```

## Data Directory Structure

```
data/
├── raw/              # Source files (user places ZIPs/CSVs here)
│   ├── census/       # Census DHC ZIP files
│   └── iris.zip      # UCI Iris dataset
└── landing/          # Extracted/processed files (auto-created)
    ├── census/       # Extracted DHC files
    └── iris/         # Extracted .data files
```

## Project Structure

```
teeny_data_framework/
├── src/
│   ├── table_functions_postgres.py  # PostgreSQL implementation
│   └── table_functions_spark.py     # Spark implementation
├── notebooks/
│   ├── iris_dataset.py              # Iris example
│   └── census_dhc_2020.py           # Census example
├── data/
│   ├── raw/                         # Place source files here
│   └── landing/                     # Extracted files go here
├── README.md
└── CLAUDE.md                        # Development notes
```

## Testing

Reset database and test:

```bash
# Drop and recreate schema
psql -U user -d mydb -c "DROP SCHEMA IF EXISTS raw CASCADE; CREATE SCHEMA raw;"

# Run a notebook
python notebooks/iris_dataset.py

# Check results
psql -U user -d mydb -c "SELECT status, COUNT(*) FROM raw.metadata GROUP BY status;"
```

## Important Notes

### Do's ✅
- Derive headers from column_mapping for headerless files
- Use `resume=True` to avoid reprocessing files
- Always validate row counts match between source and database
- Use the two-phase approach (extract → ingest)

### Don'ts ❌
- Don't duplicate column names in header_fn - derive from column_mapping
- Don't assume DHC columns are numeric - they're text, cast in queries
- Don't use non-prefixed variables across multiple marimo cells
- Don't forget to escape `%` as `%%` in marimo SQL queries
- Don't create new data directories - use existing structure

## Performance Considerations

**Memory**: Entire files are loaded into memory. For large files (>1GB), consider:
- Processing in smaller batches
- Using chunked reading (requires code modification)
- Using the Spark version for distributed processing

**Speed**: COPY is very fast for bulk loading. Typical speeds:
- Small files (<100MB): Seconds
- Medium files (100MB-1GB): Minutes
- Large files (>1GB): May hit memory limits

## Troubleshooting

**Import Error:**
```python
# Make sure to import from src/
from src.table_functions_postgres import add_files_to_metadata_table, update_table
```

**Encoding Issues:**
```python
# Specify encoding explicitly
add_files_to_metadata_table(
    encoding="utf-8-sig",  # Handles BOM markers
    ...
)
```

**Row Count Mismatch:**
- Check for blank lines in source files
- Verify `has_header` setting is correct
- Look for files with multiple header rows

## License

MIT

## Contributing

This is a personal project but suggestions are welcome!
