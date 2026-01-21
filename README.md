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
make setup  # Installs uv and syncs dependencies
```

Or manually:
```bash
pip install uv
uv sync
```

## Quick Start

### 1. Setup PostgreSQL Schema

```sql
CREATE SCHEMA raw;
```

### 2. Create a Data Ingestion Script

```python
from table_functions import add_files_to_metadata_table, update_table
import psycopg

# Connection string
conninfo = "postgresql://user@localhost:5432/mydb"

# Create schema (one-time setup)
with psycopg.connect(conninfo) as conn:
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")

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
    conninfo=conninfo,
    schema="raw",
    source_dir="data/",
    filetype="csv",
    compression_type="zip",
    archive_glob="*.csv",
    has_header=False,
    encoding="utf-8",
    resume=True,
)

# Ingest extracted files into PostgreSQL
def header_fn(file):
    return [k for k in column_mapping.keys() if k != "default"]

update_table(
    conninfo=conninfo,
    schema="raw",
    output_table="my_table",
    filetype="csv",
    sql_glob="%.csv",
    column_mapping=column_mapping,
    header_fn=header_fn,
    source_dir="data/",
    resume=True,
)
```

### Connection String Format

The `conninfo` parameter accepts standard PostgreSQL connection strings:

```python
# URI format (recommended)
conninfo = "postgresql://user:password@host:port/database"

# Examples
conninfo = "postgresql://tanner@localhost:5432/postgres"  # No password (local trust)
conninfo = "postgresql://user:secret@db.example.com/mydb"  # With password
conninfo = "postgresql://user@localhost/mydb?sslmode=require"  # With options
```

## Key Concepts

### Two-Phase Ingestion

**Phase 1: `add_files_to_metadata_table`** - Extract & Catalog
- Searches for files (or ZIPs) in `source_dir`
- Extracts compressed files to local cache
- Records metadata: file path, row count, hash, header

**Phase 2: `update_table`** - Read & Load
- Reads files based on metadata
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

Example marimo notebooks at the top level:

- `example_census_dhc_2020.py` - 2020 Census DHC data ingestion
- `example_encoding.py` - Working with non-UTF-8 encodings
- `example_iris_dataset.py` - UCI Iris dataset

**Run them:**
```bash
python example_iris_dataset.py
```

## Advanced Features

### Resume Capability

Set `resume=True` to skip already-processed files:

```python
add_files_to_metadata_table(resume=True, ...)
update_table(resume=True, ...)
```

### Archive-Level Resume (Skip Entire Archives)

For large S3 archives, use `expected_archive_file_count` to skip completed archives entirely (no download/open):

```python
add_files_to_metadata_table(
    source_dir="s3://bucket/data/",
    compression_type="zip",
    expected_archive_file_count=20,  # Expected files per archive
    resume=True,
)
```

Archives with `processed_file_count >= expected_archive_file_count` are marked `Success` and skipped on resume. This creates an `archive_metadata` table to track completion status.

### Retry Failed Files

```python
update_table(resume=True, retry_failed=True, ...)
```

### S3 Support

Load files directly from S3:

```python
add_files_to_metadata_table(
    source_dir="s3://my-bucket/path/to/data",
    filetype="csv",
    compression_type="zip",
    ...
)
```

**Requirements:**
- AWS credentials configured (via `~/.aws/credentials`, environment variables, or IAM role)

**Using AWS SSO or named profiles:**
```python
import s3fs
fs = s3fs.S3FileSystem(profile="my-sso-profile")
add_files_to_metadata_table(..., filesystem=fs)
```

### Schema Inference CLI

Infer column types and detect null values from your data files:

```bash
python table_functions.py data/my_file.csv --pretty
python table_functions.py data/my_files/ --pretty  # All files in directory
```

The CLI reads the entire file by default (use `--sample-rows N` to limit) and automatically detects common null value representations (`NA`, `N/A`, `None`, `NULL`, etc.) in your data.

### Custom Transformations

```python
def transform_fn(df):
    df['new_column'] = df['existing_column'] * 2
    return df

update_table(transform_fn=transform_fn, ...)
```

### Dynamic Column Mapping

```python
def column_mapping_fn(file_path):
    if "2020" in str(file_path):
        return {"id": ([], "string"), "value": ([], "Int64")}
    else:
        return {"id": ([], "string"), "amount": ([], "float64")}

update_table(column_mapping_fn=column_mapping_fn, ...)
```

### Additional Columns from Filename

```python
def additional_cols_fn(file):
    year = int(file.stem.split('_')[1])
    return {"year": year}

update_table(additional_cols_fn=additional_cols_fn, ...)
```

### Custom Null Values

Handle multiple null representations in your data:

```python
update_table(
    null_values=["NA", "None", "N/A", ""],  # Treat all as null
    ...
)
```

## Data Directory Structure

```
data/                 # Source files (immutable)
├── census/           # Census DHC ZIP files
├── earthquakes/      # Earthquake CSVs
└── iris.zip          # UCI Iris dataset

temp/                 # Cache for S3 files and extracted archives
└── ...
```

## Project Structure

```
teeny_data_framework/
├── table_functions.py           # Main PostgreSQL implementation
├── example_census_dhc_2020.py   # Census example notebook
├── example_encoding.py          # Encoding example notebook
├── example_iris_dataset.py      # Iris example notebook
├── data/                        # Source data files
├── tests/                       # Test suite
├── scripts/
│   ├── pre-push                 # Git pre-push hook
│   └── table_functions_spark.py # Spark implementation (archived)
├── Makefile
├── pyproject.toml
├── README.md
└── CLAUDE.md                    # AI development notes
```

## Testing

```bash
# Run all tests (requires Docker/Podman)
pytest tests/ -v

# Install git pre-push hook (runs tests before push)
make install-hooks
```

## License

MIT
