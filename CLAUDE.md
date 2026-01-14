# Claude Working Notes - Teeny Data Framework

## Project Purpose

Lightweight data ingestion framework for loading CSV/TSV/PSV files (including from ZIP archives) into PostgreSQL with metadata tracking and row count validation.

## Schema Inference CLI

Use the CLI to automatically infer column types from your data files. This generates a column_mapping that you can copy-paste directly into your code.

### Basic Usage

```bash
# Infer schema from any file
python src/table_functions_postgres.py <file_path> --pretty

# Examples
python src/table_functions_postgres.py data/raw/my_file.csv --pretty
python src/table_functions_postgres.py data/raw/my_file.psv --filetype psv --pretty
python src/table_functions_postgres.py data/raw/my_file.data --no-header --pretty
```

### CLI Options

- `--filetype {csv,tsv,psv,xlsx,parquet}` - File type (auto-detected from extension)
- `--separator ","` - Column separator for text files
- `--no-header` - File has no header row (generates col_0, col_1, etc.)
- `--encoding "utf-8-sig"` - File encoding
- `--excel-skiprows N` - Rows to skip in Excel files
- `--sample-rows 1000` - Number of rows to sample for type inference
- `--pretty` - Pretty-print JSON output

### Workflow

```bash
# 1. Run CLI to infer schema
python src/table_functions_postgres.py data/raw/earthquakes.csv --pretty

# 2. Copy the JSON output and paste as column_mapping in your notebook:
column_mapping = {
  "time": [[], "string"],
  "latitude": [[], "float"],
  "longitude": [[], "float"],
  "mag": [[], "float"],
  # ... rest of output
}

# 3. Use in update_table()
update_table(
    conninfo="postgresql://user:pass@host/db",  # Connection string
    column_mapping=column_mapping,  # <-- Pasted mapping
    ...
)
```

### Output Format

The CLI outputs JSON in the exact `column_mapping` format with **automatic snake_case conversion**:

```json
{
  "snake_case_column": [["OriginalColumnName"], "type_string"],
  "already_snake": [[], "type_string"],
  ...
}
```

**Column Name Conversion:**
- Column names are automatically converted to `snake_case`
- Original column name is included in the array (e.g., `"FirstName"` becomes `"first_name": [["FirstName"], "string"]`)
- If the column is already in snake_case, the array is empty (e.g., `"user_id": [[], "int"]`)
- This allows the framework to match either the snake_case name or the original name when reading files

**Examples:**
- `"FirstName"` → `"first_name": [["FirstName"], "string"]`
- `"User ID"` → `"user_id": [["User ID"], "int"]`
- `"totalAmount"` → `"total_amount": [["totalAmount"], "float"]`
- `"user_id"` → `"user_id": [[], "int"]` (already snake_case)

**Type strings:** `int`, `float`, `boolean`, `datetime`, `string`

## Key Patterns to Follow

### 1. Header Function Pattern (CRITICAL)

For headerless files, **ALWAYS derive the header from column_mapping**. Never duplicate column names.

```python
# Define column mapping
column_mapping = {
    "col1": ([], "string"),
    "col2": ([], "float"),
    "default": ([], "string"),
}

# Derive header from mapping keys (excluding 'default')
def my_header_fn(file):
    return [k for k in column_mapping.keys() if k != "default"]

update_table(
    column_mapping=column_mapping,
    header_fn=my_header_fn,
    ...
)
```

This maintains single source of truth for column names.

### 2. Character Encoding Support

The framework supports non-UTF-8 encodings like CP-1252 (Windows-1252), common in Excel exports.

**For `add_files_to_metadata_table()`:**
```python
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    search_dir="data/raw/",
    landing_dir="data/landing/",
    filetype="csv",
    encoding="cp1252",  # Specify encoding
)
```

**For `update_table()` - direct encoding parameter:**
```python
update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    landing_dir="data/landing/",
    column_mapping=column_mapping,
    encoding="cp1252",  # Simply specify encoding!
)
```

**Alternative - use `custom_read_fn` for custom logic:**
```python
import pandas as pd

def read_cp1252_csv(full_path):
    return pd.read_csv(
        full_path,
        encoding='cp1252',
        dtype={
            'Name': 'string',
            'City': 'string',
            'Price': 'float64',
        }
    )

update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    landing_dir="data/landing/",
    custom_read_fn=read_cp1252_csv,
)
```

**For schema inference CLI:**
```bash
python src/table_functions_postgres.py data/file.csv --encoding cp1252 --pretty
```

**Common encodings:**
- `utf-8` - Universal (default)
- `cp1252` - Windows-1252, Excel exports
- `latin1` - ISO-8859-1, Western European
- `utf-8-sig` - UTF-8 with BOM

**Detect encoding:**
```python
import chardet
with open('file.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result['encoding'])  # e.g., 'Windows-1252'
```

See `notebooks/encoding_example.py` for a complete demo.

### 3. S3 Support

Both `search_dir` and `landing_dir` can be prefixed with `s3://` to use S3 buckets:

```python
# Example: Read from S3, write to local landing
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    search_dir="s3://my-bucket/raw-data/",
    landing_dir="data/landing/",
    filetype="csv",
)

# Example: Read from local, write to S3 landing
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    search_dir="data/raw/",
    landing_dir="s3://my-bucket/landing/",
    filetype="csv",
)

# Example: Both S3
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    search_dir="s3://my-bucket/raw-data/",
    landing_dir="s3://my-bucket/landing/",
    filetype="csv",
)
```

**Requirements:**
- Install s3fs: `pip install s3fs`
- AWS credentials configured (via environment, ~/.aws/credentials, or IAM role)

**How it works:**
- Files from S3 `search_dir` are downloaded temporarily and uploaded to `landing_dir`
- Files in S3 `landing_dir` are downloaded temporarily for processing
- All metadata (paths, file hashes, row counts) are tracked in PostgreSQL

**Using AWS SSO or named profiles:**
```python
import s3fs
fs = s3fs.S3FileSystem(profile="my-sso-profile")
add_files_to_metadata_table(..., filesystem=fs)
```
Or just set `AWS_PROFILE=my-profile` and credentials are used automatically.

### 3. Data Directory Structure

```
data/
├── raw/              # Source files (user places ZIPs/CSVs here)
│   ├── census/       # Census DHC ZIP files
│   ├── earthquakes/  # Earthquake CSVs
│   └── iris.zip      # UCI Iris dataset
└── landing/          # Extracted/processed files (auto-created)
    ├── census/       # Extracted DHC files
    └── iris/         # Extracted .data files
```

Or use S3 buckets:
```
s3://my-bucket/
├── raw-data/         # Source files
└── landing/          # Extracted/processed files
```

### 4. Running Marimo Notebooks

Marimo notebooks can be run as Python scripts:
```bash
python census/dhc_2020.py
```

### 5. Marimo Cells - Variable Naming

In marimo notebooks, variables must be unique across cells OR prefixed with underscore for private:
```python
# WRONG - will error
@app.cell
def _(conn):
    cur = conn.cursor()  # 'cur' defined in multiple cells

# CORRECT
@app.cell
def _(conn):
    _cur = conn.cursor()  # private variable
```

### 6. SQL in Marimo - Escaping

When using `mo.sql()`, escape `%` as `%%` in LIKE patterns:
```python
mo.sql("""
    SELECT * FROM table
    WHERE path LIKE '%%pattern%%'  -- Note: %% not %
""", engine=conn)
```

### 7. DHC Census Files

DHC files are **pipe-delimited** (`.dhc`) with **no headers**:
- filetype: `"psv"`
- has_header: `False`
- Columns stored as `text`, cast in queries: `SUM("POP100"::bigint)`

## Typical Workflow

### ZIP Extraction + Ingestion

```python
# Step 1: Extract from ZIP
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    search_dir=str(search_dir),
    landing_dir=str(landing_dir),
    filetype="csv",  # or "psv" for pipe-delimited
    compression_type="zip",
    archive_glob="*.csv",
    has_header=False,
    encoding="utf-8",
    resume=False,
)

# Step 2: Ingest extracted files
update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    sql_glob="%.csv",
    column_mapping=column_mapping,
    header_fn=my_header_fn,  # Use pattern from section 1
    landing_dir=str(landing_dir),
    resume=False,
)
```

## update_table() Advanced Features

Quick reference for `update_table()` optional parameters:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `transform_fn` | `Callable[[DataFrame], DataFrame]` | Transform DataFrame after reading |
| `additional_cols_fn` | `Callable[[Path], Dict]` | Add columns from filename |
| `output_table_naming_fn` | `Callable[[Path], str]` | Dynamic table name per file |
| `file_list_filter_fn` | `Callable[[List], List]` | Filter files to process |
| `custom_read_fn` | `Callable[[str], DataFrame]` | Custom file reader |
| `column_mapping_fn` | `Callable[[Path], Dict]` | Dynamic column mapping per file |
| `pivot_mapping` | `Dict` | Unpivot/melt configuration |
| `header_fn` | `Callable[[Path], List[str]]` | Custom header for headerless files |
| `sample` | `int` | Process only N files |
| `null_value` | `str` | Custom null representation |
| `resume` | `bool` | Skip already-processed files |
| `retry_failed` | `bool` | Re-process failed files |

## File Type Reference

| Format | filetype | has_header | header_fn needed? | Notes |
|--------|----------|------------|-------------------|-------|
| CSV with headers | `"csv"` | `True` | No | Standard CSV |
| CSV without headers | `"csv"` | `False` | Yes - use pattern | |
| Pipe-delimited (.dhc) | `"psv"` | `False` | Yes - use pattern | Census DHC files |
| Tab-delimited | `"tsv"` | `True/False` | If False | |
| Excel | `"xlsx"` | `True` | No | Requires openpyxl |
| Parquet | `"parquet"` | N/A | No | Requires pyarrow |
| Fixed-width | `"fixed_width"` | `False` | Special format | Different column_mapping format |

## Important Don'ts

1. **Don't** duplicate column names in header_fn - derive from column_mapping
2. **Don't** use non-prefixed variables across multiple marimo cells
3. **Don't** forget to escape `%` as `%%` in marimo SQL queries
4. **Don't** assume DHC columns are numeric - they're text, cast in queries
5. **Don't** create new data directories - use existing `data/raw/` and `data/landing/`

## Notebook Locations

All notebooks are in `notebooks/` directory. **One notebook per dataset.**

- `census_dhc_2020.py` - 2020 Census DHC data ingestion
- `iris_dataset.py` - UCI Iris dataset

## Testing

### Running the Test Suite

The project has comprehensive tests using pytest and testcontainers (requires Docker):

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_table_functions_postgres.py -v

# Run just path utility tests
pytest tests/test_table_functions_postgres.py::TestPathUtilities -v
```

**Test coverage includes:**
- Path utilities (`normalize_path`, `path_join`, `path_basename`, `path_parent`)
- S3 helpers
- File readers (CSV, TSV, PSV, Excel, Parquet, Fixed-width)
- Database operations
- All `update_table()` advanced features (`transform_fn`, `custom_read_fn`, `column_mapping_fn`, `pivot_mapping`, etc.)
- `add_files_to_metadata_table()` end-to-end workflows
- CLI schema inference

### Quick Manual Testing

Reset database and test:
```bash
psql -U tanner -d postgres -c "DROP SCHEMA IF EXISTS raw CASCADE; CREATE SCHEMA raw;"
python notebooks/iris_dataset.py
```

Check results:
```bash
psql -U tanner -d postgres -c "SELECT status, COUNT(*) FROM raw.metadata GROUP BY status;"
```
