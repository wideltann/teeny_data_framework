# Claude Working Notes - Teeny Data Framework

## Project Purpose

Lightweight data ingestion framework for loading CSV/TSV/PSV files (including from ZIP archives) into PostgreSQL with metadata tracking and row count validation.

## Schema Inference CLI

Use the CLI to automatically infer column types from your data files. This generates a column_mapping that you can copy-paste directly into your code.

### Basic Usage

```bash
# Infer schema from a single file
python table_functions.py <file_path>

# Infer schema from all files in a directory (output keyed by filename)
python table_functions.py <directory_path>

# Examples
python table_functions.py data/my_file.csv
python table_functions.py data/my_file.psv --filetype psv
python table_functions.py data/my_file.data --no-header
python table_functions.py data/earthquakes/  # All files in dir
```

### CLI Options

- `--filetype {csv,tsv,psv,xlsx,parquet}` - File type (auto-detected from extension)
- `--no-header` - File has no header row (generates col_0, col_1, etc.)
- `--excel-skiprows N` - Rows to skip in Excel files
- `--sample-rows N` - Number of rows to sample for type inference (default: read entire file)

**Note:** Delimiter and encoding are auto-detected by DuckDB. The framework tries UTF-8 first, then falls back to latin-1 + ftfy for files with non-UTF-8 bytes.

### Workflow

```bash
# 1. Run CLI to infer schema
python table_functions.py data/earthquakes.csv --pretty

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

The CLI outputs JSON keyed by **original filename** with `table_name` (snake_case), `column_mapping`, detected `encoding`, and optionally `null_values`:

```json
{
  "MyDataFile.csv": {
    "table_name": "my_data_file",
    "column_mapping": {
      "snake_case_column": [["OriginalColumnName"], "type_string"],
      "already_snake": [[], "type_string"]
    },
    "null_values": ["NA", "None", "N/A"],
    "encoding": "utf-8"
  }
}
```

**Structure:**
- **Key**: Original filename (for matching in `column_mapping_fn`)
- **table_name**: Snake_case version of filename stem (for `output_table_naming_fn`)
- **column_mapping**: Column definitions with automatic snake_case conversion
- **encoding**: How the file was decoded - either `"utf-8"` or `"latin-1+ftfy"` (for non-UTF-8 files)
- **null_values**: (optional) List of detected null value representations in the data. Only included if custom null values are found. Common patterns detected: `NA`, `N/A`, `None`, `NULL`, `NaN`, `.`, `-`

**Column Name Conversion:**
- Column names are automatically converted to `snake_case` using the `inflection` library
- Original column name is included in the array (e.g., `"FirstName"` becomes `"first_name": [["FirstName"], "string"]`)
- If the column is already in snake_case, the array is empty (e.g., `"user_id": [[], "int"]`)
- This allows the framework to match either the snake_case name or the original name when reading files

**Examples:**
- `"FirstName"` → `"first_name": [["FirstName"], "string"]`
- `"User ID"` → `"user_id": [["User ID"], "int"]`
- `"totalAmount"` → `"total_amount": [["totalAmount"], "float"]`
- `"user_id"` → `"user_id": [[], "int"]` (already snake_case)

**Type strings:** `int`, `float`, `boolean`, `datetime`, `string`

### Multi-File Ingestion Pattern

The CLI output is designed for easy multi-file ingestion with dynamic mappings:

```python
# 1. Run CLI on directory
# python table_functions.py data/my_files/ --pretty > schema.json

# 2. Paste output as all_mappings dict
all_mappings = {
    "sales.csv": {
        "table_name": "sales",
        "column_mapping": {"date": [[], "datetime"], "amount": [[], "float"]}
    },
    "customers.csv": {
        "table_name": "customers",
        "column_mapping": {"name": [[], "string"], "email": [[], "string"]}
    },
}

# 3. Define lookup functions
def get_column_mapping(file_path):
    return all_mappings[file_path.name]["column_mapping"]

def get_table_name(file_path):
    return all_mappings[file_path.name]["table_name"]

# 4. Single update_table() call for all files
update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="unused",  # ignored when using output_table_naming_fn
    filetype="csv",
    source_dir="data/my_files/",
    column_mapping_fn=get_column_mapping,
    output_table_naming_fn=get_table_name,
)
```

- Processes all files matching the filetype (or all supported types if not specified)
- Non-recursive (only files directly in the directory)
- Files that fail to parse will include an `{"error": "..."}` entry

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

### 2. Automatic Encoding Detection

The framework automatically handles file encodings using a robust two-step approach:

1. **Try UTF-8 first** - most modern files use UTF-8
2. **Fall back to latin-1 + ftfy** - for files with non-UTF-8 bytes, decode as latin-1 (which accepts any byte) and use the `ftfy` library to fix encoding issues (mojibake)

This handles:
- Pure UTF-8 files
- Windows CP-1252 files (smart quotes, accented characters)
- Mixed encoding files (common when data comes from multiple sources)
- Files with problematic bytes like `0x81` (undefined in CP-1252)

**No encoding parameter needed** - encoding is always auto-detected.

**Example files that work automatically:**
```python
# All these work without specifying encoding:
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    source_dir="data/mixed_encodings/",  # UTF-8, CP-1252, latin-1 - all handled
    filetype="csv",
)

update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    source_dir="data/mixed_encodings/",
    column_mapping=column_mapping,
)
```

**For custom reading logic, use `custom_read_fn`:**
```python
import pandas as pd

def custom_reader(full_path):
    # Custom logic for special cases
    return pd.read_csv(full_path, dtype={'col': 'string'})

update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    source_dir="data/special/",
    custom_read_fn=custom_reader,
)
```

### 3. S3 Support

Source files can be stored in S3. Files are cached locally in `temp/` for processing.

```python
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    source_dir="s3://my-bucket/source/",   # S3 source (immutable)
    filetype="csv",
)
```

**Key Architecture:**
- `source_path` is the PRIMARY KEY in metadata - it uniquely identifies each file
- For archives: `source_path` uses `::` delimiter: `s3://bucket/archive.zip::inner/file.csv`
- Cache path is derived from `source_path`, mirroring the S3 structure locally

**Requirements:**
- Install s3fs: `pip install s3fs`
- AWS credentials configured (via environment, ~/.aws/credentials, or IAM role)

**How it works:**
- S3 files are downloaded to local `temp/` cache that mirrors the S3 path structure
- Cache uses size-based validation (re-downloads only if file size changed)
- Downloaded archives go to `temp/archives/` to avoid conflicts with extracted contents
- Extracted contents go to `temp/<bucket>/<archive_name>/...`
- Metadata table stores S3 paths, making pipelines portable across machines

**Cache structure:**
```
temp/
├── archives/                           # Downloaded S3 archives (separate to avoid conflicts)
│   └── my-bucket/
│       └── source/
│           └── data.zip                # The actual archive file
└── my-bucket/
    └── source/
        ├── file.csv                    # Cached non-archive S3 file
        └── data.zip/                   # Directory for extracted contents
            └── inner/
                └── file.csv            # Extracted archive contents
```

**Cache management:**
```bash
rm -rf temp/              # Clear all cached S3 files
rm -rf temp/my-bucket/    # Clear specific bucket
ls -la temp/              # See what's cached
du -sh temp/              # Check cache size
```

**Using AWS SSO or named profiles:**
```python
import s3fs
fs = s3fs.S3FileSystem(profile="my-sso-profile")
add_files_to_metadata_table(..., filesystem=fs)
```
Or set `AWS_PROFILE=my-profile` environment variable.

### 4. Data Directory Structure

**Source files can be local or S3:**
```
# Local source example
data/                 # Immutable source files
├── census/           # Census DHC ZIP files
├── earthquakes/      # Earthquake CSVs
└── iris.zip          # UCI Iris dataset

# S3 source example
s3://my-bucket/data/  # Immutable source files (never modified)
├── census/
└── earthquakes/
```

**Local cache (auto-managed, mirrors source structure):**
```
temp/                 # S3 file cache
├── archives/         # Downloaded S3 archives (kept separate)
│   └── my-bucket/
│       └── data/
│           └── archive.zip
└── my-bucket/
    └── data/
        ├── file.csv                    # Downloaded S3 files
        └── archive.zip/                # Extracted contents (directory)
            └── inner/
                └── file.csv
```

### 5. Running Marimo Notebooks

Marimo notebooks can be run as Python scripts:
```bash
python example_iris_dataset.py
```

### 6. Marimo Setup Cell Pattern

Use `with app.setup:` for imports. This runs before other cells and makes variables available to all cells:

```python
import marimo

app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import psycopg
    from table_functions import add_files_to_metadata_table, update_table


@app.cell
def _():
    # mo, psycopg, add_files_to_metadata_table, update_table are all available here
    conninfo = "postgresql://user:pass@host/db"
    # ...
```

### 7. Marimo Cells - Variable Naming

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

### 8. SQL in Marimo - Escaping

When using `mo.sql()`, escape `%` as `%%` in LIKE patterns:
```python
mo.sql("""
    SELECT * FROM table
    WHERE path LIKE '%%pattern%%'  -- Note: %% not %
""", engine=conn)
```

### 9. DHC Census Files

DHC files are **pipe-delimited** (`.dhc`) with **no headers**:
- filetype: `"psv"`
- has_header: `False`
- Columns stored as `text`, cast in queries: `SUM("POP100"::bigint)`

## Typical Workflow

### ZIP Extraction + Ingestion

```python
# Step 1: Extract from ZIP and add to metadata
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    source_dir="s3://my-bucket/data/",  # or local path
    filetype="csv",  # or "psv" for pipe-delimited
    compression_type="zip",
    archive_glob="*.csv",
    has_header=False,  # Required for accurate row count validation
    encoding="utf-8",
    resume=False,
    expected_archive_file_count=20,  # Optional: enables archive-level skip on resume
)
```

### Archive-Level Resume (Skipping Entire Archives)

When processing many archives with `resume=True`, the framework normally opens each archive to check which files have been processed. For large S3 archives, this can be slow.

The `expected_archive_file_count` parameter enables **archive-level skipping**:

```python
# First run: processes all archives, tracks completion in archive_metadata table
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    source_dir="s3://my-bucket/data/",
    filetype="csv",
    compression_type="zip",
    archive_glob="*.csv",
    expected_archive_file_count=20,  # Expected files per archive
    resume=False,
)

# Second run: skips completed archives entirely (no download, no opening)
add_files_to_metadata_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    source_dir="s3://my-bucket/data/",
    filetype="csv",
    compression_type="zip",
    archive_glob="*.csv",
    expected_archive_file_count=20,
    resume=True,  # Completed archives are skipped entirely
)
```

**How it works:**
- Creates `{schema}.archive_metadata` table to track archive completion
- After processing, archives with `processed_file_count >= expected_archive_file_count` are marked `Success`
- On `resume=True`, archives with `status='Success'` are filtered out before any downloads
- Archives with `status='Partial'` are still processed (allows incremental extraction)

**archive_metadata table schema:**
| Column | Type | Description |
|--------|------|-------------|
| archive_path | TEXT (PK) | Full path to the archive |
| source_dir | TEXT | Source directory |
| expected_file_count | INTEGER | User-provided expected count |
| processed_file_count | INTEGER | Actual files processed |
| status | TEXT | 'Success' or 'Partial' |
| ingest_datetime | TIMESTAMP | When processed |

**When to use:**
- Processing many large archives from S3
- You know the expected file count per archive
- You want fast resume without re-downloading/opening completed archives

**Without `expected_archive_file_count`:** Resume still works at file-level (opens archives but skips individual files already in metadata).

**Why `has_header` matters:** The metadata table stores a raw line count (`row_count`) for each file. During `update_table()`, this count is compared against the DataFrame row count to catch cases where pandas silently drops or misparses rows. To get an accurate line count, the framework needs to know whether to subtract 1 for a header row.

```python
# Step 2: Ingest files into target table
update_table(
    conninfo="postgresql://user:pass@host/db",
    schema="raw",
    output_table="my_table",
    filetype="csv",
    sql_glob="%.csv",
    column_mapping=column_mapping,
    header_fn=my_header_fn,  # Use pattern from section 1
    source_dir="s3://my-bucket/data/",  # matches source_dir from step 1
    resume=False,
    cleanup=True,  # Optional: delete cached files after successful ingestion
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
| `null_values` | `List[str]` | Custom null representations (e.g., `["NA", "None", "N/A"]`) |
| `resume` | `bool` | Skip already-processed files |
| `retry_failed` | `bool` | Re-process failed files |
| `cleanup` | `bool` | Delete cached files after successful ingestion |
| `ephemeral_cache` | `bool` | Use temporary directory (auto-deleted) instead of persistent `temp/` |

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
5. **Don't** create new data directories - use existing `data/` structure
6. **Don't** add helper functions that wrap simple shell commands - users can just run `rm -rf temp/` or `ls temp/` themselves. Avoid unnecessary abstraction.

## Example Notebooks

Example marimo notebooks at the top level. **One notebook per dataset.**

- `example_census_dhc_2020.py` - 2020 Census DHC data ingestion
- `example_encoding.py` - Working with non-UTF-8 encodings
- `example_iris_dataset.py` - UCI Iris dataset

## Testing

### Pre-push Hook

Tests run automatically on `git push` via a pre-push hook. You don't need to run tests manually before committing - just push and the hook will catch failures.

### Running the Test Suite

The project has comprehensive tests using pytest and testcontainers (requires Docker/Podman):

**Start Podman (if using Podman instead of Docker):**
```bash
podman machine start
```

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_table_functions.py -v

# Run just path utility tests
pytest tests/test_table_functions.py::TestPathUtilities -v
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
python example_iris_dataset.py
```

Check results:
```bash
psql -U tanner -d postgres -c "SELECT status, COUNT(*) FROM raw.metadata GROUP BY status;"
```
