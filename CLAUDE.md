# Claude Working Notes - Teeny Data Framework

## Project Purpose

Lightweight data ingestion framework for loading CSV/TSV/PSV files (including from ZIP archives) into PostgreSQL with metadata tracking and row count validation.

## Key Patterns to Follow

### 1. Header Function Pattern (CRITICAL)

For headerless files, **ALWAYS derive the header from column_mapping**. Never duplicate column names.

```python
# Define column mapping
column_mapping = {
    "col1": ([], "string"),
    "col2": ([], "float64"),
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

### 2. Data Directory Structure

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

### 3. Running Marimo Notebooks

Marimo notebooks can be run as Python scripts:
```bash
python census/dhc_2020.py
```

### 4. Marimo Cells - Variable Naming

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

### 5. SQL in Marimo - Escaping

When using `mo.sql()`, escape `%` as `%%` in LIKE patterns:
```python
mo.sql("""
    SELECT * FROM table
    WHERE path LIKE '%%pattern%%'  -- Note: %% not %
""", engine=conn)
```

### 6. DHC Census Files

DHC files are **pipe-delimited** (`.dhc`) with **no headers**:
- filetype: `"psv"`
- has_header: `False`
- Columns stored as `text`, cast in queries: `SUM("POP100"::bigint)`

## Typical Workflow

### ZIP Extraction + Ingestion

```python
# Step 1: Extract from ZIP
add_files_to_metadata_table(
    conn=conn,
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
    conn=conn,
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

## File Type Reference

| Format | filetype | has_header | header_fn needed? |
|--------|----------|------------|-------------------|
| CSV with headers | `"csv"` | `True` | No |
| CSV without headers | `"csv"` | `False` | Yes - use pattern |
| Pipe-delimited (.dhc) | `"psv"` | `False` | Yes - use pattern |
| Tab-delimited | `"tsv"` | `True/False` | If False |

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

## Quick Testing

Reset database and test:
```bash
psql -U tanner -d postgres -c "DROP SCHEMA IF EXISTS raw CASCADE; CREATE SCHEMA raw;"
python notebooks/iris_dataset.py
```

Check results:
```bash
psql -U tanner -d postgres -c "SELECT status, COUNT(*) FROM raw.metadata GROUP BY status;"
```
