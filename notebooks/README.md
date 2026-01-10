# Notebooks

This directory contains marimo notebooks for data ingestion. **One notebook per dataset.**

## Available Notebooks

### `census_dhc_2020.py` - 2020 Census DHC Data

Ingest 2020 Decennial Census Demographic and Housing Characteristics (DHC) data from ZIP archives.

**Data source:** `data/raw/census/*.zip`
**Output table:** `raw.dhc_2020`

**Usage:**
```bash
# Run as script
python notebooks/census_dhc_2020.py

# Or run interactively
marimo edit notebooks/census_dhc_2020.py
```

**Features:**
- Extracts pipe-delimited `.dhc` files from ZIP archives
- Loads geographic identifiers and 100% population/housing counts
- Includes summary statistics by geography level

---

### `iris_dataset.py` - UCI Iris Dataset

UCI Iris dataset ingestion from ZIP archive.

**Data source:** `data/raw/iris.zip`
**Output table:** `raw.iris`

**Usage:**
```bash
# Run as script
python notebooks/iris_dataset.py

# Or run interactively
marimo edit notebooks/iris_dataset.py
```

**Features:**
- Extracts `.data` files from ZIP
- Demonstrates headerless CSV ingestion
- Includes data preview and SQL queries

---

## Common Patterns

All notebooks follow similar patterns:

1. **Setup** - Connect to PostgreSQL, define paths
2. **Column Mapping** - Define expected columns and types
3. **Extract** - Extract files from ZIP (if applicable)
4. **Ingest** - Load data into PostgreSQL
5. **Explore** - Query and analyze the data

## Data Locations

- **Input:** `data/raw/` - Place ZIP files or CSVs here
- **Output:** `data/landing/` - Extracted files stored here
- **Database:** PostgreSQL `raw` schema

## Quick Testing

```bash
# Reset database
psql -U tanner -d postgres -c "DROP SCHEMA IF EXISTS raw CASCADE; CREATE SCHEMA raw;"

# Test with iris dataset (small, fast)
python notebooks/iris_dataset.py

# Check results
psql -U tanner -d postgres -c "SELECT status, COUNT(*) FROM raw.metadata GROUP BY status;"
```
