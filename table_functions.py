"""
Data ingestion framework using pandas and PostgreSQL
Pure psycopg implementation for efficient bulk loading
"""

import fnmatch
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import psycopg


# Module-level variable to override temp directory (used for ephemeral cache)
_temp_dir_override: Optional[Path] = None


# S3 Helper Functions


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path"""
    path_str = str(path)
    return path_str.startswith("s3://")


def normalize_path(path: str) -> str:
    """Normalize a path to a consistent string format (no trailing slash except for root)"""
    path_str = str(path)
    # Don't modify S3 paths with Path(), just normalize slashes
    if is_s3_path(path_str):
        return path_str.rstrip("/")
    # For local paths, use Path to normalize
    return Path(path_str).as_posix().rstrip("/")


def path_join(*parts: str) -> str:
    """Join path parts, handling both S3 and local paths"""
    if not parts:
        return ""

    # Check if first part is S3
    first_part = str(parts[0])
    is_s3 = first_part.startswith("s3://")
    is_absolute = first_part.startswith("/") and not is_s3

    # Strip slashes from parts for joining
    cleaned_parts = [str(p).strip("/") for p in parts if p and str(p).strip("/")]
    if not cleaned_parts:
        return ""

    if is_s3:
        # Reconstruct S3 path
        return "s3://" + "/".join(p.replace("s3://", "") for p in cleaned_parts)

    # Local path
    result = "/".join(cleaned_parts)
    if is_absolute:
        result = "/" + result
    return result


def path_basename(path: str) -> str:
    """Get the filename from a path (S3 or local)"""
    path_str = str(path).rstrip("/")
    return path_str.split("/")[-1]


def path_parent(path: str) -> str:
    """Get the parent directory of a path (S3 or local)"""
    path_str = str(path).rstrip("/")
    parts = path_str.split("/")
    if is_s3_path(path_str):
        # Keep s3:// prefix
        if len(parts) > 3:  # s3://bucket/path -> s3://bucket
            return "/".join(parts[:-1])
        return path_str
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def get_s3_filesystem(filesystem: Optional[Any] = None) -> Any:
    """
    Get an s3fs filesystem instance.

    Args:
        filesystem: Optional s3fs.S3FileSystem. If not provided, one will be created
                   using default credentials (environment, ~/.aws/credentials, IAM role).

    Returns:
        s3fs.S3FileSystem instance
    """
    if filesystem is not None:
        return filesystem

    try:
        import s3fs
    except ImportError:
        raise ImportError(
            "s3fs is required for S3 support. Install with: pip install s3fs"
        )

    return s3fs.S3FileSystem()


def get_persistent_temp_dir() -> Path:
    """
    Get temp directory for caching files.

    If _temp_dir_override is set (via ephemeral_cache context), uses that.
    Otherwise creates/uses temp/ directory in current working directory.

    Returns:
        Path to temp directory
    """
    global _temp_dir_override
    if _temp_dir_override is not None:
        _temp_dir_override.mkdir(exist_ok=True, parents=True)
        return _temp_dir_override

    temp_dir = Path.cwd() / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


def set_temp_dir_override(path: Optional[Path]) -> None:
    """Set the temp directory override (used internally by ephemeral_cache)"""
    global _temp_dir_override
    _temp_dir_override = path


def get_cache_path_from_s3(s3_path: str) -> Path:
    """
    Convert S3 path to local cache path that mirrors the S3 structure.

    Args:
        s3_path: S3 path (e.g., "s3://my-bucket/raw-data/file.csv")

    Returns:
        Path to cached file location (e.g., temp/my-bucket/raw-data/file.csv)

    Examples:
        s3://my-bucket/raw-data/census/file.csv
        -> temp/my-bucket/raw-data/census/file.csv

        s3://bucket/subfolder/data.csv
        -> temp/bucket/subfolder/data.csv
    """
    # Remove s3:// prefix
    path_without_prefix = s3_path.replace("s3://", "")
    # Build cache path that mirrors S3 structure
    cache_path = get_persistent_temp_dir() / path_without_prefix
    # Ensure parent directories exist
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    return cache_path


def get_archive_cache_path_from_s3(s3_path: str) -> Path:
    """
    Convert S3 archive path to local cache path in the archives subdirectory.

    Archives are stored separately in temp/archives/ to avoid conflicts with
    extracted contents which go in temp/bucket/archive.zip/...

    Args:
        s3_path: S3 path to archive (e.g., "s3://my-bucket/data/archive.zip")

    Returns:
        Path to cached archive (e.g., temp/archives/my-bucket/data/archive.zip)

    Examples:
        s3://my-bucket/data/archive.zip
        -> temp/archives/my-bucket/data/archive.zip
    """
    # Remove s3:// prefix
    path_without_prefix = s3_path.replace("s3://", "")
    # Build cache path in archives subdirectory
    cache_path = get_persistent_temp_dir() / "archives" / path_without_prefix
    # Ensure parent directories exist
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    return cache_path


def get_cache_path_from_source_path(source_path: str) -> Path:
    """
    Convert source_path to local cache path.

    source_path can be:
    - S3 file: s3://bucket/path/file.csv -> temp/bucket/path/file.csv
    - S3 archive with inner path: s3://bucket/archive.zip::inner/file.csv -> temp/bucket/archive.zip/inner/file.csv
    - Local file: /path/to/file.csv -> /path/to/file.csv (returned as-is)
    - Local archive: /path/archive.zip::inner/file.csv -> temp/path/archive.zip/inner/file.csv

    Note: Downloaded archives are stored in temp/archives/ (via get_archive_cache_path_from_s3)
    while extracted contents go in temp/bucket/... This avoids path conflicts.

    Args:
        source_path: Source path with optional ::inner_path for archives

    Returns:
        Path to cached/local file location
    """
    # Check for archive delimiter
    if "::" in source_path:
        archive_path, inner_path = source_path.split("::", 1)

        if is_s3_path(archive_path):
            # S3 archive: extracted contents go in temp/bucket/archive.zip/inner/file.csv
            # (the archive itself is cached separately in temp/archives/)
            path_without_prefix = archive_path.replace("s3://", "")
            cache_path = get_persistent_temp_dir() / path_without_prefix / inner_path
        else:
            # Local archive: temp/path/archive.zip/inner/file.csv
            # Remove leading slash for consistent temp structure
            clean_archive_path = archive_path.lstrip("/")
            cache_path = get_persistent_temp_dir() / clean_archive_path / inner_path

        cache_path.parent.mkdir(exist_ok=True, parents=True)
        return cache_path
    else:
        # Non-archive file
        if is_s3_path(source_path):
            return get_cache_path_from_s3(source_path)
        else:
            # Local file - return as-is
            return Path(source_path)


def download_s3_file_with_cache(
    s3_path: str, filesystem: Any, is_archive: bool = False
) -> Path:
    """
    Download S3 file to persistent cache, reusing cached file if it exists with matching size.
    Files are cached in temp/ directory with structure mirroring S3 paths.

    Args:
        s3_path: S3 path to download (e.g., "s3://bucket/path/file.csv")
        filesystem: s3fs.S3FileSystem instance
        is_archive: If True, cache in temp/archives/ to avoid conflicts with extracted contents

    Returns:
        Path to cached file in temp/ directory

    Cache Strategy:
        - Checks if file exists in cache
        - Compares file sizes (S3 vs cached)
        - Only re-downloads if sizes differ or file missing
        - Cached files persist across runs until manually deleted
    """
    if is_archive:
        cache_path = get_archive_cache_path_from_s3(s3_path)
    else:
        cache_path = get_cache_path_from_s3(s3_path)

    # Get S3 file size
    s3_info = filesystem.info(s3_path)
    s3_size = s3_info["size"]

    # Check if cached file exists and matches size
    if cache_path.exists():
        cached_size = cache_path.stat().st_size
        if cached_size == s3_size:
            print(f"Cache hit: {cache_path.relative_to(Path.cwd())}")
            return cache_path

    # Download from S3 to cache
    print(f"Downloading: {s3_path} -> {cache_path.relative_to(Path.cwd())}")
    filesystem.get(s3_path, str(cache_path))

    return cache_path


def prepare_column_mapping(
    header: List[str], column_mapping: Dict[str, Tuple[List[str], str]]
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Process column mapping to build dictionaries for DataFrame transformation.

    Args:
        header: List of column names from the file
        column_mapping: Dict mapping output names to ([alternative_names], dtype).
                       Use "default" key for unmapped columns.

    Returns:
        rename_dict: Maps original column names to new names
        read_dtypes: Maps original column names to their types (for pd.read_*)
        missing_cols: Columns to add with their types

    Example:
        column_mapping = {
            "block_fips": ([], "string"),              # Keep name as-is
            "pop": (["POP100", "population"], "int"),  # Rename if found
            "default": ([], "float"),                  # Type for unmapped cols
        }
    """
    rename_dict = {}
    read_dtypes = {}
    missing_cols = {}
    processed_cols = set()
    default_type = column_mapping.get("default", (None, None))[1]

    for target_name, (alt_names, dtype) in column_mapping.items():
        if target_name == "default":
            continue

        # Find column in header (use target_name if no alternatives)
        found_name = None
        if not alt_names and target_name in header:
            found_name = target_name
        else:
            found_name = next((name for name in alt_names if name in header), None)

        if found_name:
            processed_cols.add(found_name)
            read_dtypes[found_name] = _dtype_str_to_pandas(dtype)
            if found_name != target_name:
                rename_dict[found_name] = target_name
        else:
            # Column not in file, add as missing
            missing_cols[target_name] = dtype

    # Handle unmapped columns with default type
    if default_type:
        for col in header:
            if col not in processed_cols:
                read_dtypes[col] = _dtype_str_to_pandas(default_type)

    return rename_dict, read_dtypes, missing_cols


def _dtype_str_to_pandas(dtype_str: str) -> str:
    """Convert simplified dtype string to pandas dtype.

    Type mappings:
        int -> Int64 (nullable)
        float -> float64
        boolean -> boolean
        datetime -> datetime64[ns]
        string -> string
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower == "int":
        return "Int64"
    elif dtype_lower == "float":
        return "float64"
    elif dtype_lower == "boolean":
        return "boolean"
    elif dtype_lower == "datetime":
        return "datetime64[ns]"
    elif dtype_lower == "string":
        return "string"
    else:
        # Pass through pandas types directly (Int64, float64, etc.)
        return dtype_str


def _apply_column_transforms(
    df: pd.DataFrame, rename_dict: Dict[str, str], missing_cols: Dict[str, str]
) -> pd.DataFrame:
    """Apply column renames and add missing columns with proper types"""
    if rename_dict:
        df = df.rename(columns=rename_dict)
    for col_name, col_type in missing_cols.items():
        df[col_name] = None
        pandas_type = _dtype_str_to_pandas(col_type)
        if pandas_type != "object":
            df[col_name] = df[col_name].astype(pandas_type)
    return df


def read_xlsx(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    excel_skiprows: int = 0,
) -> pd.DataFrame:
    """
    Read Excel file and return pandas DataFrame with proper column mapping

    Args:
        full_path: Path to Excel file
        column_mapping: Column mapping dictionary
        excel_skiprows: Number of rows to skip at the beginning (for label rows before data)
    """
    header = pd.read_excel(full_path, skiprows=excel_skiprows, nrows=1).columns.tolist()

    # Process column mapping to get rename dict, dtypes, and missing columns
    rename_dict, read_dtypes, missing_cols = prepare_column_mapping(
        header, column_mapping
    )
    df = pd.read_excel(full_path, skiprows=excel_skiprows, dtype=read_dtypes)

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_parquet(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
) -> pd.DataFrame:
    """
    Read parquet file and return pandas DataFrame with column mapping
    """
    df = pd.read_parquet(full_path)

    # Prepare column transformations (parquet already has type info)
    rename_dict, _, missing_cols = prepare_column_mapping(
        header=df.columns.tolist(), column_mapping=column_mapping
    )

    # Keep only mapped columns (drop unmapped unless there's a default type)
    if "default" not in column_mapping:
        keep_cols = [
            col for col in df.columns if col in rename_dict or col in column_mapping
        ]
        df = df[keep_cols]

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_csv(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_values: Optional[List[str]] = None,
    separator: str = ",",
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read CSV file and return pandas DataFrame with proper schema and column mapping.

    Args:
        encoding: File encoding (e.g., "utf-8", "latin-1"). Defaults to "utf-8".
    """
    from io import StringIO

    encoding = encoding or "utf-8"
    with open(full_path, "r", encoding=encoding) as f:
        file_content = f.read()

    # Remove BOM if present
    if file_content.startswith("\ufeff"):
        file_content = file_content[1:]

    if not header:
        import csv

        reader = csv.reader(StringIO(file_content), delimiter=separator)
        header = next(reader)

    # Process column mapping to get rename dict, dtypes, and missing columns
    rename_dict, read_dtypes, missing_cols = prepare_column_mapping(
        header, column_mapping
    )

    # Determine keep_default_na: False only if "" is explicitly in null_values
    keep_default_na = True
    if null_values is not None and "" in null_values:
        keep_default_na = False

    try:
        df = pd.read_csv(
            StringIO(file_content),
            sep=separator,
            header=0 if has_header else None,
            names=header if not has_header else None,
            dtype=read_dtypes,
            na_values=null_values,
            keep_default_na=keep_default_na,
        )
    except (ValueError, TypeError) as e:
        # Try to identify which column caused the error by reading without dtypes
        df_raw = pd.read_csv(
            StringIO(file_content),
            sep=separator,
            header=0 if has_header else None,
            names=header if not has_header else None,
            na_values=null_values,
            keep_default_na=keep_default_na,
        )
        for col, dtype in read_dtypes.items():
            if col in df_raw.columns:
                try:
                    df_raw[col].astype(dtype)
                except (ValueError, TypeError):
                    # Get sample of values that failed conversion
                    sample_values = df_raw[col].unique()[:20]
                    raise ValueError(
                        f"Column '{col}' cannot be converted to {dtype}. "
                        f"Sample values: {list(sample_values)}"
                    ) from e
        raise

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_fixed_width(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[str, int, int]]] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read fixed-width file and return pandas DataFrame using pd.read_fwf

    column_mapping format:
    {'stusab': ('string', 7, 2),
     'sumlev': ('float', 9, 3),
     'geocomp': ('string', 12, 2)}

    {'column_name': (type, starting_position, field_size)}

    Note: starting_position is 1-indexed (first character is position 1)

    Args:
        encoding: File encoding (e.g., "utf-8", "latin-1"). Defaults to "utf-8".
    """
    from io import StringIO

    encoding = encoding or "utf-8"
    with open(full_path, "r", encoding=encoding) as f:
        file_content = f.read()

    # Build colspecs for pandas read_fwf
    # colspecs is a list of (start, end) tuples, 0-indexed
    colspecs = []
    names = []
    dtypes = {}

    for col_name, (col_type, start_pos, field_size) in column_mapping.items():
        # Convert 1-indexed to 0-indexed
        start_idx = start_pos - 1
        end_idx = start_idx + field_size
        colspecs.append((start_idx, end_idx))
        names.append(col_name)
        dtypes[col_name] = _dtype_str_to_pandas(col_type)

    df = pd.read_fwf(
        StringIO(file_content),
        colspecs=colspecs,
        names=names,
        dtype=dtypes,
    )

    return df


def read_using_column_mapping(
    full_path: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Any]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_values: Optional[List[str]] = None,
    excel_skiprows: int = 0,
    encoding: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Router function to read different file types with column mapping.

    Args:
        encoding: File encoding for text files (e.g., "utf-8", "latin-1"). Defaults to "utf-8".
    """
    match filetype:
        case "csv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator=",",
                null_values=null_values,
                encoding=encoding,
            )
        case "tsv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="\t",
                null_values=null_values,
                encoding=encoding,
            )
        case "psv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="|",
                null_values=null_values,
                encoding=encoding,
            )
        case "xlsx":
            return read_xlsx(
                full_path=full_path,
                column_mapping=column_mapping,
                excel_skiprows=excel_skiprows,
            )
        case "parquet":
            return read_parquet(full_path=full_path, column_mapping=column_mapping)
        case "fixed_width":
            return read_fixed_width(
                full_path=full_path,
                column_mapping=column_mapping,
                encoding=encoding,
            )
        case _:
            raise ValueError(
                f"Invalid filetype: {filetype}. "
                f"Must be one of: csv, tsv, psv, xlsx, parquet, fixed_width"
            )


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """Check if a table exists in the database"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return cur.fetchone()[0]


def _infer_postgres_type(dtype: Any) -> str:
    """Infer PostgreSQL type from pandas dtype"""
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "BIGINT"
    elif "float" in dtype_str or "decimal" in dtype_str:
        return "DOUBLE PRECISION"
    elif "bool" in dtype_str:
        return "BOOLEAN"
    elif "datetime" in dtype_str or "date" in dtype_str or "time" in dtype_str:
        return "TIMESTAMP"
    else:
        return "TEXT"


def _infer_column_mapping_type(dtype: Any) -> str:
    """Infer column_mapping type string from pandas dtype"""
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "int"
    elif "float" in dtype_str or "decimal" in dtype_str:
        return "float"
    elif "bool" in dtype_str:
        return "boolean"
    elif "datetime" in dtype_str or "date" in dtype_str or "time" in dtype_str:
        return "datetime"
    else:
        return "string"


def get_table_schema(
    conn: psycopg.Connection, schema: str, table: str
) -> Dict[str, str]:
    """
    Get existing table schema from PostgreSQL

    Returns:
        Dict mapping column names to PostgreSQL types
    """
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """

    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        rows = cur.fetchall()

    return {row[0]: row[1] for row in rows}


def validate_schema_match(
    conn: psycopg.Connection, df: pd.DataFrame, schema: str, table: str
) -> None:
    """
    Validate that DataFrame schema matches existing table schema

    Raises:
        ValueError: If schemas don't match
    """
    if not table_exists(conn, schema, table):
        return  # No validation needed for new tables

    existing_schema = get_table_schema(conn, schema, table)
    df_columns = set(df.columns)
    table_columns = set(existing_schema.keys())

    # Check for missing columns in table
    missing_in_table = df_columns - table_columns
    if missing_in_table:
        raise ValueError(
            f"Schema mismatch for {schema}.{table}: "
            f"DataFrame has columns not in table: {sorted(missing_in_table)}\n"
            f"Table columns: {sorted(table_columns)}\n"
            f"DataFrame columns: {sorted(df_columns)}"
        )

    # Check for missing columns in DataFrame
    missing_in_df = table_columns - df_columns
    if missing_in_df:
        raise ValueError(
            f"Schema mismatch for {schema}.{table}: "
            f"Table has columns not in DataFrame: {sorted(missing_in_df)}\n"
            f"Table columns: {sorted(table_columns)}\n"
            f"DataFrame columns: {sorted(df_columns)}"
        )

    # Validate column types match
    # Map pandas types to PostgreSQL equivalents for comparison
    type_mismatches = []
    for col in df.columns:
        expected_pg_type = _infer_postgres_type(df[col].dtype)
        actual_pg_type = existing_schema[col]

        # Normalize type names for comparison
        # PostgreSQL returns different names than we use in CREATE TABLE
        type_map = {
            "BIGINT": ["bigint"],
            "DOUBLE PRECISION": ["double precision", "numeric"],
            "TEXT": ["text", "character varying"],
            "BOOLEAN": ["boolean"],
            "TIMESTAMP": ["timestamp without time zone", "timestamp with time zone"],
        }

        # Check if types are compatible
        expected_variants = type_map.get(expected_pg_type, [expected_pg_type.lower()])
        if actual_pg_type.lower() not in expected_variants:
            type_mismatches.append(
                f"  {col}: expected {expected_pg_type}, got {actual_pg_type.upper()}"
            )

    if type_mismatches:
        raise ValueError(
            f"Schema mismatch for {schema}.{table}:\n"
            f"Column type mismatches:\n" + "\n".join(type_mismatches)
        )


def create_table_from_dataframe(
    conn: psycopg.Connection, df: pd.DataFrame, schema: str, table: str
) -> None:
    """Create table from DataFrame schema if it doesn't exist"""
    # Build column definitions
    columns = []
    for col in df.columns:
        pg_type = _infer_postgres_type(df[col].dtype)
        columns.append(f'"{col}" {pg_type}')

    columns_sql = ",\n    ".join(columns)
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            {columns_sql}
        )
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)


def copy_dataframe_to_table(
    conn: psycopg.Connection, df: pd.DataFrame, schema: str, table: str
) -> None:
    """
    Bulk load DataFrame to PostgreSQL using COPY with write_row()
    This is the recommended psycopg3 approach - cleaner and more memory efficient

    Note: The entire DataFrame must already be loaded in memory. This function
    does not chunk - it processes the full DataFrame that was loaded by the caller.

    Raises:
        ValueError: If table exists but schema doesn't match DataFrame
    """
    # Validate schema matches if table exists
    validate_schema_match(conn, df, schema, table)

    # Ensure table exists (creates if not exists)
    create_table_from_dataframe(conn, df, schema, table)

    # Replace pandas NA/NaT with None for psycopg3 compatibility
    # This handles pd.NA, pd.NaT, and np.nan
    df = df.fillna(value=np.nan).replace({np.nan: None})

    # Quote column names to preserve case sensitivity
    columns = ", ".join(f'"{col}"' for col in df.columns)
    copy_sql = f"COPY {schema}.{table} ({columns}) FROM STDIN"

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            # Use itertuples for efficient row iteration
            # index=False excludes the index, name=None returns plain tuples
            # NOTE: This iterates over the full DataFrame already in memory (not chunked)
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)


def row_count_check(
    conn: psycopg.Connection,
    schema: str,
    df: pd.DataFrame,
    source_path: str,
    unpivot_row_multiplier: Optional[int] = None,
) -> None:
    """
    Sanity check on flat file ingest comparing metadata row count to DataFrame row count
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT row_count FROM {schema}.metadata WHERE source_path = %s",
            (source_path,),
        )
        result = cur.fetchone()

    metadata_row_count = result[0] if result else None
    output_df_row_count = len(df)

    if unpivot_row_multiplier:
        metadata_row_count *= unpivot_row_multiplier

    if not metadata_row_count:
        pass  # No metadata row count to compare against
    elif metadata_row_count == output_df_row_count:
        pass  # Check passed
    else:
        raise ValueError(
            f"Check failed {source_path} since metadata table row count {metadata_row_count} is not equal to output table row count {output_df_row_count}"
        )


def update_metadata(
    conn: psycopg.Connection,
    source_path: str,
    schema: str,
    error_message: Optional[str] = None,
    unpivot_row_multiplier: Optional[int] = None,
    ingest_runtime: Optional[int] = None,
    output_table: Optional[str] = None,
) -> None:
    """
    Update metadata table with ingestion status and runtime
    """
    from datetime import datetime

    metadata_ingest_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "Failure" if error_message else "Success"

    if error_message:
        # Truncate error message to prevent massive output
        if len(error_message) > 500:
            error_message = error_message[:500] + "... [truncated]"
        # if these characters are in the sql string itll break
        error_message = error_message.replace("'", "").replace("`", "")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {schema}.metadata
            SET
                ingest_datetime = %s,
                status = %s,
                error_message = %s,
                unpivot_row_multiplier = %s,
                ingest_runtime = %s,
                output_table = %s
            WHERE source_path = %s
            """,
            (
                metadata_ingest_datetime,
                status,
                error_message,
                unpivot_row_multiplier,
                ingest_runtime,
                output_table,
                source_path,
            ),
        )


def update_table(
    conninfo: Optional[str] = None,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    schema: Optional[str] = None,
    metadata_schema: Optional[str] = None,
    output_table: Optional[str] = None,
    output_table_naming_fn: Optional[Callable[[Path], str]] = None,
    additional_cols_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    file_list_filter_fn: Optional[Callable[[List[Path]], List[Path]]] = None,
    custom_read_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    source_dir: Optional[str] = None,
    sql_glob: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    column_mapping_fn: Optional[
        Callable[[Path], Dict[str, Tuple[List[str], str]]]
    ] = None,
    pivot_mapping: Optional[Dict[str, Any]] = None,
    header_fn: Optional[Callable[[Path], List[str]]] = None,
    null_values: Optional[List[str]] = None,
    excel_skiprows: int = 0,
    cleanup: bool = False,
    ephemeral_cache: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main ingestion function that reads files and writes to PostgreSQL

    Args:
        conninfo: PostgreSQL connection string (e.g. "postgresql://user:pass@host/db")
        source_dir: Source directory to filter files (replaces landing_dir)
        cleanup: If True, delete cached files after successful ingest
        ephemeral_cache: If True, use a temporary directory that is deleted after processing.
                        If False (default), use persistent temp/ directory.
        encoding: File encoding (e.g. "utf-8", "latin-1", "cp1252"). Defaults to "utf-8".
        ... (other args unchanged)
    """
    import tempfile

    # Handle ephemeral cache mode
    temp_dir_context = None
    if ephemeral_cache:
        temp_dir_context = tempfile.TemporaryDirectory()
        set_temp_dir_override(Path(temp_dir_context.name))

    try:
        return _update_table_impl(
            conninfo=conninfo,
            resume=resume,
            retry_failed=retry_failed,
            sample=sample,
            schema=schema,
            metadata_schema=metadata_schema,
            output_table=output_table,
            output_table_naming_fn=output_table_naming_fn,
            additional_cols_fn=additional_cols_fn,
            file_list_filter_fn=file_list_filter_fn,
            custom_read_fn=custom_read_fn,
            transform_fn=transform_fn,
            source_dir=source_dir,
            sql_glob=sql_glob,
            filetype=filetype,
            column_mapping=column_mapping,
            column_mapping_fn=column_mapping_fn,
            pivot_mapping=pivot_mapping,
            encoding=encoding,
            header_fn=header_fn,
            null_values=null_values,
            excel_skiprows=excel_skiprows,
            cleanup=cleanup,
        )
    finally:
        if temp_dir_context:
            set_temp_dir_override(None)
            temp_dir_context.cleanup()


def _update_table_impl(
    conninfo: Optional[str] = None,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    schema: Optional[str] = None,
    metadata_schema: Optional[str] = None,
    output_table: Optional[str] = None,
    output_table_naming_fn: Optional[Callable[[Path], str]] = None,
    additional_cols_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    file_list_filter_fn: Optional[Callable[[List[Path]], List[Path]]] = None,
    custom_read_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    source_dir: Optional[str] = None,
    sql_glob: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    column_mapping_fn: Optional[
        Callable[[Path], Dict[str, Tuple[List[str], str]]]
    ] = None,
    pivot_mapping: Optional[Dict[str, Any]] = None,
    header_fn: Optional[Callable[[Path], List[str]]] = None,
    null_values: Optional[List[str]] = None,
    excel_skiprows: int = 0,
    cleanup: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Internal implementation of update_table"""
    import time

    # Validate mutually exclusive parameters
    if column_mapping is not None and column_mapping_fn is not None:
        raise ValueError("Cannot specify both column_mapping and column_mapping_fn")
    if output_table is not None and output_table_naming_fn is not None:
        raise ValueError("Cannot specify both output_table and output_table_naming_fn")

    required_params = [source_dir, schema, conninfo, filetype]
    if not custom_read_fn and not column_mapping_fn:
        required_params.append(column_mapping)

    if any(param is None for param in required_params):
        raise ValueError(
            "Required params: source_dir, filetype, column_mapping, schema, conninfo. Column mapping not required if using custom_read_fn or if using column_mapping_fn"
        )

    # Normalize source_dir to string with trailing slash for LIKE query
    source_dir = normalize_path(source_dir) + "/"

    metadata_schema = metadata_schema or schema

    # Query metadata for files to process
    sql = f"""
        SELECT source_path
        FROM {metadata_schema}.metadata
        WHERE
            source_dir LIKE %s AND
            metadata_ingest_status = 'Success'
    """

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source_dir,))
            rows = cur.fetchall()
    file_list = sorted([row[0] for row in rows])

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    if resume:
        sql = f"""
            SELECT source_path
            FROM {metadata_schema}.metadata
            WHERE
                source_dir LIKE %s AND
                ingest_datetime IS NOT NULL
                {"AND status = 'Success'" if not retry_failed else ""}
        """

        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_dir,))
                rows = cur.fetchall()
        processed_set = set(row[0] for row in rows)
        file_list = [f for f in file_list if f not in processed_set]

    total_files_to_be_processed = sample if sample else len(file_list)

    if total_files_to_be_processed == 0:
        print("No files to ingest")
    else:
        print(
            f"Output {'table' if output_table else 'schema'}: {output_table or schema} | Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
        )

    for i, source_path in enumerate(file_list):
        start_time = time.time()

        if sample and i == sample:
            break

        # Derive local cache path from source_path
        cache_path = get_cache_path_from_source_path(source_path)

        header = None
        has_header = True
        if header_fn:
            header = header_fn(Path(cache_path))
            has_header = False

        unpivot_row_multiplier = None
        table_name = (
            output_table_naming_fn(Path(cache_path))
            if output_table_naming_fn
            else output_table
        )
        try:
            if custom_read_fn:
                df = custom_read_fn(full_path=str(cache_path))
            else:
                column_mapping_use = (
                    column_mapping_fn(Path(cache_path))
                    if column_mapping_fn
                    else column_mapping
                )

                df = read_using_column_mapping(
                    full_path=str(cache_path),
                    filetype=filetype,
                    column_mapping=column_mapping_use,
                    header=header,
                    has_header=has_header,
                    null_values=null_values,
                    excel_skiprows=excel_skiprows,
                    encoding=encoding,
                )

            if pivot_mapping:
                id_vars = pivot_mapping["id_vars"]
                value_vars = [col for col in df.columns if col not in id_vars]
                unpivot_row_multiplier = len(value_vars)

                df = df.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name=pivot_mapping["variable_column_name"],
                    value_name=pivot_mapping["value_column_name"],
                )

            # Fresh connection for row count check
            with psycopg.connect(conninfo) as conn:
                row_count_check(
                    conn=conn,
                    schema=metadata_schema,
                    df=df,
                    source_path=source_path,
                    unpivot_row_multiplier=unpivot_row_multiplier,
                )

            if transform_fn:
                df = transform_fn(df=df)

            if additional_cols_fn:
                additional_cols_dict = additional_cols_fn(Path(cache_path))
                for col_name, col_value in additional_cols_dict.items():
                    df[col_name] = col_value

            df["source_path"] = source_path

            # Fresh connection for write operations
            with psycopg.connect(conninfo) as conn:
                # Check if table exists and delete existing records for this file
                if table_exists(conn, schema, table_name):
                    with conn.cursor() as cur:
                        cur.execute(
                            f"DELETE FROM {schema}.{table_name} WHERE source_path = %s",
                            (source_path,),
                        )

                # Bulk load using COPY
                copy_dataframe_to_table(conn, df, schema, table_name)
                conn.commit()

            ingest_runtime = int(time.time() - start_time)

            # Fresh connection for metadata update
            with psycopg.connect(conninfo) as conn:
                update_metadata(
                    conn=conn,
                    source_path=source_path,
                    schema=metadata_schema,
                    unpivot_row_multiplier=unpivot_row_multiplier,
                    ingest_runtime=ingest_runtime,
                    output_table=f"{schema}.{table_name}",
                )
                conn.commit()

            # Cleanup cached file if requested
            if cleanup and is_s3_path(source_path.split("::")[0]):
                try:
                    cache_path.unlink()
                except Exception:
                    pass  # Ignore cleanup failures

            print(
                f"{i + 1}/{total_files_to_be_processed} Ingested {source_path} -> {schema}.{table_name}"
            )

        except Exception as e:
            error_str = str(e)

            # Schema mismatches should fail hard - user needs to fix the table or column_mapping
            if "Schema mismatch" in error_str:
                raise

            # Fresh connection for error metadata update
            with psycopg.connect(conninfo) as conn:
                update_metadata(
                    conn=conn,
                    source_path=source_path,
                    schema=metadata_schema,
                    error_message=error_str,
                    unpivot_row_multiplier=unpivot_row_multiplier,
                    output_table=f"{schema}.{table_name}" if table_name else None,
                )
                conn.commit()

            # Truncate error for printing
            if len(error_str) > 200:
                error_str = error_str[:200] + "... [truncated]"
            print(f"Failed on {source_path} with {error_str}", file=sys.stderr)

    # Return metadata results
    # Use '%' as default glob to match all files in source_dir
    glob_pattern = sql_glob if sql_glob else "%"
    sql = f"""
        SELECT *
        FROM {metadata_schema}.metadata
        WHERE source_dir LIKE %s AND source_path LIKE %s
        ORDER BY ingest_datetime DESC
    """

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source_dir, glob_pattern))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)


# START OF METADATA FUNCTIONS #


def get_csv_header_and_row_count(
    file: Optional[str] = None,
    separator: str = ",",
    has_header: bool = True,
    encoding: Optional[str] = None,
) -> Tuple[List[str], int]:
    """
    Get header and row count from CSV file.
    Counts only non-blank lines to match pandas' skip_blank_lines=True behavior.
    File path should be a string.

    Args:
        encoding: File encoding (e.g., "utf-8", "latin-1"). Defaults to "utf-8".
    """
    import subprocess
    import csv

    file_str = str(file)
    encoding = encoding or "utf-8"

    with open(file_str, "r", encoding=encoding) as f:
        first_line = f.readline()

    # Remove BOM if present
    if first_line.startswith("\ufeff"):
        first_line = first_line[1:]

    reader = csv.reader([first_line], delimiter=separator)
    header = next(reader)

    # Count non-blank lines to match pandas behavior (skip_blank_lines=True)
    # Using grep -c to count non-empty lines
    subtract_header_row = 1 if has_header else 0
    row_count = (
        int(
            subprocess.check_output(["grep", "-c", "^[^[:space:]]", file_str]).split()[
                0
            ]
        )
        - subtract_header_row
    )

    return header, row_count


def extract_and_add_zip_files(
    file_list: Optional[List[str]] = None,
    source_path_list: Optional[List[str]] = None,
    source_dir: Optional[str] = None,
    has_header: bool = True,
    filetype: Optional[str] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    archive_glob: Optional[str] = None,
    filesystem: Optional[Any] = None,
    **kwargs,  # Accept but ignore extra kwargs for compatibility
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract files from ZIP archives and add to metadata.

    source_path format for archives: "archive_path::inner_path"
    e.g., "s3://bucket/data.zip::folder/file.csv"

    Files are extracted to temp/ cache, mirroring the source structure.

    Returns:
        Tuple of (rows, archive_stats) where:
        - rows: List of metadata dictionaries for each processed file
        - archive_stats: Dict mapping archive_path to count of files processed from that archive
    """
    try:
        import zipfile_deflate64 as zipfile
    except Exception:
        import zipfile

    import shutil

    # Normalize source_dir
    source_dir = normalize_path(source_dir) if source_dir else None

    rows = []
    archive_stats: Dict[str, int] = {}  # archive_path -> files processed count
    num_processed = 0

    # Get filesystem if any S3 source paths involved
    fs = None
    if any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    # Convert source_path_list to set for fast lookup
    source_path_set = set(source_path_list) if source_path_list else set()

    for compressed in file_list:
        if sample and num_processed == sample:
            break

        archive_path = str(compressed)
        archive_file_count = 0  # Track files processed from this archive

        # Download archive from S3 if needed (uses persistent cache in temp/archives/)
        if is_s3_path(archive_path):
            cached_zip = download_s3_file_with_cache(archive_path, fs, is_archive=True)
            zip_path = str(cached_zip)
        else:
            zip_path = archive_path

        with zipfile.ZipFile(zip_path) as zip_ref:
            namelist = [
                f for f in zip_ref.namelist() if fnmatch.fnmatch(f, archive_glob)
            ]

            if not namelist:
                print(
                    f"No files matching '{archive_glob}' in {path_basename(archive_path)}, trying next archive..."
                )
                continue

            for inner_path in namelist:
                if sample and num_processed == sample:
                    break

                # Build source_path: archive_path::inner_path
                source_path = f"{archive_path}::{inner_path}"

                num_processed += 1
                file_num = (
                    f"{num_processed}/{sample}" if sample else f"{num_processed} |"
                )

                # Check if already processed
                if resume and source_path in source_path_set:
                    print(
                        f"{file_num} Skipped (in metadata): {path_basename(inner_path)}"
                    )
                    continue

                # Get cache path for extracted file
                cache_path = get_cache_path_from_source_path(source_path)

                try:
                    # Check if already extracted to cache
                    if cache_path.exists():
                        print(
                            f"{file_num} Cache hit: {cache_path.relative_to(Path.cwd())}"
                        )
                    else:
                        # Extract to temp directory, then move to cache location
                        # Using extract() + move instead of read() to avoid loading entire file into RAM
                        import tempfile

                        with tempfile.TemporaryDirectory() as temp_extract_dir:
                            zip_ref.extract(inner_path, temp_extract_dir)
                            extracted_file = Path(temp_extract_dir) / inner_path

                            shutil.move(str(extracted_file), str(cache_path))

                        print(f"{file_num} Extracted: {source_path}")

                    row = get_file_metadata_row(
                        source_path=source_path,
                        source_dir=source_dir,
                        filetype=filetype,
                        has_header=has_header,
                    )
                    archive_file_count += 1

                except Exception as e:
                    print(f"Failed on {inner_path} with {e}", file=sys.stderr)

                    row = get_file_metadata_row(
                        source_path=source_path,
                        source_dir=source_dir,
                        filetype=filetype,
                        has_header=has_header,
                        error_message=str(e),
                    )

                    # Cleanup failed files
                    if cache_path.exists():
                        cache_path.unlink()

                    if cache_path.parent.exists() and not any(
                        cache_path.parent.iterdir()
                    ):
                        cache_path.parent.rmdir()

                rows.append(row)

        # Record stats for this archive
        if archive_file_count > 0:
            archive_stats[archive_path] = archive_file_count

    return rows, archive_stats


def add_files(
    source_dir: Optional[str] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    file_list: Optional[List[str]] = None,
    filetype: Optional[str] = None,
    has_header: bool = True,
    source_path_list: Optional[List[str]] = None,
    filesystem: Optional[Any] = None,
    **kwargs,  # Accept but ignore extra kwargs for compatibility
) -> List[Dict[str, Any]]:
    """
    Process non-archive files and add to metadata.

    For S3 files: downloads to temp/ cache (mirroring S3 structure)
    For local files: reads directly from source (no copy needed)

    source_path is the file's identity (primary key in metadata).
    """
    # Normalize source_dir
    source_dir = normalize_path(source_dir) if source_dir else None

    # Build set of already processed source paths for quick lookup
    source_path_set = set(source_path_list) if source_path_list else set()

    # Filter out already processed files first
    if source_path_set:
        file_list = [f for f in file_list if str(f) not in source_path_set]

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
    )

    # Get filesystem if any S3 source paths involved
    fs = None
    if any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    rows = []
    num_processed = 0
    for i, file in enumerate(file_list):
        source_path = str(file)  # This IS the source_path (primary key)

        if sample and num_processed == sample:
            break

        num_processed += 1

        try:
            # For S3 files, download to cache; for local files, they're already in place
            if is_s3_path(source_path):
                cache_path = get_cache_path_from_source_path(source_path)
                fs.get(source_path, str(cache_path))
                print(f"Downloaded {source_path} to {cache_path}")
            # Local files don't need copying - get_cache_path_from_source_path returns the path as-is

            row = get_file_metadata_row(
                source_path=source_path,
                source_dir=source_dir,
                filetype=filetype,
                has_header=has_header,
            )

            print(f"{num_processed}/{total_files_to_be_processed}")

        except Exception as e:
            print(f"Failed on {source_path} with {e}", file=sys.stderr)

            row = get_file_metadata_row(
                source_path=source_path,
                source_dir=source_dir,
                filetype=filetype,
                has_header=has_header,
                error_message=str(e),
            )

        rows.append(row)

    return rows


def get_file_metadata_row(
    source_path: Optional[str] = None,
    source_dir: Optional[str] = None,
    filetype: Optional[str] = None,
    has_header: Optional[bool] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate metadata row for a file.

    Args:
        source_path: Full source identity (e.g., "s3://bucket/file.csv" or "s3://bucket/archive.zip::inner/file.csv")
        source_dir: Source directory for filtering (with trailing slash)
        filetype: File type (csv, tsv, psv, etc.)
        has_header: Whether file has header row
        error_message: Error message if extraction failed
    """
    import hashlib
    import time
    from datetime import datetime

    # Normalize source_dir with trailing slash for consistent LIKE queries
    source_dir = normalize_path(source_dir) + "/" if source_dir else None

    row = {
        "source_path": source_path,
        "source_dir": source_dir,
        "filesize": None,
        "header": None,
        "row_count": None,
        "file_hash": None,
        "metadata_ingest_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_ingest_status": "Failure",
        "error_message": error_message,
    }

    if error_message:
        return row

    start_time = time.time()

    # Derive local cache path from source_path
    cache_path = get_cache_path_from_source_path(source_path)

    try:
        header, row_count = None, None
        match filetype:
            case "csv":
                header, row_count = get_csv_header_and_row_count(
                    file=str(cache_path), has_header=has_header
                )
            case "tsv":
                header, row_count = get_csv_header_and_row_count(
                    file=str(cache_path),
                    has_header=has_header,
                    separator="\t",
                )
            case "psv":
                header, row_count = get_csv_header_and_row_count(
                    file=str(cache_path),
                    has_header=has_header,
                    separator="|",
                )
            case "fixed_width":
                header, row_count = get_csv_header_and_row_count(
                    file=str(cache_path), has_header=False
                )
            case "xlsx":
                df = pd.read_excel(cache_path)
                header, row_count = list(df.columns), len(df)
            case "parquet":
                df = pd.read_parquet(cache_path)
                header, row_count = list(df.columns), len(df)
            case "xml":
                # there isnt really a standard way of getting these values
                header, row_count = None, None
            case _:
                raise Exception(f"Unsupported filetype: {filetype}")

        row["header"], row["row_count"] = header, row_count
        row["file_hash"] = hashlib.md5(open(cache_path, "rb").read()).hexdigest()
        row["filesize"] = cache_path.stat().st_size
        row["metadata_ingest_status"] = "Success"

        filename = path_basename(source_path) if source_path else "unknown"
        print(
            f"Row count: {row_count} Filename: {filename} | Runtime: {time.time() - start_time:.2f}"
        )

    except Exception as e:
        row["metadata_ingest_status"] = "Failure"
        row["error_message"] = str(e)

    return row


# Valid parameters for add_files_to_metadata_table
_ADD_FILES_VALID_PARAMS = {
    "schema",
    "source_dir",
    "filetype",
    "glob",
    "compression_type",
    "archive_glob",
    "has_header",
    "resume",
    "retry_failed",
    "sample",
    "file_list_filter_fn",
    "filesystem",
    "expected_archive_file_count",
}


def add_files_to_metadata_table(
    conninfo: str, ephemeral_cache: bool = False, **kwargs: Any
) -> pd.DataFrame:
    """
    Add files to metadata table, creating it if necessary

    Args:
        conninfo: PostgreSQL connection string (e.g. "postgresql://user:pass@host/db")
        ephemeral_cache: If True, use a temporary directory that is deleted after processing.
                        If False (default), use persistent temp/ directory.
    """
    import tempfile

    # Validate kwargs - fail fast on unknown parameters
    unknown_params = set(kwargs.keys()) - _ADD_FILES_VALID_PARAMS
    if unknown_params:
        raise TypeError(
            f"add_files_to_metadata_table() got unexpected keyword argument(s): {', '.join(sorted(unknown_params))}. "
            f"Valid parameters are: {', '.join(sorted(_ADD_FILES_VALID_PARAMS))}"
        )

    temp_dir_context = None
    if ephemeral_cache:
        temp_dir_context = tempfile.TemporaryDirectory()
        set_temp_dir_override(Path(temp_dir_context.name))

    try:
        return _add_files_to_metadata_table_impl(conninfo, **kwargs)
    finally:
        if temp_dir_context:
            set_temp_dir_override(None)
            temp_dir_context.cleanup()


def _add_files_to_metadata_table_impl(conninfo: str, **kwargs: Any) -> pd.DataFrame:
    """Internal implementation of add_files_to_metadata_table"""
    schema = kwargs.pop("schema", None)
    if not schema:
        raise Exception("You must provide the schema as a param")

    glob = kwargs.pop("glob", None)
    compression_type = kwargs.pop("compression_type", None)
    filetype = kwargs["filetype"]
    archive_glob = kwargs.get("archive_glob", None)

    if not glob:
        glob = f"*.{compression_type}" if compression_type else f"*.{filetype}"

    if compression_type:
        if not archive_glob:
            archive_glob = f"*.{filetype}"
            kwargs["archive_glob"] = archive_glob

    # Normalize source_dir
    source_dir = normalize_path(kwargs["source_dir"])

    # Handle S3 or local paths for file listing
    filesystem = kwargs.get("filesystem", None)
    if is_s3_path(source_dir):
        # S3 path - use s3fs.glob()
        fs = get_s3_filesystem(filesystem)
        s3_glob_pattern = f"{source_dir}/**/{glob}"
        s3_paths = fs.glob(s3_glob_pattern)
        # s3fs returns paths without s3:// prefix, add it back
        file_list = [f"s3://{p}" if not p.startswith("s3://") else p for p in s3_paths]
    else:
        # Local path - use Path.rglob but convert results to strings
        file_list = [f.as_posix() for f in Path(source_dir).rglob(glob)]

    kwargs["source_dir"] = source_dir

    file_list_filter_fn = kwargs.pop("file_list_filter_fn", None)
    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    file_list = sorted(file_list)
    kwargs["file_list"] = file_list

    output_table = f"{schema}.metadata"
    archive_metadata_table = f"{schema}.archive_metadata"

    # Create metadata table if it doesn't exist
    with psycopg.connect(conninfo) as conn:
        if not table_exists(conn, schema, "metadata"):
            create_table_sql = f"""
                CREATE TABLE {output_table} (
                    source_path TEXT PRIMARY KEY,
                    source_dir TEXT,
                    filesize BIGINT,
                    header TEXT[],
                    row_count BIGINT,
                    file_hash TEXT,
                    metadata_ingest_datetime TIMESTAMP,
                    metadata_ingest_status TEXT,
                    ingest_datetime TIMESTAMP,
                    ingest_runtime INTEGER,
                    status TEXT,
                    error_message TEXT,
                    unpivot_row_multiplier INTEGER,
                    output_table TEXT
                )
            """
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()

        # Create archive_metadata table if it doesn't exist (for tracking archive completion)
        if compression_type and not table_exists(conn, schema, "archive_metadata"):
            create_archive_table_sql = f"""
                CREATE TABLE {archive_metadata_table} (
                    archive_path TEXT PRIMARY KEY,
                    source_dir TEXT,
                    expected_file_count INTEGER,
                    processed_file_count INTEGER,
                    status TEXT,
                    ingest_datetime TIMESTAMP
                )
            """
            with conn.cursor() as cur:
                cur.execute(create_archive_table_sql)
            conn.commit()

    resume = kwargs.get("resume", False)
    retry_failed = kwargs.pop("retry_failed", False)
    expected_archive_file_count = kwargs.pop("expected_archive_file_count", None)

    kwargs["source_path_list"] = []
    completed_archives = set()

    if resume:
        # source_dir stored in metadata has trailing slash, so add it for matching
        source_dir_match = source_dir + "/"
        sql = f"""
            SELECT source_path
            FROM {output_table}
            WHERE
                source_dir = %s
                {"AND metadata_ingest_status = 'Success'" if not retry_failed else ""}
        """

        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_dir_match,))
                rows = cur.fetchall()
        kwargs["source_path_list"] = [row[0] for row in rows]

        # For archives: check archive_metadata for completed archives to skip entirely
        if compression_type:
            sql = f"""
                SELECT archive_path
                FROM {archive_metadata_table}
                WHERE
                    source_dir = %s
                    AND status = 'Success'
            """
            with psycopg.connect(conninfo) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (source_dir_match,))
                    archive_rows = cur.fetchall()
            completed_archives = {row[0] for row in archive_rows}

            if completed_archives:
                original_count = len(file_list)
                file_list = [f for f in file_list if f not in completed_archives]
                skipped_count = original_count - len(file_list)
                if skipped_count > 0:
                    print(f"Skipping {skipped_count} completed archive(s)")
                kwargs["file_list"] = file_list

    match compression_type:
        case "zip":
            rows, archive_stats = extract_and_add_zip_files(**kwargs)
        case None:
            rows = add_files(**kwargs)
            archive_stats = {}
        case _:
            raise Exception("Unsupported compression type")

    # Update archive_metadata for processed archives
    if compression_type and archive_stats and expected_archive_file_count is not None:
        from datetime import datetime

        source_dir_match = source_dir + "/"
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                for archive_path, processed_count in archive_stats.items():
                    status = (
                        "Success"
                        if processed_count >= expected_archive_file_count
                        else "Partial"
                    )
                    cur.execute(
                        f"""
                        INSERT INTO {archive_metadata_table}
                        (archive_path, source_dir, expected_file_count, processed_file_count, status, ingest_datetime)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (archive_path)
                        DO UPDATE SET
                            processed_file_count = {archive_metadata_table}.processed_file_count + EXCLUDED.processed_file_count,
                            status = CASE
                                WHEN {archive_metadata_table}.processed_file_count + EXCLUDED.processed_file_count >= EXCLUDED.expected_file_count
                                THEN 'Success'
                                ELSE 'Partial'
                            END,
                            ingest_datetime = EXCLUDED.ingest_datetime
                        """,
                        (
                            archive_path,
                            source_dir_match,
                            expected_archive_file_count,
                            processed_count,
                            status,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
            conn.commit()

    if len(rows) == 0:
        print("Did not add any files to metadata table")
    else:
        rows_sorted = sorted(rows, key=lambda x: x["source_path"] or "")

        # Upsert to metadata table using PostgreSQL's ON CONFLICT
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                for row in rows_sorted:
                    cur.execute(
                        f"""
                        INSERT INTO {output_table}
                        (source_path, source_dir, filesize, header, row_count,
                         file_hash, metadata_ingest_datetime, metadata_ingest_status)
                        VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_path)
                        DO UPDATE SET
                            source_dir = EXCLUDED.source_dir,
                            filesize = EXCLUDED.filesize,
                            header = EXCLUDED.header,
                            row_count = EXCLUDED.row_count,
                            file_hash = EXCLUDED.file_hash,
                            metadata_ingest_datetime = EXCLUDED.metadata_ingest_datetime,
                            metadata_ingest_status = EXCLUDED.metadata_ingest_status
                        """,
                        (
                            row["source_path"],
                            row["source_dir"],
                            row["filesize"],
                            row["header"],
                            row["row_count"],
                            row["file_hash"],
                            row["metadata_ingest_datetime"],
                            row["metadata_ingest_status"],
                        ),
                    )
            conn.commit()

    # Return metadata results for this source_dir
    # Stored source_dir has trailing slash, so match with trailing slash
    source_dir_match = normalize_path(source_dir) + "/"
    sql = f"""
        SELECT *
        FROM {output_table}
        WHERE source_dir = %s
        ORDER BY metadata_ingest_datetime DESC
    """

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source_dir_match,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)


def drop_metadata_by_source(
    conninfo: str,
    source_dir: str,
    schema: str,
) -> None:
    """
    Remove all files from a search directory from metadata

    Args:
        conninfo: PostgreSQL connection string (e.g. "postgresql://user:pass@host/db")
    """
    source_dir = Path(source_dir).as_posix()

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {schema}.metadata WHERE source_dir LIKE %s",
                (source_dir,),
            )
            count_before = cur.fetchone()[0]

        print(f"Rows before drop: {count_before}")

        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {schema}.metadata WHERE source_dir LIKE %s", (source_dir,)
            )
            print(f"Deleted {cur.rowcount} rows")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {schema}.metadata WHERE source_dir LIKE %s",
                (source_dir,),
            )
            count_after = cur.fetchone()[0]

        print(f"Rows after drop: {count_after}")


def drop_partition(
    conninfo: str,
    table: str,
    partition_key: str,
    schema: str,
) -> None:
    """
    Delete records matching partition key from table

    Args:
        conninfo: PostgreSQL connection string (e.g. "postgresql://user:pass@host/db")
    """
    print(
        f"Running: DELETE FROM {schema}.{table} WHERE source_path LIKE '{partition_key}'"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {schema}.{table} WHERE source_path LIKE %s",
                (partition_key,),
            )
            print(f"Deleted {cur.rowcount} rows")
        conn.commit()

    # PostgreSQL doesn't need vacuum the same way Spark does
    # VACUUM reclaims storage, but happens automatically in most cases


def drop_file_from_metadata_and_table(
    conninfo: str,
    table: str,
    source_path: str,
    schema: str,
) -> None:
    """
    Remove a file from both metadata and data table

    Args:
        conninfo: PostgreSQL connection string (e.g. "postgresql://user:pass@host/db")
        source_path: The source_path identifier (e.g. s3://bucket/file.csv or s3://bucket/archive.zip::inner/file.csv)
    """
    # Delete from metadata
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {schema}.metadata WHERE source_path = %s", (source_path,)
            )
            print(f"Deleted {cur.rowcount} rows from metadata")
        conn.commit()

    # Delete from data table
    drop_partition(
        conninfo=conninfo, table=table, partition_key=source_path, schema=schema
    )


# CLI SCHEMA INFERENCE FUNCTIONS


def decode_file_content(
    file_path: str, encoding: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decode file content with specified encoding.

    Args:
        file_path: Path to the file
        encoding: File encoding (e.g., "utf-8", "latin-1"). Defaults to "utf-8".

    Returns:
        Dictionary with:
        {
            "content": "decoded string content",
            "encoding": "utf-8" or specified encoding,
        }
    """
    encoding = encoding or "utf-8"

    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()

    return {
        "content": content,
        "encoding": encoding,
    }


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case using inflection library

    Examples:
        "FirstName" -> "first_name"
        "User ID" -> "user_id"
        "price-per-unit" -> "price_per_unit"
        "totalAmount" -> "total_amount"
        "café" -> "cafe"
    """
    import inflection

    # transliterate converts accented chars to ASCII (café -> cafe)
    # underscore handles camelCase (firstName -> first_name)
    # parameterize handles spaces/special chars and normalizes
    return inflection.parameterize(
        inflection.underscore(inflection.transliterate(name)), separator="_"
    )


def infer_schema_from_file(
    file_path: str,
    filetype: Optional[str] = None,
    has_header: bool = True,
    encoding: Optional[str] = None,
    excel_skiprows: int = 0,
    sample_rows: int = 20000,
) -> Dict[str, Any]:
    """
    Infer schema from file using pandas.

    Args:
        file_path: Path to the file
        filetype: File type (csv, tsv, psv, xlsx, parquet). If None, auto-detects from extension.
        has_header: Whether the file has a header row
        encoding: File encoding (e.g., "utf-8", "latin-1"). Defaults to "utf-8".
        excel_skiprows: Rows to skip in Excel files
        sample_rows: Number of rows to sample for type inference (default: 20000)

    Note: For CSV/TSV/PSV files, the delimiter is auto-detected.

    Returns:
        Dictionary with:
        {
            "column_mapping": {"column_name": ([], "type_string"), ...},
            "null_values": ["NA", "None", ...] or None if no custom nulls detected,
            "encoding": the encoding used
        }
    """
    path = Path(file_path)

    # Auto-detect filetype from extension if not provided
    if not filetype:
        ext = path.suffix.lower().lstrip(".")
        filetype = ext if ext in ["csv", "tsv", "psv", "xlsx", "parquet"] else "csv"

    if filetype in ["csv", "tsv", "psv"]:
        encoding = encoding or "utf-8"
        
        sep = None
        if filetype == "tsv":
            sep = "\t"
        elif filetype == "psv":
            sep = "|"
        else:
            # Attempt to sniff delimiter for CSV
            try:
                import csv
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    # Read a sample
                    sample = f.read(4096)
                    if sample.startswith("\ufeff"):
                        sample = sample[1:]
                    
                    if sample:
                        sniffer = csv.Sniffer()
                        # prefer commonly used delimiters
                        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
                        sep = dialect.delimiter
            except Exception:
                # Fallback to default comma if sniffing fails
                pass
        
        # Default to comma if still None (e.g. sniffing failed or filetype=csv with no sniffed sep)
        if sep is None:
            sep = ","

        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                encoding=encoding,
                header=0 if has_header else None,
                nrows=sample_rows,
                on_bad_lines="skip",
            )
        except pd.errors.EmptyDataError:
            # Handle empty file
            return {"column_mapping": {}, "null_values": None, "encoding": encoding}

        # Build column mapping
        column_mapping = {}
        for i, col in enumerate(df.columns):
            if has_header:
                original_col = str(col)
                series = df[col]
            else:
                original_col = f"col_{i}"
                series = df[col]  # df columns are integers when header=None

            if series.isna().all():
                type_string = "string"
            else:
                type_string = _infer_column_mapping_type(series.dtype)

            snake_case_col = to_snake_case(original_col)

            if snake_case_col == original_col:
                column_mapping[snake_case_col] = ([], type_string)
            else:
                column_mapping[snake_case_col] = ([original_col], type_string)

        return {
            "column_mapping": column_mapping,
            "null_values": None,
            "encoding": encoding,
        }

    elif filetype == "xlsx":
        # Excel files: use pandas (DuckDB xlsx support requires extension)
        df = pd.read_excel(
            file_path,
            skiprows=excel_skiprows,
            nrows=sample_rows,
        )

        column_mapping = {}
        for col in df.columns:
            type_string = _infer_column_mapping_type(df[col].dtype)
            original_col = str(col)
            snake_case_col = to_snake_case(original_col)
            if snake_case_col == original_col:
                column_mapping[snake_case_col] = ([], type_string)
            else:
                column_mapping[snake_case_col] = ([original_col], type_string)

        return {"column_mapping": column_mapping, "null_values": None, "encoding": None}

    elif filetype == "parquet":
        # Parquet already has type information
        df = pd.read_parquet(file_path)
        if sample_rows:
            df = df.head(sample_rows)

        column_mapping = {}
        for col in df.columns:
            type_string = _infer_column_mapping_type(df[col].dtype)
            original_col = str(col)
            snake_case_col = to_snake_case(original_col)
            if snake_case_col == original_col:
                column_mapping[snake_case_col] = ([], type_string)
            else:
                column_mapping[snake_case_col] = ([original_col], type_string)

        return {"column_mapping": column_mapping, "null_values": None, "encoding": None}

    else:
        raise ValueError(f"Unsupported filetype: {filetype}")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Infer column schema from data files and output column mapping JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python table_functions.py data/raw/my_file.csv
  python table_functions.py data/raw/my_file.psv
  python table_functions.py data/raw/my_file.xlsx --no-header
  python table_functions.py data/raw/my_file.parquet --sample-rows 5000
  python table_functions.py data/raw/  # Infer all files in directory

Note: Delimiter is auto-detected for CSV/TSV/PSV files.
        """,
    )

    parser.add_argument(
        "path",
        help="Path to the input file or directory",
    )

    parser.add_argument(
        "--filetype",
        choices=["csv", "tsv", "psv", "xlsx", "parquet"],
        help="File type (auto-detected from extension if not provided)",
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="File has no header row",
    )

    parser.add_argument(
        "--excel-skiprows",
        type=int,
        default=0,
        help="Number of rows to skip in Excel files (default: 0)",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20000,
        help="Number of rows to sample for type inference (default: 20000)",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="File encoding (e.g., utf-8, latin-1, cp1252). Defaults to utf-8.",
    )

    args = parser.parse_args()

    input_path = Path(args.path)

    # Validate path exists
    if not input_path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        exit(1)

    try:
        if input_path.is_dir():
            # Directory mode: infer schema for all files, output keyed by filename
            # Supported extensions based on filetype arg or all supported types
            if args.filetype:
                extensions = [f".{args.filetype}"]
            else:
                extensions = [".csv", ".tsv", ".psv", ".xlsx", ".parquet"]

            # Find all matching files in directory (non-recursive)
            files = sorted(
                [
                    f
                    for f in input_path.iterdir()
                    if f.is_file() and f.suffix.lower() in extensions
                ]
            )

            if not files:
                print(f"Error: No matching files found in {args.path}", file=sys.stderr)
                print(f"Looking for extensions: {extensions}", file=sys.stderr)
                exit(1)

            # Build output dictionary keyed by filename
            # Include snake_case table name in the value for convenience
            output = {}
            total_files = len(files)
            for i, file_path in enumerate(files):
                try:
                    print(
                        f"{i + 1}/{total_files} Inferring schema: {file_path.name}",
                        file=sys.stderr,
                    )
                    result = infer_schema_from_file(
                        file_path=str(file_path),
                        filetype=args.filetype,
                        has_header=not args.no_header,
                        encoding=args.encoding,
                        excel_skiprows=args.excel_skiprows,
                        sample_rows=args.sample_rows,
                    )
                    file_output = {
                        "table_name": to_snake_case(file_path.stem),
                        "column_mapping": result["column_mapping"],
                    }
                    if result.get("null_values"):
                        file_output["null_values"] = result["null_values"]
                    if result.get("encoding"):
                        file_output["encoding"] = result["encoding"]
                    output[file_path.name] = file_output
                except Exception as e:
                    print(
                        f"{i + 1}/{total_files} Failed: {file_path.name} - {e}",
                        file=sys.stderr,
                    )
                    output[file_path.name] = {"error": str(e)}

            # Output as JSON (pretty-printed)
            print(json.dumps(output, indent=2))

        else:
            # Single file mode - keyed by filename with table_name included
            print(f"Inferring schema: {input_path.name}", file=sys.stderr)
            result = infer_schema_from_file(
                file_path=str(input_path),
                filetype=args.filetype,
                has_header=not args.no_header,
                encoding=args.encoding,
                excel_skiprows=args.excel_skiprows,
                sample_rows=args.sample_rows,
            )

            file_output = {
                "table_name": to_snake_case(input_path.stem),
                "column_mapping": result["column_mapping"],
            }
            if result.get("null_values"):
                file_output["null_values"] = result["null_values"]
            if result.get("encoding"):
                file_output["encoding"] = result["encoding"]

            output = {input_path.name: file_output}

            # Output as JSON (pretty-printed)
            print(json.dumps(output, indent=2))

    except Exception as e:
        print(f"Error inferring schema: {e}", file=sys.stderr)
        exit(1)
