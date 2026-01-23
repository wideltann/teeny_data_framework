"""
Data ingestion framework using pandas and PostgreSQL
Pure psycopg implementation for efficient bulk loading
"""

import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql


# =============================================================================
# Type System - Single source of truth for type mappings
# =============================================================================


@dataclass
class TypeMapping:
    """Mapping between simplified type strings and pandas/postgres types"""

    pandas: str
    postgres: str


TYPE_SYSTEM: Dict[str, TypeMapping] = {
    "int": TypeMapping(pandas="Int64", postgres="BIGINT"),
    "float": TypeMapping(pandas="float64", postgres="DOUBLE PRECISION"),
    "boolean": TypeMapping(pandas="boolean", postgres="BOOLEAN"),
    "datetime": TypeMapping(pandas="datetime64[ns]", postgres="TIMESTAMP"),
    "string": TypeMapping(pandas="string", postgres="TEXT"),
}

# Postgres type variants for schema validation (what info_schema returns)
POSTGRES_TYPE_VARIANTS: Dict[str, List[str]] = {
    "BIGINT": ["bigint"],
    "DOUBLE PRECISION": ["double precision", "numeric"],
    "TEXT": ["text", "character varying"],
    "BOOLEAN": ["boolean"],
    "TIMESTAMP": ["timestamp without time zone", "timestamp with time zone"],
}


def dtype_str_to_pandas(dtype_str: str) -> str:
    """Convert simplified dtype string to pandas dtype."""
    dtype_lower = dtype_str.lower()
    if dtype_lower in TYPE_SYSTEM:
        return TYPE_SYSTEM[dtype_lower].pandas
    # Pass through pandas types directly (Int64, float64, etc.)
    return dtype_str


def dtype_str_to_postgres(dtype_str: str) -> str:
    """Convert simplified dtype string to PostgreSQL type."""
    dtype_lower = dtype_str.lower()
    if dtype_lower in TYPE_SYSTEM:
        return TYPE_SYSTEM[dtype_lower].postgres
    return "TEXT"


def pandas_dtype_to_postgres(dtype: Any) -> str:
    """Infer PostgreSQL type from pandas dtype."""
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "BIGINT"
    elif "float" in dtype_str or "decimal" in dtype_str:
        return "DOUBLE PRECISION"
    elif "bool" in dtype_str:
        return "BOOLEAN"
    elif "datetime" in dtype_str or "date" in dtype_str or "time" in dtype_str:
        return "TIMESTAMP"
    return "TEXT"


def pandas_dtype_to_type_str(dtype: Any) -> str:
    """Infer column_mapping type string from pandas dtype."""
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "int"
    elif "float" in dtype_str or "decimal" in dtype_str:
        return "float"
    elif "bool" in dtype_str:
        return "boolean"
    elif "datetime" in dtype_str or "date" in dtype_str or "time" in dtype_str:
        return "datetime"
    return "string"


# =============================================================================
# Column Mapping Result - Structured return type
# =============================================================================


@dataclass
class ColumnMappingResult:
    """Result of processing column mapping"""

    rename_dict: Dict[str, str]
    read_dtypes: Dict[str, str]
    missing_cols: Dict[str, str]


def prepare_column_mapping(
    header: List[str], column_mapping: Dict[str, Tuple[List[str], str]]
) -> ColumnMappingResult:
    """
    Process column mapping to build dictionaries for DataFrame transformation.

    Args:
        header: List of column names from the file
        column_mapping: Dict mapping output names to ([alternative_names], dtype).
                       Use "default" key for unmapped columns.

    Returns:
        ColumnMappingResult with rename_dict, read_dtypes, and missing_cols

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
            read_dtypes[found_name] = dtype_str_to_pandas(dtype)
            if found_name != target_name:
                rename_dict[found_name] = target_name
        else:
            # Column not in file, add as missing
            missing_cols[target_name] = dtype

    # Handle unmapped columns with default type
    if default_type:
        for col in header:
            if col not in processed_cols:
                read_dtypes[col] = dtype_str_to_pandas(default_type)

    return ColumnMappingResult(
        rename_dict=rename_dict, read_dtypes=read_dtypes, missing_cols=missing_cols
    )


def apply_column_transforms(
    df: pd.DataFrame, mapping_result: ColumnMappingResult
) -> pd.DataFrame:
    """Apply column renames and add missing columns with proper types"""
    if mapping_result.rename_dict:
        df = df.rename(columns=mapping_result.rename_dict)
    for col_name, col_type in mapping_result.missing_cols.items():
        df[col_name] = None
        pandas_type = dtype_str_to_pandas(col_type)
        if pandas_type != "object":
            df[col_name] = df[col_name].astype(pandas_type)
    return df


# =============================================================================
# Error Handling - Centralized error normalization
# =============================================================================


def normalize_error_message(error: str, max_length: int = 500) -> str:
    """Normalize error message for storage in metadata."""
    if len(error) > max_length:
        error = error[:max_length] + "... [truncated]"
    # Remove characters that could break SQL strings
    return error.replace("'", "").replace("`", "")


# =============================================================================
# Temp Directory Context Manager
# =============================================================================


class TempDirContext:
    """
    Context manager for temporary directory handling.

    Replaces global _temp_dir_override with explicit context.
    """

    _current: Optional[Path] = None

    def __init__(self, ephemeral: bool = False):
        self.ephemeral = ephemeral
        self._temp_dir = None
        self._previous = None

    def __enter__(self) -> "TempDirContext":
        self._previous = TempDirContext._current
        if self.ephemeral:
            import tempfile

            self._temp_dir = tempfile.TemporaryDirectory()
            TempDirContext._current = Path(self._temp_dir.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TempDirContext._current = self._previous
        if self._temp_dir:
            self._temp_dir.cleanup()
        return False

    @classmethod
    def get_temp_dir(cls) -> Path:
        """Get current temp directory."""
        if cls._current is not None:
            cls._current.mkdir(exist_ok=True, parents=True)
            return cls._current
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        return temp_dir


# Keep backward compatibility
def get_persistent_temp_dir() -> Path:
    """Get temp directory for caching files."""
    return TempDirContext.get_temp_dir()


def set_temp_dir_override(path: Optional[Path]) -> None:
    """Set the temp directory override (for backward compatibility)."""
    TempDirContext._current = path


# =============================================================================
# S3 Helper Functions
# =============================================================================


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path"""
    return str(path).startswith("s3://")


def normalize_path(path: str) -> str:
    """Normalize a path to a consistent string format (no trailing slash except for root)"""
    path_str = str(path)
    if is_s3_path(path_str):
        return path_str.rstrip("/")
    return Path(path_str).as_posix().rstrip("/")


def path_join(*parts: str) -> str:
    """Join path parts, handling both S3 and local paths"""
    if not parts:
        return ""

    first_part = str(parts[0])
    is_s3 = first_part.startswith("s3://")
    is_absolute = first_part.startswith("/") and not is_s3

    cleaned_parts = [str(p).strip("/") for p in parts if p and str(p).strip("/")]
    if not cleaned_parts:
        return ""

    if is_s3:
        return "s3://" + "/".join(p.replace("s3://", "") for p in cleaned_parts)

    result = "/".join(cleaned_parts)
    if is_absolute:
        result = "/" + result
    return result


def path_basename(path: str) -> str:
    """Get the filename from a path (S3 or local)"""
    return str(path).rstrip("/").split("/")[-1]


def path_parent(path: str) -> str:
    """Get the parent directory of a path (S3 or local)"""
    path_str = str(path).rstrip("/")
    parts = path_str.split("/")
    if is_s3_path(path_str):
        if len(parts) > 3:
            return "/".join(parts[:-1])
        return path_str
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


def get_s3_filesystem(filesystem: Optional[Any] = None) -> Any:
    """Get an s3fs filesystem instance."""
    if filesystem is not None:
        return filesystem

    try:
        import s3fs
    except ImportError:
        raise ImportError(
            "s3fs is required for S3 support. Install with: pip install s3fs"
        )

    return s3fs.S3FileSystem()


def get_cache_path_from_s3(s3_path: str) -> Path:
    """Convert S3 path to local cache path that mirrors the S3 structure."""
    path_without_prefix = s3_path.replace("s3://", "")
    cache_path = get_persistent_temp_dir() / path_without_prefix
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    return cache_path


def get_archive_cache_path_from_s3(s3_path: str) -> Path:
    """Convert S3 archive path to local cache path in the archives subdirectory."""
    path_without_prefix = s3_path.replace("s3://", "")
    cache_path = get_persistent_temp_dir() / "archives" / path_without_prefix
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    return cache_path


def get_cache_path_from_source_path(source_path: str) -> Path:
    """Convert source_path to local cache path."""
    if "::" in source_path:
        archive_path, inner_path = source_path.split("::", 1)

        if is_s3_path(archive_path):
            path_without_prefix = archive_path.replace("s3://", "")
            cache_path = get_persistent_temp_dir() / path_without_prefix / inner_path
        else:
            clean_archive_path = archive_path.lstrip("/")
            cache_path = get_persistent_temp_dir() / clean_archive_path / inner_path

        cache_path.parent.mkdir(exist_ok=True, parents=True)
        return cache_path
    else:
        if is_s3_path(source_path):
            return get_cache_path_from_s3(source_path)
        return Path(source_path)


def download_s3_file_with_cache(
    s3_path: str, filesystem: Any, is_archive: bool = False
) -> Path:
    """Download S3 file to persistent cache, reusing cached file if it exists with matching size."""
    if is_archive:
        cache_path = get_archive_cache_path_from_s3(s3_path)
    else:
        cache_path = get_cache_path_from_s3(s3_path)

    s3_info = filesystem.info(s3_path)
    s3_size = s3_info["size"]

    if cache_path.exists():
        cached_size = cache_path.stat().st_size
        if cached_size == s3_size:
            print(f"Cache hit: {cache_path.relative_to(Path.cwd())}")
            return cache_path

    print(f"Downloading: {s3_path} -> {cache_path.relative_to(Path.cwd())}")
    filesystem.get(s3_path, str(cache_path))

    return cache_path


# =============================================================================
# File Type Registry - Single source of truth for file type handling
# =============================================================================


@dataclass
class FileTypeConfig:
    """Configuration for a file type"""

    separator: Optional[str] = None  # For delimited files
    reader_key: str = "default"  # Key for reader dispatch


FILETYPE_REGISTRY: Dict[str, FileTypeConfig] = {
    "csv": FileTypeConfig(separator=",", reader_key="delimited"),
    "tsv": FileTypeConfig(separator="\t", reader_key="delimited"),
    "psv": FileTypeConfig(separator="|", reader_key="delimited"),
    "xlsx": FileTypeConfig(reader_key="xlsx"),
    "parquet": FileTypeConfig(reader_key="parquet"),
    "fixed_width": FileTypeConfig(reader_key="fixed_width"),
    "xml": FileTypeConfig(reader_key="xml"),
}


def get_filetype_config(filetype: str) -> FileTypeConfig:
    """Get configuration for a file type."""
    if filetype not in FILETYPE_REGISTRY:
        raise ValueError(
            f"Invalid filetype: {filetype}. "
            f"Must be one of: {', '.join(FILETYPE_REGISTRY.keys())}"
        )
    return FILETYPE_REGISTRY[filetype]


# =============================================================================
# File Readers
# =============================================================================


def read_delimited(
    full_path: str,
    column_mapping: Dict[str, Tuple[List[str], str]],
    header: Optional[List[str]] = None,
    has_header: bool = True,
    null_values: Optional[List[str]] = None,
    separator: str = ",",
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Read delimited file (CSV, TSV, PSV) and return DataFrame with column mapping."""
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

    mapping_result = prepare_column_mapping(header, column_mapping)

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
            dtype=mapping_result.read_dtypes,
            na_values=null_values,
            keep_default_na=keep_default_na,
        )
    except (ValueError, TypeError) as e:
        # Try to identify which column caused the error
        df_raw = pd.read_csv(
            StringIO(file_content),
            sep=separator,
            header=0 if has_header else None,
            names=header if not has_header else None,
            na_values=null_values,
            keep_default_na=keep_default_na,
        )
        for col, dtype in mapping_result.read_dtypes.items():
            if col in df_raw.columns:
                try:
                    df_raw[col].astype(dtype)
                except (ValueError, TypeError):
                    sample_values = df_raw[col].unique()[:20]
                    raise ValueError(
                        f"Column '{col}' cannot be converted to {dtype}. "
                        f"Sample values: {list(sample_values)}"
                    ) from e
        raise

    return apply_column_transforms(df, mapping_result)


def read_xlsx(
    full_path: str,
    column_mapping: Dict[str, Tuple[List[str], str]],
    excel_skiprows: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Read Excel file and return DataFrame with column mapping."""
    header = pd.read_excel(full_path, skiprows=excel_skiprows, nrows=1).columns.tolist()
    mapping_result = prepare_column_mapping(header, column_mapping)
    df = pd.read_excel(
        full_path, skiprows=excel_skiprows, dtype=mapping_result.read_dtypes
    )
    return apply_column_transforms(df, mapping_result)


def read_parquet(
    full_path: str,
    column_mapping: Dict[str, Tuple[List[str], str]],
    **kwargs,
) -> pd.DataFrame:
    """Read parquet file and return DataFrame with column mapping."""
    df = pd.read_parquet(full_path)
    mapping_result = prepare_column_mapping(df.columns.tolist(), column_mapping)

    # Keep only mapped columns (drop unmapped unless there's a default type)
    if "default" not in column_mapping:
        keep_cols = [
            col
            for col in df.columns
            if col in mapping_result.rename_dict or col in column_mapping
        ]
        df = df[keep_cols]

    return apply_column_transforms(df, mapping_result)


def read_xml(
    full_path: str,
    column_mapping: Dict[str, Tuple[List[str], str]],
    encoding: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read shallow XML file and return DataFrame with column mapping."""
    df = pd.read_xml(full_path, parser="etree", encoding=encoding)

    if df.empty:
        return pd.DataFrame()

    header = df.columns.tolist()
    mapping_result = prepare_column_mapping(header, column_mapping)

    # Apply dtypes - XML text is all strings, so we need to convert
    for col, dtype in mapping_result.read_dtypes.items():
        if col in df.columns:
            try:
                if dtype == "Int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "boolean":
                    # pandas may auto-convert to bool, or leave as string
                    if df[col].dtype == bool or df[col].dtype == "boolean":
                        df[col] = df[col].astype("boolean")
                    else:
                        df[col] = (
                            df[col]
                            .map(
                                {
                                    "true": True,
                                    "false": False,
                                    "1": True,
                                    "0": False,
                                    "True": True,
                                    "False": False,
                                }
                            )
                            .astype("boolean")
                        )
                elif dtype == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "string":
                    df[col] = df[col].astype("string")
            except Exception:
                df[col] = df[col].astype("string")

    return apply_column_transforms(df, mapping_result)


def read_fixed_width(
    full_path: str,
    column_mapping: Dict[str, Tuple[str, int, int]],
    encoding: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read fixed-width file and return DataFrame.

    column_mapping format for fixed_width:
    {'column_name': (type, starting_position, field_size)}

    Note: starting_position is 1-indexed (first character is position 1)
    """
    from io import StringIO

    encoding = encoding or "utf-8"
    with open(full_path, "r", encoding=encoding) as f:
        file_content = f.read()

    colspecs = []
    names = []
    dtypes = {}

    for col_name, (col_type, start_pos, field_size) in column_mapping.items():
        start_idx = start_pos - 1
        end_idx = start_idx + field_size
        colspecs.append((start_idx, end_idx))
        names.append(col_name)
        dtypes[col_name] = dtype_str_to_pandas(col_type)

    return pd.read_fwf(
        StringIO(file_content),
        colspecs=colspecs,
        names=names,
        dtype=dtypes,
    )


def read_using_column_mapping(
    full_path: str,
    filetype: str,
    column_mapping: Dict[str, Any],
    header: Optional[List[str]] = None,
    has_header: bool = True,
    null_values: Optional[List[str]] = None,
    excel_skiprows: int = 0,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Router function to read different file types with column mapping."""
    config = get_filetype_config(filetype)

    if config.reader_key == "delimited":
        return read_delimited(
            full_path=full_path,
            column_mapping=column_mapping,
            has_header=has_header,
            header=header,
            separator=config.separator,
            null_values=null_values,
            encoding=encoding,
        )
    elif config.reader_key == "xlsx":
        return read_xlsx(
            full_path=full_path,
            column_mapping=column_mapping,
            excel_skiprows=excel_skiprows,
        )
    elif config.reader_key == "parquet":
        return read_parquet(full_path=full_path, column_mapping=column_mapping)
    elif config.reader_key == "fixed_width":
        return read_fixed_width(
            full_path=full_path,
            column_mapping=column_mapping,
            encoding=encoding,
        )
    elif config.reader_key == "xml":
        return read_xml(
            full_path=full_path,
            column_mapping=column_mapping,
            encoding=encoding,
        )
    else:
        raise ValueError(f"Unknown reader_key: {config.reader_key}")


# =============================================================================
# Database Operations
# =============================================================================


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


def get_table_schema(
    conn: psycopg.Connection, schema: str, table: str
) -> Dict[str, str]:
    """Get existing table schema from PostgreSQL."""
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
    """Validate that DataFrame schema matches existing table schema."""
    if not table_exists(conn, schema, table):
        return

    existing_schema = get_table_schema(conn, schema, table)
    df_columns = set(df.columns)
    table_columns = set(existing_schema.keys())

    missing_in_table = df_columns - table_columns
    if missing_in_table:
        raise ValueError(
            f"Schema mismatch for {schema}.{table}: "
            f"DataFrame has columns not in table: {sorted(missing_in_table)}\n"
            f"Table columns: {sorted(table_columns)}\n"
            f"DataFrame columns: {sorted(df_columns)}"
        )

    missing_in_df = table_columns - df_columns
    if missing_in_df:
        raise ValueError(
            f"Schema mismatch for {schema}.{table}: "
            f"Table has columns not in DataFrame: {sorted(missing_in_df)}\n"
            f"Table columns: {sorted(table_columns)}\n"
            f"DataFrame columns: {sorted(df_columns)}"
        )

    # Validate column types match
    type_mismatches = []
    for col in df.columns:
        expected_pg_type = pandas_dtype_to_postgres(df[col].dtype)
        actual_pg_type = existing_schema[col]

        expected_variants = POSTGRES_TYPE_VARIANTS.get(
            expected_pg_type, [expected_pg_type.lower()]
        )
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
    columns = []
    for col in df.columns:
        pg_type = pandas_dtype_to_postgres(df[col].dtype)
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
    """Bulk load DataFrame to PostgreSQL using COPY with write_row()"""
    validate_schema_match(conn, df, schema, table)
    create_table_from_dataframe(conn, df, schema, table)

    # Replace pandas NA/NaT with None for psycopg3 compatibility
    df = df.fillna(value=np.nan).replace({np.nan: None})

    columns = ", ".join(f'"{col}"' for col in df.columns)
    copy_sql = f"COPY {schema}.{table} ({columns}) FROM STDIN"

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)


def row_count_check(
    conn: psycopg.Connection,
    schema: str,
    df: pd.DataFrame,
    source_path: str,
    unpivot_row_multiplier: Optional[int] = None,
) -> None:
    """Sanity check comparing metadata row count to DataFrame row count"""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT row_count FROM {}.metadata WHERE source_path = %s").format(
                sql.Identifier(schema)
            ),
            (source_path,),
        )
        result = cur.fetchone()

    metadata_row_count = result[0] if result else None
    output_df_row_count = len(df)

    if unpivot_row_multiplier:
        metadata_row_count *= unpivot_row_multiplier

    if metadata_row_count and metadata_row_count != output_df_row_count:
        raise ValueError(
            f"Check failed {source_path} since metadata table row count {metadata_row_count} "
            f"is not equal to output table row count {output_df_row_count}"
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
    """Update metadata table with ingestion status and runtime"""
    from datetime import datetime

    metadata_ingest_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "Failure" if error_message else "Success"

    if error_message:
        error_message = normalize_error_message(error_message)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                UPDATE {}.metadata
                SET
                    ingest_datetime = %s,
                    status = %s,
                    error_message = %s,
                    unpivot_row_multiplier = %s,
                    ingest_runtime = %s,
                    output_table = %s
                WHERE source_path = %s
                """
            ).format(sql.Identifier(schema)),
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


# =============================================================================
# Main Ingestion Functions
# =============================================================================


def update_table(
    conninfo: str,
    schema: str,
    source_dir: str,
    filetype: str,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    column_mapping_fn: Optional[
        Callable[[Path], Dict[str, Tuple[List[str], str]]]
    ] = None,
    output_table: Optional[str] = None,
    output_table_naming_fn: Optional[Callable[[Path], str]] = None,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    metadata_schema: Optional[str] = None,
    additional_cols_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    file_list_filter_fn: Optional[Callable[[List[Path]], List[Path]]] = None,
    custom_read_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    sql_glob: Optional[str] = None,
    pivot_mapping: Optional[Dict[str, Any]] = None,
    header_fn: Optional[Callable[[Path], List[str]]] = None,
    null_values: Optional[List[str]] = None,
    excel_skiprows: int = 0,
    cleanup: bool = False,
    ephemeral_cache: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main ingestion function that reads files and writes to PostgreSQL.

    Args:
        conninfo: PostgreSQL connection string
        schema: Target schema for data tables
        source_dir: Source directory to filter files from metadata
        filetype: File type (csv, tsv, psv, xlsx, parquet, fixed_width, xml)
        column_mapping: Column mapping dictionary
        column_mapping_fn: Function to get column mapping per file
        output_table: Target table name
        output_table_naming_fn: Function to determine table name per file
        resume: Skip already-processed files
        retry_failed: Re-process failed files when resume=True
        sample: Process only N files
        metadata_schema: Schema for metadata table (defaults to schema)
        additional_cols_fn: Function to add columns from filename
        file_list_filter_fn: Function to filter files to process
        custom_read_fn: Custom file reader function
        transform_fn: Transform DataFrame after reading
        sql_glob: SQL LIKE pattern to filter files
        pivot_mapping: Unpivot/melt configuration
        header_fn: Function to provide header for headerless files
        null_values: Custom null value representations
        excel_skiprows: Rows to skip in Excel files
        cleanup: Delete cached files after successful ingest
        ephemeral_cache: Use temporary directory (auto-deleted)
        encoding: File encoding (defaults to utf-8)

    Returns:
        DataFrame with metadata results
    """
    with TempDirContext(ephemeral=ephemeral_cache):
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


def _update_table_impl(
    conninfo: str,
    schema: str,
    source_dir: str,
    filetype: str,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    column_mapping_fn: Optional[
        Callable[[Path], Dict[str, Tuple[List[str], str]]]
    ] = None,
    output_table: Optional[str] = None,
    output_table_naming_fn: Optional[Callable[[Path], str]] = None,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    metadata_schema: Optional[str] = None,
    additional_cols_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    file_list_filter_fn: Optional[Callable[[List[Path]], List[Path]]] = None,
    custom_read_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    sql_glob: Optional[str] = None,
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

    if not custom_read_fn and not column_mapping_fn and column_mapping is None:
        raise ValueError(
            "column_mapping is required unless using custom_read_fn or column_mapping_fn"
        )

    source_dir = normalize_path(source_dir) + "/"
    metadata_schema = metadata_schema or schema
    glob_pattern = sql_glob if sql_glob else "%"

    # Query metadata for files to process
    query = sql.SQL(
        """
        SELECT source_path
        FROM {}.metadata
        WHERE
            source_dir LIKE %s AND
            source_path LIKE %s AND
            metadata_ingest_status = 'Success'
        """
    ).format(sql.Identifier(metadata_schema))

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (source_dir, glob_pattern))
            rows = cur.fetchall()
    file_list = sorted([row[0] for row in rows])

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    if resume:
        resume_query = sql.SQL(
            """
            SELECT source_path
            FROM {}.metadata
            WHERE
                source_dir LIKE %s AND
                ingest_datetime IS NOT NULL
                {}
            """
        ).format(
            sql.Identifier(metadata_schema),
            sql.SQL("AND status = 'Success'" if not retry_failed else ""),
        )

        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(resume_query, (source_dir,))
                rows = cur.fetchall()
        processed_set = set(row[0] for row in rows)
        file_list = [f for f in file_list if f not in processed_set]

    total_files_to_be_processed = sample if sample else len(file_list)

    if total_files_to_be_processed == 0:
        print("No files to ingest")
    else:
        print(
            f"Output {'table' if output_table else 'schema'}: {output_table or schema} | "
            f"Num files being processed: {total_files_to_be_processed} out of {len(file_list)} "
            f"{'new files' if resume else 'total files'}"
        )

    for i, source_path in enumerate(file_list):
        start_time = time.time()

        if sample and i == sample:
            break

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

            # Fresh connection for row count check (resilience for long loops)
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

            # Fresh connection for write operations (resilience for long loops)
            with psycopg.connect(conninfo) as conn:
                if table_exists(conn, schema, table_name):
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("DELETE FROM {}.{} WHERE source_path = %s").format(
                                sql.Identifier(schema), sql.Identifier(table_name)
                            ),
                            (source_path,),
                        )

                copy_dataframe_to_table(conn, df, schema, table_name)
                conn.commit()

            ingest_runtime = int(time.time() - start_time)

            # Fresh connection for metadata update (resilience for long loops)
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
                    pass

            print(
                f"{i + 1}/{total_files_to_be_processed} Ingested {source_path} -> "
                f"{schema}.{table_name} ({len(df)} rows)"
            )

        except Exception as e:
            error_str = str(e)

            # Fresh connection for error metadata update (resilience for long loops)
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

            print(
                f"Failed on {source_path} with {normalize_error_message(error_str, 200)}",
                file=sys.stderr,
            )

    # Return metadata results
    result_query = sql.SQL(
        """
        SELECT *
        FROM {}.metadata
        WHERE source_dir LIKE %s AND source_path LIKE %s
        ORDER BY ingest_datetime DESC
        """
    ).format(sql.Identifier(metadata_schema))

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(result_query, (source_dir, glob_pattern))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)


# =============================================================================
# Metadata Functions
# =============================================================================


def get_csv_header_and_row_count(
    file: str,
    separator: str = ",",
    has_header: bool = True,
    encoding: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Get header and row count from CSV file."""
    import subprocess
    import csv

    file_str = str(file)
    encoding = encoding or "utf-8"

    with open(file_str, "r", encoding=encoding) as f:
        first_line = f.readline()

    if first_line.startswith("\ufeff"):
        first_line = first_line[1:]

    reader = csv.reader([first_line], delimiter=separator)
    header = next(reader)

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


def get_file_metadata_row(
    source_path: str,
    source_dir: str,
    filetype: str,
    has_header: bool = True,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate metadata row for a file."""
    import hashlib
    import time
    from datetime import datetime

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
        "error_message": normalize_error_message(error_message)
        if error_message
        else None,
    }

    if error_message:
        return row

    start_time = time.time()
    cache_path = get_cache_path_from_source_path(source_path)

    try:
        config = get_filetype_config(filetype)

        if config.reader_key == "delimited":
            header, row_count = get_csv_header_and_row_count(
                file=str(cache_path),
                has_header=has_header,
                separator=config.separator,
            )
        elif config.reader_key == "fixed_width":
            header, row_count = get_csv_header_and_row_count(
                file=str(cache_path), has_header=False
            )
        elif config.reader_key == "xlsx":
            df = pd.read_excel(cache_path)
            header, row_count = list(df.columns), len(df)
        elif config.reader_key == "parquet":
            df = pd.read_parquet(cache_path)
            header, row_count = list(df.columns), len(df)
        elif config.reader_key == "xml":
            header, row_count = None, None
        else:
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
        row["error_message"] = normalize_error_message(str(e))

    return row


def extract_and_add_zip_files(
    file_list: List[str],
    source_path_list: List[str],
    source_dir: str,
    has_header: bool,
    filetype: str,
    resume: bool,
    sample: Optional[int],
    archive_glob: str,
    filesystem: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Extract files from ZIP archives and add to metadata."""
    try:
        import zipfile_deflate64 as zipfile
    except Exception:
        import zipfile

    import shutil

    source_dir = normalize_path(source_dir) if source_dir else None

    rows = []
    archive_stats: Dict[str, int] = {}
    num_processed = 0

    fs = None
    if any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    source_path_set = set(source_path_list) if source_path_list else set()

    for compressed in file_list:
        if sample and num_processed == sample:
            break

        archive_path = str(compressed)
        archive_file_count = 0

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
                    f"No files matching '{archive_glob}' in {path_basename(archive_path)}, "
                    f"trying next archive..."
                )
                continue

            for inner_path in namelist:
                if sample and num_processed == sample:
                    break

                source_path = f"{archive_path}::{inner_path}"

                num_processed += 1
                file_num = (
                    f"{num_processed}/{sample}" if sample else f"{num_processed} |"
                )

                if resume and source_path in source_path_set:
                    print(
                        f"{file_num} Skipped (in metadata): {path_basename(inner_path)}"
                    )
                    continue

                cache_path = get_cache_path_from_source_path(source_path)

                try:
                    if cache_path.exists():
                        print(
                            f"{file_num} Cache hit: {cache_path.relative_to(Path.cwd())}"
                        )
                    else:
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

                    if cache_path.exists():
                        cache_path.unlink()

                    if cache_path.parent.exists() and not any(
                        cache_path.parent.iterdir()
                    ):
                        cache_path.parent.rmdir()

                rows.append(row)

        if archive_file_count > 0:
            archive_stats[archive_path] = archive_file_count

    return rows, archive_stats


def add_files(
    source_dir: str,
    file_list: List[str],
    filetype: str,
    has_header: bool = True,
    resume: bool = False,
    sample: Optional[int] = None,
    source_path_list: Optional[List[str]] = None,
    filesystem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Process non-archive files and add to metadata."""
    source_dir = normalize_path(source_dir) if source_dir else None
    source_path_set = set(source_path_list) if source_path_list else set()

    if source_path_set:
        file_list = [f for f in file_list if str(f) not in source_path_set]

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Num files being processed: {total_files_to_be_processed} out of {len(file_list)} "
        f"{'new files' if resume else 'total files'}"
    )

    fs = None
    if any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    rows = []
    num_processed = 0

    for i, file in enumerate(file_list):
        source_path = str(file)

        if sample and num_processed == sample:
            break

        num_processed += 1

        try:
            if is_s3_path(source_path):
                cache_path = get_cache_path_from_source_path(source_path)
                fs.get(source_path, str(cache_path))
                print(f"Downloaded {source_path} to {cache_path}")

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


def add_files_to_metadata_table(
    conninfo: str,
    schema: str,
    source_dir: str,
    filetype: str,
    glob: Optional[str] = None,
    compression_type: Optional[str] = None,
    archive_glob: Optional[str] = None,
    has_header: bool = True,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    file_list_filter_fn: Optional[Callable[[List[str]], List[str]]] = None,
    filesystem: Optional[Any] = None,
    expected_archive_file_count: Optional[int] = None,
    ephemeral_cache: bool = False,
) -> pd.DataFrame:
    """
    Add files to metadata table, creating it if necessary.

    Args:
        conninfo: PostgreSQL connection string
        schema: Target schema
        source_dir: Directory containing source files
        filetype: File type (csv, tsv, psv, xlsx, parquet, xml)
        glob: Glob pattern to match files (defaults to *.{filetype} or *.{compression_type})
        compression_type: Archive type (e.g., "zip")
        archive_glob: Glob pattern for files within archives
        has_header: Whether files have header rows
        resume: Skip already-processed files
        retry_failed: Re-process failed files when resume=True
        sample: Process only N files
        file_list_filter_fn: Function to filter file list
        filesystem: s3fs filesystem for S3 access
        expected_archive_file_count: Expected files per archive (enables archive-level skip)
        ephemeral_cache: Use temporary directory (auto-deleted)

    Returns:
        DataFrame with metadata results
    """
    with TempDirContext(ephemeral=ephemeral_cache):
        return _add_files_to_metadata_table_impl(
            conninfo=conninfo,
            schema=schema,
            source_dir=source_dir,
            filetype=filetype,
            glob=glob,
            compression_type=compression_type,
            archive_glob=archive_glob,
            has_header=has_header,
            resume=resume,
            retry_failed=retry_failed,
            sample=sample,
            file_list_filter_fn=file_list_filter_fn,
            filesystem=filesystem,
            expected_archive_file_count=expected_archive_file_count,
        )


def _add_files_to_metadata_table_impl(
    conninfo: str,
    schema: str,
    source_dir: str,
    filetype: str,
    glob: Optional[str] = None,
    compression_type: Optional[str] = None,
    archive_glob: Optional[str] = None,
    has_header: bool = True,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    file_list_filter_fn: Optional[Callable[[List[str]], List[str]]] = None,
    filesystem: Optional[Any] = None,
    expected_archive_file_count: Optional[int] = None,
) -> pd.DataFrame:
    """Internal implementation of add_files_to_metadata_table"""
    if not glob:
        glob = f"*.{compression_type}" if compression_type else f"*.{filetype}"

    if compression_type and not archive_glob:
        archive_glob = f"*.{filetype}"

    source_dir = normalize_path(source_dir)

    # Handle S3 or local paths for file listing
    if is_s3_path(source_dir):
        fs = get_s3_filesystem(filesystem)
        s3_glob_pattern = f"{source_dir}/**/{glob}"
        s3_paths = fs.glob(s3_glob_pattern)
        file_list = [f"s3://{p}" if not p.startswith("s3://") else p for p in s3_paths]
    else:
        file_list = [f.as_posix() for f in Path(source_dir).rglob(glob)]

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    file_list = sorted(file_list)

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

    source_path_list = []
    completed_archives = set()

    if resume:
        source_dir_match = source_dir + "/"
        resume_query = sql.SQL(
            """
            SELECT source_path
            FROM {}.metadata
            WHERE
                source_dir = %s
                {}
            """
        ).format(
            sql.Identifier(schema),
            sql.SQL(
                "AND metadata_ingest_status = 'Success'" if not retry_failed else ""
            ),
        )

        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(resume_query, (source_dir_match,))
                rows = cur.fetchall()
        source_path_list = [row[0] for row in rows]

        if compression_type:
            archive_query = sql.SQL(
                """
                SELECT archive_path
                FROM {}.archive_metadata
                WHERE
                    source_dir = %s
                    AND status = 'Success'
                """
            ).format(sql.Identifier(schema))

            with psycopg.connect(conninfo) as conn:
                with conn.cursor() as cur:
                    cur.execute(archive_query, (source_dir_match,))
                    archive_rows = cur.fetchall()
            completed_archives = {row[0] for row in archive_rows}

            if completed_archives:
                original_count = len(file_list)
                file_list = [f for f in file_list if f not in completed_archives]
                skipped_count = original_count - len(file_list)
                if skipped_count > 0:
                    print(f"Skipping {skipped_count} completed archive(s)")

    if compression_type == "zip":
        rows, archive_stats = extract_and_add_zip_files(
            file_list=file_list,
            source_path_list=source_path_list,
            source_dir=source_dir,
            has_header=has_header,
            filetype=filetype,
            resume=resume,
            sample=sample,
            archive_glob=archive_glob,
            filesystem=filesystem,
        )
    elif compression_type is None:
        rows = add_files(
            source_dir=source_dir,
            file_list=file_list,
            filetype=filetype,
            has_header=has_header,
            resume=resume,
            sample=sample,
            source_path_list=source_path_list,
            filesystem=filesystem,
        )
        archive_stats = {}
    else:
        raise Exception(f"Unsupported compression type: {compression_type}")

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

    # Return metadata results
    source_dir_match = normalize_path(source_dir) + "/"
    result_query = sql.SQL(
        """
        SELECT *
        FROM {}.metadata
        WHERE source_dir = %s
        ORDER BY metadata_ingest_datetime DESC
        """
    ).format(sql.Identifier(schema))

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(result_query, (source_dir_match,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)


def drop_metadata_by_source(
    conninfo: str,
    source_dir: str,
    schema: str,
) -> None:
    """Remove all files from a source directory from metadata"""
    source_dir = Path(source_dir).as_posix()

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {}.metadata WHERE source_dir LIKE %s"
                ).format(sql.Identifier(schema)),
                (source_dir,),
            )
            count_before = cur.fetchone()[0]

        print(f"Rows before drop: {count_before}")

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {}.metadata WHERE source_dir LIKE %s").format(
                    sql.Identifier(schema)
                ),
                (source_dir,),
            )
            print(f"Deleted {cur.rowcount} rows")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {}.metadata WHERE source_dir LIKE %s"
                ).format(sql.Identifier(schema)),
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
    """Delete records matching partition key from table"""
    print(
        f"Running: DELETE FROM {schema}.{table} WHERE source_path LIKE '{partition_key}'"
    )

    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {}.{} WHERE source_path LIKE %s").format(
                    sql.Identifier(schema), sql.Identifier(table)
                ),
                (partition_key,),
            )
            print(f"Deleted {cur.rowcount} rows")
        conn.commit()


def drop_file_from_metadata_and_table(
    conninfo: str,
    table: str,
    source_path: str,
    schema: str,
) -> None:
    """Remove a file from both metadata and data table"""
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {}.metadata WHERE source_path = %s").format(
                    sql.Identifier(schema)
                ),
                (source_path,),
            )
            print(f"Deleted {cur.rowcount} rows from metadata")
        conn.commit()

    drop_partition(
        conninfo=conninfo, table=table, partition_key=source_path, schema=schema
    )


# =============================================================================
# CLI Schema Inference Functions
# =============================================================================


def to_snake_case(name: str) -> str:
    """Convert a string to snake_case using inflection library"""
    import inflection

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

    Returns:
        Dictionary with column_mapping, null_values, and encoding
    """
    path = Path(file_path)

    if not filetype:
        ext = path.suffix.lower().lstrip(".")
        filetype = ext if ext in FILETYPE_REGISTRY else "csv"

    config = get_filetype_config(filetype)

    if config.reader_key == "delimited":
        encoding = encoding or "utf-8"

        sep = config.separator
        if filetype == "csv":
            # Use sep=None with python engine for auto-detection
            sep = None

        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                engine="python" if sep is None else None,
                encoding=encoding,
                header=0 if has_header else None,
                nrows=sample_rows,
                on_bad_lines="skip",
            )
            # Handle pandas bug with single-column files
            if sep is None and any("Unnamed" in str(c) for c in df.columns):
                df = pd.read_csv(
                    file_path,
                    sep=",",
                    encoding=encoding,
                    header=0 if has_header else None,
                    nrows=sample_rows,
                    on_bad_lines="skip",
                    low_memory=False,
                )
        except pd.errors.EmptyDataError:
            return {"column_mapping": {}, "null_values": None, "encoding": encoding}

        column_mapping = {}
        for i, col in enumerate(df.columns):
            if has_header:
                original_col = str(col)
                series = df[col]
            else:
                original_col = f"col_{i}"
                series = df[col]

            if series.isna().all():
                type_string = "string"
            else:
                type_string = pandas_dtype_to_type_str(series.dtype)

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

    elif config.reader_key == "xlsx":
        df = pd.read_excel(
            file_path,
            skiprows=excel_skiprows,
            nrows=sample_rows,
        )

        column_mapping = {}
        for col in df.columns:
            type_string = pandas_dtype_to_type_str(df[col].dtype)
            original_col = str(col)
            snake_case_col = to_snake_case(original_col)
            if snake_case_col == original_col:
                column_mapping[snake_case_col] = ([], type_string)
            else:
                column_mapping[snake_case_col] = ([original_col], type_string)

        return {"column_mapping": column_mapping, "null_values": None, "encoding": None}

    elif config.reader_key == "parquet":
        df = pd.read_parquet(file_path)
        if sample_rows:
            df = df.head(sample_rows)

        column_mapping = {}
        for col in df.columns:
            type_string = pandas_dtype_to_type_str(df[col].dtype)
            original_col = str(col)
            snake_case_col = to_snake_case(original_col)
            if snake_case_col == original_col:
                column_mapping[snake_case_col] = ([], type_string)
            else:
                column_mapping[snake_case_col] = ([original_col], type_string)

        return {"column_mapping": column_mapping, "null_values": None, "encoding": None}

    elif config.reader_key == "xml":
        df = pd.read_xml(file_path, parser="etree")

        if sample_rows and len(df) > sample_rows:
            df = df.head(sample_rows)

        if df.empty:
            return {"column_mapping": {}, "null_values": None, "encoding": None}

        # Try to infer types by attempting numeric conversion
        for col in df.columns:
            try:
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                if numeric_vals.notna().any():
                    if (
                        numeric_vals.dropna() == numeric_vals.dropna().astype(int)
                    ).all():
                        df[col] = numeric_vals.astype("Int64")
                    else:
                        df[col] = numeric_vals
            except Exception:
                pass

        column_mapping = {}
        for col in df.columns:
            type_string = pandas_dtype_to_type_str(df[col].dtype)
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
        choices=list(FILETYPE_REGISTRY.keys()),
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

    if not input_path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    try:
        if input_path.is_dir():
            if args.filetype:
                extensions = [f".{args.filetype}"]
            else:
                extensions = [f".{ft}" for ft in FILETYPE_REGISTRY.keys()]

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
                sys.exit(1)

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

            print(json.dumps(output, indent=2))

        else:
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
            print(json.dumps(output, indent=2))

    except Exception as e:
        print(f"Error inferring schema: {e}", file=sys.stderr)
        sys.exit(1)
