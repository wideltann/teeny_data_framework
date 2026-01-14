"""
Data ingestion framework using Polars and PostgreSQL via ADBC
"""

import polars as pl
import adbc_driver_postgresql.dbapi as pg_dbapi
import json
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
from contextlib import contextmanager


# Connection management


@contextmanager
def get_connection(uri: str):
    """
    Context manager for ADBC PostgreSQL connections.

    Args:
        uri: PostgreSQL connection URI (e.g., "postgresql://user:pass@host:port/db")

    Yields:
        ADBC connection object
    """
    conn = pg_dbapi.connect(uri)
    try:
        yield conn
    finally:
        conn.close()


def execute_sql(uri: str, sql: str, params: Optional[tuple] = None) -> None:
    """Execute a SQL statement that doesn't return results (DDL, DML)"""
    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
        conn.commit()


def execute_sql_fetch(
    uri: str, sql: str, params: Optional[tuple] = None
) -> List[tuple]:
    """Execute a SQL statement and return all results"""
    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur.fetchall()


def execute_sql_fetchone(
    uri: str, sql: str, params: Optional[tuple] = None
) -> Optional[tuple]:
    """Execute a SQL statement and return first result"""
    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return cur.fetchone()


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
        read_dtypes: Maps original column names to their types (for reading)
        missing_cols: Columns to add with their types

    Example:
        column_mapping = {
            "block_fips": ([], "string"),              # Keep name as-is
            "pop": (["POP100", "population"], "Int64"), # Rename if found
            "default": ([], "float64"),                # Type for unmapped cols
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
            read_dtypes[found_name] = dtype
            if found_name != target_name:
                rename_dict[found_name] = target_name
        else:
            # Column not in file, add as missing
            missing_cols[target_name] = dtype

    # Handle unmapped columns with default type
    if default_type:
        for col in header:
            if col not in processed_cols:
                read_dtypes[col] = default_type

    return rename_dict, read_dtypes, missing_cols


def _dtype_str_to_polars(dtype_str: str) -> pl.DataType:
    """Convert dtype string to Polars dtype.

    Type mappings:
        int -> pl.Int64
        float -> pl.Float64
        boolean -> pl.Boolean
        datetime -> pl.Datetime
        string -> pl.Utf8
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower == "int":
        return pl.Int64
    elif dtype_lower == "float":
        return pl.Float64
    elif dtype_lower == "boolean":
        return pl.Boolean
    elif dtype_lower == "datetime":
        return pl.Datetime
    elif dtype_lower == "string":
        return pl.Utf8
    else:
        return pl.Utf8


def _apply_column_transforms(
    df: pl.DataFrame, rename_dict: Dict[str, str], missing_cols: Dict[str, str]
) -> pl.DataFrame:
    """Apply column renames and add missing columns with proper types"""
    if rename_dict:
        df = df.rename(rename_dict)
    for col_name, col_type in missing_cols.items():
        polars_type = _dtype_str_to_polars(col_type)
        df = df.with_columns(pl.lit(None).cast(polars_type).alias(col_name))
    return df


def read_xlsx(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    excel_skiprows: int = 0,
) -> pl.DataFrame:
    """
    Read Excel file and return polars DataFrame with proper column mapping

    Args:
        full_path: Path to Excel file
        column_mapping: Column mapping dictionary
        excel_skiprows: Number of rows to skip at the beginning (for label rows before data)
    """
    # Read Excel file with polars
    # Note: polars uses 0-based indexing for read_excel
    df = pl.read_excel(full_path, read_options={"skip_rows": excel_skiprows})
    header = df.columns

    # Process column mapping to get rename dict, dtypes, and missing columns
    rename_dict, read_dtypes, missing_cols = prepare_column_mapping(
        header, column_mapping
    )

    # Apply type casts where needed
    cast_exprs = []
    for col_name, dtype_str in read_dtypes.items():
        if col_name in df.columns:
            polars_type = _dtype_str_to_polars(dtype_str)
            cast_exprs.append(pl.col(col_name).cast(polars_type))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_parquet(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
) -> pl.DataFrame:
    """
    Read parquet file and return polars DataFrame with column mapping
    """
    df = pl.read_parquet(full_path)

    # Prepare column transformations (parquet already has type info)
    rename_dict, _, missing_cols = prepare_column_mapping(
        header=df.columns, column_mapping=column_mapping
    )

    # Keep only mapped columns (drop unmapped unless there's a default type)
    if "default" not in column_mapping:
        keep_cols = [
            col for col in df.columns if col in rename_dict or col in column_mapping
        ]
        df = df.select(keep_cols)

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_csv(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_value: Optional[str] = None,
    separator: str = ",",
    encoding: str = "utf-8-sig",
) -> pl.DataFrame:
    """
    Read CSV file and return polars DataFrame with proper schema and column mapping
    """
    if not header:
        # some files start with a BOM (byte-order-mark)
        # if we dont use sig then that BOM will get included in the header
        import csv

        with open(full_path, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=separator)
            header = next(reader)

    # Process column mapping to get rename dict, dtypes, and missing columns
    rename_dict, read_dtypes, missing_cols = prepare_column_mapping(
        header, column_mapping
    )

    # Build polars schema from read_dtypes
    schema_overrides = {}
    for col_name, dtype_str in read_dtypes.items():
        schema_overrides[col_name] = _dtype_str_to_polars(dtype_str)

    # Handle null values
    null_values = [null_value] if null_value is not None else None

    # Handle encoding for polars
    polars_encoding = encoding.replace("-sig", "").replace("-", "")
    if polars_encoding.lower() not in ["utf8", "utf-8"]:
        polars_encoding = "utf8-lossy"

    # Read CSV with polars
    try:
        df = pl.read_csv(
            full_path,
            separator=separator,
            has_header=has_header,
            new_columns=header if not has_header else None,
            null_values=null_values,
            encoding=polars_encoding,
            schema_overrides=schema_overrides,
            infer_schema_length=10000,
        )
    except Exception:
        # Fallback: read as strings first, then cast
        df = pl.read_csv(
            full_path,
            separator=separator,
            has_header=has_header,
            new_columns=header if not has_header else None,
            null_values=null_values,
            encoding=polars_encoding,
            infer_schema_length=0,  # Read all as strings
        )
        # Apply casts
        cast_exprs = []
        for col_name, dtype_str in read_dtypes.items():
            if col_name in df.columns:
                polars_type = _dtype_str_to_polars(dtype_str)
                if polars_type in (pl.Int64, pl.Float64):
                    # For numeric types, handle casting carefully
                    cast_exprs.append(
                        pl.col(col_name)
                        .str.strip_chars()
                        .cast(polars_type, strict=False)
                    )
                else:
                    cast_exprs.append(pl.col(col_name).cast(polars_type, strict=False))
        if cast_exprs:
            df = df.with_columns(cast_exprs)

    return _apply_column_transforms(df, rename_dict, missing_cols)


def read_fixed_width(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[str, int, int]]] = None,
) -> pl.DataFrame:
    """
    Read fixed-width file and return polars DataFrame

    column_mapping format:
    {'stusab': ('string', 7, 2),
     'sumlev': ('float64', 9, 3),
     'geocomp': ('string', 12, 2)}

    {'column_name': (type, starting_position, field_size)}
    """
    # Read as text file
    with open(full_path, "r") as f:
        lines = f.readlines()

    data = {}
    for col_name, value in column_mapping.items():
        col_type, starting_position, field_size = value

        # Extract column data (1-indexed position converted to 0-indexed)
        col_data = []
        for line in lines:
            # Extract substring and trim
            value_str = line[
                starting_position - 1 : starting_position - 1 + field_size
            ].strip()
            # Convert empty strings to None
            col_data.append(None if value_str == "" else value_str)

        data[col_name] = col_data

    df = pl.DataFrame(data)

    # Cast to proper types
    cast_exprs = []
    for col_name, value in column_mapping.items():
        col_type = value[0]
        if col_type != "object" and col_type != "string":
            cast_exprs.append(pl.col(col_name).cast(pl.Float64, strict=False))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def read_using_column_mapping(
    full_path: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Any]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_value: Optional[str] = None,
    excel_skiprows: int = 0,
    encoding: str = "utf-8-sig",
) -> Optional[pl.DataFrame]:
    """
    Router function to read different file types with column mapping
    """
    match filetype:
        case "csv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator=",",
                null_value=null_value,
                encoding=encoding,
            )
        case "tsv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="\t",
                null_value=null_value,
                encoding=encoding,
            )
        case "psv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="|",
                null_value=null_value,
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
            return read_fixed_width(full_path=full_path, column_mapping=column_mapping)
        case _:
            raise ValueError(
                f"Invalid filetype: {filetype}. "
                f"Must be one of: csv, tsv, psv, xlsx, parquet, fixed_width"
            )


def table_exists(uri: str, schema: str, table: str) -> bool:
    """Check if a table exists in the database"""
    sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = $2
        )
    """
    result = execute_sql_fetchone(uri, sql, (schema, table))
    return result[0] if result else False


def _infer_postgres_type(dtype: Any) -> str:
    """Infer PostgreSQL type from polars dtype"""
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
    """Infer column_mapping type string from polars dtype"""
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


def get_table_schema(uri: str, schema: str, table: str) -> Dict[str, str]:
    """
    Get existing table schema from PostgreSQL

    Returns:
        Dict mapping column names to PostgreSQL types
    """
    sql = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
    """
    rows = execute_sql_fetch(uri, sql, (schema, table))
    return {row[0]: row[1] for row in rows}


def validate_schema_match(uri: str, df: pl.DataFrame, schema: str, table: str) -> None:
    """
    Validate that DataFrame schema matches existing table schema

    Raises:
        ValueError: If schemas don't match
    """
    if not table_exists(uri, schema, table):
        return  # No validation needed for new tables

    existing_schema = get_table_schema(uri, schema, table)
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
    # Map polars types to PostgreSQL equivalents for comparison
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
    uri: str, df: pl.DataFrame, schema: str, table: str
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

    execute_sql(uri, create_sql)


def copy_dataframe_to_table(
    uri: str, df: pl.DataFrame, schema: str, table: str
) -> None:
    """
    Bulk load DataFrame to PostgreSQL using Polars write_database with ADBC.

    This uses Arrow-native data transfer for optimal performance.

    Raises:
        ValueError: If table exists but schema doesn't match DataFrame
    """
    # Validate schema matches if table exists
    validate_schema_match(uri, df, schema, table)

    # Ensure table exists (creates if not exists)
    create_table_from_dataframe(uri, df, schema, table)

    # Use Polars native write_database with ADBC for efficient bulk insert
    df.write_database(
        table_name=f"{schema}.{table}",
        connection=uri,
        engine="adbc",
        if_table_exists="append",
    )


def row_count_check(
    uri: str,
    schema: str,
    df: pl.DataFrame,
    full_path: str,
    unpivot_row_multiplier: Optional[int] = None,
) -> None:
    """
    Sanity check on flat file ingest comparing metadata row count to DataFrame row count
    """
    sql = f"SELECT row_count FROM {schema}.metadata WHERE full_path = $1"
    result = execute_sql_fetchone(uri, sql, (full_path,))

    metadata_row_count = result[0] if result else None
    output_df_row_count = len(df)

    if unpivot_row_multiplier:
        metadata_row_count *= unpivot_row_multiplier

    if not metadata_row_count:
        print("No metadata row count to compare against. Check passed.")
    elif metadata_row_count == output_df_row_count:
        print(
            f"Check passed {full_path}, metadata table row count {metadata_row_count} is equal to output table row count {output_df_row_count}"
        )
    else:
        raise ValueError(
            f"Check failed {full_path} since metadata table row count {metadata_row_count} is not equal to output table row count {output_df_row_count}"
        )


def update_metadata(
    uri: str,
    full_path: str,
    schema: str,
    error_message: Optional[str] = None,
    unpivot_row_multiplier: Optional[int] = None,
    ingest_runtime: Optional[int] = None,
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

    print(f"Updating metadata for {full_path}: {status}")

    # Build SQL dynamically to avoid ADBC issues with None parameters
    # ADBC has issues mapping None to PostgreSQL types
    set_clauses = [
        f"ingest_datetime = '{metadata_ingest_datetime}'",
        f"status = '{status}'",
    ]

    if error_message is not None:
        # Escape single quotes in error message
        escaped_error = error_message.replace("'", "''")
        set_clauses.append(f"error_message = '{escaped_error}'")
    else:
        set_clauses.append("error_message = NULL")

    if unpivot_row_multiplier is not None:
        set_clauses.append(f"unpivot_row_multiplier = {unpivot_row_multiplier}")

    if ingest_runtime is not None:
        set_clauses.append(f"ingest_runtime = {ingest_runtime}")

    # Escape single quotes in full_path
    escaped_full_path = full_path.replace("'", "''")

    sql = f"""
        UPDATE {schema}.metadata
        SET {", ".join(set_clauses)}
        WHERE full_path = '{escaped_full_path}'
    """

    execute_sql(uri, sql)


def update_table(
    uri: Optional[str] = None,
    resume: bool = False,
    retry_failed: bool = False,
    sample: Optional[int] = None,
    schema: Optional[str] = None,
    metadata_schema: Optional[str] = None,
    output_table: Optional[str] = None,
    output_table_naming_fn: Optional[Callable[[Path], str]] = None,
    additional_cols_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    file_list_filter_fn: Optional[Callable[[List[Path]], List[Path]]] = None,
    custom_read_fn: Optional[Callable[[str], pl.DataFrame]] = None,
    transform_fn: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
    landing_dir: Optional[str] = None,
    sql_glob: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    column_mapping_fn: Optional[
        Callable[[Path], Dict[str, Tuple[List[str], str]]]
    ] = None,
    pivot_mapping: Optional[Dict[str, Any]] = None,
    header_fn: Optional[Callable[[Path], List[str]]] = None,
    null_value: str = "",
    excel_skiprows: int = 0,
    encoding: str = "utf-8-sig",
    filesystem: Optional[Any] = None,
) -> pl.DataFrame:
    """
    Main ingestion function that reads files and writes to PostgreSQL
    """
    import time

    required_params = [landing_dir, schema, uri, filetype]
    if not custom_read_fn and not column_mapping_fn:
        required_params.append(column_mapping)

    if any(param is None for param in required_params):
        print(f"Missing required parameters: {required_params}")
        raise ValueError(
            "Required params: landing_dir, filetype, column_mapping, schema, uri. Column mapping not required if using custom_read_fn or if using column_mapping_fn"
        )

    # Normalize landing_dir to string with trailing slash for LIKE query
    landing_dir = normalize_path(landing_dir) + "/"

    # If the glob is omitted we create a glob using the filetype
    sql_glob = sql_glob or f"%.{filetype}"

    metadata_schema = metadata_schema or schema

    sql = f"""
        SELECT full_path
        FROM {metadata_schema}.metadata
        WHERE
            landing_dir LIKE $1 AND
            full_path LIKE $2 AND
            metadata_ingest_status = 'Success'
    """

    print(f"Metadata file search query: {sql}")

    rows = execute_sql_fetch(uri, sql, (landing_dir, sql_glob))
    file_list = sorted([row[0] for row in rows])

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    if resume:
        sql = f"""
            SELECT full_path
            FROM {metadata_schema}.metadata
            WHERE
                landing_dir LIKE $1 AND
                ingest_datetime IS NOT NULL
                {"AND status = 'Success'" if not retry_failed else ""}
        """

        rows = execute_sql_fetch(uri, sql, (landing_dir,))
        full_path_list = set(row[0] for row in rows)
        file_list = [f for f in file_list if f not in full_path_list]

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Output {'table' if output_table else 'schema'}: {output_table or schema} | Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
    )

    for i, file in enumerate(file_list):
        start_time = time.time()

        if sample and i == sample:
            break

        full_path = str(file)  # file is already a string

        # Check if file is in S3
        file_is_s3 = is_s3_path(full_path)
        temp_file_path = None

        # Download S3 file to temp location if needed
        if file_is_s3:
            import tempfile

            fs = get_s3_filesystem(filesystem)
            temp_file_path = Path(tempfile.gettempdir()) / path_basename(full_path)
            fs.get(full_path, str(temp_file_path))
            print(f"Downloaded {full_path} to temporary location for processing")
            read_path = str(temp_file_path)
        else:
            read_path = full_path

        header = None
        has_header = True
        if header_fn:
            header = header_fn(file)
            has_header = False

        unpivot_row_multiplier = None
        try:
            if custom_read_fn:
                df = custom_read_fn(full_path=read_path)
            else:
                column_mapping_use = (
                    column_mapping_fn(file) if column_mapping_fn else column_mapping
                )

                df = read_using_column_mapping(
                    full_path=read_path,
                    filetype=filetype,
                    column_mapping=column_mapping_use,
                    header=header,
                    has_header=has_header,
                    null_value=null_value,
                    excel_skiprows=excel_skiprows,
                    encoding=encoding,
                )

            if pivot_mapping:
                id_vars = pivot_mapping["id_vars"]
                value_vars = [col for col in df.columns if col not in id_vars]
                unpivot_row_multiplier = len(value_vars)

                df = df.unpivot(
                    index=id_vars,
                    on=value_vars,
                    variable_name=pivot_mapping["variable_column_name"],
                    value_name=pivot_mapping["value_column_name"],
                )

            # Raises exception on failure
            row_count_check(
                uri=uri,
                schema=metadata_schema,
                df=df,
                full_path=full_path,
                unpivot_row_multiplier=unpivot_row_multiplier,
            )

            if transform_fn:
                df = transform_fn(df=df)

            if additional_cols_fn:
                additional_cols_dict = additional_cols_fn(file)
                for col_name, col_value in additional_cols_dict.items():
                    df = df.with_columns(pl.lit(col_value).alias(col_name))

            df = df.with_columns(pl.lit(full_path).alias("full_path"))

            # Write to PostgreSQL
            table_name = (
                output_table_naming_fn(file) if output_table_naming_fn else output_table
            )

            # Check if table exists and delete existing records for this file
            if table_exists(uri, schema, table_name):
                delete_sql = f"DELETE FROM {schema}.{table_name} WHERE full_path = $1"
                execute_sql(uri, delete_sql, (full_path,))

            # Bulk load using Polars write_database
            copy_dataframe_to_table(uri, df, schema, table_name)

            ingest_runtime = int(time.time() - start_time)

            update_metadata(
                uri=uri,
                full_path=full_path,
                schema=metadata_schema,
                unpivot_row_multiplier=unpivot_row_multiplier,
                ingest_runtime=ingest_runtime,
            )
        except Exception as e:
            error_str = str(e)

            update_metadata(
                uri=uri,
                full_path=full_path,
                schema=metadata_schema,
                error_message=error_str,
                unpivot_row_multiplier=unpivot_row_multiplier,
            )

            # Truncate error for printing
            if len(error_str) > 200:
                error_str = error_str[:200] + "... [truncated]"
            print(f"Failed on {file} with {error_str}")
        finally:
            # Cleanup temp file if it was created
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()

    # Return metadata results using Polars read_database
    sql = f"""
        SELECT *
        FROM {metadata_schema}.metadata
        WHERE landing_dir LIKE $1 AND full_path LIKE $2
        ORDER BY ingest_datetime DESC
    """

    df = pl.read_database_uri(
        sql, uri, engine="adbc", execute_options={"parameters": (landing_dir, sql_glob)}
    )
    return df


# START OF METADATA FUNCTIONS #


def get_csv_header_and_row_count(
    encoding: str = "utf-8-sig",
    file: Optional[str] = None,
    separator: str = ",",
    has_header: bool = True,
) -> Tuple[List[str], int]:
    """
    Get header and row count from CSV file.
    Counts only non-blank lines to match pandas' skip_blank_lines=True behavior.
    File path should be a string.
    """
    import subprocess
    import csv

    file_str = str(file)

    # always return the first row even if it isnt a header
    # must ignore BOM mark
    with open(file_str, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=separator)
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
    full_path_list: Optional[List[str]] = None,
    search_dir: Optional[str] = None,
    landing_dir: Optional[str] = None,
    has_header: bool = True,
    filetype: Optional[str] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    encoding: Optional[str] = None,
    archive_glob: Optional[str] = None,
    num_search_dir_parents: int = 0,
    filesystem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Extract files from ZIP archives and add to metadata.
    All paths are strings (both local and S3).
    """
    try:
        import zipfile_deflate64 as zipfile
    except Exception:
        print("Missing zipfile_deflate64, using zipfile")
        import zipfile

    import fnmatch
    import tempfile

    # Normalize paths
    search_dir = normalize_path(search_dir) if search_dir else None
    landing_dir = normalize_path(landing_dir) if landing_dir else None

    rows = []
    num_processed = 0

    # Check if landing_dir is S3
    landing_is_s3 = is_s3_path(landing_dir) if landing_dir else False
    temp_landing_base = None
    if landing_is_s3:
        # Create temp directory for extraction before upload to S3
        temp_landing_base = Path(tempfile.gettempdir()) / "teeny_data_s3_landing"
        temp_landing_base.mkdir(exist_ok=True, parents=True)

    # Get filesystem if any S3 paths involved
    fs = None
    if landing_is_s3 or any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    # Convert full_path_list to set for fast lookup
    full_path_set = set(full_path_list) if full_path_list else set()

    for compressed in file_list:
        if sample and num_processed == sample:
            break

        compressed_str = str(compressed)

        # Download from S3 if needed
        temp_zip = None
        if is_s3_path(compressed_str):
            temp_zip = Path(tempfile.gettempdir()) / path_basename(compressed_str)
            fs.get(compressed_str, str(temp_zip))
            print(f"Downloaded {compressed_str} to temporary location")
            zip_path = str(temp_zip)
        else:
            zip_path = compressed_str

        # Get zip file stem (name without extension)
        zip_stem = path_basename(compressed_str).rsplit(".", 1)[0]

        with zipfile.ZipFile(zip_path) as zip_ref:
            namelist = [
                f for f in zip_ref.namelist() if fnmatch.fnmatch(f, archive_glob)
            ]

            for f in namelist:
                if sample and num_processed == sample:
                    break

                num_processed += 1
                file_num = (
                    f"{num_processed}/{sample}" if sample else f"{num_processed} |"
                )

                # Get parent directories from path
                if num_search_dir_parents > 0:
                    parts = compressed_str.replace("s3://", "").split("/")
                    parent_parts = parts[-(num_search_dir_parents + 1) : -1]
                    parents = "/".join(parent_parts) if parent_parts else ""
                else:
                    parents = ""

                compressed_file_basename = path_basename(f)

                # Build output path
                if parents:
                    raw_output_path = path_join(
                        landing_dir, parents, zip_stem, compressed_file_basename
                    )
                else:
                    raw_output_path = path_join(
                        landing_dir, zip_stem, compressed_file_basename
                    )

                # Handle S3 or local landing directory
                if landing_is_s3:
                    # Extract to temp directory first
                    if parents:
                        temp_output_dir = temp_landing_base / parents / zip_stem
                    else:
                        temp_output_dir = temp_landing_base / zip_stem
                    temp_output_dir.mkdir(exist_ok=True, parents=True)
                    temp_output_path = temp_output_dir / compressed_file_basename
                else:
                    # Local landing directory
                    raw_output_dir = path_parent(raw_output_path)
                    Path(raw_output_dir).mkdir(exist_ok=True, parents=True)
                    temp_output_path = Path(raw_output_path)

                # Check if already processed
                if resume and raw_output_path in full_path_set:
                    print(f"{file_num} Skipped extracting {compressed_str}:{f}")
                    continue

                try:
                    # Extract to temp location (or final location if local)
                    if landing_is_s3:
                        zip_ref.extract(f, str(temp_output_dir))
                        print(f"{file_num} Extracted {f} to temporary location")

                        # Upload to S3
                        fs.put(str(temp_output_path), raw_output_path)
                        print(f"{file_num} Uploaded to {raw_output_path}")
                    else:
                        zip_ref.extract(f, raw_output_dir)
                        print(f"{file_num} Extracted {f} to {raw_output_dir}")

                    row = get_file_metadata_row(
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        archive_full_path=compressed_str,
                        file=raw_output_path,
                        has_header=has_header,
                        encoding=encoding,
                        filesystem=filesystem,
                    )

                except Exception as e:
                    print(f"Failed on {f} with {e}")

                    row = get_file_metadata_row(
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        file=None,
                        archive_full_path=compressed_str,
                        has_header=has_header,
                        error_message=str(e),
                        encoding=encoding,
                        filesystem=filesystem,
                    )

                    # Cleanup failed files
                    if temp_output_path.exists():
                        temp_output_path.unlink()
                        print(f"Removed bad extracted file: {f}")

                    if not landing_is_s3:
                        raw_output_dir_path = Path(raw_output_dir)
                        if raw_output_dir_path.exists() and not any(
                            raw_output_dir_path.iterdir()
                        ):
                            raw_output_dir_path.rmdir()
                            print(f"Removed empty output dir: {raw_output_dir}")
                finally:
                    # Cleanup temp file if using S3
                    if landing_is_s3 and temp_output_path.exists():
                        temp_output_path.unlink()

                rows.append(row)

        # Cleanup downloaded zip file from S3
        if temp_zip and temp_zip.exists():
            temp_zip.unlink()

    # Cleanup temp landing directory if using S3
    if temp_landing_base and temp_landing_base.exists():
        import shutil

        shutil.rmtree(temp_landing_base, ignore_errors=True)

    return rows


def add_files(
    search_dir: Optional[str] = None,
    landing_dir: Optional[str] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    file_list: Optional[List[str]] = None,
    filetype: Optional[str] = None,
    has_header: bool = True,
    full_path_list: Optional[List[str]] = None,
    encoding: Optional[str] = None,
    num_search_dir_parents: int = 0,
    filesystem: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Copy files from search directory to landing directory and add to metadata.
    All paths are strings (both local and S3).
    """
    import shutil
    import tempfile

    # Normalize paths
    search_dir = normalize_path(search_dir) if search_dir else None
    landing_dir = normalize_path(landing_dir) if landing_dir else None

    # Filter out already processed files
    if full_path_list:
        full_path_set = set(full_path_list)
        file_list = [f for f in file_list if f not in full_path_set]

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
    )

    # Check if paths are S3
    landing_is_s3 = is_s3_path(landing_dir) if landing_dir else False

    # Get filesystem if any S3 paths involved
    fs = None
    if landing_is_s3 or any(is_s3_path(f) for f in file_list):
        fs = get_s3_filesystem(filesystem)

    rows = []
    for i, file in enumerate(file_list):
        if sample and i == sample:
            break

        try:
            file_str = str(file)
            file_is_s3 = is_s3_path(file_str)

            # Get filename and parent directories
            file_name = path_basename(file_str)
            if num_search_dir_parents > 0:
                # Extract parent directories from path
                parts = file_str.replace("s3://", "").split("/")
                parent_parts = parts[-(num_search_dir_parents + 1) : -1]
                parents = "/".join(parent_parts) if parent_parts else ""
            else:
                parents = ""

            # Build landing path
            if parents:
                landing_path = path_join(landing_dir, parents, file_name)
            else:
                landing_path = path_join(landing_dir, file_name)

            # Copy/upload file to landing location
            if landing_is_s3:
                if file_is_s3:
                    # S3 to S3 copy
                    fs.copy(file_str, landing_path)
                    print(f"Copied {file_str} to {landing_path}")
                else:
                    # Upload local file to S3
                    fs.put(file_str, landing_path)
                    print(f"Uploaded {file_str} to {landing_path}")
            else:
                # Local landing directory
                landing_path_dir = path_parent(landing_path)
                if landing_path_dir:
                    Path(landing_path_dir).mkdir(exist_ok=True, parents=True)

                if file_is_s3:
                    # Download from S3 to local
                    fs.get(file_str, landing_path)
                    print(f"Downloaded {file_str} to {landing_path}")
                elif landing_dir != search_dir:
                    # Copy local file
                    shutil.copy2(file_str, landing_path)

            row = get_file_metadata_row(
                search_dir=search_dir,
                landing_dir=landing_dir,
                filetype=filetype,
                file=landing_path,
                has_header=has_header,
                encoding=encoding,
                filesystem=filesystem,
            )

            print(f"Processing file {i + 1}/{sample if sample else len(file_list)}")

        except Exception as e:
            print(f"Failed on {file} with {e}")

            row = get_file_metadata_row(
                search_dir=search_dir,
                landing_dir=landing_dir,
                filetype=filetype,
                file=file,
                has_header=has_header,
                error_message=str(e),
                encoding=encoding,
                filesystem=filesystem,
            )

        rows.append(row)

    return rows


def get_file_metadata_row(
    search_dir: Optional[str] = None,
    landing_dir: Optional[str] = None,
    file: Optional[str] = None,
    filetype: Optional[str] = None,
    archive_full_path: Optional[str] = None,
    has_header: Optional[bool] = None,
    error_message: Optional[str] = None,
    encoding: Optional[str] = None,
    filesystem: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate metadata row for a file. All paths are strings.
    """
    import hashlib
    import time
    from datetime import datetime
    import tempfile

    # Normalize paths to strings with trailing slash for consistent LIKE queries
    search_dir = normalize_path(search_dir) + "/" if search_dir else None
    landing_dir = normalize_path(landing_dir) + "/" if landing_dir else None
    file_str = str(file) if file else None

    row = {
        "search_dir": search_dir,
        "landing_dir": landing_dir,
        "full_path": file_str,
        "filesize": None,
        "header": None,
        "row_count": None,
        "archive_full_path": archive_full_path,
        "file_hash": None,
        "metadata_ingest_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_ingest_status": "Failure",
        "error_message": error_message,
    }

    if error_message:
        return row

    start_time = time.time()

    # Check if file is in S3 and download to temp if needed
    file_is_s3 = is_s3_path(file_str) if file_str else False
    temp_file_path = None

    if file_is_s3:
        fs = get_s3_filesystem(filesystem)
        temp_file_path = Path(tempfile.gettempdir()) / path_basename(file_str)
        fs.get(file_str, str(temp_file_path))
        read_file = str(temp_file_path)
    else:
        read_file = file_str

    try:
        header, row_count = None, None
        match filetype:
            case "csv":
                header, row_count = get_csv_header_and_row_count(
                    file=read_file, has_header=has_header, encoding=encoding
                )
            case "tsv":
                header, row_count = get_csv_header_and_row_count(
                    file=read_file,
                    has_header=has_header,
                    separator="\t",
                    encoding=encoding,
                )
            case "psv":
                header, row_count = get_csv_header_and_row_count(
                    file=read_file,
                    has_header=has_header,
                    separator="|",
                    encoding=encoding,
                )
            case "fixed_width":
                header, row_count = get_csv_header_and_row_count(
                    file=read_file, has_header=False, encoding=encoding
                )
            case "xlsx":
                df = pl.read_excel(read_file)
                header, row_count = list(df.columns), len(df)
            case "parquet":
                df = pl.read_parquet(read_file)
                header, row_count = list(df.columns), len(df)
            case "xml":
                # there isnt really a standard way of getting these values
                header, row_count = None, None
            case _:
                raise Exception(f"Unsupported filetype: {filetype}")

        row["header"], row["row_count"] = header, row_count
        row["file_hash"] = hashlib.md5(open(read_file, "rb").read()).hexdigest()
        row["filesize"] = Path(read_file).stat().st_size
        row["metadata_ingest_status"] = "Success"

        filename = path_basename(file_str) if file_str else "unknown"
        print(
            f"Row count: {row_count} Filename: {filename} | Runtime: {time.time() - start_time:.2f}"
        )

    finally:
        # Cleanup temp file if it was created
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()

    return row


def add_files_to_metadata_table(uri: str, **kwargs: Any) -> pl.DataFrame:
    """
    Add files to metadata table, creating it if necessary
    """
    schema = kwargs.pop("schema", None)
    if not schema:
        raise Exception("You must provide the schema as a param")

    glob = kwargs.pop("glob", None)
    compression_type = kwargs.pop("compression_type", None)
    filetype = kwargs["filetype"]
    archive_glob = kwargs.get("archive_glob", None)

    if not glob:
        glob = f"*.{compression_type}" if compression_type else f"*.{filetype}"

    sql_glob = glob.replace("*", "%")

    if compression_type:
        if not archive_glob:
            archive_glob = f"*.{filetype}"
            kwargs["archive_glob"] = archive_glob
        sql_glob = archive_glob.replace("*", "%")

    # Normalize paths to strings
    landing_dir = normalize_path(kwargs["landing_dir"])
    search_dir = normalize_path(kwargs["search_dir"])

    # Handle S3 or local paths for file listing
    filesystem = kwargs.get("filesystem", None)
    if is_s3_path(search_dir):
        # S3 path - use s3fs.glob()
        fs = get_s3_filesystem(filesystem)
        s3_glob_pattern = f"{search_dir}/**/{glob}"
        s3_paths = fs.glob(s3_glob_pattern)
        # s3fs returns paths without s3:// prefix, add it back
        file_list = [f"s3://{p}" if not p.startswith("s3://") else p for p in s3_paths]
    else:
        # Local path - use Path.rglob but convert results to strings
        file_list = [f.as_posix() for f in Path(search_dir).rglob(glob)]

    kwargs["search_dir"] = search_dir
    kwargs["landing_dir"] = landing_dir

    file_list_filter_fn = kwargs.pop("file_list_filter_fn", None)
    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    file_list = sorted(file_list)
    kwargs["file_list"] = file_list

    output_table = f"{schema}.metadata"

    # Create metadata table if it doesn't exist
    if not table_exists(uri, schema, "metadata"):
        create_table_sql = f"""
            CREATE TABLE {output_table} (
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
        """
        execute_sql(uri, create_table_sql)

    resume = kwargs.get("resume", False)
    retry_failed = kwargs.pop("retry_failed", False)

    kwargs["full_path_list"] = []
    if resume:
        sql = f"""
            SELECT full_path
            FROM {output_table}
            WHERE
                search_dir LIKE $1 AND
                full_path LIKE $2
                {"AND metadata_ingest_status = 'Success'" if not retry_failed else ""}
        """

        rows = execute_sql_fetch(uri, sql, (search_dir, sql_glob))
        kwargs["full_path_list"] = [row[0] for row in rows]

    match compression_type:
        case "zip":
            rows = extract_and_add_zip_files(**kwargs)
        case None:
            rows = add_files(**kwargs)
        case _:
            raise Exception("Unsupported compression type")

    if len(rows) == 0:
        print("Did not add any files to metadata table")
    else:
        rows_sorted = sorted(rows, key=lambda x: x["full_path"] or "")

        # Upsert to metadata table using PostgreSQL's ON CONFLICT
        # Build SQL dynamically to avoid ADBC issues with None parameters and type mismatches
        for row in rows_sorted:
            # Helper to escape single quotes and handle None
            def sql_str(val):
                if val is None:
                    return "NULL"
                return "'" + str(val).replace("'", "''") + "'"

            def sql_int(val):
                if val is None:
                    return "NULL"
                return str(int(val))

            # Convert list to PostgreSQL array format
            header_list = row["header"] if row["header"] else None
            if header_list:
                header_sql = "ARRAY[" + ", ".join(sql_str(h) for h in header_list) + "]"
            else:
                header_sql = "NULL"

            upsert_sql = f"""
                INSERT INTO {output_table}
                (search_dir, landing_dir, full_path, filesize, header, row_count,
                 archive_full_path, file_hash, metadata_ingest_datetime, metadata_ingest_status)
                VALUES
                ({sql_str(row["search_dir"])}, {sql_str(row["landing_dir"])}, {sql_str(row["full_path"])},
                 {sql_int(row["filesize"])}, {header_sql}, {sql_int(row["row_count"])},
                 {sql_str(row["archive_full_path"])}, {sql_str(row["file_hash"])},
                 {sql_str(row["metadata_ingest_datetime"])}::timestamp, {sql_str(row["metadata_ingest_status"])})
                ON CONFLICT (full_path)
                DO UPDATE SET
                    search_dir = EXCLUDED.search_dir,
                    landing_dir = EXCLUDED.landing_dir,
                    filesize = EXCLUDED.filesize,
                    header = EXCLUDED.header,
                    row_count = EXCLUDED.row_count,
                    archive_full_path = EXCLUDED.archive_full_path,
                    file_hash = EXCLUDED.file_hash,
                    metadata_ingest_datetime = EXCLUDED.metadata_ingest_datetime,
                    metadata_ingest_status = EXCLUDED.metadata_ingest_status
            """

            execute_sql(uri, upsert_sql)

    # Return metadata results using Polars read_database
    sql = f"""
        SELECT *
        FROM {output_table}
        WHERE search_dir LIKE $1 AND full_path LIKE $2
        ORDER BY metadata_ingest_datetime DESC
    """

    df = pl.read_database_uri(
        sql, uri, engine="adbc", execute_options={"parameters": (search_dir, sql_glob)}
    )
    return df


def drop_search_dir(
    uri: str,
    search_dir: str,
    schema: str,
) -> None:
    """
    Remove all files from a search directory from metadata
    """
    search_dir = Path(search_dir).as_posix()

    sql = f"SELECT COUNT(*) FROM {schema}.metadata WHERE search_dir LIKE $1"
    result = execute_sql_fetchone(uri, sql, (search_dir,))
    count_before = result[0] if result else 0

    print(f"Rows before drop: {count_before}")

    delete_sql = f"DELETE FROM {schema}.metadata WHERE search_dir LIKE $1"

    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            cur.execute(delete_sql, (search_dir,))
            rowcount = cur.rowcount
            print(f"Deleted {rowcount} rows")
        conn.commit()

    result = execute_sql_fetchone(uri, sql, (search_dir,))
    count_after = result[0] if result else 0

    print(f"Rows after drop: {count_after}")


def drop_partition(
    uri: str,
    table: str,
    partition_key: str,
    schema: str,
) -> None:
    """
    Delete records matching partition key from table
    """
    print(
        f"Running: DELETE FROM {schema}.{table} WHERE full_path LIKE '{partition_key}'"
    )

    delete_sql = f"DELETE FROM {schema}.{table} WHERE full_path LIKE $1"

    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            cur.execute(delete_sql, (partition_key,))
            print(f"Deleted {cur.rowcount} rows")
        conn.commit()

    # PostgreSQL doesn't need vacuum the same way Spark does
    # VACUUM reclaims storage, but happens automatically in most cases
    print("Note: PostgreSQL autovacuum will reclaim space automatically")


def drop_file_from_metadata_and_table(
    uri: str,
    table: str,
    full_path: str,
    schema: str,
) -> None:
    """
    Remove a file from both metadata and data table
    """
    full_path = Path(full_path).as_posix()

    # Delete from metadata
    delete_metadata_sql = f"DELETE FROM {schema}.metadata WHERE full_path = $1"

    with get_connection(uri) as conn:
        with conn.cursor() as cur:
            cur.execute(delete_metadata_sql, (full_path,))
            print(f"Deleted {cur.rowcount} rows from metadata")
        conn.commit()

    # Delete from data table
    drop_partition(uri=uri, table=table, partition_key=full_path, schema=schema)


# CLI SCHEMA INFERENCE FUNCTIONS


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case

    Examples:
        "FirstName" -> "first_name"
        "User ID" -> "user_id"
        "price-per-unit" -> "price_per_unit"
        "totalAmount" -> "total_amount"
    """
    import re

    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s\-]+", "_", name)

    # Insert underscore before uppercase letters that follow lowercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    # Convert to lowercase
    name = name.lower()

    # Remove any duplicate underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


def infer_schema_from_file(
    file_path: str,
    filetype: Optional[str] = None,
    separator: str = ",",
    has_header: bool = True,
    encoding: str = "utf-8-sig",
    excel_skiprows: int = 0,
    sample_rows: int = 1000,
) -> Dict[str, Tuple[List[str], str]]:
    """
    Infer schema from file using polars type inference

    Args:
        file_path: Path to the file
        filetype: File type (csv, tsv, psv, xlsx, parquet)
        separator: Delimiter for text files
        has_header: Whether the file has a header row
        encoding: File encoding
        excel_skiprows: Rows to skip in Excel files
        sample_rows: Number of rows to sample for type inference

    Returns:
        Column mapping dictionary in the format:
        {
            "column_name": ([], "type_string"),
            ...
        }
    """
    path = Path(file_path)

    # Auto-detect filetype from extension if not provided
    if not filetype:
        ext = path.suffix.lower().lstrip(".")
        filetype = ext if ext in ["csv", "tsv", "psv", "xlsx", "parquet"] else "csv"

    # Read file with polars type inference
    df = None

    if filetype in ["csv", "tsv", "psv"]:
        # Determine separator
        if filetype == "tsv":
            separator = "\t"
        elif filetype == "psv":
            separator = "|"

        # Handle encoding for polars
        polars_encoding = encoding.replace("-sig", "").replace("-", "")
        if polars_encoding.lower() not in ["utf8", "utf-8"]:
            polars_encoding = "utf8-lossy"

        # Read with polars type inference
        df = pl.read_csv(
            file_path,
            separator=separator,
            has_header=has_header,
            encoding=polars_encoding,
            n_rows=sample_rows,
            infer_schema_length=sample_rows,
        )

        # If no header, generate column names
        if not has_header:
            new_columns = [f"col_{i}" for i in range(len(df.columns))]
            df = df.rename(dict(zip(df.columns, new_columns)))

    elif filetype == "xlsx":
        df = pl.read_excel(
            file_path,
            read_options={"skip_rows": excel_skiprows, "n_rows": sample_rows},
        )

    elif filetype == "parquet":
        # Parquet already has type information
        df = pl.read_parquet(file_path)
        # Sample rows
        df = df.head(sample_rows)

    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    # Build column mapping from inferred types
    column_mapping = {}

    for col in df.columns:
        type_string = _infer_column_mapping_type(df[col].dtype)

        # Convert column name to snake_case
        original_col = str(col)
        snake_case_col = to_snake_case(original_col)

        # Format: "snake_case_name": (["OriginalName"], "type_string")
        # Include original column name in the list of possible column names
        # If snake_case conversion doesn't change the name, use empty list
        if snake_case_col == original_col:
            column_mapping[snake_case_col] = ([], type_string)
        else:
            column_mapping[snake_case_col] = ([original_col], type_string)

    return column_mapping


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Infer column schema from data files and output column mapping JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/table_functions_postgres.py data/raw/my_file.csv
  python src/table_functions_postgres.py data/raw/my_file.csv --filetype csv --separator ","
  python src/table_functions_postgres.py data/raw/my_file.psv --filetype psv
  python src/table_functions_postgres.py data/raw/my_file.xlsx --filetype xlsx --no-header
  python src/table_functions_postgres.py data/raw/my_file.parquet --sample-rows 5000
        """,
    )

    parser.add_argument(
        "file_path",
        help="Path to the input file",
    )

    parser.add_argument(
        "--filetype",
        choices=["csv", "tsv", "psv", "xlsx", "parquet"],
        help="File type (auto-detected from extension if not provided)",
    )

    parser.add_argument(
        "--separator",
        default=",",
        help="Column separator for text files (default: ',')",
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="File has no header row",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="File encoding (default: utf-8-sig)",
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
        default=1000,
        help="Number of rows to sample for type inference (default: 1000)",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        exit(1)

    # Infer schema
    try:
        column_mapping = infer_schema_from_file(
            file_path=args.file_path,
            filetype=args.filetype,
            separator=args.separator,
            has_header=not args.no_header,
            encoding=args.encoding,
            excel_skiprows=args.excel_skiprows,
            sample_rows=args.sample_rows,
        )

        # Output as JSON
        if args.pretty:
            print(json.dumps(column_mapping, indent=2))
        else:
            print(json.dumps(column_mapping))

    except Exception as e:
        print(f"Error inferring schema: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
