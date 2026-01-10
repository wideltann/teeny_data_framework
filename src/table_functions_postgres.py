"""
Data ingestion framework using pandas and PostgreSQL
Pure psycopg implementation for efficient bulk loading
"""

import pandas as pd
import numpy as np
import psycopg
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from pathlib import Path


def create_pandas_selection_from_header(
    header: List[str], column_mapping: Dict[str, Tuple[List[str], str]]
) -> Tuple[List[Union[str, Tuple[str, str]]], Dict[str, str], Dict[str, str]]:
    """
    Creates column selections and dtypes for pandas DataFrame
    Handles column renaming, missing columns (fills with None), and default types
    """
    selections = []
    dtypes = {}
    processed_cols = []
    default_type = None
    missing_cols = {}  # track columns that need to be added as None

    # example mapping:
    # column_mapping = {
    #    "block_fips": ([], "string"),
    #    "county_fips": ([], "string"),
    #    "state_abbr": ([], "string"),
    #    "default": ([], "float64"),
    # }
    for alias, (possible_cols, col_type) in column_mapping.items():
        if alias == "default":
            default_type = col_type
            continue

        # we want to keep the original column name and there's no column name variation
        # if alias not in header we need to add as None
        if not possible_cols and alias in header:
            col_name = alias
            selections.append(col_name)
            dtypes[col_name] = col_type
            processed_cols.append(col_name)
        # there is variation in the column name across time so we pass
        # in a list of column names that mean the same thing
        elif any(col_name in header for col_name in possible_cols):
            col_name = next(
                col_name for col_name in possible_cols if col_name in header
            )
            selections.append((col_name, alias))  # (original, renamed)
            dtypes[alias] = col_type
            processed_cols.append(col_name)
        # in order to combine this dataframe with other dataframes in a single table
        # all columns in the table schema must exist in the data frame
        else:
            missing_cols[alias] = col_type

    # all cols in the header but not in the column mapping will be assigned this type
    # if you set a default then you cant delete columns by not selecting them
    if default_type is not None:
        remaining_cols = [
            col_name for col_name in header if col_name not in processed_cols
        ]

        for col_name in remaining_cols:
            selections.append(col_name)
            dtypes[col_name] = default_type

    return selections, dtypes, missing_cols


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

    selections, dtypes, missing_cols = create_pandas_selection_from_header(
        header=header, column_mapping=column_mapping
    )

    # Build rename mapping
    rename_dict = {}
    for sel in selections:
        if isinstance(sel, tuple):
            orig, new = sel
            rename_dict[orig] = new

    """
    We have to use dtypes because that's when the data is initially being read.
    If we did the pandas auto read then stuff like ids with left padded 0s would become ints instead of strings
    """
    df = pd.read_excel(full_path, skiprows=excel_skiprows, dtype=dtypes)

    # Apply renames
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Add missing columns with None values and proper type
    for col_name, col_type in missing_cols.items():
        df[col_name] = None
        if col_type != "object":
            df[col_name] = df[col_name].astype(col_type)

    return df


def read_parquet(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
) -> pd.DataFrame:
    """
    Read parquet file and return pandas DataFrame with column mapping
    """
    df = pd.read_parquet(full_path)

    # since parquet has type information, we dont need to set that initially
    # but we do want to run the rename and selection
    selections, dtypes, missing_cols = create_pandas_selection_from_header(
        header=df.columns.tolist(), column_mapping=column_mapping
    )

    # Apply column renames
    rename_dict = {orig: new for orig, new in selections if isinstance(orig, tuple)}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Add missing columns
    for col_name, col_type in missing_cols.items():
        df[col_name] = None
        df[col_name] = df[col_name].astype(col_type)

    return df


def read_csv(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[List[str], str]]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_value: Optional[str] = None,
    separator: str = ",",
) -> pd.DataFrame:
    """
    Read CSV file and return pandas DataFrame with proper schema and column mapping
    """
    if not header:
        # some files start with a BOM (byte-order-mark)
        # if we dont use sig then that BOM will get included in the header
        # the character will show up as \ufeff if the encoding is utf-8 (which is the default)
        with open(full_path, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip().split(separator)

    selections, dtypes, missing_cols = create_pandas_selection_from_header(
        header=header, column_mapping=column_mapping
    )

    """
    - Always include the header because otherwise the header will be in the data
    - Always provide the dtypes because otherwise it will auto infer
        - Also the inferred schema is not always what you want
            - Some ids look like they should be integers but really they should
                be varchars
    """
    # Build rename mapping
    rename_dict = {}
    keep_cols = []
    for sel in selections:
        if isinstance(sel, tuple):
            orig, new = sel
            keep_cols.append(orig)
            rename_dict[orig] = new
        else:
            keep_cols.append(sel)

    df = pd.read_csv(
        full_path,
        sep=separator,
        header=0 if has_header else None,
        names=header if not has_header else None,
        dtype=dtypes,
        na_values=null_value,
        keep_default_na=False if null_value == "" else True,
    )

    # Apply renames
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Add missing columns with None values and proper type
    for col_name, col_type in missing_cols.items():
        df[col_name] = None
        if col_type != "object":
            df[col_name] = df[col_name].astype(col_type)

    return df


def read_fixed_width(
    full_path: Optional[str] = None,
    column_mapping: Optional[Dict[str, Tuple[str, int, int]]] = None,
) -> pd.DataFrame:
    """
    Read fixed-width file and return pandas DataFrame

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

    df = pd.DataFrame(data)

    # Cast to proper types
    for col_name, value in column_mapping.items():
        col_type = value[0]
        if col_type != "object" and col_type != "string":
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    return df


def read_using_column_mapping(
    full_path: Optional[str] = None,
    filetype: Optional[str] = None,
    column_mapping: Optional[Dict[str, Any]] = None,
    header: Optional[List[str]] = None,
    has_header: bool = False,
    null_value: Optional[str] = None,
    excel_skiprows: int = 0,
) -> Optional[pd.DataFrame]:
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
            )
        case "tsv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="\t",
                null_value=null_value,
            )
        case "psv":
            return read_csv(
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="|",
                null_value=null_value,
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
            print("Invalid filetype")


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


def infer_postgres_type(dtype: Any) -> str:
    """Infer PostgreSQL type from pandas dtype"""
    dtype_str = str(dtype)
    if "int" in dtype_str:
        return "BIGINT"
    elif "float" in dtype_str:
        return "DOUBLE PRECISION"
    elif "bool" in dtype_str:
        return "BOOLEAN"
    elif "datetime" in dtype_str:
        return "TIMESTAMP"
    else:
        return "TEXT"


def create_table_from_dataframe(
    conn: psycopg.Connection, df: pd.DataFrame, schema: str, table: str
) -> None:
    """Create table from DataFrame schema if it doesn't exist"""
    # Build column definitions
    columns = []
    for col in df.columns:
        pg_type = infer_postgres_type(df[col].dtype)
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
    """
    # Ensure table exists
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
    conn: Optional[psycopg.Connection] = None,
    schema: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    full_path: Optional[str] = None,
    unpivot_row_multiplier: Optional[int] = None,
) -> None:
    """
    Sanity check on flat file ingest comparing metadata row count to DataFrame row count
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT row_count FROM {schema}.metadata WHERE full_path = %s",
            (full_path,),
        )
        result = cur.fetchone()

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
    conn: Optional[psycopg.Connection] = None,
    full_path: Optional[str] = None,
    schema: Optional[str] = None,
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
        max_length = 500
        if len(error_message) > max_length:
            error_message = error_message[:max_length] + "... [truncated]"
        # if these characters are in the sql string itll break
        error_message = error_message.replace("'", "").replace("`", "")

    query = f"""
        UPDATE {schema}.metadata
        SET
            ingest_datetime = %s,
            status = %s,
            error_message = %s,
            unpivot_row_multiplier = %s,
            ingest_runtime = %s
        WHERE full_path = %s
    """

    print(f"Updating metadata for {full_path}: {status}")

    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                metadata_ingest_datetime,
                status,
                error_message,
                unpivot_row_multiplier,
                ingest_runtime,
                full_path,
            ),
        )

        if cur.rowcount > 1:
            print(
                f"Updated more than one row, you should be concerned. File: {full_path}"
            )


def update_table(
    conn: Optional[psycopg.Connection] = None,
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
) -> pd.DataFrame:
    """
    Main ingestion function that reads files and writes to PostgreSQL
    """
    import time

    required_params = [landing_dir, schema, conn, filetype]
    if not custom_read_fn and not column_mapping_fn:
        required_params.append(column_mapping)

    if any(param is None for param in required_params):
        print(required_params)
        raise ValueError(
            "Required params: landing_dir, filetype, column_mapping, schema, conn. Column mapping not required if using custom_read_fn or if using column_mapping_fn"
        )

    # remove potential trailing slash
    landing_dir = Path(landing_dir).as_posix()

    # If the glob is omitted we create a glob using the filetype
    sql_glob = sql_glob or f"%.{filetype}"

    metadata_schema = metadata_schema or schema

    query = f"""
        SELECT full_path
        FROM {metadata_schema}.metadata
        WHERE
            landing_dir LIKE %s AND
            full_path LIKE %s AND
            metadata_ingest_status = 'Success'
    """

    print(f"metadata file search query: {query}")

    with conn.cursor() as cur:
        cur.execute(query, (landing_dir, sql_glob))
        rows = cur.fetchall()

    pdf_list = [row[0] for row in rows]
    file_list = [Path(f) for f in pdf_list]
    file_list = sorted(file_list)

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    if resume:
        query = f"""
            SELECT full_path
            FROM {metadata_schema}.metadata
            WHERE
                landing_dir LIKE %s AND
                ingest_datetime IS NOT NULL
                {"AND status = 'Success'" if not retry_failed else ""}
        """

        with conn.cursor() as cur:
            cur.execute(query, (landing_dir,))
            rows = cur.fetchall()

        full_path_list = [Path(row[0]) for row in rows]
        file_list = list(set(file_list) - set(full_path_list))

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Output {'table' if output_table else 'schema'}: {output_table or schema} | Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
    )

    for i, file in enumerate(file_list):
        start_time = time.time()

        if sample and i == sample:
            break

        full_path = file.as_posix()

        header = None
        has_header = True
        if header_fn:
            header = header_fn(file)
            has_header = False

        unpivot_row_multiplier = None
        try:
            if custom_read_fn:
                df = custom_read_fn(full_path=full_path)
            else:
                column_mapping_use = (
                    column_mapping_fn(file) if column_mapping_fn else column_mapping
                )

                df = read_using_column_mapping(
                    full_path=full_path,
                    filetype=filetype,
                    column_mapping=column_mapping_use,
                    header=header,
                    has_header=has_header,
                    null_value=null_value,
                    excel_skiprows=excel_skiprows,
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

            # Raises exception on failure
            row_count_check(
                conn=conn,
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
                    df[col_name] = col_value

            df["full_path"] = full_path

            # Write to PostgreSQL
            table_name = (
                output_table_naming_fn(file) if output_table_naming_fn else output_table
            )

            # Check if table exists and delete existing records for this file
            if table_exists(conn, schema, table_name):
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {schema}.{table_name} WHERE full_path = %s",
                        (full_path,),
                    )
                conn.commit()

            # Bulk load using COPY
            copy_dataframe_to_table(conn, df, schema, table_name)
            conn.commit()

            ingest_runtime = int(time.time() - start_time)

            update_metadata(
                conn=conn,
                full_path=full_path,
                schema=metadata_schema,
                unpivot_row_multiplier=unpivot_row_multiplier,
                ingest_runtime=ingest_runtime,
            )
            conn.commit()
        except Exception as e:
            error_str = str(e)
            conn.rollback()  # Rollback failed transaction

            update_metadata(
                conn=conn,
                full_path=full_path,
                schema=metadata_schema,
                error_message=error_str,
                unpivot_row_multiplier=unpivot_row_multiplier,
            )
            conn.commit()

            # Truncate error for printing
            max_print_length = 200
            if len(error_str) > max_print_length:
                error_str = error_str[:max_print_length] + "... [truncated]"
            print(f"Failed on {file} with {error_str}")

    query = f"""
        SELECT *
        FROM {metadata_schema}.metadata
        WHERE landing_dir LIKE %s AND full_path LIKE %s
        ORDER BY ingest_datetime DESC
    """

    with conn.cursor() as cur:
        cur.execute(query, (landing_dir, sql_glob))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    return df


# START OF METADATA FUNCTIONS #


def get_csv_header_and_row_count(
    encoding: str = "utf-8-sig",
    file: Optional[Path] = None,
    separator: str = ",",
    has_header: bool = True,
) -> Tuple[List[str], int]:
    """
    Get header and row count from CSV file
    Counts only non-blank lines to match pandas' skip_blank_lines=True behavior
    """
    import subprocess

    # always return the first row even if it isnt a header
    # must ignore BOM mark
    with open(file, "r", encoding=encoding) as f:
        header = f.readline().strip().split(separator)

    # Count non-blank lines to match pandas behavior (skip_blank_lines=True)
    # Using grep -c to count non-empty lines
    subtract_header_row = 1 if has_header else 0
    row_count = (
        int(
            subprocess.check_output(
                ["grep", "-c", "^[^[:space:]]", file.as_posix()]
            ).split()[0]
        )
        - subtract_header_row
    )

    return header, row_count


def extract_and_add_zip_files(
    conn: Optional[psycopg.Connection] = None,
    file_list: Optional[List[Path]] = None,
    full_path_list: Optional[List[Path]] = None,
    search_dir: Optional[Path] = None,
    landing_dir: Optional[Path] = None,
    compression_type: Optional[str] = None,
    has_header: bool = True,
    filetype: Optional[str] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    encoding: Optional[str] = None,
    archive_glob: Optional[str] = None,
    num_search_dir_parents: int = 0,
) -> List[Dict[str, Any]]:
    """
    Extract files from ZIP archives and add to metadata
    """
    try:
        import zipfile_deflate64 as zipfile
    except Exception:
        print(f"Missing zipfile_deflate64, using zipfile")
        import zipfile

    import fnmatch

    rows = []
    num_processed = 0

    for compressed in file_list:
        if sample and num_processed == sample:
            break

        with zipfile.ZipFile(compressed) as zip_ref:
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

                parents = (
                    Path(*compressed.parts[-(num_search_dir_parents + 1) : -1])
                    if num_search_dir_parents > 0
                    else Path()
                )
                raw_output_dir = landing_dir / parents / compressed.stem
                raw_output_dir.mkdir(exist_ok=True, parents=True)

                compressed_file_basename = Path(f).name
                raw_output_path = raw_output_dir / compressed_file_basename

                if resume and raw_output_path in full_path_list:
                    print(f"{file_num} Skipped extracting {compressed}:{f}")
                    continue

                try:
                    zip_ref.extract(f, raw_output_dir)
                    print(f"{file_num} Extracted {f} to {raw_output_dir}")

                    row = get_file_metadata_row(
                        conn=conn,
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        archive_full_path=compressed.as_posix(),
                        file=raw_output_path,
                        has_header=has_header,
                        encoding=encoding,
                    )

                except Exception as e:
                    print(f"Failed on {f} with {e}")

                    raw_output_path = None
                    row = get_file_metadata_row(
                        conn=conn,
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        file=raw_output_path,
                        archive_full_path=compressed.as_posix(),
                        has_header=has_header,
                        error_message=str(e),
                        encoding=encoding,
                    )

                    if raw_output_path and raw_output_path.exists():
                        raw_output_path.unlink()
                        print(f"Removed bad extracted file: {f}")

                    if not any(raw_output_dir.iterdir()):
                        raw_output_dir.rmdir()
                        print(f"Removed empty output dir: {raw_output_dir}")

                rows.append(row)

    return rows


def add_files(
    conn: Optional[psycopg.Connection] = None,
    search_dir: Optional[Path] = None,
    landing_dir: Optional[Path] = None,
    resume: Optional[bool] = None,
    sample: Optional[int] = None,
    file_list: Optional[List[Path]] = None,
    filetype: Optional[str] = None,
    has_header: bool = True,
    full_path_list: Optional[List[Path]] = None,
    encoding: Optional[str] = None,
    num_search_dir_parents: int = 0,
) -> List[Dict[str, Any]]:
    """
    Copy files from search directory to landing directory and add to metadata
    """
    import shutil

    if full_path_list:
        file_list = set(file_list) - set(full_path_list)

    total_files_to_be_processed = sample if sample else len(file_list)

    print(
        f"Num files being processed: {total_files_to_be_processed} out of {len(file_list)} {'new files' if resume else 'total files'}"
    )

    rows = []
    for i, file in enumerate(file_list):
        if sample and i == sample:
            break

        try:
            parents = (
                Path(*file.parts[-(num_search_dir_parents + 1) : -1])
                if num_search_dir_parents > 0
                else Path()
            )
            landing_path_dir = landing_dir / parents
            landing_path_dir.mkdir(exist_ok=True, parents=True)
            landing_path = landing_path_dir / file.name

            if landing_dir != search_dir:
                shutil.copy2(file, landing_path)

            row = get_file_metadata_row(
                conn=conn,
                search_dir=search_dir,
                landing_dir=landing_dir,
                filetype=filetype,
                file=landing_path,
                has_header=has_header,
                encoding=encoding,
            )

            print(f"{i + 1}/{sample if sample else len(file_list)}")

        except Exception as e:
            print(f"Failed on {file} with {e}")

            row = get_file_metadata_row(
                conn=conn,
                search_dir=search_dir,
                landing_dir=landing_dir,
                filetype=filetype,
                file=file,
                has_header=has_header,
                error_message=str(e),
                encoding=encoding,
            )

        rows.append(row)

    return rows


def get_file_metadata_row(
    conn: Optional[psycopg.Connection] = None,
    search_dir: Optional[Path] = None,
    landing_dir: Optional[Path] = None,
    file: Optional[Path] = None,
    filetype: Optional[str] = None,
    compression_type: Optional[str] = None,
    archive_full_path: Optional[str] = None,
    has_header: Optional[bool] = None,
    error_message: Optional[str] = None,
    encoding: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate metadata row for a file
    """
    import hashlib
    import time
    from datetime import datetime

    row = {
        "search_dir": search_dir.as_posix(),
        "landing_dir": landing_dir.as_posix(),
        "full_path": file.as_posix() if file else None,
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

    header, row_count = None, None
    match filetype:
        case "csv":
            header, row_count = get_csv_header_and_row_count(
                file=file, has_header=has_header, encoding=encoding
            )
        case "tsv":
            header, row_count = get_csv_header_and_row_count(
                file=file, has_header=has_header, separator="\t", encoding=encoding
            )
        case "psv":
            header, row_count = get_csv_header_and_row_count(
                file=file, has_header=has_header, separator="|", encoding=encoding
            )
        case "fixed_width":
            header, row_count = get_csv_header_and_row_count(
                file=file, has_header=False, encoding=encoding
            )
        case "xlsx":
            pdf = pd.read_excel(file)
            header, row_count = list(pdf.columns), len(pdf)
        case "parquet":
            pdf = pd.read_parquet(file)
            header, row_count = list(pdf.columns), len(pdf)
        case "xml":
            # there isnt really a standard way of getting these values
            header, row_count = None, None
        case _:
            raise Exception(f"Unsupported filetype: {filetype}")

    row["header"], row["row_count"] = header, row_count
    row["file_hash"] = hashlib.md5(open(file, "rb").read()).hexdigest()
    row["filesize"] = file.stat().st_size
    row["metadata_ingest_status"] = "Success"

    print(
        f"Row count: {row_count} Filename: {file.name} | Runtime: {time.time() - start_time:.2f}"
    )

    return row


def add_files_to_metadata_table(**kwargs: Any) -> pd.DataFrame:
    """
    Add files to metadata table, creating it if necessary
    """
    schema = kwargs.pop("schema", None)
    if not schema:
        raise Exception("You must provide the schema as a param")

    glob = kwargs.pop("glob", None)
    compression_type = kwargs.get("compression_type", None)
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

    landing_dir = Path(kwargs["landing_dir"])
    search_dir = Path(kwargs["search_dir"])
    file_list = [f for f in search_dir.rglob(glob)]

    kwargs["search_dir"] = search_dir
    kwargs["landing_dir"] = landing_dir

    file_list_filter_fn = kwargs.get("file_list_filter_fn", None)
    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    file_list = sorted(file_list)
    kwargs["file_list"] = file_list

    output_table = f"{schema}.metadata"
    conn = kwargs["conn"]

    # Create metadata table if it doesn't exist
    if not table_exists(conn, schema, "metadata"):
        create_table_query = f"""
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

        with conn.cursor() as cur:
            cur.execute(create_table_query)
        conn.commit()

    resume = kwargs.get("resume", False)
    retry_failed = kwargs.pop("retry_failed", False)

    kwargs["full_path_list"] = []
    if resume:
        query = f"""
            SELECT full_path
            FROM {output_table}
            WHERE
                search_dir LIKE %s AND
                full_path LIKE %s
                {"AND metadata_ingest_status = 'Success'" if not retry_failed else ""}
        """

        with conn.cursor() as cur:
            cur.execute(query, (search_dir.as_posix(), sql_glob))
            rows = cur.fetchall()

        full_path_list = [Path(row[0]) for row in rows]
        kwargs["full_path_list"] = full_path_list

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
        pdf = pd.DataFrame(sorted(rows, key=lambda x: x["full_path"]))

        # Upsert to metadata table using PostgreSQL's ON CONFLICT
        for _, row in pdf.iterrows():
            # Convert list to PostgreSQL array format
            header_list = row["header"] if row["header"] else None

            upsert_query = f"""
                INSERT INTO {output_table}
                (search_dir, landing_dir, full_path, filesize, header, row_count,
                 archive_full_path, file_hash, metadata_ingest_datetime, metadata_ingest_status)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

            with conn.cursor() as cur:
                cur.execute(
                    upsert_query,
                    (
                        row["search_dir"],
                        row["landing_dir"],
                        row["full_path"],
                        row["filesize"],
                        header_list,
                        row["row_count"],
                        row["archive_full_path"],
                        row["file_hash"],
                        row["metadata_ingest_datetime"],
                        row["metadata_ingest_status"],
                    ),
                )
        conn.commit()

    # Return metadata results
    query = f"""
        SELECT *
        FROM {output_table}
        WHERE search_dir LIKE %s AND full_path LIKE %s
        ORDER BY metadata_ingest_datetime DESC
    """

    with conn.cursor() as cur:
        cur.execute(query, (search_dir.as_posix(), sql_glob))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        rows_df = pd.DataFrame(rows, columns=columns)

    return rows_df


def drop_search_dir(
    conn: Optional[psycopg.Connection] = None,
    search_dir: Optional[str] = None,
    schema: Optional[str] = None,
) -> None:
    """
    Remove all files from a search directory from metadata
    """
    if not schema:
        raise Exception("Must specify schema")

    search_dir = Path(search_dir).as_posix()

    query = f"SELECT COUNT(*) FROM {schema}.metadata WHERE search_dir LIKE %s"

    with conn.cursor() as cur:
        cur.execute(query, (search_dir,))
        count_before = cur.fetchone()[0]

    print(f"Rows before drop: {count_before}")

    delete_query = f"DELETE FROM {schema}.metadata WHERE search_dir LIKE %s"

    with conn.cursor() as cur:
        cur.execute(delete_query, (search_dir,))
        rowcount = cur.rowcount
        print(f"Deleted {rowcount} rows")

    conn.commit()

    with conn.cursor() as cur:
        cur.execute(query, (search_dir,))
        count_after = cur.fetchone()[0]

    print(f"Rows after drop: {count_after}")


def drop_partition(
    conn: Optional[psycopg.Connection] = None,
    table: Optional[str] = None,
    partition_key: Optional[str] = None,
    schema: Optional[str] = None,
) -> None:
    """
    Delete records matching partition key from table
    """
    print(
        f"Running: DELETE FROM {schema}.{table} WHERE full_path LIKE '{partition_key}'"
    )

    query = f"DELETE FROM {schema}.{table} WHERE full_path LIKE %s"

    with conn.cursor() as cur:
        cur.execute(query, (partition_key,))
        print(f"Deleted {cur.rowcount} rows")

    conn.commit()

    # PostgreSQL doesn't need vacuum the same way Spark does
    # VACUUM reclaims storage, but happens automatically in most cases
    print("Note: PostgreSQL autovacuum will reclaim space automatically")


def drop_file_from_metadata_and_table(
    conn: Optional[psycopg.Connection] = None,
    table: Optional[str] = None,
    full_path: Optional[str] = None,
    schema: Optional[str] = None,
) -> None:
    """
    Remove a file from both metadata and data table
    """
    full_path = Path(full_path).as_posix()

    # Delete from metadata
    delete_metadata_query = f"DELETE FROM {schema}.metadata WHERE full_path = %s"

    with conn.cursor() as cur:
        cur.execute(delete_metadata_query, (full_path,))
        print(f"Deleted {cur.rowcount} rows from metadata")

    conn.commit()

    # Delete from data table
    drop_partition(conn=conn, table=table, partition_key=full_path, schema=schema)
