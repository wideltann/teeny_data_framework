def create_pandas_dtypes_and_columns_from_header(header, column_mapping):
    dtypes, columns = {}, {}
    default_type = None

    for alias, (possible_cols, col_type) in column_mapping.items():
        if alias == "default":
            default_type = col_type

        # we want to keep the original column name and there's no column name variation
        # so we pass in an empty possible cols
        if not possible_cols:
            dtypes[alias] = col_type
        # there is variation in the column name across time so we pass
        # in a list of column names that mean the same thing
        elif any(col_name in header for col_name in possible_cols):
            col_name = next(
                col_name for col_name in possible_cols if col_name in header
            )
            columns[col_name] = alias
            dtypes[alias] = col_type

    # if default type is specifed then set all remaining columns in header to this type
    if default_type is not None:
        for col_name in header:
            if col_name not in dtypes:
                dtypes[col_name] = default_type

    return dtypes, columns


def read_xlsx(full_path=None, column_mapping=None):
    # requires openpyxl to read xlsx spreadsheets
    import pandas as pd

    # sometimes there are labels as the first couple rows so this wont always work but for fcc it does
    header = pd.read_excel(full_path, nrows=1).columns.tolist()

    dtypes, columns = create_pandas_dtypes_and_columns_from_header(
        header=header, column_mapping=column_mapping
    )

    """
    1. the rename will create columns that dont exist which serves the same purpose
    as the lit none in read csv
    2. we have to use dtypes because thats when the data is initially being read.
    if we did the pandas auto read then stuff like ids with left padded 0s would become ints instead of strings
    """
    pdf = pd.read_excel(full_path, dtype=dtypes).rename(columns=columns)
    df = spark.createDataFrame(pdf)

    return df


def create_spark_schema_and_selection_from_header(header, column_mapping):
    from pyspark.sql import functions as F, types as T

    # header is a list of strings
    selections = []
    schema = []
    processed_cols = []
    default_type = None

    # if you dont include all cols and dont include default then the read will fail
    # because you havent specified the entire schema
    # if we get the schema wrong at all then the read will fail
    # because the default is strict schema enforcement
    # example mapping:
    # column_mapping = {
    #    "block_fips": ([], T.StringType()),
    #    "county_fips": ([], T.StringType()),
    #    "state_abbr": ([], T.StringType()),
    #    "default": ([], T.FloatType()),
    # }
    for alias, (possible_cols, col_type) in column_mapping.items():
        if alias == "default":
            default_type = col_type
            # must continue or else default will get added as a column
            continue

        # we want to keep the original column name and there's no column name variation
        # so we pass in an empty possible cols
        # if alias not in header we need to do lit none
        if not possible_cols and alias in header:
            col_name = alias
            selections.append(F.col(col_name))
            schema.append(T.StructField(col_name, col_type))
            processed_cols.append(col_name)
        # there is variation in the column name across time so we pass
        # in a list of column names that mean the same thing
        elif any(col_name in header for col_name in possible_cols):
            col_name = next(
                col_name for col_name in possible_cols if col_name in header
            )
            selections.append(F.col(col_name).alias(alias))
            schema.append(T.StructField(col_name, col_type))
            processed_cols.append(col_name)
        # in order to combine this dataframe with other dataframes in a single table
        # all columns in the table schema must exist in the data frame
        else:
            # we have to do this cast because if we create the table from the first file
            # and that file hits this line and we didnt cast then that columns type would be
            # null and when we try to add cols that do have that data then it will fail
            selections.append(F.lit(None).cast(col_type).alias(alias))

    # all cols in the header but not in the column mapping will be assigned this type
    # if you set a default then you cant delete columns by not selecting them
    if default_type is not None:
        remaining_cols = [
            col_name for col_name in header if col_name not in processed_cols
        ]

        for col_name in remaining_cols:
            selections.append(F.col(col_name))
            schema.append(T.StructField(col_name, default_type))

    return schema, selections


def read_parquet(spark=None, full_path=None, column_mapping=None):
    df = spark.read.parquet(full_path, mode="failfast")

    # since the shapefiles have metadata about the column types
    # the geoparquet have type information
    # so we dont need to set that initially but we do want to run the rename
    _, selections = create_spark_schema_and_selection_from_header(
        header=df.columns, column_mapping=column_mapping
    )

    result_df = df.select(*selections)

    return result_df


def read_csv(
    spark=None,
    full_path=None,
    column_mapping=None,
    header=None,
    has_header=False,
    null_value=None,
    separator=",",
):
    from pyspark.sql import functions as F, types as T

    if not header:
        # some files start with a BOM (byte-order-mark)
        # if we dont use sig then that BOM will get included in the header
        # the character will show up as \ufeff if the encoding is utf-8 (which is the default)
        with open(full_path, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip().split(separator)

    schema, selections = create_spark_schema_and_selection_from_header(
        header=header, column_mapping=column_mapping
    )

    """
    - Always include the header because otherwise the header will be in the data
        - if you have has_header as true and there is no header the first row wont
          be in the dataset
    - Use a single select because withColumn is a new transformation step
    - Always provide the schema because otherwise it will auto infer the schema
        - Which means it has to read the file twice, once to generate a schema
            and again to actually parse the file
        - Also the inferred schema is not always what you want
            - Some ids look like they should be integers but really they should
                varchars
    """

    df = spark.read.csv(
        full_path,
        sep=separator,
        header=has_header,
        schema=T.StructType(schema),
        nullValue=null_value,
        mode="failfast",
    ).select(*selections)

    return df


def read_fixed_width(spark=None, full_path=None, column_mapping=None):
    from pyspark.sql import functions as F, types as T

    """
    this function expects a dict in this format:
{'stusab': (StringType(), 7, 2),
'sumlev': (FloatType(), 9, 3),
'geocomp': (StringType(), 12, 2)}

    {'column_name': (type, starting_position, field_size)}
    """

    columns = []
    for col_name, value in column_mapping.items():
        col_type, starting_position, field_size = value

        # when you use read.text each line of the textfile is a row
        # with a single col named "value"
        # this also means that you cant really define a schema besides casting
        trimmed_column = F.trim(F.col("value").substr(starting_position, field_size))

        col = (
            F.when(trimmed_column == "", None)
            .otherwise(trimmed_column)
            .cast(col_type)
            .alias(col_name)
        )

        columns.append(col)

    df = spark.read.text(full_path).select(*columns)

    return df


def read_using_column_mapping(
    spark=None,
    filetype=None,
    full_path=None,
    column_mapping=None,
    header=None,
    has_header=False,
    null_value=None,
):
    # requires package openpyxl to parse xslx spreadsheets
    # send full path because its annoying when testing the read functions
    # to have to import pathlib to send a path
    # and also spark doesnt convert a pathlib path to a posix path
    match filetype:
        case "csv":
            # manually define separator based on filetype
            return read_csv(
                spark=spark,
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator=",",
                null_value=null_value,
            )
        case "tsv":
            # manually define separator based on filetype
            return read_csv(
                spark=spark,
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="\t",
                null_value=null_value,
            )
        case "psv":
            # manually define separator based on filetype
            return read_csv(
                spark=spark,
                full_path=full_path,
                column_mapping=column_mapping,
                has_header=has_header,
                header=header,
                separator="|",
                null_value=null_value,
            )
        case "xlsx":
            return read_xlsx(full_path=full_path, column_mapping=column_mapping)
        case "parquet":
            return read_parquet(
                spark=spark, full_path=full_path, column_mapping=column_mapping
            )
        case "fixed_width":
            return read_fixed_width(
                spark=spark, full_path=full_path, column_mapping=column_mapping
            )
        case _:
            print("Invalid filetype")


# sanity check on flat file ingest
def row_count_check(
    spark=None,
    schema=None,
    df=None,
    full_path=None,
    unpivot_row_multiplier=None,
):
    metadata_row_count = spark.sql(
        f"select row_count from {schema}.metadata where full_path = '{full_path}'"
    ).first()["row_count"]

    output_df_row_count = df.count()

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
    spark=None,
    full_path=None,
    schema=None,
    error_message=None,
    unpivot_row_multiplier=None,
    ingest_runtime=None,
):
    from datetime import datetime

    metadata_ingest_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status = "Failure" if error_message else "Success"

    if error_message:
        # if these characters are in the sql f string itll break
        # the update because they wont be properly escaped
        error_message = error_message.replace("'", "").replace("`", "")
        error_message = f"'{error_message}'"
    else:
        error_message = "null"

    unpivot_row_multiplier = (
        unpivot_row_multiplier if unpivot_row_multiplier else "null"
    )
    ingest_runtime = ingest_runtime if ingest_runtime else "null"

    query = f"""
update {schema}.metadata
set
    ingest_datetime = '{metadata_ingest_datetime}',
    status = '{status}',
    error_message = {error_message},
    unpivot_row_multiplier = {unpivot_row_multiplier},
    ingest_runtime = {ingest_runtime}
where full_path = '{full_path}'
        """

    print(query)

    pdf = spark.sql(query).toPandas()

    if pdf["num_affected_rows"].iloc[0] > 1:
        print(f"Updated more than one row, you should be concerned. File: {full_path}")


# list out all params so that when you call create table it fails if you put in param wrong
def update_table(
    spark=None,
    resume=False,
    retry_failed=False,
    sample=None,
    schema=None,
    metadata_schema=None,
    output_table=None,
    output_table_naming_fn=None,
    additional_cols_fn=None,
    file_list_filter_fn=None,  # main reason for this is that you cant do not in with a glob, you can only match a substring
    custom_read_fn=None,
    transform_fn=None,
    landing_dir=None,
    sql_glob=None,
    filetype=None,
    column_mapping=None,
    column_mapping_fn=None,
    pivot_mapping=None,
    header_fn=None,
    null_value="",
):
    import datetime
    import time
    from pathlib import Path

    import pandas as pd
    from pyspark.sql import functions as F

    required_params = [landing_dir, schema, spark, filetype]
    if not custom_read_fn and not column_mapping_fn:
        required_params.append(column_mapping)

    if any(param is None for param in required_params):
        print(required_params)
        raise ValueError(
            "Required params: landing_dir, filetype, column_mapping, schema, spark. Column mapping not required if using custom_read_fn or if using column_mapping_fn"
        )

    # remove potential trailing slash
    landing_dir = Path(landing_dir).as_posix()

    # If the glob is omitted we create a glob using the filetype
    # sometimes the glob is like "*/raw/*.csv" but the default is probably "*.csv"
    # INFO: can also have the glob match a single file
    sql_glob = sql_glob or f"%.{filetype}"

    # We use the metadata table because thats the source of truth for what
    # files have been processed
    # if publishing to a not raw schema then we cant use the same schema for the metadata
    # since the metadata table is in the raw schema
    if (metadata_schema and len(metadata_schema.split(".")) == 1) or len(
        schema.split(".")
    ) == 1:
        raise ValueError(
            "Must provide environment in schema as well as metadata_schema (if metadata schema is different than output schema)"
        )

    metadata_schema = metadata_schema or schema

    query = f"""
select full_path
from {metadata_schema}.metadata
where
    landing_dir like '{landing_dir}' and
    full_path like '{sql_glob}' and
    metadata_ingest_status = 'Success'
                    """

    print(f"metadata file search query: {query}")

    pdf = spark.sql(query).toPandas()

    pdf_list = pdf["full_path"].tolist()
    file_list = [Path(f) for f in pdf_list]
    file_list = sorted(file_list)

    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    if resume:
        pdf = spark.sql(
            f"""
select full_path
from {metadata_schema}.metadata
where
    landing_dir like '{landing_dir}' and
    ingest_datetime is not null
    {"and status = 'Success'" if retry_failed else ""}
"""
        ).toPandas()

        full_path_list = pdf["full_path"].tolist()
        full_path_list = [Path(f) for f in full_path_list]

        file_list = set(file_list) - set(full_path_list)

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
            # have to manually define has_header because otherwise the header existing
            # via the header_fn would make has_header true for read_csv
            header = header_fn(file)
            has_header = False

        unpivot_row_multiplier = None
        try:
            if custom_read_fn:
                df = custom_read_fn(full_path=full_path)
            else:
                column_mapping = (
                    column_mapping_fn(file) if column_mapping_fn else column_mapping
                )

                df = read_using_column_mapping(
                    spark=spark,
                    full_path=full_path,
                    filetype=filetype,
                    column_mapping=column_mapping,
                    header=header,
                    has_header=has_header,
                    null_value=null_value,
                )

            if pivot_mapping:
                value_vars = [
                    col for col in df.columns if col not in pivot_mapping["id_vars"]
                ]
                unpivot_row_multiplier = len(value_vars)

                df = df.unpivot(
                    ids=pivot_mapping["id_vars"],
                    values=value_vars,
                    variableColumnName=pivot_mapping["variable_column_name"],
                    valueColumnName=pivot_mapping["value_column_name"],
                )

            # Raises exception on failure
            row_count_check(
                spark=spark,
                schema=metadata_schema,
                df=df,
                full_path=full_path,
                unpivot_row_multiplier=unpivot_row_multiplier,
            )

            if transform_fn:
                df = transform_fn(df=df)

            additional_cols = additional_cols_fn(file) if additional_cols_fn else []

            df = df.select(
                *[F.col(c) for c in df.columns],
                *additional_cols,
                F.lit(full_path).alias("full_path"),
            )

            # output table naming function and output_table are mutually exclusive
            if output_table_naming_fn:
                df.write.mode("overwrite").partitionBy("full_path").saveAsTable(
                    f"{schema}.{output_table_naming_fn(file)}"
                )
            elif spark.catalog.tableExists(f"{schema}.{output_table}"):
                spark.sql(
                    f"delete from {schema}.{output_table} where full_path='{full_path}'"
                )
                df.write.mode("overwrite").insertInto(f"{schema}.{output_table}")
            else:
                df.write.mode("overwrite").partitionBy("full_path").saveAsTable(
                    f"{schema}.{output_table}"
                )

            ingest_runtime = int(time.time() - start_time)

            update_metadata(
                spark=spark,
                full_path=full_path,
                schema=metadata_schema,
                unpivot_row_multiplier=unpivot_row_multiplier,
                ingest_runtime=ingest_runtime,
            )
        except Exception as e:
            update_metadata(
                spark=spark,
                full_path=full_path,
                schema=metadata_schema,
                error_message=str(e),
                unpivot_row_multiplier=unpivot_row_multiplier,
            )

            print(f"Failed on {file} with {e}")

    df = spark.sql(
        f"""
select *
from {metadata_schema}.metadata
where landing_dir like '{landing_dir}' and full_path like '{sql_glob}'
order by ingest_datetime desc
    """
    )

    return df


# START OF METADATA FUNCTIONS #


def get_csv_header_and_row_count(
    encoding="utf-8-sig", file=None, separator=",", has_header=True
):
    import subprocess

    # always return the first row even if it isnt a header
    # this is useful for sleuthing later
    # must ignore BOM mark
    with open(file, "r", encoding=encoding) as f:
        header = f.readline().strip().split(separator)

    subtract_header_row = 1 if has_header else 0
    row_count = (
        int(subprocess.check_output(["wc", "-l", file.as_posix()]).split()[0])
        - subtract_header_row
    )

    return header, row_count


def extract_and_add_zip_files(
    spark=None,
    file_list=None,
    full_path_list=None,
    search_dir=None,
    landing_dir=None,
    compression_type=None,
    has_header=True,
    filetype=None,
    resume=None,
    sample=None,
    encoding=None,
    archive_glob=None,
    num_search_dir_parents=0,
):
    # this function requires installing: zipfile-deflate64
    # Recent versions of Microsoft Windows Explorer use Deflate64 compression when creating ZIP files larger than 2GB. With the ubiquity of Windows and the ease of using "Sent to compressed folder", a majority of newly-created large ZIP files use Deflate64 compression.
    # https://github.com/brianhelba/zipfile-deflate64
    try:
        import zipfile_deflate64 as zipfile
    except Exception as e:
        print(f"Missing zipfile_deflate64, using zipfile")
    finally:
        import zipfile

    import fnmatch
    from pathlib import Path

    import pandas as pd

    rows = []

    # have to use num processed here because we want total number of files
    # if we enumerate the inner for loop then that will just be the num files within a single zip
    # if we do the outer loop then thats the num zip files
    num_processed = 0
    for compressed in file_list:
        if sample and num_processed == sample:
            break

        with zipfile.ZipFile(compressed) as zip_ref:
            # fnmatch allows for globbing
            namelist = [
                f for f in zip_ref.namelist() if fnmatch.fnmatch(f, archive_glob)
            ]

            for f in namelist:
                # have to break twice otherwise we wont be able to loop through
                # other zipfiles
                if sample and num_processed == sample:
                    break

                num_processed += 1

                file_num = (
                    f"{num_processed}/{sample}" if sample else f"{num_processed} |"
                )

                # name output dir after archive since sometimes different archives
                # have their compressed file be named the same
                # using the archive dir makes our full_path primary key more likely to be unique
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
                        spark=spark,
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        archive_full_path=compressed.as_posix(),  # have to go posix here because this isnt an arg that exists for regular files
                        file=raw_output_path,
                        has_header=has_header,
                        encoding=encoding,
                    )

                except Exception as e:
                    print(f"Failed on {f} with {e} hi")

                    raw_output_path = None
                    row = get_file_metadata_row(
                        spark=spark,
                        search_dir=search_dir,
                        landing_dir=landing_dir,
                        filetype=filetype,
                        file=raw_output_path,
                        archive_full_path=compressed.as_posix(),
                        has_header=has_header,
                        error_message=str(e),
                        encoding=encoding,
                    )

                    # if it extracted with an error, the file will still be there
                    # so we need to manually remove it
                    if raw_output_path.exists():
                        raw_output_path.unlink()
                        print(f"Removed bad extracted file: {f}")

                    if not any(raw_output_dir.iterdir()):
                        raw_output_dir.rmdir()
                        print(f"Removed empty output dir: {raw_output_dir}")

                rows.append(row)

    return rows


def add_files(
    spark=None,
    search_dir=None,
    landing_dir=None,
    resume=None,
    sample=None,
    file_list=None,
    filetype=None,
    has_header=True,
    full_path_list=None,
    encoding=None,
    num_search_dir_parents=0,
):
    import shutil
    from pathlib import Path

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

            # for if raw files are already in path
            if landing_dir != search_dir:
                shutil.copy2(file, landing_path)

            # file metadata should return complete set of rows to add to table
            row = get_file_metadata_row(
                spark=spark,
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
                spark=spark,
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
    spark=None,
    search_dir=None,
    landing_dir=None,
    file=None,
    filetype=None,
    compression_type=None,
    archive_full_path=None,
    has_header=None,
    error_message=None,
    encoding=None,
):
    import hashlib
    import time
    from datetime import datetime

    import pandas as pd

    row = {
        "search_dir": search_dir.as_posix(),
        "landing_dir": landing_dir.as_posix(),
        "full_path": file.as_posix(),
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
            df = spark.read.option("mode", "failfast").parquet(file.as_posix())
            header, row_count = df.columns, df.count()
        case "xml":
            # there isnt really a standard way of gettting these values
            header, row_count = None, None
        case _:
            raise Exception("Unsupported filetype")

    row["header"], row["row_count"] = header, row_count
    row["file_hash"] = hashlib.md5(open(file, "rb").read()).hexdigest()
    row["filesize"] = file.stat().st_size
    row["metadata_ingest_status"] = "Success"

    print(
        f"Row count: {row_count} Filename: {file.name} | Runtime: {time.time() - start_time:2f}"
    )

    return row


# the point of this is to try to read the row count of the source file
# in as raw a format as possible to compare against our ingestion process
# and protect against ingestion mistakes
def add_files_to_metadata_table(**kwargs):
    from pathlib import Path

    import pandas as pd
    from delta.tables import DeltaTable

    schema = kwargs.pop("schema", None)
    if not schema:
        raise Exception("You must provide the schema as a param")

    if len(schema.split(".")) == 1:
        raise ValueError("Must provide environment in schema")

    # If the glob is omitted we create a glob using the filetype
    # sometimes the glob is like "*/raw/*.csv" but the default is probably "*.csv"
    # INFO: can also have the glob match a single file
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

    # we name this search dir because its a recursive glob
    landing_dir = Path(kwargs["landing_dir"])
    search_dir = Path(kwargs["search_dir"])
    file_list = [f for f in search_dir.rglob(glob)]

    # overwrite the input so that potential trailing slashes
    # are standardized for the functions we pass kwargs into
    kwargs["search_dir"] = search_dir
    kwargs["landing_dir"] = landing_dir

    file_list_filter_fn = kwargs.get("file_list_filter_fn", None)
    if file_list_filter_fn:
        file_list = file_list_filter_fn(file_list)

    file_list = sorted(file_list)

    kwargs["file_list"] = file_list

    output_table = f"{schema}.metadata"

    spark = kwargs["spark"]

    if not spark.catalog.tableExists(output_table):
        spark.sql(
            f"""
                create table {output_table} (
                    search_dir string,
                    landing_dir string,
                    full_path string,
                    filesize string,
                    header array<string>,
                    row_count bigint,
                    archive_full_path string,
                    file_hash string,
                    metadata_ingest_datetime string,
                    metadata_ingest_status string,
                    ingest_datetime string,
                    ingest_runtime int,
                    status string,
                    error_message string,
                    unpivot_row_multiplier int
                )
                        """
        )

    resume = kwargs.get("resume", False)
    retry_failed = kwargs.pop("retry_failed", False)

    # have to define up here or else we cant check within it if the table doesnt exist yet
    kwargs["full_path_list"] = []
    if resume:
        pdf = spark.sql(
            f"""
select full_path
from {output_table}
where
    search_dir like '{search_dir}' and
    full_path like '{sql_glob}'
    {"and metadata_ingest_status = 'Success'" if retry_failed else ""}
        """
        ).toPandas()
        full_path_list = pdf["full_path"].tolist()
        kwargs["full_path_list"] = [Path(f) for f in full_path_list]

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

        df = spark.createDataFrame(pdf)

        # create the view so we can explore after if desired
        df.createOrReplaceTempView("tmp")

        """
        - i would use sql here but you cannot merge from a temp view
        - when matched update is required so that we can update if calling the metadata update with resume=False
        - also required for idempotency across runs
        - since the pk is full_path, different search_dirs can be overridden
        - error_message is the only column shared between metadata_ingest/ingest
        - youll never have an error message for metadata ingest after doing the table ingest
        """
        ingest_cols = [
            "ingest_datetime",
            "status",
            "error_message",
            "unpivot_row_multiplier",
            "ingest_runtime",
        ]

        DeltaTable.forName(spark, output_table).alias("old").merge(
            df.alias("new"), f"old.full_path = new.full_path"
        ).whenMatchedUpdate(
            set={col: f"new.{col}" for col in df.columns if col not in ingest_cols}
        ).whenNotMatchedInsert(
            values={col: f"new.{col}" for col in df.columns}
        ).execute()

    # we run a query here so that even if resume is enabled and no files are processed
    # we show the metadata result from the search dir
    rows_df = spark.sql(
        f"""
select *
from {output_table}
where search_dir like '{search_dir}' and full_path like '{sql_glob}'
order by metadata_ingest_datetime desc
    """
    )

    return rows_df


def drop_search_dir(spark=None, search_dir=None, schema=None):
    from pathlib import Path

    if not schema:
        raise ("Must specify schema")

    # standardize path
    search_dir = Path(search_dir).as_posix()

    df = spark.sql(
        f"select * from {schema}.metadata where search_dir like '{search_dir}'"
    )

    print(f"Rows before drop {df.count()}")

    df = spark.sql(
        f"delete from {schema}.metadata where search_dir like '{search_dir}'"
    )

    df.show()

    df = spark.sql(
        f"select * from {schema}.metadata where search_dir like '{search_dir}'"
    )

    print(f"Rows after drop {df.count()}")


def drop_partition(spark=None, table=None, partition_key=None):
    if len(table.split(".")) < 3:
        raise ValueError("Must provide full path for table")

    print(f"Running: DELETE from {table} where full_path like '{partition_key}'")

    spark.sql(f"delete from {table} where full_path like '{partition_key}'")

    print("Vacuuming to remove the files within the partition")

    df = spark.sql(f"select * from {table} where full_path like '{partition_key}'")

    spark.sql(f"vacuum {table}")

    empty_df = spark.sql(
        f"select * from {table} where full_path like '{partition_key}'"
    )

    print(df.count())

    return df


def drop_file_from_metadata_and_table(spark=None, table=None, full_path=None):
    from pathlib import Path

    if len(table.split(".")) < 3:
        raise ValueError("Must provide full path for table")

    # standardize path
    full_path = Path(full_path).as_posix()

    schema = table.split(".")[1]

    df = spark.sql(f"delete from {schema}.metadata where full_path = '{full_path}'")

    print(df.collect())

    drop_partition(spark=spark, table=table, partition_key=full_path)
