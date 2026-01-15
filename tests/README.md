# Test Suite for Teeny Data Framework

Comprehensive test suite for `table_functions_postgres.py` with 61 tests covering all major functionality.

## Running Tests

### Quick Start (using justfile)

The easiest way to run tests:

```bash
# Run all tests (assumes Podman is set up)
just test

# Check if Podman is running
just check-podman

# Other useful commands
just test-verbose       # Verbose output
just test-fast          # Stop on first failure
just test-failed        # Re-run only failed tests
just --list             # See all available commands
```

### Manual Testing

The tests use **testcontainers** to automatically manage PostgreSQL containers.

**Requirements:**
- Podman (or Docker) must be running
- No manual PostgreSQL setup needed!

```bash
# With Podman (recommended)
export DOCKER_HOST=unix:///tmp/podman.sock
pytest tests/test_table_functions_postgres.py -v

# Run specific test class
pytest tests/test_table_functions_postgres.py::TestFileReading -v

# Run specific test
pytest tests/test_table_functions_postgres.py::TestFileReading::test_read_csv_with_header -v
```

**How it works:**
- First test run: Downloads PostgreSQL Docker image (one-time, ~80MB)
- Starts a PostgreSQL container automatically
- Runs all tests against the container
- Cleans up container when tests finish
- Container is reused across the test session for speed

**Using Podman instead of Docker:**

testcontainers works with Podman too! Just set the `DOCKER_HOST` environment variable:

```bash
# Enable Podman socket (one-time setup)
podman machine start
podman system service --time=0 unix:///tmp/podman.sock &

# Set environment variable and run tests
export DOCKER_HOST=unix:///tmp/podman.sock
pytest tests/test_table_functions_postgres.py -v

# Or set it in your shell config (~/.zshrc or ~/.bashrc)
echo 'export DOCKER_HOST=unix:///tmp/podman.sock' >> ~/.zshrc
```

Alternatively, if using Podman Desktop with Docker compatibility:
```bash
# Podman creates a Docker-compatible socket
export DOCKER_HOST=unix:///var/run/docker.sock
pytest tests/test_table_functions_postgres.py -v
```

**Troubleshooting:**
- If you see `DockerException: Error while fetching server API version`, neither Docker nor Podman is running
- For Docker: Start Docker Desktop
- For Podman: Ensure Podman machine is running and socket is enabled
- Check your `DOCKER_HOST` environment variable is set correctly

## Test Coverage

### S3 Helper Functions (6 tests)
- `is_s3_path()` - S3 path detection
- `parse_s3_path()` - S3 URL parsing
- `list_s3_files()` - S3 file listing with mocked boto3

### Column Mapping (4 tests)
- `create_pandas_selection_from_header()` - Column selection, renaming, missing columns, default types

### File Reading (8 tests)
- `read_csv()` - CSV with/without headers, pipe-delimited, missing columns
- `read_using_column_mapping()` - Router for CSV, TSV, PSV
- `read_fixed_width()` - Fixed-width file parsing

### Database Operations (13 tests)
- `table_exists()` - Table existence checks
- `infer_postgres_type()` - Type inference for all pandas dtypes
- `get_table_schema()` - Schema retrieval
- `validate_schema_match()` - Schema validation (success/failure cases)
- `create_table_from_dataframe()` - Table creation
- `copy_dataframe_to_table()` - Bulk loading with COPY, NULL handling

### Metadata Functions (9 tests)
- `get_csv_header_and_row_count()` - Header/row count extraction
- `get_file_metadata_row()` - Metadata row generation
- `row_count_check()` - Row count validation with/without unpivot multiplier
- `update_metadata()` - Success/failure status updates

### Schema Inference (9 tests)
- `pandas_dtype_to_column_mapping_type()` - Type string conversion
- `infer_schema_from_file()` - Schema inference from CSV, PSV, auto-detection

### Integration Tests (4 tests)
- `add_files()` - File copying and metadata generation
- `extract_and_add_zip_files()` - ZIP extraction
- `drop_metadata_by_source()` - Metadata cleanup
- `drop_partition()` - Partition deletion

### Edge Cases (4 tests)
- BOM (byte-order mark) handling
- Empty string null values
- Error message truncation
- Schema type mismatches

### Additional Functions (4 tests)
- `drop_file_from_metadata_and_table()` - Combined deletion
- Invalid filetype handling
- Unpivot multiplier functionality

## Key Testing Patterns

### Fixtures
- `postgres_container` - PostgreSQL Docker container (session-scoped, reused)
- `db_conn` - PostgreSQL connection with automatic cleanup (function-scoped)
- `temp_dir` - Temporary directory for test files
- `sample_csv_file` - CSV file with headers
- `sample_csv_no_header` - Headerless CSV
- `sample_psv_file` - Pipe-delimited file
- `sample_zip_file` - ZIP archive with CSV files

### Mocking
- S3 operations use `unittest.mock.patch` to mock boto3
- No actual S3 calls are made during testing

### Edge Cases Covered
- Files with BOM (byte-order mark)
- NULL/None value handling
- Large error message truncation
- Schema validation failures
- Missing columns in CSV vs mapping
- Type mismatches between DataFrame and database

## Notes

- Tests use numpy dtypes rather than pandas extension dtypes (Int64, StringDtype) where `infer_postgres_type()` is involved, since that function works with numpy dtypes
- Some pandas extension dtypes fall through to TEXT type - this is expected behavior
- All database operations use transactions that are automatically rolled back on test failure
- The PostgreSQL container uses the `postgres:16-alpine` image (lightweight)
