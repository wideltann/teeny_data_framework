"""
Tests for S3 support in table_functions_postgres

Uses unittest.mock to mock s3fs operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile

# Import functions to test
from src.table_functions_postgres import (
    is_s3_path,
    get_s3_filesystem,
)


class TestS3HelperFunctions:
    """Test S3 helper functions"""

    def test_is_s3_path_with_s3_url(self):
        """Test is_s3_path returns True for S3 URLs"""
        assert is_s3_path("s3://bucket/path/to/file.csv") is True
        assert is_s3_path("s3://my-bucket/") is True

    def test_is_s3_path_with_local_path(self):
        """Test is_s3_path returns False for local paths"""
        assert is_s3_path("/local/path/file.csv") is False
        assert is_s3_path("data/raw/file.csv") is False
        assert is_s3_path(Path("/local/path")) is False

    def test_get_s3_filesystem_returns_passed(self):
        """Test get_s3_filesystem returns passed filesystem"""
        mock_fs = MagicMock()
        result = get_s3_filesystem(mock_fs)
        assert result is mock_fs

    def test_get_s3_filesystem_creates_new(self):
        """Test get_s3_filesystem creates new filesystem if not provided"""
        # This test verifies that get_s3_filesystem returns an s3fs object when None is passed
        # We can't easily mock the import inside the function, so we just verify the function works
        import s3fs
        result = get_s3_filesystem(None)
        assert isinstance(result, s3fs.S3FileSystem)


class TestGetFileMetadataRow:
    """Test get_file_metadata_row with S3 support"""

    @patch("src.table_functions_postgres.is_s3_path")
    def test_metadata_from_local_file(self, mock_is_s3):
        """Test getting metadata from local file (no S3)"""
        from src.table_functions_postgres import get_file_metadata_row

        # Setup mock
        mock_is_s3.return_value = False

        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\nval1,val2\nval3,val4\n")
            temp_file = Path(f.name)

        try:
            # Call function
            result = get_file_metadata_row(
                search_dir=Path("data/raw"),
                landing_dir=Path("data/landing"),
                file=temp_file,
                filetype="csv",
                archive_full_path=None,
                has_header=True,
                error_message=None,
                encoding="utf-8",
            )

            # Assertions
            assert result["metadata_ingest_status"] == "Success"
            assert result["full_path"] == temp_file.as_posix()
            assert result["header"] == ["col1", "col2"]
            assert result["row_count"] == 2
            assert result["file_hash"] is not None
            assert result["filesize"] == temp_file.stat().st_size

        finally:
            temp_file.unlink()

    def test_metadata_with_error_message(self):
        """Test that error message prevents processing"""
        from src.table_functions_postgres import get_file_metadata_row

        result = get_file_metadata_row(
            search_dir=Path("data/raw"),
            landing_dir=Path("data/landing"),
            file=Path("nonexistent.csv"),
            filetype="csv",
            archive_full_path=None,
            has_header=True,
            error_message="Test error",
            encoding="utf-8",
        )

        # Assertions
        assert result["metadata_ingest_status"] == "Failure"
        assert result["error_message"] == "Test error"
        assert result["header"] is None
        assert result["row_count"] is None
        assert result["file_hash"] is None

    def test_metadata_from_s3_csv(self):
        """Test getting metadata from S3 CSV file"""
        from src.table_functions_postgres import get_file_metadata_row

        # Create temp file to simulate downloaded S3 file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\nval1,val2\n")
            temp_file = Path(f.name)

        try:
            with patch("src.table_functions_postgres.is_s3_path") as mock_is_s3:
                with patch("src.table_functions_postgres.get_s3_filesystem") as mock_get_fs:
                    # Setup mocks
                    mock_is_s3.side_effect = lambda path: str(path).startswith("s3://")

                    mock_fs = MagicMock()
                    mock_get_fs.return_value = mock_fs

                    def get_side_effect(s3_path, local_path):
                        # Simulate download by copying temp file
                        import shutil
                        shutil.copy(temp_file, local_path)

                    mock_fs.get.side_effect = get_side_effect

                    # Call function
                    result = get_file_metadata_row(
                        search_dir="s3://my-bucket/raw/",
                        landing_dir=Path("data/landing"),
                        file="s3://my-bucket/data/file.csv",
                        filetype="csv",
                        archive_full_path=None,
                        has_header=True,
                        error_message=None,
                        encoding="utf-8",
                    )

                    # Assertions
                    assert result["metadata_ingest_status"] == "Success"
                    assert result["full_path"] == "s3://my-bucket/data/file.csv"
                    assert result["header"] == ["col1", "col2"]
                    assert result["row_count"] == 1
                    assert result["file_hash"] is not None
                    assert result["filesize"] is not None

                    # Verify fs.get was called
                    assert mock_fs.get.called

        finally:
            temp_file.unlink()


class TestAddFilesWithS3:
    """Test add_files function with S3 support"""

    def test_add_files_local_to_local(self):
        """Test adding local files to local landing directory (no S3)"""
        from src.table_functions_postgres import add_files
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            search_dir = tmpdir / "search"
            landing_dir = tmpdir / "landing"
            search_dir.mkdir()
            landing_dir.mkdir()

            # Create temp file
            test_file = search_dir / "test.csv"
            test_file.write_text("col1,col2\n1,2\n")

            # Call function
            result = add_files(
                search_dir=search_dir,
                landing_dir=landing_dir,
                resume=False,
                sample=None,
                file_list=[test_file],
                filetype="csv",
                has_header=True,
                full_path_list=[],
                encoding="utf-8",
                num_search_dir_parents=0,
            )

            # Assertions
            assert len(result) == 1
            assert result[0]["metadata_ingest_status"] == "Success"
            # Verify file was copied
            assert (landing_dir / "test.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
