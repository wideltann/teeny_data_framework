"""
Tests for S3 support in table_functions

Uses unittest.mock to mock s3fs operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile

# Import functions to test
from table_functions import (
    is_s3_path,
    get_s3_filesystem,
    get_persistent_temp_dir,
    get_cache_path_from_s3,
    get_cache_path_from_source_path,
    download_s3_file_with_cache,
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
    """Test get_file_metadata_row with new schema"""

    def test_metadata_from_local_file(self):
        """Test getting metadata from local file"""
        from table_functions import get_file_metadata_row

        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\nval1,val2\nval3,val4\n")
            temp_file = Path(f.name)

        try:
            # Call function with new schema - source_path is the local path
            result = get_file_metadata_row(
                source_path=temp_file.as_posix(),
                source_dir="data/raw/",
                filetype="csv",
                has_header=True,
                error_message=None,
                encoding="utf-8",
            )

            # Assertions
            assert result["metadata_ingest_status"] == "Success"
            assert result["source_path"] == temp_file.as_posix()
            assert result["header"] == ["col1", "col2"]
            assert result["row_count"] == 2
            assert result["file_hash"] is not None
            assert result["filesize"] == temp_file.stat().st_size

        finally:
            temp_file.unlink()

    def test_metadata_with_error_message(self):
        """Test that error message prevents processing"""
        from table_functions import get_file_metadata_row

        result = get_file_metadata_row(
            source_path="nonexistent.csv",
            source_dir="data/raw/",
            filetype="csv",
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


class TestAddFilesWithS3:
    """Test add_files function"""

    def test_add_files_local(self):
        """Test adding local files"""
        from table_functions import add_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()

            # Create temp file
            test_file = source_dir / "test.csv"
            test_file.write_text("col1,col2\n1,2\n")

            # Call function - now uses source_path_list, no landing_dir
            result = add_files(
                source_dir=str(source_dir) + "/",
                resume=False,
                sample=None,
                file_list=[str(test_file)],
                filetype="csv",
                has_header=True,
                source_path_list=[],
                encoding="utf-8",
            )

            # Assertions
            assert len(result) == 1
            assert result[0]["metadata_ingest_status"] == "Success"
            assert result[0]["source_path"] == test_file.as_posix()


class TestPersistentCaching:
    """Test persistent S3 caching functions"""

    def test_get_persistent_temp_dir_creates_directory(self):
        """Test that get_persistent_temp_dir creates temp/ in cwd"""
        import os
        import shutil

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_persistent_temp_dir()
                assert result.exists()
                assert result.name == "temp"
                # Use resolve() to handle symlinks (macOS /var -> /private/var)
                assert result.parent.resolve() == Path(tmpdir).resolve()
            finally:
                os.chdir(original_cwd)

    def test_get_persistent_temp_dir_idempotent(self):
        """Test that calling get_persistent_temp_dir multiple times is safe"""
        import os
        import shutil

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result1 = get_persistent_temp_dir()
                result2 = get_persistent_temp_dir()
                assert result1 == result2
                assert result1.exists()
            finally:
                os.chdir(original_cwd)

    def test_get_cache_path_from_s3_simple(self):
        """Test converting S3 path to cache path"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_s3("s3://my-bucket/data/file.csv")
                expected = Path(tmpdir).resolve() / "temp" / "my-bucket" / "data" / "file.csv"
                assert result.resolve() == expected
            finally:
                os.chdir(original_cwd)

    def test_get_cache_path_from_s3_nested(self):
        """Test converting nested S3 path to cache path"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_s3("s3://bucket/a/b/c/d/file.zip")
                expected = Path(tmpdir).resolve() / "temp" / "bucket" / "a" / "b" / "c" / "d" / "file.zip"
                assert result.resolve() == expected
            finally:
                os.chdir(original_cwd)

    def test_get_cache_path_from_s3_creates_parents(self):
        """Test that get_cache_path_from_s3 creates parent directories"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_s3("s3://bucket/nested/path/file.csv")
                # Parent directories should be created
                assert result.parent.exists()
                expected_parent = Path(tmpdir).resolve() / "temp" / "bucket" / "nested" / "path"
                assert result.parent.resolve() == expected_parent
            finally:
                os.chdir(original_cwd)

    def test_download_s3_file_with_cache_first_download(self):
        """Test downloading S3 file for the first time (cache miss)"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                # Create a mock S3 filesystem
                mock_fs = MagicMock()
                mock_fs.info.return_value = {"size": 100}

                def mock_get(s3_path, local_path):
                    # Simulate download by writing content
                    Path(local_path).write_text("x" * 100)

                mock_fs.get.side_effect = mock_get

                result = download_s3_file_with_cache("s3://bucket/file.csv", mock_fs)

                # Verify result
                assert result.exists()
                assert result.stat().st_size == 100
                assert "bucket" in str(result)

                # Verify fs.get was called
                mock_fs.get.assert_called_once()

            finally:
                os.chdir(original_cwd)

    def test_download_s3_file_with_cache_hit(self):
        """Test cache hit when file already exists with matching size"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                # Create a mock S3 filesystem
                mock_fs = MagicMock()
                mock_fs.info.return_value = {"size": 50}

                # Pre-create the cached file with matching size
                cache_path = get_cache_path_from_s3("s3://bucket/cached.csv")
                cache_path.write_text("x" * 50)

                result = download_s3_file_with_cache("s3://bucket/cached.csv", mock_fs)

                # Verify result
                assert result == cache_path
                assert result.exists()

                # Verify fs.get was NOT called (cache hit)
                mock_fs.get.assert_not_called()

            finally:
                os.chdir(original_cwd)

    def test_download_s3_file_with_cache_size_mismatch(self):
        """Test cache miss when file exists but size differs"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                # Create a mock S3 filesystem
                mock_fs = MagicMock()
                mock_fs.info.return_value = {"size": 200}  # S3 file is 200 bytes

                # Pre-create the cached file with DIFFERENT size
                cache_path = get_cache_path_from_s3("s3://bucket/stale.csv")
                cache_path.write_text("x" * 100)  # Cached is only 100 bytes

                def mock_get(s3_path, local_path):
                    # Simulate re-download with new size
                    Path(local_path).write_text("y" * 200)

                mock_fs.get.side_effect = mock_get

                result = download_s3_file_with_cache("s3://bucket/stale.csv", mock_fs)

                # Verify result
                assert result.exists()
                assert result.stat().st_size == 200  # Updated to new size

                # Verify fs.get WAS called (cache miss due to size mismatch)
                mock_fs.get.assert_called_once()

            finally:
                os.chdir(original_cwd)


class TestGetCachePathFromSourcePath:
    """Test the new get_cache_path_from_source_path function"""

    def test_s3_simple_file(self):
        """Test S3 file without archive"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_source_path("s3://bucket/path/file.csv")
                expected = Path(tmpdir).resolve() / "temp" / "bucket" / "path" / "file.csv"
                assert result.resolve() == expected
            finally:
                os.chdir(original_cwd)

    def test_s3_archive_with_inner_path(self):
        """Test S3 archive with :: delimiter"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_source_path("s3://bucket/archive.zip::inner/file.csv")
                expected = Path(tmpdir).resolve() / "temp" / "bucket" / "archive.zip" / "inner" / "file.csv"
                assert result.resolve() == expected
            finally:
                os.chdir(original_cwd)

    def test_local_file_returned_as_is(self):
        """Test local file path is returned unchanged"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_source_path("/local/path/file.csv")
                assert result == Path("/local/path/file.csv")
            finally:
                os.chdir(original_cwd)

    def test_local_archive_with_inner_path(self):
        """Test local archive with :: delimiter"""
        import os

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                result = get_cache_path_from_source_path("/local/archive.zip::inner/file.csv")
                expected = Path(tmpdir).resolve() / "temp" / "local" / "archive.zip" / "inner" / "file.csv"
                assert result.resolve() == expected
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
