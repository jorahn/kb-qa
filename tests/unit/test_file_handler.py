"""Unit tests for file handler - meaningful business logic only."""


from dresokb.utils.file_handler import FileHandler


def test_get_files_to_process_filters_supported_formats(tmp_path) -> None:
    """Test that only supported file formats are included in processing."""
    handler = FileHandler()

    # Create various files
    (tmp_path / "supported.pdf").write_text("pdf content")
    (tmp_path / "supported.docx").write_text("docx content")
    (tmp_path / "supported.xlsx").write_text("xlsx content")
    (tmp_path / "unsupported.txt").write_text("txt content")
    (tmp_path / "unsupported.png").write_text("png content")

    files = handler.get_files_to_process(tmp_path)

    assert len(files) == 3
    extensions = {f.suffix for f in files}
    assert extensions == {".pdf", ".docx", ".xlsx"}


def test_get_files_to_process_recursive_search(tmp_path) -> None:
    """Test that file search works recursively through subdirectories."""
    handler = FileHandler()

    # Create nested structure
    (tmp_path / "root.pdf").write_text("root pdf")
    subdir1 = tmp_path / "subdir1"
    subdir1.mkdir()
    (subdir1 / "sub1.pdf").write_text("sub1 pdf")

    subdir2 = subdir1 / "subdir2"
    subdir2.mkdir()
    (subdir2 / "sub2.docx").write_text("sub2 docx")

    files = handler.get_files_to_process(tmp_path)

    assert len(files) == 3
    # Files should be sorted
    assert files == sorted(files)


def test_get_files_to_process_single_file_validation(tmp_path) -> None:
    """Test processing single file validates format correctly."""
    handler = FileHandler()

    # Supported file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("dummy content")
    assert len(handler.get_files_to_process(pdf_file)) == 1

    # Unsupported file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("dummy content")
    assert len(handler.get_files_to_process(txt_file)) == 0


def test_ensure_data_structure_creates_required_directories(tmp_path) -> None:
    """Test that all required data directories are created."""
    handler = FileHandler()
    data_dir = tmp_path / "data"

    handler.ensure_data_structure(data_dir)

    required_dirs = ["input", "processed", "output"]
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        assert dir_path.exists()
        assert dir_path.is_dir()
