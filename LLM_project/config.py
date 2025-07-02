"""Configuration settings and path utilities for the application.

This module handles file system operations and path validations,
ensuring secure file access within the designated data folder.
"""
from pathlib import Path

DATA_FOLDER = Path("./data")
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

def validate_file_path(filename: str) -> str:
    """Validate and sanitize file paths to prevent directory traversal.

    Args:
        filename: Name of the file to validate

    Returns:
        str: Full validated path to the file

    Raises:
        FileNotFoundError: If the file doesn't exist in the data folder

    """
    clean_name = Path(filename.strip()).name
    filepath = DATA_FOLDER / clean_name

    if not filepath.exists():
        error_msg = f"File '{clean_name}' not found in {DATA_FOLDER}"
        raise FileNotFoundError(error_msg)

    return str(filepath)
