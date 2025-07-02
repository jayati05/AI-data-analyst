"""Module for loading and listing data files with comprehensive error handling."""

from __future__ import annotations

import os

import pandas as pd

from config import DATA_FOLDER, validate_file_path
from logging_client import log_debug, log_error, log_info, log_warning


def list_files() -> dict[str, str | list[str] | bool]:
    """List available CSV/Excel files in the data folder with logging."""
    log_info("Listing available data files")
    try:
        files = [
            f for f in os.listdir(DATA_FOLDER)
            if f.endswith((".csv", ".xlsx", ".xls"))
        ]

        if not files:
            log_warning("No data files found in directory")
            return {
                "output": "No data files found.",
                "files": [],
                "should_stop": True,
            }

        log_debug(f"Found {len(files)} files: {files}")
        return {
            "output": f"Available data files: {', '.join(files)}",
            "files": files,
            "should_stop": True,
        }
    except OSError as e:
        log_error(f"Error listing files in {DATA_FOLDER}: {e!s}")
        return {
            "output": f"Error listing files: {e!s}",
            "should_stop": True,
        }


def load_data(filename: str) -> pd.DataFrame | str:
    """Load data from file with comprehensive error handling and logging."""
    log_info(f"Loading data file: {filename}")
    try:
        filepath = validate_file_path(filename)
        log_debug(f"Validated file path: {filepath}")

        if filename.endswith(".csv"):
            log_debug("Loading CSV file")
            data_frame = pd.read_csv(filepath)
        elif filename.endswith((".xlsx", ".xls")):
            log_debug("Loading Excel file")
            data_frame = pd.read_excel(filepath)
        else:
            error_msg = f"Unsupported file format for {filename}"
            log_error(error_msg)
            return error_msg

    except FileNotFoundError:
        error_msg = f"File not found: {filename}"
        log_error(error_msg)
        return error_msg
    except pd.errors.EmptyDataError:
        error_msg = f"Empty file: {filename}"
        log_error(error_msg)
        return error_msg
    except (pd.errors.ParserError, ValueError) as e:
        error_msg = f"Error loading file {filename}: {e!s}"
        log_error(error_msg)
        return error_msg
    else:
        log_info(f"Successfully loaded {filename} with shape {data_frame.shape}")
        log_debug(f"Sample data:\n{data_frame.head(2)}")
        return data_frame
