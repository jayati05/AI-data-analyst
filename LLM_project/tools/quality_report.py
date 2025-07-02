"""Generate data quality reports with statistical analysis."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from logging_client import log_critical, log_debug, log_error, log_info
from tools.data_loading import load_data

T = TypeVar("T", bound=object)
def convert_to_serializable(
    obj: np.integer | np.floating | T,
) -> int | float | T:
    """Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Value to convert, can be numpy type or any other type

    Returns:
        Converted value if numpy type, otherwise original value

    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def calculate_numeric_stats(
    df: pd.DataFrame, numeric_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """Calculate statistics for numeric columns.

    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric column names

    Returns:
        Dictionary of statistics for each numeric column

    """
    return {
        col: {
            "mean": convert_to_serializable(df[col].mean()),
            "median": convert_to_serializable(df[col].median()),
            "std": convert_to_serializable(df[col].std()),
            "min": convert_to_serializable(df[col].min()),
            "max": convert_to_serializable(df[col].max()),
            "skew": convert_to_serializable(df[col].skew()),
            "kurtosis": convert_to_serializable(df[col].kurtosis()),
        }
        for col in numeric_cols
    }


def detect_quality_issues(
    df: pd.DataFrame, numeric_cols: list[str],
) -> list[dict[str, Any]]:
    """Detect and categorize quality issues in the DataFrame.

    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric column names

    Returns:
        List of quality issues found

    """
    quality_issues = []
    low_cardinality_threshold = 5

    for col in df.columns:
        issues = []
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_pct = null_count / len(df) * 100
            issues.append(f"{null_count} missing values ({null_pct:.2f}%)")

        if col in numeric_cols:
            unique_count = df[col].nunique()
            if unique_count == 1:
                issues.append("Constant value (no variation)")
            elif (unique_count < low_cardinality_threshold
                  and unique_count < len(df) / 2):
                issues.append("Low cardinality (few unique values)")

        if issues:
            quality_issues.append({
                "column": col,
                "issues": issues,
                "data_type": str(df[col].dtype),
            })

    return quality_issues


def data_quality(file_name: str) -> dict[str, Any]:
    """Generate a comprehensive data quality report with centralized logging.

    Args:
        file_name: Path to the data file to analyze

    Returns:
        Dictionary containing the quality report and metadata

    """
    log_info(f"Starting data quality assessment for: {file_name}")

    try:
        # Data loading phase
        log_debug(f"Loading dataset: {file_name}")
        data_frame = load_data(file_name)
        if isinstance(data_frame, str):
            log_error(f"Data loading failed: {data_frame}")
            return {"output": data_frame, "should_stop": True}

        log_info(f"Data loaded successfully. Dimensions: {data_frame.shape}")
        log_debug(f"Columns detected: {list(data_frame.columns)}")
        log_debug(f"Data sample:\n{data_frame.head(2).to_string()}")

        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns.tolist()
        log_debug(f"Numeric columns identified: {numeric_cols}")

        # Report generation
        log_debug("Calculating quality metrics")
        report = {
            "file_name": file_name,
            "total_rows": len(data_frame),
            "total_columns": len(data_frame.columns),
            "missing_values": {
                col: int(data_frame[col].isna().sum())
                for col in data_frame.columns
            },
            "missing_percentage": {
                col: float(data_frame[col].isna().mean() * 100)
                for col in data_frame.columns
            },
            "duplicate_rows": int(data_frame.duplicated().sum()),
            "data_types": {
                col: str(dtype)
                for col, dtype in data_frame.dtypes.to_dict().items()
            },
            "numeric_stats": calculate_numeric_stats(data_frame, numeric_cols),
            "unique_values": {
                col: int(data_frame[col].nunique())
                for col in data_frame.columns
            },
            "memory_usage": convert_to_serializable(
                data_frame.memory_usage(deep=True).sum(),
            ),
            "sample_data": {
                col: convert_to_serializable(data_frame[col].head().tolist())
                for col in data_frame.columns
            },
        }

        quality_issues = detect_quality_issues(data_frame, numeric_cols)
        report["quality_issues"] = quality_issues
        score_calc = len(quality_issues) / len(data_frame.columns) * 50
        report["quality_score"] = 100 - score_calc

        log_info(f"Quality report generated. Score: {report['quality_score']:.1f}")
        log_debug(f"Quality issues found: {len(quality_issues)}")

        return {
            "output": (
                f"Data Quality Report for {file_name}:\n"
                f"{json.dumps(report, indent=2)}"
            ),
            "report": report,
            "should_stop": True,
        }

    except Exception as e:  # noqa: BLE001
        log_critical(f"Critical error during quality assessment: {e!s}")
        return {"output": f"Quality report error: {e!s}", "should_stop": True}
