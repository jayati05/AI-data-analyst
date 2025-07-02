"""Data analysis module for performing various operations on datasets."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from logging_client import log_debug, log_error, log_info, log_warning
from tools.data_loading import load_data


def get_columns(file_name: str) -> dict[str, Any]:
    """List columns and data types with better formatting."""
    log_info(f"Getting columns for file: {file_name}")
    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error(f"Data loading error: {dataframe}")
            return {"output": f"Data loading error: {dataframe}", "should_stop": True}

        column_info = [f"Columns in {file_name}:"]
        for col, dtype in dataframe.dtypes.items():
            column_info.append(f"- {col}: {dtype!s}")

        sample = dataframe.head(3).to_dict("records")
        log_debug(f"Retrieved sample data: {sample[:1]}")

        result = {
            "output": "\n".join(column_info),
            "columns": list(dataframe.columns),
            "dtypes": dict(dataframe.dtypes),
            "sample": sample,
            "should_stop": True,
        }
        log_info("Successfully retrieved column information")
    except Exception as e:  # noqa: BLE001
        log_error(f"Error getting columns for {file_name}")
        return {"output": f"Error getting columns: {e!s}", "should_stop": True}
    else:
        return result


def check_missing_values(file_name: str) -> dict[str, Any]:
    """Analyze and report missing values in a data file."""
    log_info(f"Checking missing values for file: {file_name}")
    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error(f"Data loading error: {dataframe}")
            return {"output": f"Data loading error: {dataframe}", "should_stop": True}

        missing = dataframe.isna().sum()
        missing_pct = (missing / len(dataframe)) * 100
        log_debug(f"Missing values calculated: {missing.to_dict()}")

        results = [
            f"- {col}: {missing[col]} missing ({missing_pct[col]:.1f}%)"
            for col in dataframe.columns
        ]

        result = {
            "output": f"Missing values in {file_name}:\n" + "\n".join(results),
            "missing_counts": missing.to_dict(),
            "missing_percentages": missing_pct.to_dict(),
            "should_stop": True,
        }
        log_info("Missing values analysis completed")
    except Exception as e:  # noqa: BLE001
        log_error(f"Error checking missing values in {file_name}")
        return {"output": f"Error checking missing values: {e!s}", "should_stop": True}
    else:
        return result


def detect_outliers(
    file_name: str,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict:
    """Detect outliers in specified column of a data file."""
    log_info("Detecting outliers")
    try:
        dataframe = load_data(file_name)
        log_info("Data loaded successfully")

        dataframe.columns = dataframe.columns.str.strip().str.lower()
        column = column.strip().lower()
        log_debug("Standardized column name")

        if column not in dataframe.columns:
            error_msg = (
                f"Column '{column}' not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        try:
            data = pd.to_numeric(dataframe[column], errors="raise")
            log_debug("Column converted to numeric successfully")
        except ValueError:
            error_msg = f"Column '{column}' contains non-numeric values"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = dataframe[(data < lower_bound) | (data > upper_bound)]
            stats = f"IQR: {iqr:.2f}, Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
            log_debug("IQR method stats")
        elif method == "zscore":
            z_scores = (data - data.mean()) / data.std()
            outliers = dataframe[abs(z_scores) > threshold]
            stats = f"Mean: {data.mean():.2f}, Std: {data.std():.2f}"
            log_debug("Z-score method")
        else:
            error_msg = f"Invalid method '{method}'. Use 'iqr' or 'zscore'"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        if outliers.empty:
            result = {
                "output": (
                    f"No outliers found in '{column}' "
                    f"using {method} (threshold: {threshold})"
                ),
                "should_stop": True,
            }
            log_info("No outliers found")
            return result

        output = (
            f"Found {len(outliers)} outliers in '{column}':\n"
            f"Method: {method} (threshold: {threshold})\n"
            f"Statistics: {stats}\n"
            f"Sample outliers:\n{outliers.head().to_string()}"
        )

        result = {
            "output": output,
            "outliers": outliers.to_dict("records"),
            "should_stop": True,
        }
        log_info("Outlier detection completed: %s outliers found", len(outliers))
    except Exception as e:  # noqa: BLE001
        log_error(f"Error detecting outliers in {file_name}")
        return {
            "output": f"Error detecting outliers: {e!s}",
            "should_stop": True,
        }
    else:
        return result


def show_data_sample(
    file_name: str,
    num_rows: int = 5,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Show sample rows from a data file."""
    log_info(
        "Showing data sample",
    )
    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error("Data loading error: %s", dataframe)
            return {"output": f"Data loading error: {dataframe}", "should_stop": True}

        if columns:
            missing = [col for col in columns if col not in dataframe.columns]
            if missing:
                error_msg = f"Columns not found: {missing}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
            dataframe = dataframe[columns]
            log_debug("Filtered to specified columns")

        sample = dataframe.head(num_rows)
        sample_list = sample.to_dict("records")
        log_debug("Retrieved sample data")

        output_lines = [f"First {num_rows} rows of {file_name}:"]
        for record in sample_list:
            output_lines.append("\n".join(f"{k}: {v}" for k, v in record.items()))
            output_lines.append("-" * 20)

        result = {
            "output": "\n".join(output_lines),
            "sample": sample_list,
            "should_stop": True,
        }
        log_info("Data sample retrieved successfully")
    except Exception as e:  # noqa: BLE001
        log_error(f"Error showing data from {file_name}")
        return {"output": f"Error showing data: {e!s}", "should_stop": True}
    else:
        return result


def transform_data(
    file_name: str,
    operations: list[dict[str, Any]] | list[str],
    output_file: str | None = None,
) -> dict[str, Any]:
    """Transform data by creating new columns or modifying existing ones."""
    log_info(f"Transforming data in {file_name}")

    try:
        # Load and validate data
        dataframe = _load_and_validate_data(file_name)
        if isinstance(dataframe, dict):
            return dataframe

        # Format operations
        formatted_ops = _format_operations(operations)
        if isinstance(formatted_ops, dict):
            return formatted_ops

        # Apply transformations
        error = _apply_transformations(dataframe, formatted_ops)
        if error:
            return error

        # Handle output
        return _handle_output(dataframe, output_file)

    except OSError as e:
        error_msg = f"Transformation error: {e!s}"
        log_error(error_msg)
        return {"output": error_msg, "should_stop": True}


def _load_and_validate_data(file_name: str) -> pd.DataFrame | dict[str, Any]:
    """Load and validate input data."""
    dataframe = load_data(file_name)
    if isinstance(dataframe, str):
        error_msg = f"Data loading error: {dataframe}"
        log_error(error_msg)
        return {"output": error_msg, "should_stop": True}
    return dataframe


def _format_operations(
    operations: list[dict[str, Any]] | list[str],
) -> list[str] | dict[str, Any]:
    """Format operations into consistent string format."""
    formatted_ops = []
    for operation in operations:
        if isinstance(operation, dict):
            if "column" not in operation or "expression" not in operation:
                error_msg = f"Invalid operation format: {operation}"
                log_error(error_msg)
                return {"output": error_msg, "should_stop": True}
            formatted_ops.append(f"{operation['expression']} AS {operation['column']}")
        elif isinstance(operation, str):
            formatted_ops.append(operation)
        else:
            error_msg = f"Unsupported operation format: {operation}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}
    return formatted_ops


def _apply_transformations(
    dataframe: pd.DataFrame, operations: list[str],
) -> dict[str, Any] | None:
    """Apply all transformations to the dataframe."""
    for operation in operations:
        expr, new_col = operation.split(" AS ")
        expr = expr.strip()
        new_col = new_col.strip()

        result = dataframe.eval(expr, engine="python")
        if not isinstance(result, pd.Series):
            error_msg = f"Error executing '{operation}'"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        dataframe[new_col] = result
        log_debug(f"Applied transformation: {operation}")
    return None


def _handle_output(
    dataframe: pd.DataFrame, output_file: str | None,
) -> dict[str, Any]:
    """Handle the output of the transformation."""
    if output_file:
        save_path = f"data/{output_file}"
        dataframe.to_csv(save_path, index=False)
        result = {
            "output": f"Transformed data saved to {save_path}",
            "file_path": save_path,
            "should_stop": False,
        }
        log_info(f"Data transformed and saved to {save_path}")
    else:
        sample = dataframe.head().to_dict("records")
        result = {
            "output": f"Transformation done. Sample: {json.dumps(sample, indent=2)}",
            "sample": sample,
            "should_stop": True,
        }
        log_info("Data transformation completed")
    return result


def _apply_filter_operation(
    dataframe: pd.DataFrame,
    column: str,
    operator: str,
    value: float,
) -> pd.DataFrame:
    """Apply a single filter operation to the dataframe."""
    col_data = pd.to_numeric(dataframe[column], errors="ignore")

    if operator == ">":
        return dataframe[col_data > value]
    if operator == ">=":
        return dataframe[col_data >= value]
    if operator == "<":
        return dataframe[col_data < value]
    if operator == "<=":
        return dataframe[col_data <= value]
    if operator == "==":
        return dataframe[col_data == value]
    if operator == "!=":
        return dataframe[col_data != value]
    error_msg = f"Invalid operator '{operator}'"
    raise ValueError(error_msg)

def filter_data(file_name: str, operations: list[dict]) -> dict:
    """Filter data rows based on operations."""
    log_info("Filtering data")
    try:
        dataframe = load_data(file_name)
        log_debug(f"Data loaded successfully, shape: {dataframe.shape}")
        dataframe.columns = dataframe.columns.str.strip().str.lower()

        if not operations or not all(
            isinstance(op, dict) and
            all(k in op for k in ["column", "operator", "value"])
            for op in operations
        ):
            error_msg = "Each operation must be {column, operator, value}"
            log_error(error_msg)  # ✅ Fixed incorrect logging
            return {"output": error_msg, "should_stop": True}

        filtered_df = dataframe.copy()
        for op in operations:
            column = op["column"].strip().lower()
            if column not in filtered_df.columns:
                error_msg = (
                    f"Column '{column}' not found. "
                    f"Available: {list(filtered_df.columns)}"
                )
                log_error(error_msg)  # ✅ Fixed incorrect logging
                return {"output": error_msg, "should_stop": True}

            try:
                filtered_df = _apply_filter_operation(
                    filtered_df,
                    column,
                    op["operator"],
                    op["value"],
                )
                log_debug(
                    f"Applied filter: {column} {op['operator']} {op['value']}",
                )
            except ValueError as e:
                error_msg = f"Error filtering {column}: {e}"
                log_error(error_msg)  # ✅ Fixed incorrect logging
                return {"output": error_msg, "should_stop": True}

        result = {
            "output": f"Found {len(filtered_df)} matching rows",
            "filtered_data": filtered_df.to_dict("records"),
            "should_stop": True,
        }

        if not filtered_df.empty:
            result["output"] += ":\n" + filtered_df.head().to_string()
            log_debug(f"Filter results sample:\n{filtered_df.head().to_string()}")

        log_info("Data filtering completed successfully")

    except (OSError) as e:
        log_error(f"Error filtering data in {file_name}")  # ✅ Fixed incorrect logging
        return {"output": f"Data processing error: {e!s}", "should_stop": True}

    else:
        return result

def _format_aggregation_results(column: str, results: dict) -> str:
    """Format aggregation results into readable output."""
    output = [f"Aggregation results for column '{column}':"]
    for func, value in results.items():
        if isinstance(value, dict):
            output.append(f"- {func} (grouped):")
            output.extend(f"  • {k}: {v}" for k, v in value.items())
        else:
            output.append(f"- {func}: {value}")
    return "\n".join(output)

def aggregate_data(file_name: str, column: str = None,
                   agg_funcs: list = None, group_by: str = None) -> dict:
    """Perform aggregation operations on a specified column."""
    log_info(f"Aggregating data from {file_name}")

    valid_funcs = {
        "count": "count", "mean": "mean", "max": "max",
        "min": "min", "sum": "sum", "std": "std",
        "median": "median", "nunique": "nunique",
    }

    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error(f"Data loading error: {dataframe}")
            return {"output": f"Data loading error: {dataframe}", "should_stop": True}

        log_debug(f"Columns available in {file_name}: {list(dataframe.columns)}")
        dataframe.columns = dataframe.columns.str.strip().str.lower()

        # If column is None or empty, perform row count
        if not column:
            if "count" in agg_funcs:
                count_result = len(dataframe)
                return {"output": f"Total row count: {count_result}",
                        "results": {"count": count_result}, "should_stop": True}
            error_msg = "Column is required unless performing a row count with 'count'."
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        column = column.strip().lower()
        if group_by:
            group_by = group_by.strip().lower()

        if column not in dataframe.columns:
            error_msg = f"Column '{column}' not found. Available: {list(dataframe.columns)}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        if group_by and group_by not in dataframe.columns:
            error_msg = f"Group-by column '{group_by}' not found."
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        invalid_funcs = [f for f in agg_funcs if f.lower() not in valid_funcs]
        if invalid_funcs:
            error_msg = f"Unsupported aggregation functions: {invalid_funcs}"
            log_error(error_msg)
            return {"output": error_msg, "should_stop": True}

        if group_by:
            results = dataframe.groupby(group_by)[column].agg(
                [valid_funcs[f.lower()] for f in agg_funcs])
            results = results.to_dict(orient="index")
        else:
            results = {func: getattr(dataframe[column],
                                     valid_funcs[func.lower()])() for func in agg_funcs}

        output = _format_aggregation_results(column, results)
    except (OSError) as e:
        log_error(f"Error aggregating data: {e!s}")
        return {"output": f"Aggregation error: {e!s}", "should_stop": True}
    else:
        return {"output": output, "results": results, "should_stop": True}

def _apply_filter_to_dataframe(
    dataframe: pd.DataFrame,
    filter_dict: dict[str, dict[str, str | int | float]],
) -> pd.DataFrame:
    """Apply filtering to dataframe based on filter_dict."""
    filter_column, filter_details = next(iter(filter_dict.items()))
    filter_column = filter_column.strip().lower()
    filter_operator = filter_details.get("op", "").strip()
    filter_value = filter_details.get("val")

    if filter_column not in dataframe.columns:
        error_msg = f"Filter column '{filter_column}' not found"
        raise ValueError(error_msg)

    dataframe[filter_column] = pd.to_numeric(dataframe[filter_column], errors="coerce")
    dataframe = dataframe.dropna(subset=[filter_column])

    if filter_operator == ">":
        return dataframe[dataframe[filter_column] > filter_value]
    if filter_operator == ">=":
        return dataframe[dataframe[filter_column] >= filter_value]
    if filter_operator == "<":
        return dataframe[dataframe[filter_column] < filter_value]
    if filter_operator == "<=":
        return dataframe[dataframe[filter_column] <= filter_value]
    if filter_operator in ["=", "=="]:
        return dataframe[dataframe[filter_column] == filter_value]
    error_msg = f"Unsupported filter operator: {filter_operator}"
    raise ValueError(error_msg)

def sort_data(
    file_name: str | dict,
    columns: list[str],
    order: str,
    filter_dict: dict[str, dict[str, str | int | float]] | None = None,
) -> dict:
    """Sort data by specified columns with optional filtering."""
    try:
        if isinstance(file_name, dict) and "value" in file_name:
            file_name = file_name["value"]

        dataframe = load_data(file_name.strip())
        if isinstance(dataframe, str):
            return {"error": dataframe}

        dataframe.columns = dataframe.columns.str.strip().str.lower()
        columns = [col.strip().lower() for col in columns]

        missing_cols = [col for col in columns if col not in dataframe.columns]
        if missing_cols:
            return {"error": f"Columns not found: {missing_cols}"}

        if filter_dict:
            try:
                dataframe = _apply_filter_to_dataframe(dataframe, filter_dict)
            except ValueError as e:
                return {"error": str(e)}

        ascending = order.lower() == "asc"
        df_sorted = dataframe.sort_values(by=columns, ascending=ascending)

        result = {
            "output": f"Sorted by {columns} in {order} order",
            "data": df_sorted.to_dict("records"),
            "should_stop": True,
        }
        if not dataframe.empty:
            result["output"] += ":\n" + df_sorted.head().to_string()
        log_debug(f"Filter results sample: {df_sorted.head().to_string()}")

        log_info("Data sorting completed successfully")

    except Exception as e:  # noqa: BLE001
        return {"error": f"Sorting error: {e!s}"}

    else:
        return result


def summary_statistics(file_name: str) -> dict:
    """Calculate summary statistics for all numeric columns in a file."""
    log_info("Calculating summary statistics")
    if isinstance(file_name, dict):
        file_name = file_name.get("title", "")
    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error(f"Data loading error: {dataframe}")
            return {"output": f"Data loading error: {dataframe}", "should_stop": True}

        numeric_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            log_warning("No numeric columns found")
            return {"output": "No numeric columns found", "should_stop": True}

        results = {}
        for col in numeric_cols:
            results[col] = {
                "mean": dataframe[col].mean(),
                "max": dataframe[col].max(),
                "min": dataframe[col].min(),
                "sum": dataframe[col].sum(),
                "count": dataframe[col].count(),
            }
            log_debug("Calculated stats")

        output = "Summary statistics:\n"
        for col, stats in results.items():
            output += f"\nColumn: {col}\n"
            output += "\n".join(
                [
                    f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in stats.items()
                ],
            )
            output += "\n"

        result = {
            "output": output,
            "results": results,
            "should_stop": True,
        }
        log_info("Summary statistics calculated successfully")
    except Exception as e:  # noqa: BLE001
        log_error("Error calculating summary statistics")
        return {"output": f"Summary statistics error: {e!s}", "should_stop": True}
    else:
        return result


def correlation_analysis(file_name: str, cols: list[str]) -> dict:
    """Calculate correlation between specified columns."""
    log_info(f"Calculating correlations in {file_name}, columns: {cols}")

    try:
        dataframe = load_data(file_name)
        if isinstance(dataframe, str):
            log_error(f"Data loading error: {dataframe}")
            return {"error": dataframe}

        dataframe.columns = dataframe.columns.str.strip().str.lower()
        cols = [col.strip().lower() for col in cols]
        log_debug(f"Standardized column names: {cols}")

        missing_cols = [col for col in cols if col not in dataframe.columns]
        if missing_cols:
            error_msg = (
                f"Columns not found: {missing_cols}. "
                f"Available: {list(dataframe.columns)}"
            )
            log_error(error_msg)
            return {"error": error_msg}

        numeric_cols = [
            col for col in cols if pd.api.types.is_numeric_dtype(dataframe[col])
        ]
        if not numeric_cols:
            log_warning("No numeric columns found for correlation analysis")
            return {"error": "No numeric columns found for correlation analysis"}

        # Convert columns to numeric if they are not
        for col in numeric_cols:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

        dataframe = dataframe.dropna(subset=numeric_cols)
        corr_matrix = dataframe[numeric_cols].corr()
        log_debug(f"Correlation matrix:\n{corr_matrix}")

        result = {
            "status": "success",
            "correlation_matrix": corr_matrix.to_dict(),
            "columns_used": numeric_cols,
            "should_stop": True,
        }
    except (OSError) as e:
        log_error(f"Error calculating correlations in {file_name}: {e}")
        return {"error": f"Correlation analysis error: {e!s}"}
    else:
        log_info("Correlation analysis completed successfully")
        return result
