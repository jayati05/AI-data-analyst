"""Visualize data with various plot types."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from logging_client import log_debug, log_error, log_info
from tools.data_loading import load_data


def visualize_data(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0911
    file_name: str,
    plot_type: str,
    y_col: str,
    x_col: str | None = None,
    group_by: str | None = None,
    title: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, str | bool | None]:
    """Create visualizations from data with robust error handling and logging."""
    log_info(f"Starting visualization: {plot_type} plot for {file_name}")

    # Initialize response
    response = {
        "output": "",
        "plot_file": None,
        "should_stop": True,
    }

    try:
        # 1. Data Loading
        log_debug(f"Loading data from {file_name}")
        data_frame = load_data(file_name)
        if isinstance(data_frame, str):
            log_error(f"Data loading failed: {data_frame}")
            response["output"] = f"Data loading error: {data_frame}"
            return response

        log_info(f"Data loaded successfully. Shape: {data_frame.shape}")
        log_debug(f"Data sample:\n{data_frame.head(2).to_string()}")

        # 2. Input Validation
        if not isinstance(data_frame, pd.DataFrame):
            error_msg = "Loaded data is not a pandas DataFrame"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # Standardize column names
        data_frame.columns = data_frame.columns.str.strip().str.lower()
        y_col = y_col.strip().lower()
        x_col = x_col.strip().lower() if x_col else None
        group_by = group_by.strip().lower() if group_by else None
        log_debug(
            "Standardized columns - "
            f"y_col: {y_col}, x_col: {x_col}, group_by: {group_by}",
        )

        # 3. Plot Type Validation
        plot_type = plot_type.lower().strip()
        valid_plot_types = {
            "histogram", "hist", "bar", "bar chart",
            "scatter", "scatter plot", "line", "line chart",
            "box", "boxplot", "box plot", "pie", "pie chart",
        }

        if plot_type not in valid_plot_types:
            error_msg = f"Unsupported plot type: {plot_type}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # Map to canonical plot types
        plot_type_map = {
            "histogram": "hist", "hist": "hist",
            "bar": "bar", "bar chart": "bar",
            "scatter": "scatter", "scatter plot": "scatter",
            "line": "line", "line chart": "line",
            "box": "box", "boxplot": "box", "box plot": "box",
            "pie": "pie", "pie chart": "pie",
        }
        plot_type = plot_type_map[plot_type]
        log_debug(f"Canonical plot type: {plot_type}")

        # 4. Column Validation
        required_cols = [y_col]
        if plot_type in ["scatter", "line"] and not x_col:
            error_msg = f"x_col is required for {plot_type} plot"
            log_error(error_msg)
            response["output"] = error_msg
            return response
        if x_col:
            required_cols.append(x_col)
        if group_by:
            required_cols.append(group_by)

        missing_cols = [col for col in required_cols if col not in data_frame.columns]
        if missing_cols:
            error_msg = f"Columns not found: {missing_cols}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # 5. Data Preparation
        try:
            if plot_type in ["hist", "box", "scatter", "line"]:
                data_frame[y_col] = pd.to_numeric(data_frame[y_col], errors="raise")
                if x_col:
                    data_frame[x_col] = pd.to_numeric(data_frame[x_col], errors="raise")
            log_debug("Numeric conversion successful")
        except ValueError as e:
            error_msg = f"Numeric conversion error: {e!s}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

        # 6. Plot Generation
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        plot_filename = plot_dir / f"{plot_type}_plot_{timestamp}.png"
        log_debug(f"Preparing to save plot to: {plot_filename}")
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        plt.grid(visible=True, linestyle="--", alpha=0.7)

        # Generate default title if not provided
        if not title:
            title_map = {
                "bar": f"Average {y_col} by {x_col if x_col else group_by}",
                "hist": f"Distribution of {y_col}",
                "scatter": f"{y_col} vs {x_col}",
                "line": f"{y_col} over {x_col}",
                "box": f"Distribution of {y_col}",
                "pie": f"Proportion of {y_col}",
            }
            title = title_map.get(plot_type, f"{plot_type} plot of {y_col}")
        log_debug(f"Using title: {title}")

        try:
            if plot_type == "bar":
                if group_by:
                    plot_data = data_frame.groupby(group_by)[y_col].mean()
                elif x_col:
                    plot_data = data_frame.groupby(x_col)[y_col].mean()
                else:
                    plot_data = data_frame[y_col].value_counts()

                plot_data.plot(
                    kind="bar",
                    color=kwargs.get("color", "skyblue"),
                    edgecolor="black",
                )
                plt.ylabel(y_col)

            elif plot_type == "hist":
                data_frame[y_col].plot(
                    kind="hist",
                    bins=kwargs.get("bins", "auto"),
                    color=kwargs.get("color", "lightgreen"),
                    edgecolor="black",
                )
                plt.xlabel(y_col)

            elif plot_type == "scatter":
                data_frame.plot.scatter(
                    x=x_col,
                    y=y_col,
                    color=kwargs.get("color", "coral"),
                    alpha=kwargs.get("alpha", 0.7),
                )

            elif plot_type == "line":
                data_frame.plot.line(
                    x=x_col,
                    y=y_col,
                    marker=kwargs.get("marker", "o"),
                    color=kwargs.get("color", "royalblue"),
                )

            elif plot_type == "box":
                if group_by:
                    data_frame.boxplot(
                        column=y_col,
                        by=group_by,
                        patch_artist=True,
                        boxprops={"facecolor": kwargs.get("color", "lightyellow")},
                    )
                else:
                    data_frame[y_col].plot(
                        kind="box",
                        patch_artist=True,
                        boxprops={"facecolor": kwargs.get("color", "lightyellow")},
                    )

            elif plot_type == "pie":
                if group_by:
                    plot_data = data_frame.groupby(group_by)[y_col].sum()
                else:
                    plot_data = data_frame[y_col].value_counts()

                plot_data.plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=kwargs.get("colors", plt.cm.Pastel1.colors),
                    startangle=90,
                )
                plt.ylabel("")

            plt.title(title)
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            success_msg = f"Successfully created {plot_type} plot: {plot_filename}"
            log_info(success_msg)
            response.update({
                "output": success_msg,
                "plot_file": str(plot_filename),
                "should_stop": True,
            })

        except RuntimeError as plot_error:
            plt.close()
            error_msg = f"Plot generation error: {plot_error!s}"
            log_error(error_msg)
            response["output"] = error_msg
            return response

    except (ValueError, TypeError) as e:
        error_msg = f"Visualization error: {e!s}"
        log_error(error_msg)
        response["output"] = error_msg
        return response

    return response


def aggregate_and_visualize(  # noqa: PLR0915, PLR0911
    file_name: str,
    value_col: str,
    group_by: str,
    plot_type: str = "bar",
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Aggregate data and create visualization with comprehensive logging."""
    log_info(f"Starting aggregate_and_visualize for {file_name}")
    log_debug(
        f"Params - value_col: {value_col}, "
        f"group_by: {group_by}, plot_type: {plot_type}",
    )
    try:
        # 1. Load and validate data
        log_debug(f"Loading data from {file_name}")
        data_frame = load_data(file_name)
        if isinstance(data_frame, str):
            log_error(f"Data loading failed: {data_frame}")
            return {"output": data_frame, "plot_file": None, "should_stop": True}

        log_info(f"Data loaded successfully. Shape: {data_frame.shape}")

        # 2. Standardize column names
        try:
            data_frame.columns = data_frame.columns.str.strip().str.lower()
            value_col = value_col.strip().lower()
            group_by = group_by.strip().lower()
            log_debug(
                f"Standardized columns - value_col: {value_col}, group_by: {group_by}",
            )
        except AttributeError:
            error_msg = "Invalid column names provided"
            log_error(error_msg)
            return {"output": error_msg, "plot_file": None, "should_stop": True}

        # 3. Validate columns exist
        missing_cols = [
            col
            for col in [value_col, group_by]
            if col not in data_frame.columns
        ]
        if missing_cols:
            error_msg = f"Columns not found: {missing_cols}"
            log_error(error_msg)
            return {
                "output": error_msg,
                "plot_file": None,
                "should_stop": True,
            }

        # 4. Determine appropriate aggregation
        if pd.api.types.is_numeric_dtype(data_frame[value_col]):
            log_debug(f"{value_col} is numeric, computing mean.")
            aggregated = data_frame.groupby(
                        group_by, as_index=False)[value_col].mean().reset_index()
            agg_col_name = f"avg_{value_col}"
        else:
            log_debug(f"{value_col} is categorical, computing count.")
            aggregated = data_frame.groupby(
                        group_by, as_index=False)[value_col].count().reset_index()
            agg_col_name = f"count_{value_col}"


        if aggregated.shape[1] == 1:
            aggregated[agg_col_name] = 0

        aggregated.columns = [group_by, agg_col_name]
        log_debug(f"Aggregation successful. Result shape: {aggregated.shape}")

        # 5. Generate visualization
        try:
            plot_dir = Path("plots")
            plot_dir.mkdir(exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            plot_filename = plot_dir / f"agg_{plot_type}plot{timestamp}.png"
            log_debug(f"Preparing to save plot to: {plot_filename}")

            plt.figure(figsize=kwargs.get("figsize", (10, 6)))
            plt.grid(visible=True, linestyle="--", alpha=0.7)

            if plot_type == "bar":
                aggregated.plot.bar(
                    x=group_by,
                    y=agg_col_name,
                    color=kwargs.get("color", "skyblue"),
                    edgecolor="black",
                )
                plt.title(
                    f"{agg_col_name.replace('_', ' ').title()} "
                    f"by {group_by.title()}",
                )
                plt.ylabel(agg_col_name.replace("_", " ").title())
            else:
                error_msg = f"Unsupported plot type for aggregation: {plot_type}"
                log_error(error_msg)
                return {
                    "output": error_msg,
                    "plot_file": None,
                    "should_stop": True,
                }

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            success_msg = f"Successfully created {plot_type} plot: {plot_filename}"
            log_info(success_msg)
            return {
                "output": success_msg,
                "plot_file": str(plot_filename),
                "should_stop": True,
            }

        except RuntimeError as e:
            plt.close()
            error_msg = f"Plot generation error: {e!s}"
            log_error(error_msg)
            return {"output": error_msg, "plot_file": None, "should_stop": True}

    except (ValueError, TypeError) as e:
        error_msg = f"Unexpected error in aggregate_and_visualize: {e!s}"
        log_error(error_msg)
        return {"output": error_msg, "plot_file": None, "should_stop": True}
