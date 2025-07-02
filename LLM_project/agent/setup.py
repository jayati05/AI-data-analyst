"""Define tools for the agent.

Includes configuration for structured tools and callbacks.
"""

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMMathChain
from langchain.tools import StructuredTool
from langchain_ollama import OllamaLLM

from agent.executor import CustomAgentExecutor
from logging_client import log_debug, log_info

# Import functions from tools.py
from tools import (
    correlation_analysis,
    data_quality,
    list_files,
)
from tools.data_analysis import (
    aggregate_data,
    check_missing_values,
    detect_outliers,
    filter_data,
    get_columns,
    show_data_sample,
    sort_data,
    summary_statistics,
    transform_data,
)
from tools.visualization import aggregate_and_visualize, visualize_data


def final_answer(output: str) -> str:
    """Stop execution and return the final answer."""
    log_debug(f"FinalAnswer invoked with output: {output}")
    return output


def initialize_custom_agent() -> CustomAgentExecutor:
    """Initialize the custom agent.

    Sets up the callback manager and LLM, and configures all available tools.
    Returns a configured CustomAgentExecutor instance.
    """
    log_debug("Initializing callback manager and LLM...")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = OllamaLLM(
        model="llama3:8b",
        temperature=0,
        callback_manager=callback_manager,
    )

    log_debug("Initializing Math tool...")
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    log_info("Setting up tools...")
    tools = [
        StructuredTool.from_function(
            func=list_files,
            name="ListFiles",
            description="List all available data files in the data directory. "
                    "No input required. Returns a list of filenames.",
        ),
        StructuredTool.from_function(
            func=check_missing_values,
            name="CheckMissingValues",
            description="Analyze missing values in a data file. Input should be a "
                    "JSON object with: 'file_name' (string, required)",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=detect_outliers,
            name="DetectOutliers",
            description="Identify outliers in data columns using statistical methods. "
                    "Input should be a JSON object with: 'file_name' (string, "
                    "required), 'column' (string, required), 'method' (string, "
                    "optional: 'iqr' or 'zscore', default='iqr'), 'threshold' "
                    "(number, optional: default 1.5 for IQR, 3 for Z-score)",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=filter_data,
            name="FilterData",
            description="Filter data based on conditions (mapped to 'operations' automatically). "
        "Input should be a JSON object with:\n"
        "- 'file_name' (string, required)\n"
        "- 'operations' (list of dictionaries, required)\n"
        "Each dictionary must contain:\n"
        "  - 'column' (string)\n"
        "  - 'operator' (string: '==', '!=', '>', '<', '>=', '<=')\n"
        "  - 'value' (any type)\n"
        "If 'conditions' is provided instead of 'operations', it will be used automatically.",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=aggregate_data,
            name="AggregateData",
            description=(
                "Perform aggregation on a dataset to compute summary statistics. "
                "Use this tool to count rows, calculate min/max, sum, or mean values. "
                "Input should be a JSON object with: 'file_name' (string, required), "
                "'column' (string, optional; required for all functions except 'count'), "
                "'agg_funcs' (list of strings, required: supported functions are 'mean', "
                "'max', 'min', 'sum', and 'count'). "
                "If 'count' is used without specifying a column, it counts the total number of rows."
            ),
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=sort_data,
            name="SortData",
            description="Sorts data by specified columns in ascending or descending "
                    "order, with optional filtering. Input should be a JSON "
                    "object with: 'file_name' (string, required), 'columns' "
                    "(array of strings, required), 'order' (string, required: "
                    "'asc' for ascending, 'desc' for descending), 'filter' "
                    "(dictionary, optional, containing filtering rules).",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=summary_statistics,
            name="SummaryStatistics",
            description="Calculate summary statistics for all numeric columns in a "
                    "file. Input should be a JSON object with: 'file_name' "
                    "(string, required)",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=aggregate_and_visualize,
            name="AggregateAndVisualize",
            description="Aggregates data by specified column and creates "
                    "visualization. Required parameters: file_name, value_col "
                    "(column to average), group_by (column to group by). "
                    "Optional: plot_type (default: 'bar').",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=visualize_data,
            name="VisualizeData",
            description="Create visualizations. Input should be a dictionary with "
                    "'file_name' (str), 'plot_type' (str), 'y_col' (str), and "
                    "optionally 'x_col' (str).",
        ),
        StructuredTool.from_function(
            func=data_quality,
            name="DataQualityReport",
            description="Generate a data quality report. Input should be a "
                    "dictionary with 'file_name' (str).",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=correlation_analysis,
            name="CorrelationAnalysis",
            description="Calculate correlations between columns. Input should be a "
                    "dictionary with 'file_name' (str) and 'cols' (list of str).",
        ),
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="Useful for math calculations. Input should be a math "
                    "expression as a string.",
        ),
        StructuredTool.from_function(
            func=transform_data,
            name="TransformData",
            description="Create new columns or modify data. Input: dict with "
                    "'file_name', 'operations' (list of transforms like "
                    "'salary*0.1 as bonus')",
        ),
        StructuredTool.from_function(
            func=show_data_sample,
            name="ShowDataSample",
            description="Show sample rows from a file. Input: dict with 'file_name' "
                    "and optionally 'num_rows', 'columns'",
        ),
        StructuredTool.from_function(
            func=get_columns,
            name="GetColumns",
            description="List columns and data types for a file. Input: dict with "
                    "'file_name'",
            return_direct=True,
        ),
        StructuredTool.from_function(
            func=final_answer,
            name="FinalAnswer",
            description="Stops execution and returns the final answer. Input should "
                    "be the final answer string.",
            return_direct=True,
        ),
    ]

    log_info("Initializing agent...")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )

    log_info("Creating custom agent executor...")
    custom_agent_executor = CustomAgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools=tools,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True,
    )
    log_info("Custom agent executor successfully initialized.")
    return custom_agent_executor
