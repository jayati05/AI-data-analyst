"""FastAPI application for data analysis with agent integration."""
from __future__ import annotations

import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import execute_agent_query, initialize_custom_agent
from logging_client import log_debug, log_error, log_info, log_success, log_warning

app = FastAPI(title="Data Analysis API")

class QueryRequest(BaseModel):
    """Request model for query execution."""

    query: str
    session_id: str | None = None

class TestQueryResponse(BaseModel):
    """Response model for test queries."""

    query: str
    output: str
    steps: list[str]
    success: bool
    processing_time: float | None = None

def _raise_initialization_error(msg: str) -> None:
    """Raise initialization errors with logging.

    Args:
        msg: Error message to log and raise

    """
    log_error(msg)
    raise RuntimeError(msg)

def _create_test_query_response(
    query: str,
    result: dict[str, Any],
    processing_time: float,
) -> TestQueryResponse:
    """Create a successful test query response.

    Args:
        query: The original query string
        result: Dictionary containing query results
        processing_time: Time taken to process the query

    Returns:
        TestQueryResponse: Formatted response object

    """
    return TestQueryResponse(
        query=query,
        output=result.get("output", "No results"),
        steps=result.get("intermediate_steps", []),
        success=True,
        processing_time=processing_time,
    )

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the agent with detailed logging."""
    try:
        log_info("Starting API initialization")
        start_time = time.time()

        app.state.agent = initialize_custom_agent()

        if not app.state.agent:
            _raise_initialization_error("Agent initialization returned None")

        init_time = time.time() - start_time
        log_success(f"API initialized successfully in {init_time:.2f} seconds")
        log_debug(f"Agent configuration: {str(app.state.agent)[:200]}...")

    except Exception as e:
        log_error(f"API startup failed: {e!s}")
        raise

def _handle_test_query_failure(query: str, error: Exception) -> TestQueryResponse:
    """Create a failed test query response with logging.

    Args:
        query: The original query string
        error: Exception that occurred

    Returns:
        TestQueryResponse: Formatted error response

    """
    log_warning(f"Test query failed: {error!s}")
    return TestQueryResponse(
        query=query,
        output=str(error),
        steps=[],
        success=False,
    )

def _process_single_query(query: str) -> TestQueryResponse:
    """Process a single test query with error handling.

    Args:
        query: Query string to process

    Returns:
        TestQueryResponse: Result of the query processing

    """
    try:
        log_debug(f"Processing test query: {query[:50]}...")
        start_time = time.time()
        result = execute_agent_query(app.state.agent, query)
        processing_time = time.time() - start_time
        log_info(f"Query completed in {processing_time:.2f}s")
        return _create_test_query_response(query, result, processing_time)
    except (ValueError, RuntimeError) as e:
        return _handle_test_query_failure(query, e)

@app.post("/execute-query", response_model=TestQueryResponse)
async def execute_query(request: QueryRequest) -> TestQueryResponse:
    """Execute a single query with full logging.

    Args:
        request: QueryRequest object containing the query

    Returns:
        TestQueryResponse: Result of the query execution

    Raises:
        HTTPException: If query execution fails

    """
    try:
        log_info(f"New query received from session {request.session_id}")
        log_debug(f"Query content: {request.query[:100]}...")
        start_time = time.time()

        result = execute_agent_query(app.state.agent, request.query)
        processing_time = time.time() - start_time

        log_info(f"Query processed in {processing_time:.2f} seconds")
        log_debug(f"Query result sample: {str(result)[:200]}...")

        if not result.get("output"):
            log_warning("Query returned no output")

        return _create_test_query_response(request.query, result, processing_time)

    except Exception as e:
        log_error(f"Query execution failed: {e!s}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "query": request.query,
                "error": str(e),
                "success": False,
            },
        ) from e

@app.get("/run-test-suite", response_model=list[TestQueryResponse])
async def run_test_suite() -> list[TestQueryResponse]:
    """Execute test suite with detailed logging.

    Returns:
        list[TestQueryResponse]: List of test query results

    """
    test_queries = [
        "What is the average and maximum salary in 'test.csv'?",
        "Count employees by department in 'employees.csv'",
    ]

    log_info(f"Starting test suite with {len(test_queries)} queries")
    results = [_process_single_query(query) for query in test_queries]

    success_count = len([r for r in results if r.success])
    log_info(f"Test suite completed: {success_count}/{len(results)} successful")
    return results

@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with logging.

    Returns:
        dict: Health status information

    """
    try:
        status = {
            "status": "healthy" if app.state.agent else "unhealthy",
            "agent_initialized": bool(app.state.agent),
            "timestamp": time.time(),
        }
        log_debug(f"Health check: {status}")
    except Exception as e:
        log_error(f"Health check failed: {e!s}")
        raise HTTPException(status_code=500, detail="Health check failed") from e
    else:
        return status

if __name__ == "__main__":
    log_info("Starting API server")
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=False,
        )
    except Exception as e:
        log_error(f"Server failed to start: {e!s}")
        raise
