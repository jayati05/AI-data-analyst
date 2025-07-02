"""Executor module for handling tool execution in the system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain.agents import AgentExecutor

from logging_client import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
    from langchain.callbacks.base import BaseCallbackManager


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the tool execution failure

        """
        super().__init__(message)
        self.message = message


@dataclass
class ExecutionResult:
    """Dataclass representing the result of an agent execution.

    Attributes:
        output: The final output of the execution
        intermediate_steps: List of intermediate steps taken
        should_stop: Boolean indicating if execution should stop

    """

    output: str
    intermediate_steps: list[tuple]
    should_stop: bool = False


class CustomAgentExecutor(AgentExecutor):
    """Custom agent executor with enhanced logging and error handling."""

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: str | dict[str, str],
        run_manager: BaseCallbackManager | None = None,  # type: ignore[name-defined]
    ) -> str | dict[str, Any]:
        """Execute the specified tool with the given input.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input for the tool as either a string or dictionary
            run_manager: Optional callback manager for the execution run

        Returns:
            The result from tool execution as either a string or dictionary

        Raises:
            ToolExecutionError: If the specified tool is not found or execution fails

        """
        log_info(f"Executing tool: {tool_name} with input: {tool_input}")

        tool_map = {tool.name: tool for tool in self.tools}
        if tool_name not in tool_map:
            error_msg = f"Tool '{tool_name}' not found"
            log_warning(error_msg)
            raise ToolExecutionError(error_msg)

        try:
            result = tool_map[tool_name].run(
                tool_input,
                verbose=self.verbose,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            log_info(f"Tool {tool_name} execution successful. Result: {result}")
        except Exception as exc:
            error_msg = f"Error executing tool {tool_name}: {exc!s}"
            log_error(error_msg)
            raise ToolExecutionError(error_msg) from exc
        else:
            return result

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: BaseCallbackManager | None = None,  # type: ignore[name-defined]
    ) -> dict[str, Any]:
        """Run the agent loop until final answer or max iterations.

        Args:
            inputs: Input dictionary for the agent
            run_manager: Optional run manager for callbacks

        Returns:
            Dictionary containing output and intermediate steps

        """
        log_info(f"Agent execution started with inputs: {inputs}")
        intermediate_steps: list[tuple] = []

        for step in range(self.max_iterations):
            try:
                output = self.agent.plan(
                    intermediate_steps,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **inputs,
                )
                log_debug(f"Step {step}: Agent plan output - {output}")
            except (AttributeError, ValueError, RuntimeError) as exc:
                error_msg = f"Planning error at step {step}: {exc!s}"
                log_error(error_msg)
                return ExecutionResult(
                    output=error_msg,
                    intermediate_steps=intermediate_steps,
                ).__dict__

            if getattr(output, "should_stop", False):
                log_info("Agent stopping due to should_stop flag.")
                return ExecutionResult(
                    output=getattr(output, "output", ""),
                    intermediate_steps=intermediate_steps,
                    should_stop=True,
                ).__dict__

            if hasattr(output, "tool") and output.tool == "FinalAnswer":
                log_info("FinalAnswer tool selected. Returning output.")
                return ExecutionResult(
                    output=getattr(output, "tool_input", ""),
                    intermediate_steps=intermediate_steps,
                ).__dict__

            if hasattr(output, "tool"):
                try:
                    observation = self._execute_tool(
                        output.tool,
                        output.tool_input,
                        run_manager,
                    )
                    intermediate_steps.append((output, observation))
                    log_debug(f"Step {step}: Tool execution result - {observation}")

                    if isinstance(observation, dict) and observation.get("should_stop"):
                        log_info("Stopping execution based on observation.")
                        return ExecutionResult(
                            output=observation.get("output", ""),
                            intermediate_steps=intermediate_steps,
                            should_stop=True,
                        ).__dict__
                except ToolExecutionError as exc:
                    error_msg = f"Tool execution failed: {exc!s}"
                    log_error(error_msg)
                    return ExecutionResult(
                        output=error_msg,
                        intermediate_steps=intermediate_steps,
                    ).__dict__

            if run_manager:
                run_manager.on_agent_action(output, verbose=self.verbose)

        log_warning("Maximum iterations reached without final answer.")
        return ExecutionResult(
            output="Maximum iterations reached",
            intermediate_steps=intermediate_steps,
        ).__dict__


def execute_agent_query(agent: CustomAgentExecutor, query: str) -> dict[str, Any]:
    """Execute an agent query with error handling.

    Args:
        agent: CustomAgentExecutor instance
        query: Input query string

    Returns:
        Dictionary containing output or error message

    """
    try:
        log_info(f"Executing agent query: {query}")
        response = agent.invoke({"input": query})
        log_info(f"Agent response: {response}")

        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if (isinstance(step, tuple) and len(step) > 1 and
                        isinstance(step[1], dict) and step[1].get("should_stop")):
                    log_info("Agent stopping due to should_stop in steps.")
                    return {"output": step[1].get("output", str(step[1]))}

        return {"output": response.get("output", str(response))}
    except (RuntimeError, ValueError, ToolExecutionError) as exc:
        error_msg = f"Error processing query: {exc!s}"
        log_error(error_msg)
        return {"output": error_msg}
