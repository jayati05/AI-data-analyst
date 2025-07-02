"""Unified logging client that sends logs.

To a centralized logging server using ZeroMQ.
"""

import zmq
from loguru import logger

LOG_SERVER_ADDRESS: str = "localhost"
LOG_SERVER_PORT: int = 5555


class UnifiedLogger:
    """Client for sending log messages to a unified logging server."""

    def __init__(self,
            address: str = LOG_SERVER_ADDRESS,
            port: int = LOG_SERVER_PORT,
        ) -> None:
        """Initialize the logging client."""
        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket = self.context.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{address}:{port}")

    def send_log(self, level: str, message: str) -> None:
        """Send a log message to the logging server."""
        try:
            self.socket.send_json({"level": level, "message": message})
        except zmq.ZMQError as e:
            logger.error(f"Logging error: {e}")


# Initialize the logger instance
unified_logger = UnifiedLogger()


# Helper functions for logging
def log_info(message: str) -> None:
    """Log an informational message."""
    unified_logger.send_log("INFO", message)


def log_debug(message: str) -> None:
    """Log a debug message."""
    unified_logger.send_log("DEBUG", message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    unified_logger.send_log("WARNING", message)


def log_error(message: str) -> None:
    """Log an error message."""
    unified_logger.send_log("ERROR", message)


def log_critical(message: str) -> None:
    """Log a critical error message."""
    unified_logger.send_log("CRITICAL", message)


def log_success(message: str) -> None:
    """Log a success message."""
    unified_logger.send_log("SUCCESS", message)


def log_trace(message: str) -> None:
    """Log a trace message (for very detailed debugging)."""
    unified_logger.send_log("TRACE", message)
