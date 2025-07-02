"""ZeroMQ-based unified logging server to collect logs from multiple services."""

import zmq
from loguru import logger

# Configuration
LOG_FILE = "logs/unified_logs.log"
LOG_LEVEL = "INFO"
LOG_ROTATION = "10MB"
LOG_COMPRESSION = "zip"
LOG_SERVER_PORT = 5555  # Port for the logging server

# Setup logging
logger.remove()
logger.add(
    LOG_FILE,
    rotation=LOG_ROTATION,
    compression=LOG_COMPRESSION,
    level=LOG_LEVEL,
)


def start_logging_server() -> None:
    """Start the ZeroMQ logging server to collect and store logs."""
    context = zmq.Context()
    socket = context.socket(zmq.PULL)

    try:
        socket.bind(f"tcp://0.0.0.0:{LOG_SERVER_PORT}")
        logger.info(f"Logging server started on port {LOG_SERVER_PORT}")

        while True:
            log_message: dict = socket.recv_json()
            level: str = log_message.get("level", "INFO")
            message: str = log_message.get("message", "No message")

            if level == "DEBUG":
                logger.debug(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
            elif level == "CRITICAL":
                logger.critical(message)
            else:
                logger.info(message)


    except (zmq.ZMQError, ValueError) as e:
        logger.error(f"Logging server error: {e}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_logging_server()
