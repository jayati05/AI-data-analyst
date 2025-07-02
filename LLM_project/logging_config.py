"""Configuration settings for application logging.

This module defines the LoggingConfig class which contains all configurable parameters
for the application's logging system.
"""
from __future__ import annotations

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Configuration model for application logging settings.

    Attributes:
        log_file_name: Name of the log file (default: "app.log")
        min_log_level: Minimum logging level (default: "INFO")
        log_rotation: Time rotation for log files (default: "00:00")
        log_compression: Compression format for rotated logs (default: "zip")
        log_server_address: Optional network address for remote logging
        enable_network_logging: Flag to enable/disable network logging

    """

    log_file_name: str = "app.log"
    min_log_level: str = "INFO"
    log_rotation: str = "00:00"  # Rotate at midnight
    log_compression: str = "zip"
    log_server_address: str | None = "tcp://127.0.0.1:5555"
    enable_network_logging: bool = False
