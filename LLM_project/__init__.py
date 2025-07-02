"""Exposes the main components of the system for external use."""

# Package initialization
from .agent.setup import initialize_agent
from .config import DATA_FOLDER
from .main import run_test_queries

__all__ = ["DATA_FOLDER", "initialize_agent", "run_test_queries"]
