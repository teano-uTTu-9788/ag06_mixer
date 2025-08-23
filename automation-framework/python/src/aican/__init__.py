"""AiCan Automation Framework - Python utilities."""

__version__ = "0.1.0"
__author__ = "AiCan Team"
__email__ = "team@aican.dev"

from .notion_client import NotionClient
from .utils import logger

__all__ = ["NotionClient", "logger"]