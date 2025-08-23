"""Utility functions and shared components."""

import logging
import os
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Global console instance
console = Console()

# Configure rich logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("aican")


def setup_logging(level: Optional[str] = None, debug: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        debug: Enable debug mode
    """
    if debug:
        level = "DEBUG"
    
    if level:
        logger.setLevel(level.upper())
        logging.getLogger().setLevel(level.upper())


def is_ci() -> bool:
    """Check if running in a CI environment."""
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def get_env_or_error(key: str, default: Optional[str] = None) -> str:
    """Get environment variable or raise error if not found.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If environment variable not found and no default
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is required")
    return value


def confirm(message: str, default: bool = False) -> bool:
    """Ask user for confirmation.
    
    Args:
        message: Confirmation message
        default: Default response if user just presses enter
        
    Returns:
        True if user confirms, False otherwise
    """
    if is_ci():
        logger.info(f"CI environment detected, using default: {default}")
        return default
    
    suffix = " [Y/n]" if default else " [y/N]"
    
    try:
        response = input(f"{message}{suffix}: ").lower().strip()
        if not response:
            return default
        return response in ("y", "yes", "1", "true")
    except KeyboardInterrupt:
        console.print("\nOperation cancelled.")
        return False


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"✅ {message}", style="green")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"❌ {message}", style="red")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"⚠️  {message}", style="yellow")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"ℹ️  {message}", style="blue")