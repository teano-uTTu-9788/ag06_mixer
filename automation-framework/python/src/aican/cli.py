"""Command-line interface for AiCan automation."""

import os
from typing import Optional

import click
from rich.console import Console

from .notion_client import NotionClient
from .utils import logger, print_error, print_success, setup_logging

console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--log-level", help="Set log level")
def main(debug: bool, log_level: Optional[str]) -> None:
    """AiCan Automation CLI - Python utilities for workflow automation."""
    setup_logging(log_level, debug)


@main.command()
@click.argument("page_id")
@click.argument("status")
@click.option("--token", help="Notion API token", envvar="NOTION_TOKEN")
def notion_status(page_id: str, status: str, token: Optional[str]) -> None:
    """Update Notion page status.
    
    PAGE_ID: The Notion page ID to update
    STATUS: The status value to set
    """
    try:
        client = NotionClient(token)
        client.update_page_status(page_id, status)
        print_success(f"Updated page status to: {status}")
    except Exception as e:
        print_error(f"Failed to update Notion status: {e}")
        raise click.Abort()


@main.command()
@click.argument("page_id")
@click.argument("property_name")
@click.argument("value")
@click.option("--property-type", default="rich_text", 
              help="Property type (rich_text, number, checkbox, select)")
@click.option("--token", help="Notion API token", envvar="NOTION_TOKEN")
def notion_property(
    page_id: str, 
    property_name: str, 
    value: str, 
    property_type: str, 
    token: Optional[str]
) -> None:
    """Update Notion page property.
    
    PAGE_ID: The Notion page ID to update
    PROPERTY_NAME: Name of the property to update
    VALUE: The value to set
    """
    try:
        client = NotionClient(token)
        client.update_page_property(page_id, property_name, value, property_type)
        print_success(f"Updated property {property_name} to: {value}")
    except Exception as e:
        print_error(f"Failed to update Notion property: {e}")
        raise click.Abort()


@main.command()
@click.argument("parent_id")
@click.argument("title")
@click.option("--content", help="Page content")
@click.option("--token", help="Notion API token", envvar="NOTION_TOKEN")
def notion_create(
    parent_id: str, 
    title: str, 
    content: Optional[str], 
    token: Optional[str]
) -> None:
    """Create a new Notion page.
    
    PARENT_ID: Parent database or page ID
    TITLE: Page title
    """
    try:
        client = NotionClient(token)
        result = client.create_page(parent_id, title, content)
        page_url = result.get("url", "N/A")
        print_success(f"Created page: {title}")
        console.print(f"URL: {page_url}")
    except Exception as e:
        print_error(f"Failed to create Notion page: {e}")
        raise click.Abort()


@main.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"AiCan Automation CLI v{__version__}")


if __name__ == "__main__":
    main()