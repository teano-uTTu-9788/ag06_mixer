"""Notion API client for automation tasks."""

import os
from typing import Any, Dict, Optional

import requests
from rich.console import Console

from .utils import logger

console = Console()


class NotionClient:
    """A robust Notion API client with error handling and retry logic."""

    def __init__(self, token: Optional[str] = None, version: str = "2022-06-28"):
        """Initialize the Notion client.
        
        Args:
            token: Notion API token. If not provided, will use NOTION_TOKEN env var.
            version: Notion API version to use.
        """
        self.token = token or os.environ.get("NOTION_TOKEN")
        if not self.token:
            raise ValueError("Notion token must be provided or set via NOTION_TOKEN env var")
        
        self.version = version
        self.base_url = "https://api.notion.com/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.version,
            "Content-Type": "application/json",
        })

    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the Notion API with error handling.
        
        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint (without base URL)
            data: Request data for POST/PATCH requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        logger.debug(f"Making {method} request to {url}")
        
        response = self.session.request(method, url, json=data)
        response.raise_for_status()
        
        return response.json()

    def update_page_status(self, page_id: str, status: str) -> Dict[str, Any]:
        """Update the Status property of a page.
        
        Args:
            page_id: The page ID to update
            status: The status value to set
            
        Returns:
            Updated page data
        """
        data = {
            "properties": {
                "Status": {
                    "status": {
                        "name": status
                    }
                }
            }
        }
        
        logger.info(f"Updating page {page_id} status to: {status}")
        
        try:
            result = self._make_request("PATCH", f"pages/{page_id}", data)
            logger.info("Page status updated successfully")
            return result
        except requests.RequestException as e:
            logger.error(f"Failed to update page status: {e}")
            raise

    def update_page_property(
        self, 
        page_id: str, 
        property_name: str, 
        value: Any, 
        property_type: str = "rich_text"
    ) -> Dict[str, Any]:
        """Update a property of a page.
        
        Args:
            page_id: The page ID to update
            property_name: Name of the property to update
            value: The value to set
            property_type: Type of property (rich_text, number, checkbox, select)
            
        Returns:
            Updated page data
        """
        # Build property payload based on type
        if property_type == "rich_text":
            prop_data = {
                "rich_text": [{
                    "text": {
                        "content": str(value)
                    }
                }]
            }
        elif property_type == "number":
            prop_data = {"number": float(value)}
        elif property_type == "checkbox":
            prop_data = {"checkbox": bool(value)}
        elif property_type == "select":
            prop_data = {"select": {"name": str(value)}}
        else:
            raise ValueError(f"Unsupported property type: {property_type}")

        data = {
            "properties": {
                property_name: prop_data
            }
        }
        
        logger.info(f"Updating page {page_id} property {property_name} to: {value}")
        
        try:
            result = self._make_request("PATCH", f"pages/{page_id}", data)
            logger.info("Page property updated successfully")
            return result
        except requests.RequestException as e:
            logger.error(f"Failed to update page property: {e}")
            raise

    def create_page(
        self, 
        parent_id: str, 
        title: str, 
        content: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new page in a database or as a child of another page.
        
        Args:
            parent_id: Parent database or page ID
            title: Page title
            content: Optional page content
            properties: Additional properties to set
            
        Returns:
            Created page data
        """
        # Basic page structure
        data: Dict[str, Any] = {
            "parent": {"database_id": parent_id},
            "properties": {
                "Name": {
                    "title": [{
                        "text": {
                            "content": title
                        }
                    }]
                }
            }
        }
        
        # Add custom properties
        if properties:
            data["properties"].update(properties)
        
        # Add content as blocks
        if content:
            data["children"] = [{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {
                            "content": content
                        }
                    }]
                }
            }]
        
        logger.info(f"Creating page: {title}")
        
        try:
            result = self._make_request("POST", "pages", data)
            logger.info("Page created successfully")
            return result
        except requests.RequestException as e:
            logger.error(f"Failed to create page: {e}")
            raise

    def get_page(self, page_id: str) -> Dict[str, Any]:
        """Retrieve a page by ID.
        
        Args:
            page_id: The page ID to retrieve
            
        Returns:
            Page data
        """
        logger.debug(f"Retrieving page: {page_id}")
        
        try:
            result = self._make_request("GET", f"pages/{page_id}")
            return result
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve page: {e}")
            raise

    def search_pages(self, query: str, filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Search for pages matching a query.
        
        Args:
            query: Search query
            filter_type: Optional filter by object type (page or database)
            
        Returns:
            Search results
        """
        data = {"query": query}
        
        if filter_type:
            data["filter"] = {"property": "object", filter_type: {"equals": filter_type}}
        
        logger.info(f"Searching for pages: {query}")
        
        try:
            result = self._make_request("POST", "search", data)
            return result
        except requests.RequestException as e:
            logger.error(f"Failed to search pages: {e}")
            raise