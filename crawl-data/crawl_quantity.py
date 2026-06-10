import asyncio
import json
from typing import Dict, Any, List
from playwright.async_api import async_playwright

async def crawl_quantity_from_page(page) -> Dict[str, Any]:
  """
    Crawls the quantity of each variant
    
  """