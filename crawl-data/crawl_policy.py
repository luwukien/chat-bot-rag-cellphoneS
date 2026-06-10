import json
from typing import Dict
from playwright.async_api import async_playwright
import asyncio

async def crawl_policy(page) -> Dict[str, str]:
    policy = {}
    try:
        await page.goto("https://cellphones.vn/bao-hanh")
        await page.wait_for_timeout(2000)
        
    except Exception as e:
        print(f"Lỗi khi crawl policy: {e}")
    return policy