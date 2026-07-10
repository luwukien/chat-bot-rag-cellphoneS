import asyncio
import json
from urllib.parse import urljoin
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from urllib.parse import urlparse


async def crawl_cellphones_url_and_name(url: str) -> Dict[str, Optional[str]]:
    """Crawl product links and names from the given category URL.

    Returns a list of dicts: {"name": str | None, "url": str}
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # click "show more" until there is no more
        while True:
            try:
                button = page.locator(".button__show-more-product")
                if await button.is_visible():
                    await button.click()
                    await page.wait_for_timeout(2000)
                else:
                    break
            except Exception:
                break

        elements = await page.locator("a.product__link").all()
        products: List[Dict[str, Optional[str]]] = []



        for el in elements:
            href = await el.get_attribute("href")
            if not href:
                continue
            full_url_phone = urljoin(url, href)

            # Create ID
            product_id = urlparse(full_url_phone).path.split("/")[-1].replace(".html", "")
            name_locator = el.locator(".product__name")

            if await name_locator.count() > 0:
                name = await name_locator.text_content()
                name = name.strip()
            else:
                name = None

            products.append({"id": product_id, "name": name, "url": full_url_phone})

        await browser.close()
        return products


import os
from urllib.parse import urlparse

async def main() -> None:
    # Check if list_iphone_links.txt exists (either in crawl-data/ or current directory)
    txt_path = "crawl-data/list_iphone_links.txt"
    if not os.path.exists(txt_path):
        txt_path = "list_iphone_links.txt"
        
    if os.path.exists(txt_path):
        print(f"Found link list file: {txt_path}. Parsing URLs...")
        with open(txt_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
            
        products = []
        for url in urls:
            product_id = urlparse(url).path.split("/")[-1].replace(".html", "")
            products.append({"id": product_id, "name": None, "url": url})
        print(f"Generated {len(products)} product entries from {txt_path}.")
    else:
        print("No link list file found. Crawling catalog page...")
        products = await crawl_cellphones_url_and_name("https://cellphones.com.vn/mobile/apple.html")
        print(f"Crawled {len(products)} products from catalog page.")

    # Save directly to ./data/list_product_details.json
    out_dir = "./data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "list_product_details.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    print(f"Saved initial products list to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())