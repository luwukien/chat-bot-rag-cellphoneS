import asyncio
import json
import os
from playwright.async_api import async_playwright

async def crawl_policy():
    url = "https://cellphones.com.vn/tos"
    tos_data = []

    print(f"Starting crawl policy from: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Set a standard desktop viewport size
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        
        try:
            # Go to the page and wait for it to load
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_selector(".warranty-options-item", timeout=15000)
            print("Page loaded successfully.")
            
            # Locate all the policy menu items in the sidebar
            menu_locator = page.locator(".warranty-options-item")
            menu_count = await menu_locator.count()
            print(f"Found {menu_count} policy sections in the sidebar.")
            
            if menu_count == 0:
                print("Error: No menu items found. The selector might have changed or content is blocked.")
                return
            
            for idx in range(menu_count):
                item = menu_locator.nth(idx)
                title = (await item.inner_text()).strip()
                print(f"[{idx + 1}/{menu_count}] Clicking section index {idx + 1}...")
                
                # Click the option to load its content
                await item.click()
                
                # Wait for content to load and stabilize
                await page.wait_for_timeout(2000)
                
                # Extract content from the content area
                content_locator = page.locator(".warranty-content")
                
                # Double check if text is loaded
                content_text = ""
                for _ in range(5):
                    content_text = (await content_locator.inner_text()).strip()
                    if content_text:
                        break
                    await page.wait_for_timeout(500)
                
                print(f"  -> Extracted {len(content_text)} characters of text.")
                
                tos_data.append({
                    "section_index": idx + 1,
                    "title": title,
                    "content": content_text
                })
                
        except Exception as e:
            print(f"An error occurred during crawling: {e}")
        finally:
            await browser.close()
            
    # Ensure the output directory exists
    os.makedirs("./data", exist_ok=True)
    output_file = "./data/policy.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tos_data, f, ensure_ascii=False, indent=2)
        
    print(f"Finished. Saved TOS data to: {output_file}")

if __name__ == "__main__":
    asyncio.run(crawl_policy())
