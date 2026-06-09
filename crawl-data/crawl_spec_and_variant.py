import asyncio
import json
import traceback
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from urllib.parse import urlparse

async def crawls_specs_from_page(page) -> Dict[str, Any]:
    """Extract specs from all tabs in technical specification modal using Locators.

    Use Locators to avoid ElementHandle adoption errors when the page re-renders DOM.
    """
    full_specs: Dict[str, Any] = {}

    try:
        button = page.locator(".button__show-modal-technical")
        if await button.count() and await button.is_visible():
            await button.click()
            await page.wait_for_selector(".teleport-modal", timeout=15000)
            await page.wait_for_selector("section.technical-content-section", timeout=15000)
    except Exception as e:
        print(f"  -> Lỗi khi mở modal: {e}")
        return full_specs

    # Use Locators (resolve-on-use) instead of ElementHandles
    try:
        tabs_locator = page.locator("li.technical-tab-item")
        tabs_count = await tabs_locator.count()
        print(f"    Debug: tìm thấy {tabs_count} tabs")

        for tab_idx in range(tabs_count):
            try:
                tab = tabs_locator.nth(tab_idx)
                if not (await tab.count()):
                    continue

                # Ensure tab is visible
                try:
                    await tab.scroll_into_view_if_needed()
                except Exception:
                    pass

                # Try clicking with a short timeout, fallback to force click if needed
                clicked = False
                try:
                    await tab.click(timeout=5000)
                    clicked = True
                except Exception:
                    try:
                        await tab.click(force=True, timeout=3000)
                        clicked = True
                    except Exception as e_click:
                        print(f"    -> Không thể click tab {tab_idx}: {e_click}")

                if not clicked:
                    continue

                # Wait a bit and retry reading sections until they appear (small retry loop)
                sections_locator = page.locator("section.technical-content-section")
                sections_count = 0
                for _ in range(5):
                    sections_count = await sections_locator.count()
                    if sections_count:
                        break
                    await page.wait_for_timeout(400)

                print(f"    Debug tab {tab_idx}: tìm thấy {sections_count} sections")

                for sec_idx in range(sections_count):
                    section = sections_locator.nth(sec_idx)
                    title_locator = section.locator("p.title")
                    if not (await title_locator.count()):
                        continue
                    category_name = (await title_locator.inner_text()).strip()

                    rows_locator = section.locator("table.technical-content tr.technical-content-item")
                    rows_count = await rows_locator.count()
                    for row_idx in range(rows_count):
                        row = rows_locator.nth(row_idx)
                        key_loc = row.locator("td:nth-child(1)")
                        val_loc = row.locator("td:nth-child(2)")
                        
                        if (await key_loc.count()) and (await val_loc.count()):
                            key = (await key_loc.inner_text()).strip()
                            val = (await val_loc.inner_text()).strip()
                            if key:
                                full_key = f"{key} {category_name}" if category_name not in key else key
                                full_specs[full_key] = val

            except Exception as e:
                print(f"    -> Lỗi khi xử lý tab {tab_idx}: {e}")
                continue

    except Exception as e:
        print(f"  -> Lỗi khi lấy tabs: {e}")

    return full_specs


async def main():
    file_name = 'test_result.json'
    
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        products = []

    if not products:
        print("The list spec is empty. Cannot crawl anything!")
        return

    # KHỞI CHẠY TRÌNH DUYỆT
    async with async_playwright() as p: 
        browser = await p.chromium.launch(headless=False)

        for idx, product in enumerate(products):
            url = product.get('url')
            if not url:
                continue

            # Create ID
            product_id = urlparse(url).path.split("/")[-1].replace(".html", "")
            product['id'] = product_id

            print(f"[{idx + 1}/{len(products)}] Đang xử lý: {product.get('name')}")

            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                print(f"  -> Bắt đầu cào specs cho {url}")

                # Gọi hàm cào dữ liệu và gán vào field 'specs'
                product['specs'] = await crawls_specs_from_page(page)

                if not product['specs']:
                    print(f"  -> Cảnh báo: specs trống cho {url}")

            except Exception as e:
                # Save error and try to recover if browser/page was closed
                print(f"  -> Lỗi khi tải trang {url}: {e}")
                print(traceback.format_exc())
                product['specs'] = {}
                product['error'] = str(e)

                err_text = str(e)
                if 'Target page, context or browser has been closed' in err_text or 'TargetClosedError' in type(e).__name__:
                    try:
                        await browser.close()
                    except Exception:
                        pass
                    # recreate browser to continue
                    browser = await p.chromium.launch(headless=False)
            finally:
                try:
                    await page.close()
                except Exception:
                    pass

        try:
            await browser.close()
        except Exception:
            pass

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    
    print("FINISH")

if __name__ == "__main__":
    asyncio.run(main())