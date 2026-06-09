import asyncio
import json
from typing import Dict, Any
from playwright.async_api import async_playwright

async def crawls_specs_from_page(page) -> Dict[str, Any]:
    """Extract specs directly from sections, bypassing redundant tab clicks."""
    full_specs: Dict[str, Any] = {}

    try:
        button = page.locator(".button__show-modal-technical")
        if await button.count() > 0 and await button.is_visible():
            await button.click()
            await page.wait_for_selector(".teleport-modal", timeout=15000)
            await page.wait_for_selector("section.technical-content-section", timeout=15000)
            
    except Exception as e:
        print(f"  -> Lỗi khi mở modal: {e}")
        return full_specs

    try:
        sections_locator = page.locator("section.technical-content-section")
        
        # Wait briefly to ensure DOM has fully populated the sections
        sections_count = 0
        for _ in range(5):
            sections_count = await sections_locator.count()
            if sections_count > 0:
                break
            await page.wait_for_timeout(400)

        print(f"    Debug: Tìm thấy tổng cộng {sections_count} sections. Đang trích xuất dữ liệu...")

        for sec_idx in range(sections_count):
            section = sections_locator.nth(sec_idx)
            
            title_locator = section.locator("p.title")
            if await title_locator.count() == 0:
                continue
                
            category_name = (await title_locator.first.inner_text()).strip()

            rows_locator = section.locator("table.technical-content tr.technical-content-item")
            rows_count = await rows_locator.count()
            
            for row_idx in range(rows_count):
                row = rows_locator.nth(row_idx)
                key_loc = row.locator("td").nth(0)
                val_loc = row.locator("td").nth(1)
                
                if await key_loc.count() > 0 and await val_loc.count() > 0:
                    key = (await key_loc.first.inner_text()).strip()
                    val = (await val_loc.first.inner_text()).strip()
                    
                    if key:
                        full_specs[key] = val

    except Exception as e:
        print(f"  -> Lỗi khi trích xuất dữ liệu: {e}")

    return full_specs

async def main():
    file_name = 'list_product_details.json'
    
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
        browser = await p.chromium.launch(headless=True)

        for idx, product in enumerate(products):
            url = product.get('url')
            if not url:
                continue

            print(f"[{idx + 1}/{len(products)}] Đang xử lý: {product.get('name')}")

            page = await browser.new_page()

            
            try:
                await page.goto(url, wait_until="load", timeout=60000)
                print(f"  -> Bắt đầu cào specs cho {url}")

                # Gọi hàm cào dữ liệu và gán vào field 'specs'
                product['specs'] = await crawls_specs_from_page(page)

                if not product['specs']:
                    print(f"  -> Cảnh báo: specs trống cho {url}")

            except Exception as e:
                try:
                    await browser.close()
                except Exception:
                    pass
                # recreate browser to continue
                browser = await p.chromium.launch(headless=True)
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