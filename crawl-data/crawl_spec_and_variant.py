import asyncio
import json
from typing import Dict, Any, List
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
        
        # Đợi một chút để DOM render đầy đủ các section specs
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

async def crawls_variants_from_page(page) -> List[Dict[str, Any]]:
    """Extract color and price variants directly from the product page."""
    variants: List[Dict[str, Any]] = []
    try:
        # Tìm tất cả các phần tử variant màu sắc
        variant_locators = page.locator("a.button__change-color")
        count = await variant_locators.count()
        
        for idx in range(count):
            loc = variant_locators.nth(idx)
            
            # Lấy tên màu
            name_loc = loc.locator(".item-variant-name")
            color_name = ""
            if await name_loc.count() > 0:
                color_name = (await name_loc.first.inner_text()).strip()
            
            # Lấy giá
            price_loc = loc.locator(".item-variant-price")
            price = ""
            if await price_loc.count() > 0:
                price = (await price_loc.first.inner_text()).strip()
            
            # Trạng thái hàng của variant này
            stock = ""
            out_of_stock_locator = page.locator(".order-button:has-text('TẠM HẾT HÀNG')")
            is_out_of_stock = await out_of_stock_locator.is_visible()
            if not is_out_of_stock:
                stock = "Còn hàng"
            else:
                stock = "Tạm hết hàng"
            variants.append({
                "color": color_name,
                "price": price,
                "stock": stock,
            })
    except Exception as e:
        print(f"  -> Lỗi khi trích xuất variants: {e}")
        
    return variants

async def main():
    file_name = './data/list_product_details.json'
    
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        products = []

    if not products:
        print("The list spec is empty. Cannot crawl anything!")
        return

    # KHỞI CHẠY TRÌNH DUYỆT VỚI CONTEXT GIẢ LẬP TRÌNH DUYỆT THẬT
    async with async_playwright() as p: 
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            extra_http_headers={
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )

        for idx, product in enumerate(products):
            url = product.get('url')
            if not url:
                continue

            print(f"[{idx + 1}/{len(products)}] Đang xử lý: {product.get('name')}")

            success = False
            max_attempts = 3
            for attempt in range(max_attempts):
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    
                    # Chờ thêm 2 giây để các component React/JS được kích hoạt đầy đủ
                    await page.wait_for_timeout(2000)
                    
                    print(f"  -> Bắt đầu cào specs và variants cho {url} (Lần thử {attempt + 1})")

                    # Gọi hàm cào dữ liệu và gán vào field 'specs' và 'variants'
                    specs = await crawls_specs_from_page(page)
                    variants = await crawls_variants_from_page(page)

                    if not specs:
                        print(f"  -> Specs trống ở lần thử {attempt + 1}.")
                        await page.close()
                        continue
                        
                    if not variants:
                        print(f"  -> Variants trống ở lần thử {attempt + 1}.")
                        await page.close()
                        continue

                    product['specs'] = specs
                    product['variants'] = variants
                    success = True
                    await page.close()
                    break # Thành công, thoát khỏi vòng lặp thử lại cho URL này

                except Exception as e:
                    print(f"  -> Lỗi ở lần thử {attempt + 1}: {e}")
                    try:
                        await page.close()
                    except Exception:
                        pass
                    
                    # Khởi tạo lại browser/context nếu trình duyệt bị crash hoặc bị đóng đột ngột
                    if "Target page, context or browser has been closed" in str(e) or "Target closed" in str(e):
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        browser = await p.chromium.launch(headless=True)
                        context = await browser.new_context(
                            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            viewport={"width": 1280, "height": 800},
                            extra_http_headers={
                                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                            }
                        )
                    await asyncio.sleep(2) # Đợi một lát trước khi thử lại

            if not success:
                print(f"  -> Thất bại sau {max_attempts} lần thử cho {url}.")

        try:
            await browser.close()
        except Exception:
            pass

    # Lọc lại danh sách sản phẩm hợp lệ trước khi lưu file JSON
    valid_products = []
    for p in products:
        # Phải có specs và variants
        if not p.get("specs") or not p.get("variants"):
            continue
        
        # Kiểm tra xem tất cả các variant có phải đều không có giá và tạm hết hàng hay không
        all_variants_empty_or_inactive = all(
            (v.get("price").strip() == "") and (v.get("stock") == "Tạm hết hàng")
            for v in p.get("variants")
        )
        
        # Nếu KHÔNG PHẢI tất cả đều trống/hết hàng (tức là còn ít nhất 1 variant hoạt động hoặc có giá), giữ lại sản phẩm
        if not all_variants_empty_or_inactive:
            valid_products.append(p)
    
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(valid_products, f, ensure_ascii=False, indent=2)
    
    print("FINISH")

if __name__ == "__main__":
    asyncio.run(main())