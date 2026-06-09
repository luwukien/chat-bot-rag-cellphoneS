import asyncio
import json
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from urllib.parse import urlparse

async def main():
    file_name = 'list_product_details.json'
    test_output_file = 'test_result.json' # Đổi tên file lưu kết quả test
    
    # 1. ĐỌC DỮ LIỆU TỪ FILE GỐC
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        products = []

    if not products:
        print("Danh sách rỗng, không có gì để cào!")
        return

    # ---> CẮT DANH SÁCH: CHỈ LẤY 10 SẢN PHẨM ĐẦU TIÊN ĐỂ TEST <---
    test_products = products[:10] 

    # KHỞI CHẠY TRÌNH DUYỆT
    async with async_playwright() as p: 
        browser = await p.chromium.launch(headless=False) # Bật False để nhìn trình duyệt click mượt không
        page = await browser.new_page()

        # DUYỆT QUA 10 SẢN PHẨM TEST
        for idx, product in enumerate(test_products):
            url = product.get('url')
            if not url:
                continue

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Đảm bảo bạn đang dùng phiên bản extract_specs_from_page "hạng nặng" ở câu trả lời trước nhé
                product['specs'] = await (page)
                
            except Exception as e:
                print(f"  -> Lỗi khi tải trang {url}: {e}")
                product['specs'] = {} 

        await browser.close()

    # 2. GHI DỮ LIỆU ĐÃ CẬP NHẬT VÀO FILE TEST (KHÔNG GHI ĐÈ FILE GỐC)
    with open(test_output_file, 'w', encoding='utf-8') as f:
        json.dump(test_products, f, ensure_ascii=False, indent=2)
    
    print(f"HOÀN THÀNH TEST! Đã lưu dữ liệu 10 sản phẩm vào file {test_output_file}.")

if __name__ == "__main__":
    asyncio.run(main())