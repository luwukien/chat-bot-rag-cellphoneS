import asyncio
import json
import sys
from playwright.async_api import async_playwright

async def crawl_description_from_page(page) -> str:
    """Extract description directly from the product page."""
    # Thử quét cả hai selector mỗi 200ms trong tối đa 5 giây
    for _ in range(25):
        for selector in ["#cpsKsp", ".ksp-content"]:
            try:
                locator = page.locator(selector)
                if await locator.count() > 0:
                    text = (await locator.first.inner_text()).strip()
                    if text:
                        return text
            except Exception:
                pass
        await page.wait_for_timeout(200)
        
    return ""


async def main():
  file_name = './data/list_product_details.json'
  
  try:
      with open(file_name, 'r', encoding='utf-8') as f:
          products = json.load(f)
  except (FileNotFoundError, json.JSONDecodeError):
      products = []
  
  if not products:
    print("The list product is empty. Cannot crawl anything!")
    return
  
  # Khởi chạy trình duyệt
  async with async_playwright() as p:
      browser = await p.chromium.launch(headless=True)
      for idx, product in enumerate(products):
        url = product.get('url')
        if not url:
            continue
        print(f"[{idx + 1}/{len(products)}] Đang xử lý: {product.get('name')}")
        page = await browser.new_page()
        try:
          await page.goto(url, wait_until="domcontentloaded", timeout=60000)
          print(f"  -> Bắt đầu cào description cho {url}")
          # Gọi hàm cào dữ liệu và gán vào field 'description'
          desc_text = await crawl_description_from_page(page)
          if not desc_text:
            print(f"  ->description trống cho {url}")
            continue
          product['description'] = desc_text
        except Exception as e:
          try:
            await browser.close()
          except Exception:
            pass
          # Recreate browser to continue
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
  
  # Lọc lại danh sách chỉ lấy các sản phẩm có description hợp lệ
  valid_products = [p for p in products if p.get("description")]
  
  with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(valid_products, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())