import asyncio
import json
import sys
from playwright.async_api import async_playwright

async def crawl_description_from_page(page) -> str:
    """Extract description directly from the product page."""
    try:
        description_locator = page.locator(".ksp-content")
        if await description_locator.count() > 0:
            text = await description_locator.first.inner_text()
    except Exception as e:
        print(f"  -> Lỗi khi trích xuất description: {e}")
        
    return text


async def main():
  file_name = './data/test_product_details.json'
  
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
      browser = await p.chromium.launch(headless=False)
      for idx, product in enumerate(products):
        url = product.get('url')
        if not url:
            continue
        print(f"[{idx + 1}/{len(products)}] Đang xử lý: {product.get('name')}")
        page = await browser.new_page()
        try:
          await page.goto(url, wait_until="load", timeout=60000)
          print(f"  -> Bắt đầu cào description cho {url}")
          # Gọi hàm cào dữ liệu và gán vào field 'description'
          product['description'] = await crawl_description_from_page(page)
          if not product['description'] or not product['description'].get('description'):
            print(f"  ->description trống cho {url}")
            continue;
        except Exception as e:
          try:
            await browser.close()
          except Exception:
            pass
          # Recreate browser to continue
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
  
  # Lưu lại kết quả cào vào file
  with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())