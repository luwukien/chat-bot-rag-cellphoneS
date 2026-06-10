import json
from typing import Dict
from playwright.async_api import async_playwright
import asyncio

async def crawl_faq(page) -> Dict[str, str]:
  faq = {}
  
  try:
    # Locating all items under the main container of the FAQ section.
    # Adjust the selector based on the actual structure you see.
    faq_questions = page.locator('.accordion-item')
    for i in range(await faq_questions.count()):
      # Expanding the FAQ item to reveal the answer
      faq_item = faq_questions.nth(i)
      await faq_item.click()
      await page.wait_for_timeout(1000)
      
      # Extracting question text
      question_locator = faq_item.locator('.accordion-label') # Adjust selector if needed
      question_text = await question_locator.first.wait_for(state="visible")
      
      # Extracting answer text
      answer_locator = faq_item.locator('.accordion-content') # Adjust selector if needed
      answer_text = await answer_locator.first.wait_for(state="visible")
      
      faq[question_text.inner_text().strip()] = answer_text.inner_text().strip()
      
  except Exception as e:
    print(f"Lỗi khi crawl FAQ: {e}")

  return faq

async def main():
  input_file = './data/test_product_details.json'
  output_file = './data/faq.json'
  
  try:
      with open(input_file, 'r', encoding='utf-8') as f:
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
          print(f"  -> Bắt đầu cào FAQ cho {url}")
          # Gọi hàm cào dữ liệu và gán vào field 'faq'
          product['faq'] = await crawl_faq(page)
          if not product['faq']:
            print(f"  ->faq trống cho {url}")
            continue
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
  
  # Lọc danh sách chỉ lấy các sản phẩm có FAQ hợp lệ
  faq_products = [
      {
          "id": p.get("id"),
          "name": p.get("name"),
          "faq": p.get("faq")
      }
      for p in products if p.get("faq")
  ]

  # Lưu lại kết quả cào vào file faq.json
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(faq_products, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
