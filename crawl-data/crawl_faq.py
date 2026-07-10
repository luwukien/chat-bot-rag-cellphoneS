import json
from typing import Dict
from playwright.async_api import async_playwright
import asyncio

async def crawl_faq(page) -> Dict[str, str]:
  faq = {}
  
  try:
    # Cuộn xuống cuối trang để kích hoạt Lazy Loading của phần FAQ
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(1500)
    
    # Ẩn cookie banner nếu có để tránh che khuất các phần tử khác
    try:
      await page.add_style_tag(content="#teleport-modal, .teleport-modal-cookie-consent { display: none !important; }")
    except Exception:
      pass
      
    faq_questions = page.locator('.accordion-item')
    # Đợi cho phần tử FAQ đầu tiên xuất hiện trong DOM
    try:
      await faq_questions.first.wait_for(state="attached", timeout=3000)
    except Exception:
      # Nếu quá 3 giây không tìm thấy .accordion-item, trang này không có FAQ
      return faq
    
    for i in range(await faq_questions.count()):
      faq_item = faq_questions.nth(i)
      
      # Lấy selector câu hỏi và câu trả lời
      q_locator = faq_item.locator('.accordion-label')
      a_locator = faq_item.locator('.accordion-content')
      
      try:
        # Đợi hai phần tử này đính kèm vào DOM
        await q_locator.first.wait_for(state="attached", timeout=2000)
        await a_locator.first.wait_for(state="attached", timeout=2000)
        
        # Click mở rộng accordion bằng JavaScript để tránh bị che khuất bởi cookie banner/quảng cáo
        await q_locator.first.evaluate("el => el.click()")
        await page.wait_for_timeout(300)
        
        # Lấy text câu hỏi và câu trả lời
        q_text = await q_locator.first.inner_text()
        a_text = await a_locator.first.inner_text()
        
        # Nếu inner_text rỗng, thử dùng text_content làm phương án dự phòng
        if not q_text.strip():
          q_text = await q_locator.first.text_content() or ""
        if not a_text.strip():
          a_text = await a_locator.first.text_content() or ""
          
        if q_text.strip() and a_text.strip():
          faq[q_text.strip()] = a_text.strip()
      except Exception as item_err:
        print(f"Lỗi khi cào item FAQ thứ {i}: {item_err}")
        # Phương án dự phòng cuối cùng nếu có lỗi xảy ra trong quá trình click/đợi
        try:
          q_text = await q_locator.first.text_content()
          a_text = await a_locator.first.text_content()
          if q_text and a_text and q_text.strip() and a_text.strip():
            faq[q_text.strip()] = a_text.strip()
        except Exception:
          pass
      
  except Exception as e:
    print(f"Lỗi khi crawl FAQ: {e}")
 
  return faq

async def main():
  input_file = './data/list_product_details.json'
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
