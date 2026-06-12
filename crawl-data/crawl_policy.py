import asyncio
import json
import os
import re
import sys
from playwright.async_api import async_playwright

def normalize_text(text):
    """Làm sạch các ký tự đặc biệt, dấu cách ẩn cào từ web."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ')
    text = text.replace('\u2002', ' ')
    text = text.replace('\u2003', ' ')
    text = text.replace('\u2009', ' ')
    text = text.replace('\u202f', ' ')
    text = text.replace('\u200b', '') # Zero-width space
    
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    text = text.replace('●', '•')
    return text

def render_markdown_table(rows):
    """Render mảng hai chiều thành bảng Markdown chuẩn."""
    if not rows:
        return ""
    col_count = len(rows[0])
    markdown_lines = []
    
    # Hàng tiêu đề
    markdown_lines.append("| " + " | ".join(rows[0]) + " |")
    # Dòng phân cách
    markdown_lines.append("| " + " | ".join(["---"] * col_count) + " |")
    
    # Các hàng dữ liệu
    for row in rows[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")
        
    return "\n" + "\n".join(markdown_lines) + "\n"

def convert_tab_to_markdown_table(text):
    """
    Chuyển đổi bảng phân tách bằng tab (\t) thành bảng Markdown.
    Hỗ trợ gộp các dòng xuống dòng không có tab vào cột cuối cùng.
    """
    lines = text.split('\n')
    new_lines = []
    current_table = []
    col_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_table:
                new_lines.append(render_markdown_table(current_table))
                current_table = []
                col_count = 0
            new_lines.append(line)
            continue
            
        if '\t' in line:
            columns = [col.strip() for col in line.split('\t')]
            if not current_table:
                col_count = len(columns)
                current_table.append(columns)
            else:
                if len(columns) < col_count:
                    columns += [""] * (col_count - len(columns))
                else:
                    columns = columns[:col_count]
                current_table.append(columns)
        else:
            if current_table and (stripped.startswith('-') or stripped.startswith('*') or len(stripped) < 150):
                if current_table[-1][-1]:
                    current_table[-1][-1] += "<br>" + stripped
                else:
                    current_table[-1][-1] = stripped
            else:
                if current_table:
                    new_lines.append(render_markdown_table(current_table))
                    current_table = []
                    col_count = 0
                new_lines.append(line)
                
    if current_table:
        new_lines.append(render_markdown_table(current_table))
        
    return '\n'.join(new_lines)

async def crawl_policy():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    url = "https://cellphones.com.vn/tos"
    tos_data = []

    print(f"Starting crawl policy from: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_selector(".warranty-options-item", timeout=15000)
            print("Page loaded successfully.")
            
            menu_locator = page.locator(".warranty-options-item")
            menu_count = await menu_locator.count()
            print(f"Found {menu_count} policy sections in the sidebar.")
            
            if menu_count == 0:
                print("Error: No menu items found.")
                return
            
            for idx in range(menu_count):
                item = menu_locator.nth(idx)
                title = (await item.inner_text()).strip()
                title_cleaned = normalize_text(title)
                print(f"[{idx + 1}/{menu_count}] Clicking section: {title_cleaned}...")
                
                await item.click()
                await page.wait_for_timeout(2000)
                
                content_locator = page.locator(".warranty-content")
                
                content_text = ""
                for _ in range(5):
                    content_text = (await content_locator.inner_text()).strip()
                    if content_text:
                        break
                    await page.wait_for_timeout(500)
                
                # Làm sạch và định dạng bảng biểu ngay khi cào xong
                normalized_text = normalize_text(content_text)
                final_content = convert_tab_to_markdown_table(normalized_text)
                
                print(f"  -> Extracted and cleaned {len(final_content)} characters.")
                
                tos_data.append({
                    "section_index": idx + 1,
                    "title": title_cleaned,
                    "content": final_content
                })
                
        except Exception as e:
            print(f"An error occurred during crawling: {e}")
        finally:
            await browser.close()
            
    os.makedirs("./data", exist_ok=True)
    output_file = "./data/policy.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tos_data, f, ensure_ascii=False, indent=2)
        
    print(f"Finished. Saved clean TOS data to: {output_file}")

if __name__ == "__main__":
    asyncio.run(crawl_policy())

