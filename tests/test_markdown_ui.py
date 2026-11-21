"""Browser regression checks for Markdown answers; no RAG models or Groq key needed.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -v
Requires the Playwright Chromium browser used by the project's crawlers.
"""

import functools
import json
import os
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from playwright.sync_api import sync_playwright


SAMPLE_ANSWER = """### 1. Thiết kế & màn hình
| Đặc điểm | iPhone 13 Pro | iPhone 14 Pro |
|----------|---------------|---------------|
| Khung viền | Thép không gỉ, mặt kính nhám mờ | (không có trong tài liệu) |
| Kích thước màn hình | **6.1 inch**, công nghệ ProMotion 120 Hz | (không có trong tài liệu) |
| Notch / Dynamic Island | Notch truyền thống | Dynamic Island – phần cắt hình “viên thuốc” mới, có thể biến đổi để hiển thị thông tin và tương tác đồng thời với các ứng dụng (FAQ iPhone 14 Pro). |

### 2. Chip & hiệu năng
| Đặc điểm | iPhone 13 Pro | iPhone 14 Pro |
|----------|---------------|---------------|
| Chip xử lý | A15 Bionic | Chip được xây dựng trên tiến trình 4 nm, mang lại hiệu năng cao hơn và tiêu thụ ít năng lượng hơn 20 % so với A15 Bionic (FAQ iPhone 14 Pro). |
| Khả năng chơi game / xử lý đồ họa | Đáp ứng tốt các game đồ họa nặng | Chip mới (FAQ iPhone 14 Pro). |

### 3. Camera
| Đặc điểm | iPhone 13 Pro | iPhone 14 Pro |
|----------|---------------|---------------|
| Cảm biến | Cảm biến lớn hơn so với thế hệ trước | (không có chi tiết cụ thể trong tài liệu) |

### 4. Dung lượng lưu trữ
| Model | Dung lượng |
|-------|------------|
| iPhone 13 Pro | 128 GB, 1 TB |
| iPhone 14 Pro | (không có thông tin dung lượng trong tài liệu) |

### 5. Các tính năng nổi bật khác
- **iPhone 13 Pro:** Màn hình ProMotion 120 Hz.
- **iPhone 14 Pro:** Dynamic Island.

---

> Quý khách vui lòng liên hệ tổng đài 1800.2097 để biết thêm thông tin.
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class MarkdownUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        static = Path(__file__).resolve().parents[1] / "static"
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(static))
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        )
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.context.route("https://fonts.googleapis.com/**", lambda route: route.abort())
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.url)

    def tearDown(self):
        self.context.close()

    def submit_answer(self, answer):
        def respond(route):
            self.assertEqual(route.request.method, "POST")
            self.assertEqual(route.request.post_data_json["message"], "So sánh iPhone 13 Pro và 14 Pro")
            route.fulfill(
                content_type="application/json",
                body=json.dumps({
                    "answer": answer,
                    "sources": [],
                    "cleaned_query": "So sánh iPhone 13 Pro và iPhone 14 Pro",
                    "sub_queries": [],
                }),
            )
        self.page.route("**/api/chat", respond)
        self.page.locator("#userInput").fill("So sánh iPhone 13 Pro và 14 Pro")
        self.page.locator("#sendBtn").click()
        self.page.locator(".markdown-content").wait_for()

    def test_chat_renders_multiple_tables_and_markdown(self):
        self.submit_answer(SAMPLE_ANSWER)
        answer = self.page.locator(".markdown-content")
        self.assertEqual(answer.locator("table").count(), 4)
        self.assertEqual(answer.locator("h3").count(), 5)
        self.assertEqual(answer.locator("table").first.locator("th[scope=col]").count(), 3)
        self.assertEqual(answer.locator("td strong").inner_text(), "6.1 inch")
        self.assertEqual(answer.locator("li").count(), 2)
        self.assertIn("1800.2097", answer.locator("blockquote").inner_text())
        self.assertNotIn("|----------", answer.inner_text())
        self.assertEqual(self.errors, [])

    def test_mobile_scrolls_tables_without_widening_page(self):
        self.submit_answer(SAMPLE_ANSWER)
        for width in (390, 320):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 844})
                sizes = self.page.evaluate("""() => ({
                    page: document.documentElement.scrollWidth,
                    viewport: innerWidth,
                    tables: [...document.querySelectorAll('.markdown-table')].map(el => ({
                        visible: el.clientWidth, content: el.scrollWidth
                    }))
                })""")
                self.assertLessEqual(sizes["page"], sizes["viewport"])
                self.assertTrue(all(t["content"] > t["visible"] for t in sizes["tables"]))
        table = self.page.locator(".markdown-table").first
        table.focus()
        self.page.keyboard.press("ArrowRight")
        self.page.wait_for_function("document.querySelector('.markdown-table').scrollLeft > 0")

    def test_untrusted_html_and_escaped_table_cells(self):
        self.submit_answer("""| Model | Chi tiết |
|:---|---:|
| iPhone \\| VN/A | **A15** và `128GB`<br>Thông tin |

<img src=x onerror="window.injected=true">
<script>window.injected=true</script>
[Liên kết](javascript:alert(1))
""")
        answer = self.page.locator(".markdown-content")
        self.assertEqual(answer.locator("tbody td").count(), 2)
        self.assertEqual(answer.locator("tbody td").first.inner_text(), "iPhone | VN/A")
        self.assertEqual(answer.locator("code").inner_text(), "128GB")
        self.assertEqual(answer.locator("img, script, [onerror], a[href^='javascript:']").count(), 0)
        self.assertFalse(self.page.evaluate("Boolean(window.injected)"))
        self.assertEqual(answer.locator("td").last.evaluate("el => getComputedStyle(el).textAlign"), "right")
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
