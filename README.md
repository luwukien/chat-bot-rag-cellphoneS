# CellphoneS RAG Chatbot - Crawler & Data Preparation

This repository contains the scraping and data crawling pipeline designed to extract product specifications, variants, pricing, and stock status from the CellphoneS e-commerce website. The crawled data is structured into JSON format to serve as the knowledge base for a Retrieval-Augmented Generation (RAG) chatbot.

## 📁 Project Structure

```text
├── crawl-data/                    # Python crawler scripts powered by Playwright
│   ├── crawl_url_and_name.py       # Crawls listing pages to gather product URLs, IDs, and names
│   ├── crawl_spec_and_variant.py   # Opens each product page to extract tech specs & variants
│   ├── crawl_description.py        # Extracts detailed promotional descriptions and key features
│   ├── crawl_faq.py                # (Placeholder) To crawl Frequently Asked Questions (FAQs)
│   └── list_iphone_links.txt       # Helper text file containing target product links
├── data/                          # Crawled datasets
│   ├── list_product_details.json   # Comprehensive crawled data of all products
│   └── test_product_details.json   # Subset of product data used for testing
├── .gitignore
└── README.md                       # Project documentation
```

---

## 🛠️ Prerequisites & Setup

Ensure you have Python 3.8+ installed on your machine.

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   Make sure you have Playwright and its asynchronous dependencies installed:
   ```bash
   pip install playwright
   ```

3. **Install Playwright Browsers:**
   ```bash
   playwright install
   ```

---

## 🚀 How to Run the Crawler Pipeline

The crawling process should be executed sequentially:

### Step 1: Gather Product URLs and Names
Run `crawl_url_and_name.py` to extract all target product URLs from a category page (e.g., Apple mobile products) and generate a base JSON template.
```bash
python crawl-data/crawl_url_and_name.py
```
*Output: Generates/updates `./data/list_product_details.json`.*

### Step 2: Crawl Detailed Specifications & Variants
Run `crawl_spec_and_variant.py` to iterate through the collected product URLs, open their technical modals, and capture specs (CPU, GPU, RAM, screen size, etc.) and variants (price, color, simplified stock status).
```bash
python crawl-data/crawl_spec_and_variant.py
```
*Output: Appends `specs` and `variants` details inside `./data/list_product_details.json`.*

### Step 3: Crawl Product Key Features & Descriptions
Run `crawl_description.py` to fetch key highlights/descriptions for your products.
```bash
python crawl-data/crawl_description.py
```
*Output: Updates `./data/test_product_details.json`.*

---

## 📊 Data Schema Example

Below is the clean JSON structure produced by the crawler pipeline:

```json
{
  "id": "iphone-17-pro",
  "name": "iPhone 17 Pro 256GB | Chính hãng",
  "url": "https://cellphones.com.vn/iphone-17-pro.html",
  "specs": {
    "Hệ điều hành": "iOS 26",
    "Chipset": "Chip A19 Pro",
    "Bộ nhớ trong": "256 GB",
    "Loại CPU": "CPU 6 lõi với 2 lõi hiệu năng...",
    "Kích thước màn hình": "6.3 inches"
  },
  "variants": [
    {
      "color": "Bạc",
      "price": "33.890.000₫",
      "stock": ["Còn hàng"]
    },
    {
      "color": "Cam Vũ Trụ",
      "price": "33.790.000₫",
      "stock": ["Tạm hết hàng"]
    }
  ]
}
```

---

## 🧠 RAG Design & Best Practices

When building the vector search/RAG pipeline with this data, keep in mind:

*   **Stock Status Simplification:** Highly dynamic store addresses have been simplified to generic status lists (e.g., `["Còn hàng"]` or `["Tạm hết hàng"]`) to prevent static vector database chunking issues and out-of-date store listings.
*   **Document Transformation:** Before embedding the JSON, transform structured nodes into natural language sentences (e.g., *"Sản phẩm iPhone 17 Pro 256GB màu Bạc có giá 33.890.000₫ hiện đang Còn hàng"*). This maintains context (Product Name + Variant + Status) in a single retrieval chunk.
*   **FAQ Separation:** FAQ data should be stored in a separate collection (`faq.json`) rather than inside product specifications. Each Q&A pair acts as a perfect individual chunk for semantic retrieval.
