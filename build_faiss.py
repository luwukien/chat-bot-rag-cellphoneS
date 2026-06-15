import json
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
import time
from dotenv import load_dotenv

# Cấu hình đường dẫn đầu ra cho FAISS và Metadata
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PKL_PATH = "metadata.pkl"

import re

def get_embeddings(texts, model_name="gemini-embedding-001"):
    """
    Gọi API Gemini để sinh vector embedding cho một danh sách văn bản.
    Tự động bắt lỗi 429 Rate Limit và ngủ đúng số giây yêu cầu trước khi thử lại.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return np.array(response['embedding'], dtype=np.float32)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Tìm số giây cần đợi trong thông báo lỗi (ví dụ: 'Please retry in 52.21s')
                wait_match = re.search(r"Please retry in ([\d\.]+)s", err_msg)
                wait_time = int(float(wait_match.group(1))) + 2 if wait_match else 60
                
                print(f"\n[Rate Limit] Bị giới hạn cuộc gọi (429). Tự động dừng chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Vượt quá số lần thử lại tối đa do lỗi Rate Limit liên tục.")

def main():
    # 1. Kiểm tra API Key
    load_dotenv()  # Nạp biến môi trường từ .env
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LỖI: Chưa cấu hình GEMINI_API_KEY ở biến môi trường hoặc file .env.")
        return
    genai.configure(api_key=api_key)

    # 2. Đọc dữ liệu từ 2 file JSON chunks
    docs = []
    
    # Đọc chính sách
    policy_path = "data/prepared_policy_chunks.json"
    if os.path.exists(policy_path):
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy_chunks = json.load(f)
            docs.extend(policy_chunks)
            print(f"Đã đọc {len(policy_chunks)} chunks từ chính sách.")

    # Đọc sản phẩm
    products_path = "data/prepared_products_chunks.json"
    if os.path.exists(products_path):
        with open(products_path, 'r', encoding='utf-8') as f:
            product_chunks = json.load(f)
            docs.extend(product_chunks)
            print(f"Đã đọc {len(product_chunks)} chunks từ sản phẩm.")

    if not docs:
        print("Lỗi: Không tìm thấy dữ liệu để xử lý.")
        return

    # Tách riêng phần text và metadata
    texts = [d["text"] for d in docs]
    # Lưu text gộp chung vào metadata để khi query có dữ liệu hiển thị
    metadatas = [{"text": d["text"], **d["metadata"]} for d in docs]

    print(f"\nTổng số lượng chunks cần embedding: {len(texts)}")
    
    # 3. Tạo Embeddings theo batch
    print("Bắt đầu sinh embedding qua Gemini API...")
    all_embeddings = []
    batch_size = 100
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i:i+batch_size]
        print(f" -> Đang embed batch {batch_num}/{total_batches}...")
        batch_embeds = get_embeddings(batch_texts)
        all_embeddings.append(batch_embeds)
        
        # Nghỉ nhẹ 2 giây giữa các batch, nếu bị giới hạn (429) hàm get_embeddings sẽ tự xử lý
        if batch_num < total_batches:
            time.sleep(2)

    # Gộp tất cả các mảng numpy lại thành 1 ma trận duy nhất
    embeddings = np.vstack(all_embeddings)
    print(f"Kích thước ma trận vector: {embeddings.shape}") # Ví dụ: (Số lượng chunks, 768)

    # 4. Khởi tạo FAISS Index
    # IndexFlatL2 sử dụng khoảng cách Euclidean (L2). 
    # Kích thước chiều của vector (dimension) là số cột của ma trận (ví dụ: 768)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Nạp toàn bộ các vector vào index FAISS
    print("Đang nạp vector vào FAISS Index...")
    index.add(embeddings)

    # 5. Lưu Index và Metadata xuống ổ đĩa
    print(f"Đang lưu FAISS index vào: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"Đang lưu metadata tương ứng vào: {METADATA_PKL_PATH}")
    with open(METADATA_PKL_PATH, "wb") as f:
        pickle.dump(metadatas, f)

    print("\nHOÀN THÀNH NHIỆM VỤ 3: Đã lưu thành công FAISS index và Metadata!")

if __name__ == "__main__":
    main()
