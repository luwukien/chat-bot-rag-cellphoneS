import json
import os
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Cấu hình đường dẫn đầu ra cho FAISS và Metadata
FAISS_INDEX_PATH = "./embeddings/faiss_index.bin"
METADATA_PKL_PATH = "./embeddings/metadata.pkl"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

def main():
    # 1. Khởi tạo mô hình Embedding cục bộ
    print(f"Khởi tạo mô hình Embedding cục bộ [{EMBEDDING_MODEL_NAME}]...")
    start_model = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Khởi tạo mô hình thành công sau {time.time() - start_model:.2f} giây!")

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
    
    # 3. Tạo Embeddings
    print("Bắt đầu sinh embedding cục bộ...")
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Hoàn thành tạo vector sau {time.time() - start_time:.2f} giây!")
    
    # Ép kiểu dữ liệu về float32 để tương thích tốt nhất với FAISS
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Kích thước ma trận vector: {embeddings.shape}")

    # 4. Khởi tạo FAISS Index
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

    print("\n=== HOÀN THÀNH: Đã lưu thành công FAISS index và Metadata offline! ===")

if __name__ == "__main__":
    main()
