import json
import os
import time
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Định cấu hình đường dẫn lưu trữ Database của ChromaDB
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

def main():
    # 2. Khởi tạo mô hình Embedding chạy cục bộ (Tự động tải về máy trong lần chạy đầu tiên)
    print(f"Khởi tạo mô hình Embedding cục bộ [{EMBEDDING_MODEL_NAME}]...")
    print("Lưu ý: Lần chạy đầu tiên sẽ mất vài phút để tải mô hình từ Hugging Face (~540MB).")
    start_model = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Khởi tạo mô hình thành công sau {time.time() - start_model:.2f} giây!")

    # 3. Khởi tạo Persistent Client của ChromaDB
    print(f"Khởi tạo cơ sở dữ liệu ChromaDB tại: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 4. Làm sạch database cũ bằng cách xóa các collection hiện hành (nếu có)
    print("Làm sạch các collection cũ để tránh trùng lặp dữ liệu...")
    for col_name in ["policy_collection", "product_collection"]:
        try:
            chroma_client.delete_collection(col_name)
            print(f"  -> Đã xóa collection cũ: {col_name}")
        except Exception:
            pass

    # 5. Khởi tạo các collection mới
    print("Khởi tạo các collection mới...")
    policy_col = chroma_client.get_or_create_collection(
        name="policy_collection",
        metadata={"hnsw:space": "cosine"} # Sử dụng cosine similarity
    )
    
    product_col = chroma_client.get_or_create_collection(
        name="product_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Nạp dữ liệu offline
    ingest_json_to_chroma("data/prepared_policy_chunks.json", policy_col, model)
    ingest_json_to_chroma("data/prepared_products_chunks.json", product_col, model)
    
    print("\n=== HOÀN THÀNH: Dữ liệu đã được nạp offline thành công vào ChromaDB! ===")
    print(f"  policy_collection  : {policy_col.count()} chunks")
    print(f"  product_collection : {product_col.count()} chunks")

def ingest_json_to_chroma(file_path, collection, model):
    """Đọc file JSON chứa chunks và nạp dữ liệu vào collection tương ứng sử dụng local embedding"""
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    print(f"\n--- Bắt đầu đọc file {file_path} ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Tìm thấy {len(chunks)} chunks dữ liệu. Bắt đầu sinh vector cục bộ...")

    ids = []
    documents = []
    metadatas = []

    for item in chunks:
        ids.append(item["chunk_id"])
        documents.append(item["text"])
        
        # Đảm bảo metadata không chứa kiểu dữ liệu phức tạp
        clean_metadata = {}
        for k, v in item["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)
        metadatas.append(clean_metadata)

    # Sinh embedding hàng loạt trên CPU/GPU của bạn
    print(f"  -> Đang tính toán {len(documents)} vector embedding cục bộ...")
    start_time = time.time()
    embeddings = model.encode(documents, show_progress_bar=True)
    print(f"  -> Hoàn thành tạo vector sau {time.time() - start_time:.2f} giây!")

    # Vì chạy local không lo rate limit, chúng ta có thể nạp thẳng toàn bộ mà không cần chia batch nhỏ
    print(f"  -> Đang lưu {len(ids)} phần tử vào ChromaDB [{collection.name}]...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(), # Chuyển mảng numpy sang list trước khi truyền vào Chroma
        documents=documents,
        metadatas=metadatas
    )
    print(f"Đã lưu xong vào collection {collection.name}!")

if __name__ == "__main__":
    main()
