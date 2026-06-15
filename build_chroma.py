import json
import os
import chromadb
import google.genai as genai
import time
from dotenv import load_dotenv

# Load biến môi trường từ file .env TRƯỚC TIÊN
load_dotenv()

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100

import time
import re

def get_embeddings(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Gọi Gemini API để lấy vector embedding cho một danh sách text.
    Nếu gặp lỗi Rate Limit (429), tự động trích xuất số giây cần đợi và thử lại.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Tìm số giây cần đợi trong thông báo lỗi (ví dụ: 'Please retry in 52.21s')
                wait_match = re.search(r"Please retry in ([\d\.]+)s", err_msg)
                wait_time = int(float(wait_match.group(1))) + 2 if wait_match else 60
                
                print(f"\n[Rate Limit] Bị giới hạn cuộc gọi (429). Tự động dừng chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                # Nếu là lỗi khác thì quăng lỗi ra ngoài
                raise e
    raise RuntimeError("Vượt quá số lần thử lại tối đa do lỗi Rate Limit liên tục.")


def ingest_json_to_chroma(
    file_path: str,
    collection: chromadb.Collection,
    genai_client: genai.Client,
):
    """Đọc file JSON, tạo embedding và nạp vào ChromaDB collection."""
    if not os.path.exists(file_path):
        print(f"LOI: Khong tim thay file {file_path}")
        return

    print(f"\n--- Doc file: {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Tim thay {len(chunks)} chunks. Bat dau xu ly...")

    ids, documents, metadatas = [], [], []
    for item in chunks:
        ids.append(item["chunk_id"])
        documents.append(item["text"])

        # ChromaDB chỉ nhận metadata dạng str/int/float/bool
        clean_meta = {}
        for k, v in item["metadata"].items():
            clean_meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        metadatas.append(clean_meta)

    # Nạp theo batch: tạo embedding rồi add thẳng vào Chroma
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(ids), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        b_ids = ids[i : i + BATCH_SIZE]
        b_docs = documents[i : i + BATCH_SIZE]
        b_metas = metadatas[i : i + BATCH_SIZE]

        print(f"  Batch {batch_num}/{total_batches} ({len(b_ids)} chunks) - dang embed...")
        b_embeddings = get_embeddings(genai_client, b_docs)

        collection.add(
            ids=b_ids,
            embeddings=b_embeddings,  # Truyền vector trực tiếp, không nhờ Chroma tự embed
            documents=b_docs,
            metadatas=b_metas,
        )
        print(f"  Batch {batch_num}/{total_batches} - da luu vao ChromaDB!")
        
        # Nghỉ nhẹ 2 giây giữa các batch, nếu bị giới hạn (429) hàm get_embeddings sẽ tự xử lý
        if batch_num < total_batches:
            time.sleep(2)

    print(f"Hoan thanh: {len(ids)} chunks -> [{collection.name}]")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("LOI: Khong tim thay GEMINI_API_KEY trong file .env hoac bien moi truong.")
        return

    # Khởi tạo Gemini client (dùng thư viện google-genai mới nhất)
    print("Khoi tao Gemini client...")
    gemini_client = genai.Client(api_key=api_key)

    # Khởi tạo ChromaDB Persistent Client (lưu dữ liệu xuống đĩa)
    print(f"Khoi tao ChromaDB tai: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Tạo 2 collections riêng biệt cho policy và product
    # get_or_create_collection: nếu đã tồn tại thì lấy lại, không tạo mới trùng lặp
    print("Tao/Lay cac collections...")
    policy_col = chroma_client.get_or_create_collection(
        name="policy_collection",
        metadata={"hnsw:space": "cosine"},
    )
    product_col = chroma_client.get_or_create_collection(
        name="product_collection",
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  policy_collection  : {policy_col.count()} chunks hien co")
    print(f"  product_collection : {product_col.count()} chunks hien co")

    # Nạp dữ liệu
    ingest_json_to_chroma("data/prepared_policy_chunks.json", policy_col, gemini_client)
    ingest_json_to_chroma("data/prepared_products_chunks.json", product_col, gemini_client)

    print("\n=== HOAN THANH: Du lieu da duoc nap vao ChromaDB! ===")
    print(f"  policy_collection  : {policy_col.count()} chunks")
    print(f"  product_collection : {product_col.count()} chunks")


if __name__ == "__main__":
    main()
