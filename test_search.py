import os
import pickle
import time
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

# Cấu hình đường dẫn
CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./embeddings/faiss_index.bin"
METADATA_PKL_PATH = "./embeddings/metadata.pkl"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

def search_chroma(query_text, collection_name, model, n_results=2, metadata_filter=None):
    """Tìm kiếm trên một collection của ChromaDB bằng vector tạo cục bộ"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy collection {collection_name}. Bạn đã chạy build_chroma.py chưa?")
        return []
        
    # Tạo vector cho câu hỏi cục bộ
    query_vector = model.encode(query_text).tolist()
    
    # Truy vấn bằng vector
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where=metadata_filter
    )
    
    formatted_results = []
    if results and 'documents' in results and results['documents']:
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
    return formatted_results

def search_faiss(query_text, model, n_results=2):
    """Tìm kiếm trên Index FAISS bằng vector tạo cục bộ"""
    # 1. Đọc FAISS Index và Metadata
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PKL_PATH):
        print("Lỗi: Không tìm thấy file index của FAISS. Bạn đã chạy build_faiss.py chưa?")
        return []
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PKL_PATH, "rb") as f:
        metadatas = pickle.load(f)
        
    # 2. Tạo Embedding cho câu hỏi cục bộ
    query_vector = model.encode(query_text)
    query_vector = np.array([query_vector], dtype=np.float32)
    
    # 3. Thực hiện tìm kiếm L2 trong FAISS
    distances, indices = index.search(query_vector, n_results)
    
    formatted_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadatas):
            meta = metadatas[idx]
            # Tạo bản sao sâu để tránh xóa mất trường text của đối tượng gốc lưu trong RAM
            meta_copy = meta.copy()
            text = meta_copy.pop("text", "")
            formatted_results.append({
                "text": text,
                "metadata": meta_copy,
                "distance": float(dist)
            })
    return formatted_results

def classify_query(query_text):
    

def main():
    # Khởi tạo mô hình Embedding cục bộ chung
    print(f"Đang khởi tạo mô hình embedding cục bộ [{EMBEDDING_MODEL_NAME}]...")
    start_model = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Khởi tạo mô hình thành công sau {time.time() - start_model:.2f} giây!")

    # Các câu hỏi thử nghiệm tiếng Việt
    queries = [
        "Chính sách đổi trả điện thoại trong 30 ngày đầu như thế nào?",
        "iPhone 16 Pro có giá bán bao nhiêu?",
    ]

    for q in queries:
        print("\n" + "="*80)
        print(f"CÂU HỎI TRUY VẤN: '{q}'")
        print("="*80)

        # 1. CHROME DB SEARCH
        print("\n>>> KẾT QUẢ TỪ CHROMADB (TÌM KIẾM TRÊN COLLECTION THÍCH HỢP):")
        if "chính sách" in q.lower() or "đổi trả" in q.lower() or "bảo hành" in q.lower():
            col_name = "policy_collection"
        else:
            col_name = "product_collection"
            
        chroma_res = search_chroma(q, col_name, model, n_results=2)
        for i, item in enumerate(chroma_res):
            print(f"\n[ChromaDB] Top {i+1} (Cosine Distance: {item['distance']:.4f})")
            print(f"Metadata: {item['metadata']}")
            print(f"Content: {item['text'][:250]}...")

        # 2. FAISS SEARCH
        print("\n>>> KẾT QUẢ TỪ FAISS (TÌM KIẾM TRÊN MỘT INDEX CHUNG):")
        faiss_res = search_faiss(q, model, n_results=2)
        for i, item in enumerate(faiss_res):
            print(f"\n[FAISS] Top {i+1} (L2 Distance: {item['distance']:.4f})")
            print(f"Metadata: {item['metadata']}")
            print(f"Content: {item['text'][:250]}...")

if __name__ == "__main__":
    main()
