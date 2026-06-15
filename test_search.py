import os
import pickle
import numpy as np
import faiss
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()

# Cấu hình đường dẫn
CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PKL_PATH = "metadata.pkl"

def search_chroma(query_text, collection_name, api_key, n_results=2):
    """Tìm kiếm trên một collection của ChromaDB"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/embedding-001"
    )
    
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=gemini_ef)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy collection {collection_name}. Bạn đã chạy build_chroma.py chưa?")
        return []
        
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    formatted_results = []
    # Trả về kết quả đầu tiên (query_texts là list, lấy phần tử [0])
    if results and 'documents' in results and results['documents']:
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
    return formatted_results

def search_faiss(query_text, api_key, n_results=2):
    """Tìm kiếm trên Index FAISS bằng cách tự tạo Embedding của câu hỏi"""
    # 1. Đọc FAISS Index và Metadata
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PKL_PATH):
        print("Lỗi: Không tìm thấy file index của FAISS. Bạn đã chạy build_faiss.py chưa?")
        return []
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PKL_PATH, "rb") as f:
        metadatas = pickle.load(f)
        
    # 2. Tạo Embedding cho câu hỏi truy vấn
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/embedding-001",
        content=query_text,
        task_type="retrieval_query" # Lưu ý: query dùng task_type retrieval_query, còn tài liệu dùng retrieval_document
    )
    query_vector = np.array([response['embedding']], dtype=np.float32)
    
    # 3. Thực hiện tìm kiếm L2 trong FAISS
    # distances là khoảng cách L2 (càng nhỏ càng giống), indices là các chỉ số hàng tìm thấy
    distances, indices = index.search(query_vector, n_results)
    
    formatted_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadatas):
            meta = metadatas[idx]
            text = meta.pop("text", "") # Lấy text gốc ra
            formatted_results.append({
                "text": text,
                "metadata": meta,
                "distance": float(dist)
            })
    return formatted_results

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LỖI: Chưa cấu hình GEMINI_API_KEY ở biến môi trường.")
        return

    # Các câu hỏi thử nghiệm
    queries = [
        "Chính sách đổi trả điện thoại trong 30 ngày đầu như thế nào?",
        "iPhone 16 Pro có giá bao nhiêu?",
    ]

    for q in queries:
        print("\n" + "="*80)
        print(f"CÂU HỎI TRUY VẤN: '{q}'")
        print("="*80)

        # 1. CHROME DB SEARCH
        print("\n>>> KẾT QUẢ TỪ CHROMADB (TÌM KIẾM TRÊN COLLECTION THÍCH HỢP):")
        # Chọn collection dựa theo nội dung câu hỏi một cách đơn giản để mô phỏng định tuyến
        if "chính sách" in q.lower() or "đổi trả" in q.lower() or "bảo hành" in q.lower():
            col_name = "policy_collection"
        else:
            col_name = "product_collection"
            
        chroma_res = search_chroma(q, col_name, api_key, n_results=2)
        for i, item in enumerate(chroma_res):
            print(f"\n[ChromaDB] Top {i+1} (Cosine Distance: {item['distance']:.4f})")
            print(f"Metadata: {item['metadata']}")
            print(f"Content: {item['text'][:200]}...")

        # 2. FAISS SEARCH
        print("\n>>> KẾT QUẢ TỪ FAISS (TÌM KIẾM TRÊN MỘT INDEX CHUNG):")
        faiss_res = search_faiss(q, api_key, n_results=2)
        for i, item in enumerate(faiss_res):
            print(f"\n[FAISS] Top {i+1} (L2 Distance: {item['distance']:.4f})")
            print(f"Metadata: {item['metadata']}")
            print(f"Content: {item['text'][:200]}...")

if __name__ == "__main__":
    main()
