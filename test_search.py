import os
import pickle
import time
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize, text_normalize

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
    """
    This method help determine metadata and choose collection for query of user
    """
    query_lower = query_text.lower()
    
    policy_keywords = [
        "chính sách", "điều khoản", "quy định", "bảo hành", "đổi trả", 
        "hoàn tiền", "trả góp", "lỗi"
    ]
    
    variant_keywords = [
        "giá", "mua", "bán", "cửa hàng", "sản phẩm", "dịch vụ", "màu sắc", "còn hàng",
        "hết hàng"
    ]
    specs_keywords = [
        "thông số", "cấu hình", "ram", "cpu", "chip", "màn hình", 
        "camera", "pin", "nặng", "kích thước", "bộ nhớ", "rom"
    ]


    # Kiểm tra xem câu hỏi có thuộc nhóm chính sách (policy) không
    is_policy = any(keyword in query_lower for keyword in policy_keywords)
    
    if is_policy:
        collection_name = "policy_collection"
        metadata_filter = {"type": "policy"}  # Chỉ lấy các chunk thuộc loại chính sách
        return collection_name, metadata_filter

    # Mặc định tìm trong product_collection
    collection_name = "product_collection"
    metadata_filter = {}
    
    # Ưu tiên kiểm tra từ khóa hỏi về Giá/Màu (variants) trước
    if any(keyword in query_lower for keyword in variant_keywords):
        metadata_filter = {"type": "variants"}
        
    # Tiếp theo kiểm tra từ khóa hỏi về Cấu hình/Thông số (specs)
    elif any(keyword in query_lower for keyword in specs_keywords):
        metadata_filter = {"type": "specs"}
        
    # Nếu không khớp nhóm nào, mặc định tìm trong phần mô tả chi tiết (description)
    else:
        metadata_filter = {"type": "description"}
        
    return collection_name, metadata_filter


def search_bm25(query_text, collection_name, n_results=5, metadata_filter=None):
    """Tìm kiếm từ khóa bằng thuật toán BM25 trên tập dữ liệu đã lọc metadata"""
    # 1. Kết nối ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy collection {collection_name}")
        return []
        
    # 2. Lấy danh sách tài liệu thỏa mãn điều kiện lọc metadata
    # Hàm .get() sẽ lấy văn bản gốc chứ không cần tính toán vector
    all_data = collection.get(where=metadata_filter)
    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    
    # Nếu không có tài liệu nào thỏa mãn bộ lọc, trả về danh sách rỗng
    if not documents:
        return []

    #Normalized document and query
    documents_normalized = [text_normalize(doc) for doc in documents]
    query_normalized = text_normalize(query_text)

    # 3. Tiến hành tách từ tiếng Việt cho toàn bộ danh sách tài liệu (Corpus)
    tokenized_corpus = []
    for doc in documents_normalized:
        # word_tokenize trả về danh sách các từ đã tách, ví dụ: ["điện thoại", "iphone"]
        tokens = word_tokenize(doc.lower())
        tokenized_corpus.append(tokens)
        
    # Tách từ tiếng Việt cho câu hỏi (Query)
    tokenized_query = word_tokenize(query_normalized.lower())

    # 4. Khởi tạo mô hình BM25 và tính điểm mức độ liên quan
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # 5. Lấy chỉ số (index) của tài liệu có điểm từ cao xuống thấp
    top_indices = np.argsort(scores)[::-1]
    
    formatted_results = []
    for idx in top_indices:
        score = scores[idx]
        
        # Chỉ lấy tài liệu có điểm > 0 (có khớp từ khóa)
        if score > 0:
            formatted_results.append({
                "text": documents[idx],
                "metadata": metadatas[idx],
                "bm25_score": float(score),
                "id": metadatas[idx].get("chunk_id", "")
            })
            
            # Dừng lại khi đã đủ số lượng yêu cầu
            if len(formatted_results) >= n_results:
                break
                
    return formatted_results

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



