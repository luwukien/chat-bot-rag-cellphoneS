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
        for doc, meta, dist, doc_id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "id": doc_id
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

def search_bm25_with_chroma(query_text, collection_name, n_results=5, metadata_filter=None):
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
    ids = all_data.get('ids', [])
    
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
                "id": ids[idx]
            })
            
            # Dừng lại khi đã đủ số lượng yêu cầu
            if len(formatted_results) >= n_results:
                break
                
    return formatted_results

def hybrid_search(query_text, collection_name, model, n_results=2, metadata_filter=None, k=60):
    """Tìm kiếm kết hợp Vector Search và BM25, dùng thuật toán RRF để dung hợp kết quả"""
    if metadata_filter is None:
        metadata_filter = {}

    # Chạy đồng thời 2 bộ tìm kiếm với tập ứng viên rộng hơn (n_results * 2)
    # Tại sao n_result * 2? Đại khái là nó giúp tìm ra những thằng tiềm năng nó dung hợp cả 2 giữa tìm kiếm theo ngữ nghĩa và tìm kiếm theo bm25
    vector_results = search_chroma(query_text, collection_name, model, n_results=n_results * 2, metadata_filter=metadata_filter)
    bm25_results = search_bm25_with_chroma(query_text, collection_name, n_results=n_results * 2, metadata_filter=metadata_filter)

    rrf_scores = {}
    doc_map = {}

    # Tính điểm RRF cho nhánh Vector Search
    for rank, doc in enumerate(vector_results):
        # Lấy ID duy nhất của chunk
        doc_id = doc.get("id")
        if not doc_id:
            continue
            
        # Áp dụng công thức RRF: 1 / (k + rank_1based)
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        
        # Lưu lại tài liệu gốc
        doc_map[doc_id] = doc

    # Tính điểm RRF cho nhánh BM25 Search
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get("id")
        if not doc_id:
            continue
            
        # Cộng dồn điểm RRF
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        
        # Lưu lại tài liệu nếu nó chưa từng xuất hiện ở nhánh Vector
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
    # Sắp xếp các tài liệu theo điểm RRF giảm dần
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top n_results tài liệu
    top_docs = sorted_docs[:n_results]
    
    hybrid_results = []
    for doc_id, score in top_docs:
        # Lấy thông tin tài liệu từ map
        final_doc = doc_map[doc_id].copy()
        # Lưu lại điểm RRF vào kết quả để tiện theo dõi
        final_doc["rrf_score"] = score
        # Chuẩn hóa lại cấu trúc kết quả trả về
        hybrid_results.append({
            "text": final_doc["text"],
            "metadata": final_doc["metadata"],
            "rrf_score": score
        })
        
    return hybrid_results


def main():
    # 1. Khởi tạo mô hình Embedding cục bộ
    print(f"Đang khởi tạo mô hình embedding cục bộ [{EMBEDDING_MODEL_NAME}]...")
    start_model = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Khởi tạo mô hình thành công sau {time.time() - start_model:.2f} giây!\n")

    # 2. Danh sách các câu hỏi test bao trùm nhiều nhóm thông tin khác nhau
    queries = [
        "Cấu hình chi tiết camera và chip xử lý của iPhone 16 Pro là gì?",
        "iPhone 13 Pro giá bao nhiêu và có những màu gì?",
        "Chính sách đổi trả và hoàn tiền của cửa hàng trong 30 ngày đầu như thế nào?"
    ]

    for q in queries:
        print("\n" + "="*90)
        print(f"🎯 CÂU HỎI TRUY VẤN: '{q}'")
        print("="*90)

        # 3. Phân loại câu hỏi tự động để lấy Collection và Metadata Filter
        collection_name, metadata_filter = classify_query(q)
        print(f"[Định tuyến RAG] -> Collection: {collection_name} | Bộ lọc Metadata: {metadata_filter}\n")

        # 4. CHẠY THỬ PHƯƠNG PHÁP 1: VECTOR SEARCH (CHROMA)
        print(">>> 1. KẾT QUẢ TỪ VECTOR SEARCH (CHROMA):")
        chroma_res = search_chroma(q, collection_name, model, n_results=2, metadata_filter=metadata_filter)
        if not chroma_res:
            print("   (Không tìm thấy kết quả)")
        for i, item in enumerate(chroma_res):
            print(f"   [{i+1}] (Distance: {item['distance']:.4f})")
            print(f"       Metadata: {item['metadata']}")
            print(f"       Nội dung: {item['text'][:180]}...")

        # 5. CHẠY THỬ PHƯƠNG PHÁP 2: KEYWORD SEARCH (BM25)
        print("\n>>> 2. KẾT QUẢ TỪ KEYWORD SEARCH (BM25):")
        bm25_res = search_bm25_with_chroma(q, collection_name, n_results=2, metadata_filter=metadata_filter)
        if not bm25_res:
            print("   (Không tìm thấy kết quả)")
        for i, item in enumerate(bm25_res):
            print(f"   [{i+1}] (BM25 Score: {item['bm25_score']:.2f})")
            print(f"       Metadata: {item['metadata']}")
            print(f"       Nội dung: {item['text'][:180]}...")

        # 6. CHẠY THỬ PHƯƠNG PHÁP 3: HYBRID SEARCH (RRF FUSION)
        print("\n>>> 3. KẾT QUẢ TỪ HYBRID SEARCH (RRF DUNG HỢP):")
        hybrid_res = hybrid_search(q, collection_name, model, n_results=2, metadata_filter=metadata_filter)
        if not hybrid_res:
            print("   (Không tìm thấy kết quả)")
        for i, item in enumerate(hybrid_res):
            print(f"   [{i+1}] (RRF Score: {item['rrf_score']:.6f})")
            print(f"       Metadata: {item['metadata']}")
            print(f"       Nội dung: {item['text'][:180]}...")

if __name__ == "__main__":
    main()
    # Nếu muốn test riêng hàm BM25, bạn có thể comment main() và mở comment dòng dưới:
    # test_search_bm25()



