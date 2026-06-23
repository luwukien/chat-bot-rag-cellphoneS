import os
import re
import json
import pickle
import sys
import time
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize, text_normalize
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Cấu hình đường dẫn
CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./embeddings/faiss_index.bin"
METADATA_PKL_PATH = "./embeddings/metadata.pkl"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
RERANKER_MODEL_NAME = "itdainb/PhoRanker"


# Tải biến môi trường từ file .env trong thư mục làm việc hiện tại
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))
# Cấu hình UTF-8 cho Windows Terminal để không bị lỗi mã hóa chữ tiếng Việt
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

print("Đang khởi tạo Gemini API và nạp các mô hình cục bộ...")
client = genai.Client()
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
print("Đã tải xong các mô hình!")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def search_chroma(query_text, collection_name, model, n_results=2, metadata_filter=None):
    """Tìm kiếm trên một collection của ChromaDB bằng vector tạo cục bộ"""
    
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

def hybrid_search(query_text, collection_name, model, n_results=5, metadata_filter=None, k=60):
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
            "rrf_score": score,
            "id": doc_id
        })
        
    return hybrid_results

def build_model_mappings():
    collection = chroma_client.get_collection(name="product_collection")
    all_data = collection.get()
    metadatas = all_data.get('metadatas', [])
    
    base_to_pids = {}
    pid_to_name = {}
    
    def clean_model_name(name):
        name = name.replace("Điện thoại ", "")
        # Tách bỏ phần dung lượng (ví dụ: 128GB) để lấy tên dòng máy gốc
        parts = re.split(r'\s+\d+(?:GB|TB|gb|tb)', name, flags=re.IGNORECASE)
        cleaned = parts[0]
        cleaned = re.split(r'\s*(?:\||I)\s*', cleaned)[0]
        return cleaned.strip()

    for meta in metadatas:
        pid = meta.get("product_id")
        pname = meta.get("product_name")
        if pid and pname:
            pid_to_name[pid] = pname
            base_name = clean_model_name(pname)
            if base_name not in base_to_pids:
                base_to_pids[base_name] = set()
            base_to_pids[base_name].add(pid)
            
    sorted_base_names = sorted(base_to_pids.keys(), key=len, reverse=True)
    return base_to_pids, sorted_base_names, pid_to_name

print("Đang xây dựng ánh xạ sản phẩm...")
base_to_pids, sorted_base_names, pid_to_name = build_model_mappings()

def llm_decompose_query(query_text):
    prompt = f"""
    Bạn là một trợ lý RAG Planner thông minh. Hãy phân rã câu hỏi phức tạp của người dùng thành danh sách các câu hỏi đơn (sub-queries) để tìm kiếm hiệu quả hơn.

    Hướng dẫn phân rã:
    1. Tách theo thực thể: Nếu câu hỏi hỏi về nhiều sản phẩm (ví dụ: so sánh A và B), hãy tạo các câu hỏi riêng cho từng sản phẩm.
    2. Tách theo khía cạnh thông tin: Trong cơ sở dữ liệu của chúng tôi:
    - Thông tin về Giá bán, màu sắc, tình trạng hàng nằm ở phần "Giá/Biến thể" (variants).
    - Thông tin về Thông số kỹ thuật, camera, chip, pin, màn hình nằm ở phần "Cấu hình/Thông số" (specs).
    Vì vậy, NẾU câu hỏi hỏi đồng thời cả giá bán VÀ cấu hình/pin/camera, bạn BẮT BUỘC phải tách thành các câu hỏi phụ riêng biệt (một câu hỏi về giá, một câu hỏi về cấu hình/pin). KHÔNG gộp chung giá và cấu hình/pin vào cùng một câu hỏi phụ.

    Ví dụ 1:
    Câu hỏi: "So sánh iPhone 13 Pro và iPhone 14 Pro về giá và pin"
    Trả về định dạng JSON:
    {{
    "sub_queries": [
        "iPhone 13 Pro giá bao nhiêu",
        "iPhone 13 Pro dung lượng pin thế nào",
        "iPhone 14 Pro giá bao nhiêu",
        "iPhone 14 Pro dung lượng pin thế nào"
    ]
    }}

    Ví dụ 2:
    Câu hỏi: "Cấu hình chi tiết camera và chip xử lý của iPhone 16 Pro 128gb là gì?"
    Trả về định dạng JSON:
    {{
    "sub_queries": [
        "cấu hình chi tiết camera iPhone 16 Pro 128gb",
        "vi xử lý chip của iPhone 16 Pro 128gb"
    ]
    }}

    Ví dụ 3:
    Câu hỏi: "Chính sách bảo hành đổi trả của CellphoneS"
    Trả về định dạng JSON:
    {{
    "sub_queries": [
        "Chính sách bảo hành đổi trả của CellphoneS"
    ]
    }}

    Câu hỏi của người dùng: "{query_text}"
    Trả về JSON chứa danh sách "sub_queries".
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        return data.get("sub_queries", [query_text])
    except Exception as e:
        print(f"Lỗi phân tách truy vấn bằng LLM: {e}")
        return [query_text]

def retrieve_and_rerank(query_text, n_results=5):
    # 1. Phân rã câu hỏi bằng Gemini
    sub_queries = llm_decompose_query(query_text)
    print(f"   [LLM Planner] Sub-queries generated: {sub_queries}")
    
    # 2. Tìm kiếm ứng viên cho từng câu hỏi phụ
    all_candidates = []
    retrieved_by = {}  # Đánh dấu: chunk_id -> danh sách các sub_query tìm ra nó
    
    for sq in sub_queries:
        collection_name, metadata_filter = classify_query(sq)
        candidates = hybrid_search(sq, collection_name, embed_model, n_results=20, metadata_filter=metadata_filter)
        
        for cand in candidates:
            # Lấy hoặc tự tạo unique id cho chunk
            cid = cand.get("id") or cand.get("metadata", {}).get("chunk_id")
            if not cid:
                import hashlib
                cid = hashlib.md5(cand["text"].encode('utf-8')).hexdigest()
                cand["id"] = cid
            
            # Lưu vết để phục vụ rerank chéo sau này
            if cid not in retrieved_by:
                retrieved_by[cid] = []
                all_candidates.append(cand)
            retrieved_by[cid].append(sq)
            
    print(f"   [Merge] Merged & deduplicated to {len(all_candidates)} unique candidates.")
    if not all_candidates:
        return []
        
    # 3. Chuẩn bị các cặp (sub_query, chunk_text) để PhoRanker chấm điểm
    pairs = []
    pair_mapping = []
    for cand in all_candidates:
        cid = cand["id"]
        # Chỉ chấm điểm chunk với các sub-query thực tế đã tìm ra nó
        for sq in retrieved_by[cid]:
            pairs.append([sq, cand["text"]])
            pair_mapping.append((cand, sq))
            
    # Chạy PhoRanker
    start_rerank = time.time()
    scores = reranker_model.predict(pairs)
    print(f"   [Rerank] Scored {len(pairs)} pairs using PhoRanker in {(time.time() - start_rerank)*1000:.1f} ms")
    
    # Gán điểm lớn nhất (max) cho mỗi chunk
    cand_scores = {}
    for score, (cand, sq) in zip(scores, pair_mapping):
        cid = cand["id"]
        cand_scores[cid] = max(cand_scores.get(cid, -9999.0), float(score))
        
    for cand in all_candidates:
        cand["rerank_score"] = cand_scores[cand["id"]]
        
    # Sắp xếp và trả về Top N kết quả
    ranked_results = sorted(all_candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked_results[:n_results]


def main():
    queries = [
        "Cấu hình chi tiết camera và chip xử lý của iPhone 16 Pro 128gb là gì?",
        "iPhone 13 Pro giá bao nhiêu và có những màu gì?",
        "So sánh iPhone 13 Pro và iPhone 14 Pro về giá và pin",
        "Chính sách bảo hành đổi trả của CellphoneS"
    ]
    
    for q in queries:
        print("\n" + "="*90)
        print(f"🎯 CÂU HỎI TRUY VẤN: '{q}'")
        print("="*90)
        
        start_time = time.time()
        final_results = retrieve_and_rerank(q, n_results=5)
        duration = time.time() - start_time
        
        print(f"\n[Kết quả RAG sau Rerank] - Thời gian xử lý: {duration:.2f} giây")
        for i, item in enumerate(final_results):
            snippet = item['text'].replace('\n', ' ')[:100]
            mtype = item['metadata'].get('type', 'N/A')
            pid = item['metadata'].get('product_id', 'N/A')
            score = item.get('rerank_score', 0.0)
            print(f"   [{i+1}] Score: {score:.4f} | Type: {mtype} | PID: {pid} | Snippet: {snippet}...")

if __name__ == "__main__":
    main()



