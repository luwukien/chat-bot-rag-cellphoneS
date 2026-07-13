import os
import re
import json
import sys
import time
import logging
import warnings
import numpy as np
import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize, text_normalize
from dotenv import load_dotenv

# ==========================================
# 0. CẤU HÌNH & KHỞI TẠO HỆ THỐNG
# ==========================================

# Tắt các cảnh báo không cần thiết từ thư viện transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

# Cấu hình đường dẫn dữ liệu & tên model
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
RERANKER_MODEL_NAME = "itdainb/PhoRanker"

# Tải biến môi trường (.env)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Cấu hình UTF-8 cho console Windows để in tiếng Việt không bị lỗi font
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

print("Đang khởi tạo Groq API và nạp các mô hình cục bộ...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
print("Đã tải xong các mô hình!")


# ==========================================
# 1. TIỀN XỬ LÝ TRUY VẤN BẰNG LLM (GROQ)
# ==========================================

def llm_process_query(query_text):
    """
    Sử dụng LLM (Groq Llama-3.3) để:
    1. Sửa lỗi chính tả tiếng Việt, khôi phục dấu và viết tắt chuẩn (ip -> iPhone, pm -> Pro Max...)
    2. Phân rã câu hỏi phức tạp thành danh sách các sub-queries đơn giản.
    
    *Hỗ trợ Retry tự động* khi gặp lỗi Rate Limit (HTTP 429) của Groq API.
    """
    prompt = f"""
    Bạn là một trợ lý RAG Planner thông minh và chuyên gia chuẩn hóa tiếng Việt cho chatbot của cửa hàng công nghệ CellphoneS.
    
    Nhiệm vụ của bạn là nhận vào câu hỏi của người dùng và trả về một đối tượng JSON có cấu trúc như sau:
    {{
      "cleaned_query": "Câu hỏi gốc đã được sửa hết lỗi chính tả, viết tắt, và khôi phục dấu tiếng Việt chuẩn",
      "need_decomposition": true hoặc false (true nếu là câu hỏi phức tạp cần phân rã, false nếu là câu hỏi đơn giản),
      "sub_queries": [
         "Danh sách các câu hỏi phụ đơn giản đã được sửa lỗi chính tả. Nếu need_decomposition là false, danh sách này chỉ chứa duy nhất cleaned_query."
      ]
    }}
    
    Quy tắc chuẩn hóa tiếng Việt và sản phẩm:
    - Sửa các từ viết tắt thông dụng: "ip", "iphon", "ipone" -> "iPhone"; "pm", "pr max", "promax" -> "Pro Max"; "đt" -> "điện thoại"; "bh" -> "bảo hành"; "dt" -> "điện thoại"; "km" -> "khuyến mãi".
    - Khôi phục dấu tiếng Việt đầy đủ và tự nhiên.
    - Giữ nguyên các thông số kỹ thuật (ví dụ: 128GB, 256GB, LTE, 5G).
    - Lưu ý đặc biệt về ngữ cảnh công nghệ khi khôi phục dấu: "sac" trong ngữ cảnh điện thoại luôn là "sạc" (sạc pin), KHÔNG phải "sắc" (màu sắc). Ví dụ: "pin va sac" -> "pin và sạc"; "sac nhanh" -> "sạc nhanh"; "cong sac" -> "cổng sạc"; "cap sac" -> "cáp sạc".
    
    Quy tắc phân rã (decomposition):
    - Phân rã nếu câu hỏi hỏi về từ 2 sản phẩm trở lên (so sánh, đối chiếu) HOẶC hỏi đồng thời cả giá bán (variants) VÀ cấu hình/pin/camera (specs) của một sản phẩm.
    - ĐẶC BIỆT: Nếu câu hỏi vừa hỏi về sản phẩm cụ thể vừa có từ khóa chính sách chung (ví dụ: bảo hành, đổi trả, trả góp, lỗi kỹ thuật...), hãy phân rã phần chính sách thành một câu hỏi phụ chung chung và LƯỢC BỎ tên sản phẩm cụ thể ra khỏi câu hỏi phụ đó. Ví dụ: chuyển "iPhone 16 bảo hành thế nào" thành "Chính sách bảo hành điện thoại tại CellphoneS" để hệ thống định tuyến đúng đến tài liệu chính sách chung của cửa hàng.
    - Mỗi sub-query phải là một câu hỏi độc lập, rõ nghĩa và đã được chuẩn hóa.

    Ví dụ 1 (Câu hỏi phức tạp, viết tắt, không dấu):
    Câu hỏi: "so sanh ip 13 pro vs iphon 14 pro ve gia va pin"
    Trả về định dạng JSON:
    {{
      "cleaned_query": "So sánh iPhone 13 Pro và iPhone 14 Pro về giá bán và dung lượng pin",
      "need_decomposition": true,
      "sub_queries": [
        "iPhone 13 Pro giá bao nhiêu",
        "iPhone 13 Pro dung lượng pin thế nào",
        "iPhone 14 Pro giá bao nhiêu",
        "iPhone 14 Pro dung lượng pin thế nào"
      ]
    }}

    Ví dụ 2 (Câu hỏi đơn giản vừa hỏi sản phẩm vừa hỏi chính sách):
    Câu hỏi: "ip16 promax 128gb gia bao nhieu và co bh ko"
    Trả về định dạng JSON:
    {{
      "cleaned_query": "iPhone 16 Pro Max 128GB giá bao nhiêu và có bảo hành không",
      "need_decomposition": true,
      "sub_queries": [
        "iPhone 16 Pro Max 128GB giá bao nhiêu",
        "Chính sách bảo hành điện thoại tại CellphoneS"
      ]
    }}

    Ví dụ 3 (Câu hỏi chính sách đơn giản):
    Câu hỏi: "quy dinh doi tra tai cellphones"
    Trả về định dạng JSON:
    {{
      "cleaned_query": "Quy định đổi trả tại CellphoneS",
      "need_decomposition": false,
      "sub_queries": [
        "Quy định đổi trả tại CellphoneS"
      ]
    }}

    Câu hỏi của người dùng: "{query_text}"
    Hãy trả về duy nhất một đối tượng JSON hợp lệ theo đúng cấu trúc trên. Không thêm bất kỳ văn bản giải thích nào ngoài JSON.
    """
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Cảnh báo: GROQ_API_KEY chưa được thiết lập. Sử dụng câu hỏi gốc không chuẩn hóa.")
        return {
            "cleaned_query": query_text,
            "need_decomposition": False,
            "sub_queries": [query_text]
        }
        
    max_retries = 3
    backoff_factor = 2.0
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12
            )
            response.raise_for_status()
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
            return json.loads(content)
            
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response is not None else 500
            if status_code == 429 and attempt < max_retries - 1:
                # Tính toán thời gian đợi tăng dần (exponential backoff)
                sleep_time = (backoff_factor ** (attempt + 1)) + 1.0
                print(f"   [Cảnh báo] Groq Rate Limit (429). Đang chờ {sleep_time:.1f}s để thử lại... (Lần {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"Lỗi HTTP Groq API: {http_err}")
                break
        except Exception as e:
            print(f"Lỗi kết nối Groq API: {e}")
            break
            
    # Fallback khi lỗi: Giữ nguyên câu hỏi của người dùng
    return {
        "cleaned_query": query_text,
        "need_decomposition": False,
        "sub_queries": [query_text]
    }


# ==========================================
# 2. XÂY DỰNG BẢN ĐỒ SẢN PHẨM & MAPPING
# ==========================================
#Đầu tiên sẽ tách tên sản phẩm để có thể tìm kiếm nhiều sản phẩm hơn
def clean_model_name(name):
    """
    Rút gọn tên sản phẩm về dòng máy gốc (Base Name).
    Ví dụ: 'Điện thoại iPhone 14 Pro Max 256GB | Chính hãng VN/A' -> 'iPhone 14 Pro Max'
    """
    name = name.replace("Điện thoại ", "")
    # Tách phần chứa dung lượng (128GB, 256GB...) ra
    parts = re.split(r'\s+\d+(?:GB|TB|gb|tb)', name, flags=re.IGNORECASE)
    cleaned = parts[0]
    # Bỏ phần VN/A hoặc ký hiệu đặc biệt phía sau
    cleaned = re.split(r'\s*(?:\||I)\s*', cleaned)[0]
    return cleaned.strip()

#Xây dựng cái hàm này để mapping tất cả các product_id của biến thể liên quan
#Ví dụ iPhone 14 Pro Max -> [iPhone 14 Pro Max 128GB, iPhone 14 Pro Max 256GB, iPhone 14 Pro Max 512GB]
def build_model_mappings():
    """
    Tải thông tin từ ChromaDB để tạo ánh xạ từ tên dòng máy gốc (base name)
    sang tất cả các product_id của biến thể liên quan.
    """
    try:
        collection = chroma_client.get_collection(name="product_collection")
        all_data = collection.get()
        metadatas = all_data.get('metadatas', [])
    except Exception as e:
        print(f"Cảnh báo: Không thể tải collection product_collection để tạo ánh xạ: {e}")
        return {}, [], {}
        
    base_to_pids = {}
    pid_to_name = {}
    
    for meta in metadatas:
        pid = meta.get("product_id")
        pname = meta.get("product_name")
        if pid and pname:
            pid_to_name[pid] = pname
            base_name = clean_model_name(pname)
            if base_name not in base_to_pids:
                base_to_pids[base_name] = set()
            base_to_pids[base_name].add(pid)
            
    # Sắp xếp base name giảm dần theo chiều dài để tìm kiếm khớp chính xác nhất
    sorted_base_names = sorted(base_to_pids.keys(), key=len, reverse=True)
    return base_to_pids, sorted_base_names, pid_to_name

print("Đang xây dựng ánh xạ sản phẩm...")
base_to_pids, sorted_base_names, pid_to_name = build_model_mappings()


# ==========================================
# 3. PHÂN LOẠI & TRÍCH XUẤT THÔNG TIN (ROUTING)
# ==========================================

def classify_query(query_text):
    """
    Định tuyến câu hỏi tới collection phù hợp (Chính sách cửa hàng vs Sản phẩm).
    Chiến lược lọc mềm (Soft Filtering):
    - Nếu là câu hỏi chính sách chung (không có tên sản phẩm cụ thể) -> policy_collection.
    - Các trường hợp còn lại -> product_collection (quét toàn bộ, không lọc cứng theo loại tài liệu).
    """
    query_lower = query_text.lower()
    policy_keywords = [
        "chính sách", "điều khoản", "quy định", "bảo hành", "đổi trả", 
        "hoàn tiền", "trả góp", "lỗi"
    ]

    # Kiểm tra xem có chứa tên sản phẩm nào không
    has_product = False
    for base_name in sorted_base_names:
        if base_name.lower() in query_lower:
            has_product = True
            break

    is_policy = any(keyword in query_lower for keyword in policy_keywords)
    if is_policy and not has_product:
        return "policy_collection", {"type": "policy"}

    return "product_collection", None

def extract_product_ids_from_query(query_text):
    """
    Phát hiện và trích xuất các product_id liên quan đến tên dòng sản phẩm trong câu hỏi.
    Có hỗ trợ nhận diện câu hỏi dạng dòng/series (ví dụ: 'iphone 16 series').
    """
    query_lower = query_text.lower()
    matched_pids = set()
    matched_name = None
    
    is_series = any(w in query_lower for w in ["series", "seris", "dòng"])
    
    # Duyệt từ tên dài nhất đến ngắn nhất để tránh trùng lặp đè (như iPhone 14 Pro Max khớp sang iPhone 14)
    for base_name in sorted_base_names:
        name_lower = base_name.lower()
        if name_lower in query_lower:
            matched_name = base_name
            if is_series:
                # Nếu hỏi dòng máy (series), lấy toàn bộ sản phẩm thuộc dòng đó
                for other_base in sorted_base_names:
                    if base_name.lower() in other_base.lower():
                        matched_pids.update(base_to_pids[other_base])
            else:
                matched_pids.update(base_to_pids[base_name])
            break
            
    return list(matched_pids), matched_name


# ==========================================
# 4. TÌM KIẾM HYBRID SEARCH (VECTOR + BM25 + RRF)
# ==========================================

def search_chroma(query_text, collection_name, model, n_results=5, metadata_filter=None):
    """Tìm kiếm bằng Vector Similarity trên ChromaDB"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy collection {collection_name}.")
        return []
        
    query_vector = model.encode(query_text).tolist()
    
    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter
        
    results = collection.query(**query_kwargs)
    
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

def search_bm25_with_chroma(query_text, collection_name, n_results=5, metadata_filter=None):
    """Tìm kiếm từ khóa bằng thuật toán BM25 trên tập dữ liệu đã qua bộ lọc metadata"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy collection {collection_name}")
        return []
        
    # Lấy toàn bộ dữ liệu thô thỏa mãn bộ lọc metadata để xây dựng BM25 Index cục bộ
    get_kwargs = {}
    if metadata_filter:
        get_kwargs["where"] = metadata_filter
    all_data = collection.get(**get_kwargs)
    
    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    ids = all_data.get('ids', [])
    
    if not documents:
        return []

    # Chuẩn hóa tiếng Việt văn bản gốc & câu truy vấn
    documents_normalized = [text_normalize(doc) for doc in documents]
    query_normalized = text_normalize(query_text)

    # Tiến hành tách từ (tokenization)
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents_normalized]
    tokenized_query = word_tokenize(query_normalized.lower())

    # Tính toán BM25
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1]
    
    formatted_results = []
    for idx in top_indices:
        score = scores[idx]
        # Chỉ lấy các tài liệu có từ khóa khớp thực sự (score > 0)
        if score > 0:
            formatted_results.append({
                "text": documents[idx],
                "metadata": metadatas[idx],
                "bm25_score": float(score),
                "id": ids[idx]
            })
            if len(formatted_results) >= n_results:
                break
                
    return formatted_results

def hybrid_search(query_text, collection_name, model, n_results=5, metadata_filter=None, k=60):
    """
    Tìm kiếm kết hợp Vector Search & BM25.
    Sử dụng thuật toán RRF (Reciprocal Rank Fusion) để gộp và sắp xếp lại kết quả.
    """
    # Lấy số lượng ứng viên rộng gấp đôi (n_results * 2) từ mỗi nhánh để tăng cơ hội kết hợp
    vector_results = search_chroma(query_text, collection_name, model, n_results=n_results * 2, metadata_filter=metadata_filter)
    bm25_results = search_bm25_with_chroma(query_text, collection_name, n_results=n_results * 2, metadata_filter=metadata_filter)

    rrf_scores = {}
    doc_map = {}

    # Tính điểm RRF cho nhánh Vector Search
    for rank, doc in enumerate(vector_results):
        doc_id = doc.get("id")
        if not doc_id:
            continue
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        doc_map[doc_id] = doc

    # Tính điểm RRF cho nhánh BM25 Search
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get("id")
        if not doc_id:
            continue
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sắp xếp theo điểm RRF giảm dần
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = sorted_docs[:n_results]
    
    hybrid_results = []
    for doc_id, score in top_docs:
        final_doc = doc_map[doc_id].copy()
        final_doc["rrf_score"] = score
        hybrid_results.append({
            "text": final_doc["text"],
            "metadata": final_doc["metadata"],
            "rrf_score": score,
            "id": doc_id
        })
        
    return hybrid_results


# ==========================================
# 5. ĐIỀU PHỐI VÀ CHẤM ĐIỂM (RERANKING & RETRIEVAL)
# ==========================================

def retrieve_and_rerank(query_text, n_results=5):
    """
    Luồng xử lý RAG hoàn chỉnh:
    1. Chuẩn hóa & Phân rã câu hỏi (LLM)
    2. Trích xuất sản phẩm & Lọc cứng (Soft Filter)
    3. Tìm kiếm lai Hybrid Search trên từng sub-query
    4. Cân bằng Pool tài liệu (Capping description)
    5. Đánh giá lại độ liên quan ngữ nghĩa bằng PhoRanker Reranker
    """
    # Bước 1: Chuẩn hóa & Phân rã câu hỏi
    processed = llm_process_query(query_text)
    cleaned_query = processed.get("cleaned_query", query_text)
    sub_queries = processed.get("sub_queries", [query_text])
    need_decomp = processed.get("need_decomposition", False)
    
    if cleaned_query.lower() != query_text.lower():
        print(f"   [Spell Correction] '{query_text}' -> '{cleaned_query}'")
        
    if need_decomp:
        print(f"   [Router] Phức tạp -> Phân rã thành: {sub_queries}")
    else:
        print(f"   [Router] Đơn giản -> Không phân rã")
        
    all_candidates = []
    retrieved_by = {}
    
    # Bước 2: Truy vấn tài liệu cho từng sub-query
    for sq in sub_queries:
        collection_name, base_filter = classify_query(sq)
        
        # Thiết lập bộ lọc mềm theo sản phẩm
        metadata_filter = base_filter
        extracted_name = None
        
        if collection_name == "product_collection":
            product_ids, extracted_name = extract_product_ids_from_query(sq)
            if product_ids:
                if base_filter:
                    metadata_filter = {
                        "$and": [
                            base_filter,
                            {"product_id": {"$in": product_ids}}
                        ]
                    }
                else:
                    metadata_filter = {"product_id": {"$in": product_ids}}
                print(f"   [Filter cứng] Áp bộ lọc cho sản phẩm: {extracted_name} ({len(product_ids)} PIDs)")
            else:
                print(f"   [Filter cứng] Không trích xuất được sản phẩm cho: '{sq}'")
        else:
            print(f"   [Filter] Collection chính sách: {base_filter}")
            
        # Tìm kiếm 20 ứng viên tốt nhất cho từng câu hỏi phụ
        candidates = hybrid_search(sq, collection_name, embed_model, n_results=20, metadata_filter=metadata_filter)
        
        for cand in candidates:
            cid = cand.get("id") or cand.get("metadata", {}).get("chunk_id")
            if not cid:
                import hashlib
                cid = hashlib.md5(cand["text"].encode('utf-8')).hexdigest()
                cand["id"] = cid
            
            if cid not in retrieved_by:
                retrieved_by[cid] = []
                all_candidates.append(cand)
            retrieved_by[cid].append(sq)
            
    print(f"   [Merge] Gộp lại còn {len(all_candidates)} ứng viên.")
    if not all_candidates:
        return []

    # Bước 3: Cân bằng Pool ứng viên (Capping description)
    # Giới hạn tối đa 4 chunk có type là description để đảm bảo specs, faq, variants không bị lấn át.
    MAX_DESCRIPTION = 4
    balanced_candidates = []
    desc_count = 0
    for cand in all_candidates:
        ctype = cand.get("metadata", {}).get("type", "")
        if ctype == "description":
            if desc_count < MAX_DESCRIPTION:
                balanced_candidates.append(cand)
                desc_count += 1
        else:
            balanced_candidates.append(cand)
    all_candidates = balanced_candidates
    print(f"   [Balance] Sau cân bằng type: {len(all_candidates)} ứng viên (desc capped={desc_count})")

    # Bước 4: Sắp xếp và đánh giá độ liên quan bằng PhoRanker
    pairs = []
    pair_mapping = []
    for cand in all_candidates:
        cid = cand["id"]
        for sq in retrieved_by[cid]:
            pairs.append([sq, cand["text"]])
            pair_mapping.append((cand, sq))
            
    scores = reranker_model.predict(pairs)
    
    # Gom điểm theo ID chunk (lấy điểm lớn nhất nếu một chunk khớp với nhiều sub-queries)
    cand_scores = {}
    for score, (cand, sq) in zip(scores, pair_mapping):
        cid = cand["id"]
        cand_scores[cid] = max(cand_scores.get(cid, -9999.0), float(score))
        
    for cand in all_candidates:
        cand["rerank_score"] = cand_scores[cand["id"]]
        
    # Trả về kết quả đã xếp hạng
    ranked_results = sorted(all_candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked_results[:n_results]


# ==========================================
# 6. HÀM CHẠY KIỂM THỬ THỦ CÔNG (MAIN)
# ==========================================

def main():
    queries = [
        "Cấu hình chi tiết camera và chip xử lý của iPhone 16 Pro 128gb là gì?",
        "chính sách bảo hành của iphoen 16 pro max có được đổi trả kh?",
        "iphone 16 series bao nhiêu tiền?",
        "so sanh ip 13 pro vs iphon 14 pro ve gia va pin",
        "ip 16 pr max 128gb gia bao nhieu và co bh ko",
        "quy dinh doi tra tai cellphones"
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
