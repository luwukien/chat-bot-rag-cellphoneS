import os
import re
import json
import sys
import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize, text_normalize
from dotenv import load_dotenv
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

import transformers
transformers.logging.set_verbosity_error()

# Tắt cảnh báo trùng lặp token của transformers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# Cấu hình đường dẫn
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
RERANKER_MODEL_NAME = "itdainb/PhoRanker"


# Tải biến môi trường từ file .env trong thư mục làm việc hiện tại
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))
# Cấu hình UTF-8 cho Windows Terminal để không bị lỗi mã hóa chữ tiếng Việt
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

print("Đang khởi tạo Groq API và nạp các mô hình cục bộ...")
#Chỉ sử dụng tới nhánh Encoder
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
    #Khi gọi model.encode() thì nó sẽ thực hiện các công việc sau đây:
    #Tokenizer câu hỏi
    #Padding/Truncation câu hỏi -> Căn chỉnh kích thước
    #Chuyển thành tensor
    #Transformer Encoder xử lý câu hỏi và tài liệu
    #Content-aware Tokens Embeddings
    #Poolings
    #Normalize
    #Sentence Embeddings


    # Truy vấn bằng vector
    # Lưu ý: ChromaDB không chấp nhận where={} (dict rỗng), chỉ truyền khi filter có giá trị
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

def classify_query(query_text):
    """
    Xác định collection và bộ lọc metadata phù hợp cho câu hỏi.
    
    Chiến lược (Soft Filter):
    - Nếu câu hỏi rõ ràng là chính sách cửa hàng chung (không nhắc tên sản phẩm cụ thể)
      -> Định tuyến tới policy_collection.
    - Còn lại -> Tìm trong product_collection KHÔNG lọc theo type,
      chỉ lọc theo product_id (nếu trích xuất được).
      PhoRanker Cross-Encoder sẽ tự quyết định chunk nào phù hợp nhất.
    """
    query_lower = query_text.lower()
    
    policy_keywords = [
        "chính sách", "điều khoản", "quy định", "bảo hành", "đổi trả", 
        "hoàn tiền", "trả góp", "lỗi"
    ]

    # Kiểm tra câu hỏi có chứa thực thể sản phẩm nào không
    has_product = False
    for base_name in sorted_base_names:
        if base_name.lower() in query_lower:
            has_product = True
            break

    # Chỉ định tuyến sang policy_collection nếu là câu hỏi chính sách CHUNG
    # (không đề cập tên sản phẩm cụ thể)
    is_policy = any(keyword in query_lower for keyword in policy_keywords)
    if is_policy and not has_product:
        return "policy_collection", {"type": "policy"}

    # Mặc định: Tìm trong product_collection, KHÔNG lọc theo type
    # Để PhoRanker Cross-Encoder tự quyết định chunk nào phù hợp nhất
    return "product_collection", None


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
    # Lưu ý: ChromaDB không chấp nhận where={} (dict rỗng), chỉ truyền khi filter có giá trị
    get_kwargs = {}
    if metadata_filter:
        get_kwargs["where"] = metadata_filter
    all_data = collection.get(**get_kwargs)
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
    
    #Nếu lọc như này thì mất đi bao nhiêu thông tin có ích rồi
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

def llm_process_query(query_text):
    """
    Sử dụng LLM (Groq Llama-3.3) để:
    1. Sửa lỗi chính tả tiếng Việt, khôi phục dấu và viết tắt chuẩn (ip -> iPhone, pm -> Pro Max, bh -> bảo hành...)
    2. Phân rã câu hỏi phức tạp thành danh sách các sub-queries đơn giản (đã sửa lỗi)
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
    - Lưu ý đặc biệt về ngữ cảnh công nghệ khi khôi phục dấu: "sac" trong ngữ cảnh điện thoại luôn là "ạc" (đục), KHÔNG phải "ắc" (màu). Ví dụ: "pin va sac" -> "pin và sạc"; "sac nhanh" -> "sạc nhanh"; "cong sac" -> "cổng sạc"; "cap sac" -> "cáp sạc".
    
    Quy tắc phân rã (decomposition):
    - Phân rã nếu câu hỏi hỏi về từ 2 sản phẩm trở lên (so sánh, đối chiếu) HOẶC hỏi đồng thời cả giá bán (variants) VÀ cấu hình/pin/camera (specs) của một sản phẩm.
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

    Ví dụ 2 (Câu hỏi đơn giản, viết tắt, sai chính tả):
    Câu hỏi: "ip16 promax 128gb gia bao nhieu và co bh ko"
    Trả về định dạng JSON:
    {{
      "cleaned_query": "iPhone 16 Pro Max 128GB giá bao nhiêu và có bảo hành không",
      "need_decomposition": true,
      "sub_queries": [
        "iPhone 16 Pro Max 128GB giá bao nhiêu",
        "iPhone 16 Pro Max 128GB chính sách bảo hành"
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
        
    import requests
    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        res_data = response.json()
        content = res_data["choices"][0]["message"]["content"]
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Lỗi chuẩn hóa truy vấn bằng Groq: {e}")
        return {
            "cleaned_query": query_text,
            "need_decomposition": False,
            "sub_queries": [query_text]
        }

def check_need_decomposition(query_text):
    query_lower = query_text.lower()
    
    # 1. Kiểm tra từ khóa so sánh
    compare_keywords = ["so sánh", "khác gì", "vs", "khác nhau", "so với", "sánh với"]
    if any(kw in query_lower for kw in compare_keywords):
        return True
        
    # 2. Kiểm tra xem có chứa từ 2 dòng sản phẩm trở lên không
    detected_products = []
    temp_query = query_lower
    for base_name in sorted_base_names:
        name_lower = base_name.lower()
        if name_lower in temp_query:
            detected_products.append(base_name)
            temp_query = temp_query.replace(name_lower, "")
            
    if len(set(detected_products)) >= 2:
        return True
        
    return False

def extract_product_ids_from_query(query_text):
    query_lower = query_text.lower()
    matched_pids = set()
    matched_name = None
    
    # Kiểm tra xem câu hỏi có đề cập đến dòng sản phẩm/series không
    is_series = any(w in query_lower for w in ["series", "seris", "dòng"])
    
    # Duyệt từ dài đến ngắn để tìm sản phẩm khớp chính xác nhất
    for base_name in sorted_base_names:
        name_lower = base_name.lower()
        if name_lower in query_lower:
            matched_name = base_name
            if is_series:
                # Nếu hỏi dòng/series, lấy tất cả sản phẩm chứa/bắt đầu bằng tên dòng máy này
                for other_base in sorted_base_names:
                    if base_name.lower() in other_base.lower():
                        matched_pids.update(base_to_pids[other_base])
            else:
                matched_pids.update(base_to_pids[base_name])
            break
            
    return list(matched_pids), matched_name

def retrieve_and_rerank(query_text, n_results=5):
    # 1. Gọi LLM để chuẩn hóa câu hỏi và phân rã nếu cần thiết
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
    
    # 2. Tìm kiếm ứng viên cho từng sub-query
    for sq in sub_queries:
        collection_name, base_filter = classify_query(sq)
        
        # Áp dụng Metadata Filtering (Lọc cứng) nếu là collection sản phẩm
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

    # Cân bằng pool ứng viên: giới hạn tối đa 4 chunk type=description để
    # tránh sản phẩm nhiều mô tả (30+ chunks) lấn át specs/faq/variants.
    # Giữ toàn bộ specs, variants, faq; chỉ cap description.
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

    # 3. Chuẩn bị các cặp để PhoRanker chấm điểm
    pairs = []
    pair_mapping = []
    for cand in all_candidates:
        cid = cand["id"]
        for sq in retrieved_by[cid]:
            pairs.append([sq, cand["text"]])
            pair_mapping.append((cand, sq))
            
    scores = reranker_model.predict(pairs)
    
    cand_scores = {}
    for score, (cand, sq) in zip(scores, pair_mapping):
        cid = cand["id"]
        cand_scores[cid] = max(cand_scores.get(cid, -9999.0), float(score))
        
    for cand in all_candidates:
        cand["rerank_score"] = cand_scores[cand["id"]]
        
    ranked_results = sorted(all_candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked_results[:n_results]

def main():
    queries = [
        "Cấu hình chi tiết camera và chip xử lý của iPhone 16 Pro 128gb là gì?",
        "iPhone 13 Pro max giá bao nhiêu và có những màu gì?",
        "So sánh iPhone 13 Pro và iPhone 14 Pro về giá và pin",
        "Chính sách bảo hành đổi trả của CellphoneS",
        "iphone 16 plus màu nào đẹp nhất?",
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



