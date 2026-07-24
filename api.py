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
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize, text_normalize
from dotenv import load_dotenv

# Tắt cảnh báo không cần thiết từ transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

# Nạp biến môi trường
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Cấu hình đường dẫn & Model
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
RERANKER_MODEL_NAME = "itdainb/PhoRanker"

# Các biến toàn cục lưu trữ model và database index
embed_model = None
reranker_model = None
chroma_client = None
base_to_pids = {}
sorted_base_names = []
pid_to_name = {}

# ==========================================
# 1. KHỞI TẠO BẢN ĐỒ SẢN PHẨM & MODELS
# ==========================================

def clean_model_name(name: str) -> str:
    """Rút gọn tên sản phẩm về dòng máy gốc (Base Name)."""
    name = name.replace("Điện thoại ", "")
    parts = re.split(r'\s+\d+(?:GB|TB|gb|tb)', name, flags=re.IGNORECASE)
    cleaned = parts[0]
    cleaned = re.split(r'\s*(?:\||I)\s*', cleaned)[0]
    return cleaned.strip()

def build_model_mappings():
    """Tạo ánh xạ từ tên dòng máy gốc sang danh sách product_id biến thể."""
    global base_to_pids, sorted_base_names, pid_to_name
    try:
        collection = chroma_client.get_collection(name="product_collection")
        all_data = collection.get()
        metadatas = all_data.get('metadatas', [])
    except Exception as e:
        logging.warning(f"Không thể nạp product_collection để tạo mapping: {e}")
        return

    b2p = {}
    p2n = {}
    for meta in metadatas:
        pid = meta.get("product_id")
        pname = meta.get("product_name")
        if pid and pname:
            p2n[pid] = pname
            base_name = clean_model_name(pname)
            if base_name not in b2p:
                b2p[base_name] = set()
            b2p[base_name].add(pid)

    base_to_pids = b2p
    pid_to_name = p2n
    sorted_base_names = sorted(base_to_pids.keys(), key=len, reverse=True)
    logging.info(f"Đã tạo ánh xạ cho {len(base_to_pids)} dòng sản phẩm.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, reranker_model, chroma_client
    logging.info("Đang khởi tạo các mô hình RAG & kết nối ChromaDB...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    build_model_mappings()
    logging.info("Khởi tạo hệ thống thành công!")
    yield

app = FastAPI(
    title="CellphoneS RAG Chatbot API",
    description="RESTful API Backend cho hệ thống tư vấn sản phẩm CellphoneS",
    version="1.0.0",
    lifespan=lifespan
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. XỬ LÝ TRUY VẤN VÀ TÌM KIẾM HYBRID
# ==========================================

def llm_process_query(query_text: str) -> dict:
    """Gọi Groq Llama-3.3 để sửa lỗi chính tả và phân rã câu hỏi."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return {
            "cleaned_query": query_text,
            "need_decomposition": False,
            "sub_queries": [query_text]
        }

    prompt = f"""
    Bạn là một trợ lý RAG Planner thông minh và chuyên gia chuẩn hóa tiếng Việt cho chatbot của cửa hàng công nghệ CellphoneS.
    
    Nhiệm vụ của bạn là nhận vào câu hỏi của người dùng và trả về một đối tượng JSON có cấu trúc như sau:
    {{
      "cleaned_query": "Câu hỏi gốc đã được sửa hết lỗi chính tả, viết tắt, và khôi phục dấu tiếng Việt chuẩn",
      "need_decomposition": true hoặc false,
      "sub_queries": [
         "Danh sách các câu hỏi phụ đơn giản đã được sửa lỗi chính tả."
      ]
    }}
    
    Quy tắc chuẩn hóa:
    - Sửa các từ viết tắt: "ip", "iphon" -> "iPhone"; "pm", "promax" -> "Pro Max"; "đt" -> "điện thoại"; "bh" -> "bảo hành"; "sac" -> "sạc".
    - Khôi phục dấu tiếng Việt đầy đủ và tự nhiên.
    - Giữ nguyên thông số kỹ thuật (128GB, 256GB, 5G).
    
    Quy tắc phân rã:
    - Phân rã nếu hỏi về từ 2 sản phẩm trở lên HOẶC hỏi vừa giá vừa cấu hình.
    - Nếu có từ khóa chính sách chung (bảo hành, đổi trả...), phân rã thành câu hỏi chính sách chung (lược bỏ tên sản phẩm cụ thể khỏi câu hỏi chính sách đó).

    Câu hỏi của người dùng: "{query_text}"
    Hãy trả về duy nhất một đối tượng JSON hợp lệ theo đúng cấu trúc trên.Không giải thích ngoài JSON.
    """

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    max_retries = 3
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
            if http_err.response is not None and http_err.response.status_code == 429 and attempt < max_retries - 1:
                time.sleep((2.0 ** (attempt + 1)) + 1.0)
            else:
                break
        except Exception:
            break

    return {
        "cleaned_query": query_text,
        "need_decomposition": False,
        "sub_queries": [query_text]
    }


def classify_query(query_text: str):
    query_lower = query_text.lower()
    policy_keywords = [
        "chính sách", "điều khoản", "quy định", "bảo hành", "đổi trả", 
        "hoàn tiền", "trả góp", "lỗi"
    ]
    has_product = False
    for base_name in sorted_base_names:
        if base_name.lower() in query_lower:
            has_product = True
            break

    is_policy = any(k in query_lower for k in policy_keywords)
    if is_policy and not has_product:
        return "policy_collection", {"type": "policy"}

    return "product_collection", None


def extract_product_ids_from_query(query_text: str):
    query_lower = query_text.lower()
    matched_pids = set()
    matched_name = None
    is_series = any(w in query_lower for w in ["series", "seris", "dòng"])

    for base_name in sorted_base_names:
        if base_name.lower() in query_lower:
            matched_name = base_name
            if is_series:
                for other_base in sorted_base_names:
                    if base_name.lower() in other_base.lower():
                        matched_pids.update(base_to_pids[other_base])
            else:
                matched_pids.update(base_to_pids[base_name])
            break

    return list(matched_pids), matched_name


def search_chroma(query_text, collection_name, n_results=5, metadata_filter=None):
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return []

    query_vector = embed_model.encode(query_text).tolist()
    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    results = collection.query(**query_kwargs)
    formatted = []
    if results and 'documents' in results and results['documents']:
        for doc, meta, dist, doc_id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
            formatted.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "id": doc_id
            })
    return formatted


def search_bm25_with_chroma(query_text, collection_name, n_results=5, metadata_filter=None):
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return []

    get_kwargs = {}
    if metadata_filter:
        get_kwargs["where"] = metadata_filter
    all_data = collection.get(**get_kwargs)

    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    ids = all_data.get('ids', [])

    if not documents:
        return []

    documents_normalized = [text_normalize(doc) for doc in documents]
    query_normalized = text_normalize(query_text)

    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents_normalized]
    tokenized_query = word_tokenize(query_normalized.lower())

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1]

    formatted = []
    for idx in top_indices:
        score = scores[idx]
        if score > 0:
            formatted.append({
                "text": documents[idx],
                "metadata": metadatas[idx],
                "bm25_score": float(score),
                "id": ids[idx]
            })
            if len(formatted) >= n_results:
                break
    return formatted


def hybrid_search(query_text, collection_name, n_results=5, metadata_filter=None, k=60):
    vector_results = search_chroma(query_text, collection_name, n_results=n_results * 2, metadata_filter=metadata_filter)
    bm25_results = search_bm25_with_chroma(query_text, collection_name, n_results=n_results * 2, metadata_filter=metadata_filter)

    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.get("id")
        if not doc_id:
            continue
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get("id")
        if not doc_id:
            continue
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]

    hybrid_results = []
    for doc_id, score in sorted_docs:
        final_doc = doc_map[doc_id].copy()
        final_doc["rrf_score"] = score
        hybrid_results.append({
            "text": final_doc["text"],
            "metadata": final_doc["metadata"],
            "rrf_score": score,
            "id": doc_id
        })
    return hybrid_results


def retrieve_and_rerank(query_text: str, n_results: int = 5):
    processed = llm_process_query(query_text)
    cleaned_query = processed.get("cleaned_query", query_text)
    sub_queries = processed.get("sub_queries", [query_text])

    all_candidates = []
    retrieved_by = {}

    for sq in sub_queries:
        collection_name, base_filter = classify_query(sq)
        metadata_filter = base_filter

        if collection_name == "product_collection":
            product_ids, _ = extract_product_ids_from_query(sq)
            if product_ids:
                if base_filter:
                    metadata_filter = {"$and": [base_filter, {"product_id": {"$in": product_ids}}]}
                else:
                    metadata_filter = {"product_id": {"$in": product_ids}}

        candidates = hybrid_search(sq, collection_name, n_results=20, metadata_filter=metadata_filter)
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

    if not all_candidates:
        return [], cleaned_query, sub_queries

    # Capping description
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

    # Reranking bằng PhoRanker
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
    return ranked_results[:n_results], cleaned_query, sub_queries


# ==========================================
# 3. LLM ANSWER GENERATION
# ==========================================

def generate_answer(query: str, contexts: List[dict]) -> str:
    """
    Sử dụng Groq Llama-3.3-70b để tổng hợp context và sinh câu trả lời tự nhiên.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return "Xin lỗi, hiện tại hệ thống chưa được cấu hình API Key để trả lời tự động."

    context_text_blocks = []
    for idx, ctx in enumerate(contexts):
        mtype = ctx["metadata"].get("type", "Thông tin")
        pname = ctx["metadata"].get("product_name", ctx["metadata"].get("section_title", ""))
        block = f"[Tài liệu {idx+1} | {mtype} - {pname}]\n{ctx['text']}"
        context_text_blocks.append(block)

    context_str = "\n\n".join(context_text_blocks)

    system_prompt = """
Bạn là Trợ lý AI chính thức của cửa hàng bán lẻ công nghệ CellphoneS tại Việt Nam.
Nhiệm vụ của bạn là tư vấn thông tin sản phẩm (giá bán, thông số kỹ thuật, màu sắc, ưu đãi) và chính sách mua hàng (bảo hành, đổi trả, bảo vệ thiết bị) một cách lịch sự, chính xác và chuyên nghiệp.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin được cung cấp trong danh sách [Tài liệu ngữ cảnh] bên dưới.
2. KHÔNG tự bịa đặt giá bán, quà tặng, hay thông số không có trong ngữ cảnh.
3. Trình bày rõ ràng, dễ đọc (sử dụng gạch đầu dòng, in đậm tên sản phẩm, giá tiền).
4. Nếu thông tin không có trong tài liệu ngữ cảnh, hãy trả lời lịch sự rằng hiện tại chưa có thông tin chi tiết và khuyên khách hàng liên hệ tổng đài 1800.2097 hoặc tới cửa hàng CellphoneS gần nhất.
5. Luôn giữ thái độ thân thiện, lịch sự chuẩn tư vấn viên CellphoneS.
"""

    user_prompt = f"""
[Tài liệu ngữ cảnh]
{context_str if context_str else "Không tìm thấy tài liệu phù hợp."}

[Câu hỏi của khách hàng]
{query}

Hãy trả lời câu hỏi của khách hàng một cách chi tiết và chính xác dựa trên tài liệu ngữ cảnh trên.
"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )
            response.raise_for_status()
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                break
        except Exception as e:
            logging.error(f"Lỗi khi gọi Groq generation: {e}")
            break

    return "Xin lỗi, hệ thống tư vấn đang quá tải trong giây lát. Bạn vui lòng thử lại sau ít phút!"


# ==========================================
# 4. RESTFUL API ENDPOINTS & SCHEMAS
# ==========================================

class ChatRequest(BaseModel):
    message: str

class SourceItem(BaseModel):
    id: str
    text: str
    type: str
    product_name: Optional[str] = None
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    cleaned_query: str
    sub_queries: List[str]


@app.get("/api/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống và số lượng vector trong ChromaDB."""
    try:
        policy_col = chroma_client.get_collection("policy_collection")
        policy_count = policy_col.count()
    except Exception:
        policy_count = 0

    try:
        product_col = chroma_client.get_collection("product_collection")
        product_count = product_col.count()
    except Exception:
        product_count = 0

    return {
        "status": "healthy",
        "chroma_db": {
            "status": "connected",
            "policy_chunks": policy_count,
            "product_chunks": product_count
        },
        "models_loaded": {
            "embedding": EMBEDDING_MODEL_NAME,
            "reranker": RERANKER_MODEL_NAME
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint xử lý tin nhắn chat từ người dùng."""
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Nội dung tin nhắn không được để trống.")

    # Luồng RAG
    results, cleaned_query, sub_queries = retrieve_and_rerank(user_msg, n_results=5)

    # Sinh câu trả lời
    answer = generate_answer(cleaned_query, results)

    # Format nguồn tham khảo
    sources = []
    for item in results:
        sources.append(SourceItem(
            id=item.get("id", ""),
            text=item.get("text", "")[:200] + "...",
            type=item.get("metadata", {}).get("type", "N/A"),
            product_name=item.get("metadata", {}).get("product_name") or item.get("metadata", {}).get("section_title"),
            score=round(float(item.get("rerank_score", 0.0)), 4)
        ))

    return ChatResponse(
        answer=answer,
        sources=sources,
        cleaned_query=cleaned_query,
        sub_queries=sub_queries
    )

# Serve file tĩnh (Frontend Web UI)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
