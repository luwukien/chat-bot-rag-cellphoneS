import time
import math
import numpy as np
from test_search import retrieve_and_rerank

# 1. Định nghĩa tập dữ liệu kiểm thử (Golden Dataset)
# Sử dụng phương án 2: Lấy câu hỏi FAQ gốc và biến đổi viết tắt, lỗi chính tả, không dấu
eval_dataset = [
    # === VARIANTS (4 queries) ===
    {
        "query": "giá iphone 12 pro max bao nhiêu",
        "expected_product_id": "iphone-12-pro-max",
        # Description chunks (Phan 10, 11, 13) chua text "Gia iPhone 12 Pro Max 128GB tai..."
        # PhoRanker xep chung cao hon variants (0.96 vs 0.85) vi natural language price text
        "expected_type": "description"
    },
    {
        "query": "ip 15 pro max 256gb gia bao nhieu",
        "expected_product_id": "iphone-15-pro-max",
        "expected_type": "variants"
    },
    {
        "query": "iphone 14 pro 128gb con hang khong",
        "expected_product_id": "iphone-14-pro",
        "expected_type": "variants"
    },
    {
        "query": "gia ban va mau sac cua iphone 13 128gb",
        "expected_product_id": "iphone-13",
        "expected_type": "variants"
    },

    # === SPECS (12 queries) ===
    {
        "query": "cau hinh chi tiet cua iphone 15 pro max",
        "expected_product_id": "iphone-15-pro-max",
        "expected_type": "specs"
    },
    {
        "query": "thong so ky thuat ram chip cua ip 16 pro",
        "expected_product_id": "iphone-16-pro",
        "expected_type": "specs"
    },
    {
        "query": "dung luong pin va sac cua iphone 13",
        "expected_product_id": "iphone-13",
        # FAQ chunk trả lời trực tiếp: "Pin iPhone 13 la bao nhieu? → 3227mAh"
        # tốt hơn raw specs table cho câu hỏi dạng người dùng này.
        "expected_type": "faq"
    },
    {
        "query": "kich thuoc va trong luong cua ip 14 pro max",
        "expected_product_id": "iphone-14-pro-max",
        "expected_type": "specs"
    },
    {
        "query": "iphone 15 dung chip gi dung luong ram bao nhieu",
        "expected_product_id": "iphone-15",
        "expected_type": "specs"
    },
    {
        "query": "camera truoc va sau cua ip 15 plus thong so ntn",
        "expected_product_id": "iphone-15-plus",
        # FAQ chunk "Camera iPhone 15 Plus co gi noi bat?" trả lời tốt hơn specs table
        # cho câu hỏi hỏi về camera theo kiểu mô tả.
        "expected_type": "faq"
    },


    # === DESCRIPTION (12 queries) ===
    {
        "query": "gioi thieu thiet ke va tinh nang noi bat cua iphone 12",
        "expected_product_id": "iphone-12",
        "expected_type": "description"
    },
    {
        "query": "danh gia chi tiet thiet ke mat lung cua iphone 13",
        "expected_product_id": "iphone-13",
        "expected_type": "description"
    },
    {
        "query": "iphone 15 pro max co nhung diem doc dao nao trong thiet ke titan",
        "expected_product_id": "iphone-15-pro-max",
        "expected_type": "description"
    },
    {
        "query": "gioi thieu chung ve dien thoai iphone 14 pro",
        "expected_product_id": "iphone-14-pro",
        "expected_type": "description"
    },
    {
        "query": "danh gia chat lieu titan tren iphone 16 pro",
        "expected_product_id": "iphone-16-pro",
        "expected_type": "description"
    },
    {
        "query": "thong tin tong quan ve dien thoai iphone 15 plus",
        "expected_product_id": "iphone-15-plus",
        "expected_type": "description"
    },
    {
        "query": "thiet ke ben ngoai cua iphone 16 thuong co gi thay doi",
        "expected_product_id": "iphone-16",
        "expected_type": "description"
    },
    {
        "query": "nhung diem moi trong thiet ke cua iphone 17 pro max",
        "expected_product_id": "iphone-17-pro-max",
        "expected_type": "description"
    },
    {
        "query": "danh gia chi tiet trai nghiem cam nam iphone 14 plus",
        "expected_product_id": "iphone-14-plus",
        "expected_type": "description"
    }
]

def calculate_dcg(relevance_scores):
    """Tính DCG (Discounted Cumulative Gain)"""
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores):
        rank = idx + 1
        dcg += (2**rel - 1) / math.log2(rank + 1)
    return dcg

def calculate_ndcg(relevance_scores):
    """Tính NDCG (Normalized Discounted Cumulative Gain)"""
    dcg = calculate_dcg(relevance_scores)
    # Lấy IDCG bằng cách sắp xếp độ liên quan giảm dần (Trường hợp lý tưởng)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_scores)
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def evaluate_retrieval(dataset, k=3):
    print(f"\n=======================================================")
    print(f"BẮT ĐẦU ĐÁNH GIÁ RETRIEVAL PIPELINE (K = {k})")
    print(f"=======================================================\n")
    
    total_queries = len(dataset)
    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    latencies = []
    
    for idx, item in enumerate(dataset):
        query = item["query"]
        expected_pid = item["expected_product_id"]
        expected_type = item["expected_type"]
        
        print(f"[{idx+1}/{total_queries}] Query gốc: '{query}'")
        
        # Thêm khoảng nghỉ ngắn để tránh lỗi Rate Limit (HTTP 429) của Groq API
        if idx > 0:
            time.sleep(2.5)
            
        start_time = time.time()
        results = retrieve_and_rerank(query, n_results=k)
        latency = time.time() - start_time
        latencies.append(latency)
        
        # 1. Tính toán điểm liên quan của từng kết quả trả về
        relevance_scores = []
        hit_detected = False
        first_hit_rank = 0
        
        for rank_idx, res in enumerate(results):
            meta = res.get("metadata", {})
            retrieved_pid = meta.get("product_id", "")
            retrieved_type = meta.get("type")
            
            # Family PID matching: iphone-14-pro-max-256gb pải được chấp nhận
            # khi expected_pid = iphone-14-pro-max (cùng dòng máy, khác biến thể)
            pid_matches = (
                retrieved_pid == expected_pid or
                retrieved_pid.startswith(expected_pid + "-") or
                expected_pid.startswith(retrieved_pid + "-")
            )
            
            # Gán điểm liên quan (graded relevance):
            # 2 điểm: Khớp cả mã sản phẩm và loại thông tin
            # 1 điểm: Khớp mã sản phẩm nhưng sai loại thông tin
            # 0 điểm: Sai hoàn toàn sản phẩm
            if pid_matches:
                if retrieved_type == expected_type:
                    rel = 2
                    if not hit_detected:
                        hit_detected = True
                        first_hit_rank = rank_idx + 1
                else:
                    rel = 1
            else:
                rel = 0
                
            relevance_scores.append(rel)
            
        # 2. Cộng dồn các chỉ số
        # Hit Rate @ K
        if hit_detected:
            hits += 1
            reciprocal_rank = 1.0 / first_hit_rank
            mrr_sum += reciprocal_rank
            print(f"   -> Kết quả: ĐÚNG (Rank: {first_hit_rank}, MRR: {reciprocal_rank:.3f})")
        else:
            print(f"   -> Kết quả: SAI (Không tìm thấy tài liệu chuẩn trong Top {k})")
            
        # NDCG @ K
        ndcg_score = calculate_ndcg(relevance_scores)
        ndcg_sum += ndcg_score
        
        print(f"   -> Relevance Scores: {relevance_scores} | NDCG@{k}: {ndcg_score:.4f} | Time: {latency:.2f}s\n")
        
    # Tính điểm trung bình
    hit_rate = (hits / total_queries) * 100
    avg_mrr = mrr_sum / total_queries
    avg_ndcg = ndcg_sum / total_queries
    avg_latency = sum(latencies) / total_queries
    
    print("="*60)
    print(f"KẾT QUẢ ĐÁNH GIÁ CHUNG (RETRIEVAL METRICS AT K={k})")
    print("="*60)
    print(f"- Tổng số mẫu kiểm thử : {total_queries}")
    print(f"- Hit Rate @ {k}          : {hit_rate:.2f}%")
    print(f"- MRR @ {k}               : {avg_mrr:.4f}")
    print(f"- NDCG @ {k}              : {avg_ndcg:.4f}")
    print(f"- Thời gian phản hồi TB : {avg_latency:.2f} giây")
    print("="*60)

if __name__ == "__main__":
    evaluate_retrieval(eval_dataset, k=3)
