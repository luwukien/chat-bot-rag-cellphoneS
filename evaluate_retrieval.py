import time
import math
import numpy as np
from test_search import retrieve_and_rerank

# 1. Định nghĩa tập dữ liệu kiểm thử (Golden Dataset)
# Sử dụng phương án 2: Lấy câu hỏi FAQ gốc và biến đổi viết tắt, lỗi chính tả, không dấu
# 1. Định nghĩa tập dữ liệu kiểm thử phức tạp (Golden Dataset)
# Dành cho việc đánh giá khả năng xử lý viết tắt, không dấu, series, và định tuyến hỗn hợp (Product + Policy)
eval_dataset = [
    # === NHÓM 1: GIÁ BÁN & BIẾN THỂ (VARIANTS) - Viết tắt mạnh, sai chính tả, không dấu ===
    {
        "query": "ip 16 pr max 128gb gia bao nhieu",
        "expected_product_id": "iphone-16-pro-max",
        "expected_type": "variants"
    },
    {
        "query": "iphone 15 pr 256 gb thoi diem nay con hang mau titan tu nhien k",
        "expected_product_id": "iphone-15-pro",
        "expected_type": "variants"
    },
    {
        "query": "gia ban ip 14 thuong ban thap nhat bn tien",
        "expected_product_id": "iphone-14",
        "expected_type": "variants"
    },
    {
        "query": "iphone 16 series ban 256gb gia bao nhieu",
        "expected_product_id": "iphone-16",  # Tìm kiếm dòng máy (series)
        "expected_type": "variants"
    },

    # === NHÓM 2: THÔNG SỐ KỸ THUẬT (SPECS) - Từ lóng công nghệ, không dấu ===
    {
        "query": "cau hinh chi tiet camera va chip cua ip 16 pr 128gb",
        "expected_product_id": "iphone-16-pro",
        "expected_type": "specs"
    },
    {
        "query": "man hinh va chip xu ly cua ip 15 plus thong so ntn",
        "expected_product_id": "iphone-15-plus",
        "expected_type": "specs"
    },
    {
        "query": "trong luong va kich thuoc cua dien thoai ip 13 pro max la bao nhieu",
        "expected_product_id": "iphone-13-pro-max",
        "expected_type": "specs"
    },

    # === NHÓM 3: HỎI ĐÁP FAQ SẢN PHẨM (FAQ) - Câu hỏi tự nhiên về Pin/Sạc/Tính năng ===
    {
        "query": "ip 13 thuong co sac nhanh k va pin dung duoc bn lau",
        "expected_product_id": "iphone-13",
        "expected_type": "faq"
    },
    {
        "query": "camera cua iphone 15 plus chup dem co tot khong",
        "expected_product_id": "iphone-15-plus",
        "expected_type": "faq"
    },
    {
        "query": "iphone 16 pro max sac nhanh toi da bao nhieu w",
        "expected_product_id": "iphone-16-pro-max",
        "expected_type": "faq"
    },

    # === NHÓM 4: MÔ TẢ & TRẢI NGHIỆM (DESCRIPTION) - Đánh giá sâu về thiết kế, chất liệu ===
    {
        "query": "danh gia chat lieu khung vien titan tren dong ip 15 pro max",
        "expected_product_id": "iphone-15-pro-max",
        "expected_type": "description"
    },
    {
        "query": "thiet ke nut bam camera control moi cua ip 16 co gi dac biet",
        "expected_product_id": "iphone-16",
        "expected_type": "description"
    },
    {
        "query": "trai nghiem cam nam va mau sac moi cua iphone 16 pro ntn",
        "expected_product_id": "iphone-16-pro",
        "expected_type": "description"
    },

    # === NHÓM 5: CHÍNH SÁCH CHUNG (POLICY) - Viết tắt, không dấu, định tuyến chính sách ===
    {
        "query": "quy dinh doi tra va hoan tien tai cellphones khi may bi loi phan cung",
        "expected_product_id": "N/A",  # Chính sách chung không thuộc product cụ thể nào
        "expected_type": "policy"
    },
    {
        "query": "chinh sach bh cua cua hang khi mua dt tra gop",
        "expected_product_id": "N/A",
        "expected_type": "policy"
    },
    {
        "query": "huong dan thu tuc doi tra san pham loi trong 30 ngay dau",
        "expected_product_id": "N/A",
        "expected_type": "policy"
    },

    # === NHÓM 6: HỖN HỢP (PRODUCT + POLICY) - Test khả năng phân rã của LLM & routing ===
    {
        "query": "ip 16 pr max 128gb gia bao nhieu va co bh 1 doi 1 ko",
        "expected_product_id": ["iphone-16-pro-max", "N/A"],
        "expected_type": ["variants", "policy"]
    },
    {
        "query": "mua iphone 15 pro 128gb tra gop duoc ko va thu tuc ntn",
        "expected_product_id": ["iphone-15-pro", "N/A"],
        "expected_type": ["variants", "policy"]
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
        
        # Cho phép expected_product_id là list hoặc string
        expected_pids = item["expected_product_id"]
        if isinstance(expected_pids, str):
            expected_pids = [expected_pids]
            
        # Cho phép expected_type là list hoặc string
        expected_types = item["expected_type"]
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        
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
            
            # Kiểm tra xem retrieved_pid có khớp với bất kỳ exp_pid nào trong list mong đợi
            pid_matches = False
            for exp_pid in expected_pids:
                # Chuẩn hóa các giá trị rỗng/None/N/A về dạng đồng nhất "N/A"
                norm_retrieved = retrieved_pid if retrieved_pid not in ("", None) else "N/A"
                norm_expected = exp_pid if exp_pid not in ("", None) else "N/A"
                
                if (norm_retrieved == norm_expected or
                    (exp_pid != "N/A" and exp_pid != "" and (
                        retrieved_pid.startswith(exp_pid + "-") or
                        exp_pid.startswith(retrieved_pid + "-")
                    ))):
                    pid_matches = True
                    break
            
            # Gán điểm liên quan (graded relevance):
            # 2 điểm: Khớp cả mã sản phẩm và loại thông tin mong đợi
            # 1 điểm: Khớp mã sản phẩm nhưng sai loại thông tin
            # 0 điểm: Sai hoàn toàn sản phẩm
            if pid_matches:
                if retrieved_type in expected_types:
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
