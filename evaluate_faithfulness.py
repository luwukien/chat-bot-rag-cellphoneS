import os
import json
import time
import requests
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

def get_groq_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Lỗi: GROQ_API_KEY chưa được cấu hình trong file .env!")
    return api_key

def call_groq_json(prompt, model="llama-3.3-70b-versatile", max_retries=3):
    """Gửi yêu cầu tới Groq API và yêu cầu phản hồi dạng JSON, có cơ chế retry khi bị rate limit."""
    api_key = get_groq_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Xử lý rate limit (HTTP 429) - chờ rồi thử lại
            if response.status_code == 429:
                wait_time = int(response.headers.get("retry-after", attempt * 5))
                print(f"   [⏳ Rate limit] Chờ {wait_time}s trước khi thử lại (lần {attempt}/{max_retries})...", flush=True)
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
            return json.loads(content)
        except requests.exceptions.HTTPError as e:
            print(f"   [Lỗi HTTP] Lần {attempt}/{max_retries}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(attempt * 3)
        except requests.exceptions.Timeout:
            print(f"   [Timeout] Lần {attempt}/{max_retries}: API không phản hồi trong 30s.", flush=True)
            if attempt < max_retries:
                time.sleep(attempt * 2)
        except Exception as e:
            print(f"   [Lỗi] Gọi Groq API thất bại: {e}", flush=True)
            return None
    
    print("   [Lỗi] Đã hết số lần thử lại. Bỏ qua bước này.", flush=True)
    return None

def extract_statements(actual_output):
    """
    BƯỚC 1: Tách câu trả lời của chatbot thành các khẳng định (statements) độc lập.
    """
    prompt = f"""
    Nhiệm vụ của bạn là phân tích câu trả lời của chatbot và tách nó ra thành danh sách các khẳng định (statements) thực tế, độc lập, và ngắn gọn nhất có thể.
    Mỗi khẳng định chỉ nên nói về một sự thật hoặc thuộc tính duy nhất.

    Ví dụ:
    Câu trả lời: "iPhone 16 Pro có giá bán từ 28.990.000đ và hiện tại đang còn hàng tại cửa hàng CellphoneS."
    Trả về định dạng JSON:
    {{
      "statements": [
        "iPhone 16 Pro có giá bán từ 28.990.000đ",
        "iPhone 16 Pro hiện tại đang còn hàng tại cửa hàng CellphoneS"
      ]
    }}

    Câu trả lời cần phân tích:
    "{actual_output}"

    Hãy trả về duy nhất một đối tượng JSON có khóa "statements" chứa danh sách các khẳng định trên. Không giải thích gì thêm.
    """
    result = call_groq_json(prompt)
    if result and "statements" in result:
        return result["statements"]
    return []

def verify_statements(statements, retrieval_context):
    """
    BƯỚC 2: Đối chiếu từng khẳng định với ngữ cảnh tài liệu để xem có bằng chứng hay không.
    """
    context_str = "\n---\n".join(retrieval_context)
    
    prompt = f"""
    Bạn là một giám khảo thông thái. Nhiệm vụ của bạn là kiểm tra xem từng khẳng định (statement) dưới đây có được chứng minh trực tiếp bởi ngữ cảnh tài liệu (context) cung cấp hay không.

    Ngữ cảnh tài liệu (Context):
    {context_str}

    Danh sách khẳng định cần kiểm tra:
    {json.dumps(statements, ensure_ascii=False, indent=2)}

    Quy tắc chấm điểm:
    - Trả về 'true' cho mỗi khẳng định NẾU thông tin của khẳng định đó xuất hiện hoặc có thể suy luận trực tiếp từ Context.
    - Trả về 'false' NẾU khẳng định đó KHÔNG được nhắc đến trong Context (ảo giác, tự bịa thông tin) hoặc trái ngược với Context.

    Hãy trả về duy nhất một đối tượng JSON có định dạng như sau, không thêm văn bản ngoài JSON:
    {{
      "verifications": [
        {{
          "statement": "khẳng định 1",
          "supported": true hoặc false,
          "reason": "Giải thích ngắn gọn lý do vì sao đúng hoặc sai dựa trên context"
        }},
        ...
      ]
    }}
    """
    result = call_groq_json(prompt)
    if result and "verifications" in result:
        return result["verifications"]
    return []

def calculate_faithfulness(actual_output, retrieval_context):
    """
    BƯỚC 3: Tính toán điểm Faithfulness Score tổng thể.
    """
    print(f"\n[Faithfulness Judge] Bắt đầu đánh giá câu trả lời...", flush=True)
    
    # 1. Tách ý
    statements = extract_statements(actual_output)
    print(f"   -> Tách được {len(statements)} ý khẳng định thực tế:", flush=True)
    for i, s in enumerate(statements):
        print(f"      {i+1}. {s}", flush=True)
        
    if not statements:
        print("   -> Không trích xuất được ý khẳng định nào. Mặc định 1.0 (Không vi phạm ảo giác).", flush=True)
        return 1.0, []

    # 2. Đối chiếu kiểm tra bằng chứng
    verifications = verify_statements(statements, retrieval_context)
    
    if not verifications:
        print("   -> Không thể đối chiếu. Bỏ qua đánh giá.", flush=True)
        return None, []

    # 3. Tính điểm
    supported_count = 0
    for v in verifications:
        if v.get("supported") is True:
            supported_count += 1
            print(f"   [ĐÚNG] {v['statement']} (Lý do: {v.get('reason')})", flush=True)
        else:
            print(f"   [ẢO GIÁC ❌] {v['statement']} (Lý do: {v.get('reason')})", flush=True)

    score = supported_count / len(statements)
    print(f"   => ĐIỂM FAITHFULNESS CUỐI CÙNG: {score:.2f} ({supported_count}/{len(statements)} ý hợp lệ)", flush=True)
    
    return score, verifications

if __name__ == "__main__":
    # --- TEST RUN MINH HỌA ---
    print("=== CHẠY THỬ NGHIỆM ĐÁNH GIÁ FAITHFULNESS ===")
    
    # Giả lập context tìm kiếm được từ database
    mock_context = [
        "Sản phẩm iPhone 16 Pro phiên bản màu Titan Sa Mạc có giá bán là 28.990.000₫ và tình trạng kho hàng là Còn hàng.",
        "Chính sách bảo hành CellphoneS: Điện thoại được bảo hành chính hãng 12 tháng kể từ ngày kích hoạt."
    ]
    
    # Case 1: Chatbot trả lời trung thực (Faithfulness = 1.0)
    print("\n--- CASE 1: Chatbot Trả Lời Trung Thực ---")
    answer_honest = "iPhone 16 Pro màu Titan Sa Mạc có giá 28.990.000đ và được bảo hành chính hãng 12 tháng."
    calculate_faithfulness(answer_honest, mock_context)
    
    # Chờ giữa 2 case để tránh rate limit từ Groq API
    print("\n   [⏳] Chờ 3s trước khi chạy Case 2 (tránh rate limit)...", flush=True)
    time.sleep(3)

    # Case 2: Chatbot bị ảo giác - tự bịa thông tin giá khuyến mãi (Faithfulness < 1.0)
    print("\n--- CASE 2: Chatbot Bị Ảo Giác (Bịa Thông Tin) ---", flush=True)
    answer_hallucinated = "iPhone 16 Pro màu Titan Sa Mạc có giá bán là 28.990.000đ. Đặc biệt, nếu mua hôm nay bạn sẽ được tặng kèm củ sạc nhanh 20W trị giá 500k."
    calculate_faithfulness(answer_hallucinated, mock_context)
