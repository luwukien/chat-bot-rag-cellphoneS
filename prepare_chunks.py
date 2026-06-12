import json
import os
import re

def clean_text(text):
    """Làm sạch khoảng trắng và các dòng trống thừa thãi."""
    if not text:
        return ""
    # Thay thế nhiều dấu xuống dòng bằng 1 dấu xuống dòng
    text = re.sub(r'\n+', '\n', text)
    # Loại bỏ khoảng trắng thừa ở đầu/cuối mỗi dòng
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text

def split_text_by_paragraphs(text, max_chars=800, overlap=150):
    """
    Chia nhỏ đoạn văn bản dài thành các chunk bằng cách gom các đoạn văn (paragraphs) lại với nhau.
    Đảm bảo kích thước tối đa của mỗi chunk khoảng max_chars và có overlap giữa các chunk.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        # Nếu một đoạn văn lẻ quá dài, ta cắt nhỏ nó theo câu hoặc ký tự
        if para_len > max_chars:
            # Lưu chunk hiện tại lại trước
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Cắt nhỏ đoạn văn quá dài này
            sub_paras = re.split(r'(?<=[.!?]) +', para)
            sub_chunk = []
            sub_len = 0
            for sub_p in sub_paras:
                if sub_len + len(sub_p) > max_chars:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [sub_p]
                    sub_len = len(sub_p)
                else:
                    sub_chunk.append(sub_p)
                    sub_len += len(sub_p) + 1
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        # Gom đoạn văn vào chunk hiện tại
        if current_length + para_len > max_chars:
            chunks.append("\n".join(current_chunk))
            
            # Tính toán overlap (lấy các dòng cuối của chunk trước làm overlap)
            overlap_chunk = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) < overlap:
                    overlap_chunk.insert(0, p)
                    overlap_len += len(p) + 1
                else:
                    break
            
            current_chunk = overlap_chunk + [para]
            current_length = sum(len(p) for p in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(para)
            current_length += para_len + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def process_product_data(input_file, output_file, max_chars=800, overlap=150):
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        products = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for product in products:
        product_id = product.get("id")
        name = product.get("name")
        url = product.get("url", "")
        specs = product.get("specs", {})
        description = product.get("description", "")

        # ----------------------------------------------------
        # 1. XỬ LÝ PHẦN THÔNG SỐ KỸ THUẬT (SPECS)
        # Chuyển specs dạng JSON thành đoạn văn xuôi có cấu trúc ngữ cảnh
        # ----------------------------------------------------
        if specs:
            specs_lines = []
            for key, val in specs.items():
                if val:
                    # Thay thế dấu xuống dòng trong specs thành dấu phẩy để dữ liệu liền mạch
                    val_cleaned = str(val).replace('\n', ', ')
                    specs_lines.append(f"- {key}: {val_cleaned}")
            
            specs_text = f"Thông số kỹ thuật của sản phẩm {name}:\n" + "\n".join(specs_lines)
            
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "product_id": product_id,
                "text": specs_text,
                "metadata": {
                    "product_id": product_id,
                    "product_name": name,
                    "product_url": url,
                    "type": "specs"
                }
            })

        # ----------------------------------------------------
        # 2. XỬ LÝ PHẦN BIẾN THỂ (VARIANTS)
        # Tạo chunk riêng biệt cho giá bán, màu sắc và kho hàng
        # ----------------------------------------------------
        product_variants = product.get("variants", [])
        if product_variants:
            variant_lines = []
            for var in product_variants:
                color = var.get("color", "Không xác định")
                price = var.get("price", "Liên hệ")
                stock = var.get("stock", "Hết hàng")
                # Xử lý nếu stock là một list hoặc string
                if isinstance(stock, list):
                    stock_str = ", ".join(stock)
                else:
                    stock_str = str(stock)
                variant_lines.append(f"- Màu {color}: Giá {price} (Trạng thái: {stock_str})")
            
            variants_text = f"Các phiên bản màu sắc, giá bán và tình trạng kho hàng hiện tại của {name}:\n" + "\n".join(variant_lines)
            
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "product_id": product_id,
                "text": variants_text,
                "metadata": {
                    "product_id": product_id,
                    "product_name": name,
                    "product_url": url,
                    "type": "variants"
                }
            })

        # ----------------------------------------------------
        # 2. XỬ LÝ PHẦN MÔ TẢ CHI TIẾT (DESCRIPTION)
        # ----------------------------------------------------
        cleaned_desc = clean_text(description)
        if cleaned_desc:
            # Loại bỏ các phần quảng cáo mua hàng phổ biến ở cuối mô tả
            cleaned_desc = re.sub(r'Mua\s+.*?\s+giá\s+rẻ\s+tại\s+CellphoneS.*', '', cleaned_desc, flags=re.IGNORECASE)
            
            # Nếu description ngắn (< max_chars), giữ nguyên làm 1 chunk
            if len(cleaned_desc) < max_chars:
                desc_chunks = [cleaned_desc]
            else:
                # Nếu dài, tiến hành chia nhỏ
                desc_chunks = split_text_by_paragraphs(cleaned_desc, max_chars=max_chars, overlap=overlap)

            for part_idx, desc_chunk in enumerate(desc_chunks):
                # Bơm ngữ cảnh tên sản phẩm vào đầu mỗi đoạn chunk mô tả
                contextual_text = (
                    f"Mô tả chi tiết và tính năng sản phẩm {name} (Phần {part_idx + 1}):\n"
                    f"{desc_chunk}"
                )
                
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "product_id": product_id,
                    "text": contextual_text,
                    "metadata": {
                        "product_id": product_id,
                        "product_name": name,
                        "product_url": url,
                        "type": "description"
                    }
                })

    # Ghi dữ liệu chunk đã chuẩn bị ra file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    import sys
    # Cấu hình stdout sang utf-8 để print tiếng Việt trên Windows không lỗi
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"Hoàn thành! Đã tạo thành công {len(all_chunks)} chunks sạch từ {len(products)} sản phẩm.")
        print(f"Dữ liệu được lưu tại: {output_file}")
    except AttributeError:
        # Fallback nếu môi trường không hỗ trợ reconfigure
        print(f"Hoan thanh! Da tao thanh cong {len(all_chunks)} chunks tu {len(products)} san pham.")
        print(f"File duoc luu tai: {output_file}")

if __name__ == "__main__":
    input_path = "data/list_product_details.json"
    output_path = "data/prepared_chunks.json"
    
    # BẠN CÓ THỂ ĐIỀU CHỈNH 2 THÔNG SỐ NÀY TẠI ĐÂY:
    MAX_CHARS = 800
    OVERLAP = 150
    
    process_product_data(input_path, output_path, max_chars=MAX_CHARS, overlap=OVERLAP)
