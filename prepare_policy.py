import json
import os
import re

def normalize_text(text):
    """
    Dọn dẹp và chuẩn hóa các ký tự đặc biệt, khoảng trắng ẩn
    thường xuất hiện trong văn bản tiếng Việt cào từ web.
    """
    if not text:
        return ""
    # Thay thế các loại dấu cách đặc biệt bằng dấu cách thường
    text = text.replace('\xa0', ' ')
    text = text.replace('\u2002', ' ')
    text = text.replace('\u2003', ' ')
    text = text.replace('\u2009', ' ')
    text = text.replace('\u202f', ' ')
    # Loại bỏ zero-width space (ký tự ẩn)
    text = text.replace('\u200b', '')
    
    # Chuẩn hóa các dấu gạch ngang lạ (en-dash, em-dash) thành dấu gạch ngang thường
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    
    # Chuẩn hóa các dấu ngoặc kép cong thành ngoặc kép thẳng chuẩn
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    # Chuẩn hóa dấu chấm tròn to danh sách đầu dòng để hiển thị thống nhất
    text = text.replace('●', '•')
    
    return text

def convert_tab_to_markdown_table(text):
    """
    Chuyển đổi văn bản chứa tab (\t) thành bảng Markdown chuẩn.
    Giải quyết trường hợp các dòng xuống dòng không chứa tab (ví dụ: gạch đầu dòng chi tiết)
    vẫn được gộp vào ô cuối cùng của hàng trước đó thay vì làm hỏng cấu trúc bảng.
    """
    lines = text.split('\n')
    new_lines = []
    current_table = [] # Lưu các hàng dưới dạng danh sách các cột
    col_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_table:
                # Kết thúc bảng khi gặp dòng trống
                new_lines.append(render_markdown_table(current_table))
                current_table = []
                col_count = 0
            new_lines.append(line)
            continue
            
        if '\t' in line:
            # Tách cột
            columns = [col.strip() for col in line.split('\t')]
            
            # Nếu bảng đang chạy và số lượng cột khớp, hoặc đây là hàng mới
            if not current_table:
                col_count = len(columns)
                current_table.append(columns)
            else:
                # Nếu hàng tiếp theo, chuẩn hóa số cột
                if len(columns) < col_count:
                    columns += [""] * (col_count - len(columns))
                else:
                    columns = columns[:col_count]
                current_table.append(columns)
        else:
            # Nếu dòng không chứa tab nhưng bảng đang mở, và dòng này có vẻ là nội dung tiếp nối
            # (ví dụ bắt đầu bằng dấu gạch ngang '-', hoặc dòng mô tả phụ)
            if current_table and (stripped.startswith('-') or stripped.startswith('*') or len(stripped) < 150):
                # Gộp dòng này vào ô cuối cùng của hàng trước đó (dùng <br> để xuống dòng trong ô Markdown)
                if current_table[-1][-1]:
                    current_table[-1][-1] += "<br>" + stripped
                else:
                    current_table[-1][-1] = stripped
            else:
                # Kết thúc bảng nếu có dòng text bình thường ngắt quãng
                if current_table:
                    new_lines.append(render_markdown_table(current_table))
                    current_table = []
                    col_count = 0
                new_lines.append(line)
                
    if current_table:
        new_lines.append(render_markdown_table(current_table))
        
    return '\n'.join(new_lines)

def render_markdown_table(rows):
    """Render mảng hai chiều thành bảng Markdown chuẩn."""
    if not rows:
        return ""
    col_count = len(rows[0])
    markdown_lines = []
    
    # Hàng tiêu đề
    markdown_lines.append("| " + " | ".join(rows[0]) + " |")
    # Dòng phân cách
    markdown_lines.append("| " + " | ".join(["---"] * col_count) + " |")
    
    # Các hàng dữ liệu
    for row in rows[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")
        
    return "\n" + "\n".join(markdown_lines) + "\n"

def split_policy_into_chunks(content, section_title):
    """
    Chia nhỏ nội dung chính sách dựa trên mọi cấp độ đề mục:
    - Điều: Điều 1, Điều 8, Điều 12
    - Số: 1., 2., 9., 11.1., 11.2., 2.1.
    - Chữ cái: a., b., c.
    - La mã: I., II., IV.
    - La mã trong ngoặc: (i), (ii), (iii)
    """
    # Regex phát hiện các đề mục ở đầu dòng:
    # 1. Từ "Điều [Số]" (ví dụ: Điều 8., Điều 10)
    # 2. Số dạng: 1., 2.1., 11.1.2.
    # 3. La mã dạng: I., II., IV.
    # 4. Chữ cái dạng: a., b., c.
    # 5. Ký hiệu la mã ngoặc đơn dạng: (i), (ii), (iii)
    pattern = r'\n(?=(?:Điều\s+\d+)|(?:(?:\d+(?:\.\d+)*|[A-ZĐ]+|[a-z]|\([i-vx]+\))\.\s+)|(?:\([i-vx]+\)\s+))'
    
    # Chia nhỏ văn bản
    sub_sections = re.split(pattern, content)
    chunks = []
    
    # Chunk giới thiệu
    intro = sub_sections[0].strip()
    if intro:
        chunks.append({
            "sub_title": "Giới thiệu chung",
            "text": intro
        })
        
    for sub in sub_sections[1:]:
        sub = sub.strip()
        if not sub:
            continue
            
        # Lấy dòng đầu tiên của cụm text làm tiêu đề của chunk đó
        lines = sub.split('\n')
        sub_title = lines[0].strip()
        
        chunks.append({
            "sub_title": sub_title,
            "text": sub
        })
        
    return chunks

def process_policy_data(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        sections = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        title = normalize_text(section.get("title", ""))
        content = normalize_text(section.get("content", ""))
        
        # 1. Chuyển đổi bảng tab thành bảng Markdown
        formatted_content = convert_tab_to_markdown_table(content)
        
        # 2. Chia nhỏ phần nội dung chính sách theo điều khoản
        sub_chunks = split_policy_into_chunks(formatted_content, title)
        
        # 3. Tạo cấu trúc chunk RAG chuẩn kèm Context Injection
        for sub in sub_chunks:
            chunk_counter += 1
            
            # Inject ngữ cảnh chính sách và tiêu đề lớn vào đầu mỗi chunk
            contextual_text = (
                f"[Chính sách CellphoneS] - {title}\n"
                f"Chủ đề: {sub['sub_title']}\n"
                f"Nội dung:\n{sub['text']}"
            )
            
            all_chunks.append({
                "chunk_id": f"policy_chunk_{chunk_counter:03d}",
                "text": contextual_text,
                "metadata": {
                    "section_title": title,
                    "sub_title": sub['sub_title'],
                    "type": "policy"
                }
            })

    # Ghi kết quả ra file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"Hoàn thành! Đã tạo thành công {len(all_chunks)} chunks chính sách sạch.")
        print(f"Dữ liệu được lưu tại: {output_file}")
    except AttributeError:
        print(f"Hoan thanh! Da tao {len(all_chunks)} chunks chinh sach.")
        print(f"Luu tai: {output_file}")

if __name__ == "__main__":
    input_path = "data/policy.json"
    output_path = "data/prepared_policy_chunks.json"
    process_policy_data(input_path, output_path)
