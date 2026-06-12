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
    Giải quyết chính xác lỗi lệch cột: dòng phụ nối tiếp chỉ được ghép vào
    cột cuối cùng thực tế của hàng đó và chỉ ghép nếu là dấu gạch đầu dòng.
    """
    lines = text.split('\n')
    new_lines = []
    current_table = [] # Lưu các hàng dưới dạng danh sách các cột (chưa đệm cột)
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_table:
                new_lines.append(render_markdown_table(current_table))
                current_table = []
            new_lines.append(line)
            continue
            
        if '\t' in line:
            # Tách cột thô của hàng hiện tại
            columns = [col.strip() for col in line.split('\t')]
            current_table.append(columns)
        else:
            # Chỉ ghép nếu bảng đang mở và dòng này thực sự bắt đầu bằng dấu gạch đầu dòng list
            if current_table and (stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*')):
                # Ghép vào cột cuối cùng thực tế được cung cấp của hàng trước đó
                if current_table[-1][-1]:
                    current_table[-1][-1] += "<br>" + stripped
                else:
                    current_table[-1][-1] = stripped
            else:
                # Kết thúc bảng nếu gặp tiêu đề phụ hoặc dòng bình thường
                if current_table:
                    new_lines.append(render_markdown_table(current_table))
                    current_table = []
                new_lines.append(line)
                
    if current_table:
        new_lines.append(render_markdown_table(current_table))
        
    return '\n'.join(new_lines)

def render_markdown_table(rows):
    """Render mảng hai chiều thành bảng Markdown chuẩn và đệm cột đồng đều."""
    if not rows:
        return ""
    
    # Tìm số cột lớn nhất của bảng
    max_cols = max(len(row) for row in rows)
    
    markdown_lines = []
    
    # Đệm cột cho hàng tiêu đề
    header = rows[0] + [""] * (max_cols - len(rows[0]))
    markdown_lines.append("| " + " | ".join(header) + " |")
    
    # Dòng phân cách
    markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    # Các hàng dữ liệu được đệm cột đồng đều
    for row in rows[1:]:
        padded_row = row + [""] * (max_cols - len(row))
        markdown_lines.append("| " + " | ".join(padded_row) + " |")
        
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
            # LỌC BỎ CHUNK RÁC:
            # Nếu nội dung quá ngắn (< 80 ký tự) và chỉ trùng lặp với tiêu đề phần, bỏ qua.
            cleaned_text = sub['text'].strip()
            if len(cleaned_text) < 80 and (cleaned_text in title or title in cleaned_text):
                continue
                
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
