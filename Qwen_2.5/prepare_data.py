"""
Script để chuẩn bị dữ liệu từ PDF documents cho fine-tuning
"""

import json
import os
import PyPDF2
from typing import List, Dict
import re

class DataPreparator:
    def __init__(self, source_docs_path: str = "../source_docs"):
        self.source_docs_path = source_docs_path
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Làm sạch text"""
        # Loại bỏ whitespace thừa
        text = re.sub(r'\s+', ' ', text)
        # Loại bỏ ký tự đặc biệt
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chia text thành các chunks nhỏ"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def generate_qa_pairs(self, text_chunks: List[str]) -> List[Dict]:
        """
        Tạo cặp câu hỏi-trả lời từ text chunks
        Đây là một phương pháp đơn giản, bạn có thể cải thiện bằng cách sử dụng LLM khác
        """
        qa_pairs = []
        
        # Một số template câu hỏi mẫu
        question_templates = [
            "Luật quy định như thế nào về {}?",
            "Quyền và nghĩa vụ của {} là gì?",
            "Điều kiện {} theo luật là gì?",
            "Thủ tục {} được thực hiện như thế nào?",
            "Hậu quả pháp lý của {} là gì?",
        ]
        
        for chunk in text_chunks:
            # Tìm các từ khóa chính trong chunk
            keywords = self.extract_keywords(chunk)
            
            for keyword in keywords[:2]:  # Chỉ lấy 2 từ khóa đầu
                for template in question_templates[:2]:  # Chỉ lấy 2 template đầu
                    question = template.format(keyword)
                    qa_pairs.append({
                        "instruction": question,
                        "response": chunk
                    })
        
        return qa_pairs
    
    def extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa từ text"""
        # Danh sách từ khóa luật phổ biến
        law_keywords = [
            "công ty", "doanh nghiệp", "hợp đồng", "lao động", 
            "bảo hiểm", "thuế", "đất đai", "tài sản", "quyền sở hữu",
            "nghĩa vụ", "trách nhiệm", "vi phạm", "xử phạt", "bồi thường"
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for keyword in law_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
                
        return keywords
    
    def process_pdfs(self, output_file: str = "training_data.json"):
        """Xử lý tất cả PDF và tạo dữ liệu training"""
        all_qa_pairs = []
        
        pdf_files = [f for f in os.listdir(self.source_docs_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            
            pdf_path = os.path.join(self.source_docs_path, pdf_file)
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Làm sạch text
                clean_text = self.clean_text(text)
                
                # Chia thành chunks
                chunks = self.split_into_chunks(clean_text)
                
                # Tạo QA pairs
                qa_pairs = self.generate_qa_pairs(chunks)
                all_qa_pairs.extend(qa_pairs)
                
                print(f"Generated {len(qa_pairs)} QA pairs from {pdf_file}")
        
        # Lưu vào file JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            
        print(f"Total {len(all_qa_pairs)} QA pairs saved to {output_file}")
        return all_qa_pairs

def create_sample_data():
    """Tạo dữ liệu mẫu để test"""
    sample_data = [
        {
            "instruction": "Luật doanh nghiệp quy định như thế nào về thành lập công ty?",
            "response": "Theo Luật Doanh nghiệp 2020, để thành lập công ty trách nhiệm hữu hạn, các thành viên góp vốn cần thực hiện: 1) Chuẩn bị hồ sơ đăng ký thành lập công ty, 2) Nộp hồ sơ tại Phòng Đăng ký kinh doanh cấp tỉnh, 3) Nhận Giấy chứng nhận đăng ký doanh nghiệp sau 15 ngày làm việc."
        },
        {
            "instruction": "Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động là gì?",
            "response": "Người lao động có quyền: được làm việc trong môi trường an toàn, được trả lương đúng hạn và đầy đủ, được nghỉ phép, được bảo hiểm xã hội. Nghĩa vụ: thực hiện đúng hợp đồng lao động, tuân thủ nội quy công ty, bảo vệ tài sản của người sử dụng lao động."
        },
        {
            "instruction": "Thủ tục ly hôn theo luật hôn nhân và gia đình được thực hiện như thế nào?",
            "response": "Theo Luật Hôn nhân và Gia đình 2014, ly hôn có thể thực hiện bằng: 1) Ly hôn thuận tình: cả hai vợ chồng đồng ý, có thể đăng ký tại UBND xã/phường, 2) Ly hôn tranh chấp: một bên yêu cầu, phải làm thủ tục tại Tòa án nhân dân cấp huyện."
        },
        {
            "instruction": "Điều kiện mua bán nhà đất theo luật đất đai là gì?",
            "response": "Theo Luật Đất đai 2013, để mua bán nhà đất cần: 1) Có Giấy chứng nhận quyền sử dụng đất hợp pháp, 2) Người mua phải có đủ điều kiện được sử dụng đất theo quy định, 3) Thực hiện thủ tục chuyển nhượng tại UBND cấp có thẩm quyền, 4) Nộp thuế chuyển nhượng bất động sản."
        },
        {
            "instruction": "Hậu quả pháp lý của việc vi phạm hợp đồng thương mại là gì?",
            "response": "Vi phạm hợp đồng thương mại có thể bị: 1) Buộc thực hiện đúng nghĩa vụ trong hợp đồng, 2) Bồi thường thiệt hại cho bên bị vi phạm, 3) Trả tiền phạt vi phạm nếu có thỏa thuận trong hợp đồng, 4) Bị đơn phương chấm dứt hợp đồng trong trường hợp vi phạm nghiêm trọng."
        }
    ]
    
    with open("sample_training_data.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("Sample data created: sample_training_data.json")
    return sample_data

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data for Qwen fine-tuning")
    parser.add_argument("--source_path", default="../source_docs",
                       help="Path to source documents")
    parser.add_argument("--output", default="training_data.json",
                       help="Output JSON file")
    parser.add_argument("--sample", action="store_true",
                       help="Create sample data instead of processing PDFs")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data()
    else:
        preparator = DataPreparator(args.source_path)
        preparator.process_pdfs(args.output)

if __name__ == "__main__":
    main()
