import json
import os
import re
from typing import List, Dict
import PyPDF2
from transformers import AutoTokenizer


class DataPreparator:
    def __init__(self, source_docs_path: str = "../source_docs", model_name="Qwen/Qwen2.5-0.5B"):
        self.source_docs_path = source_docs_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks_by_tokens(self, text: str, chunk_size: int = 512) -> List[str]:
        """Chia text thành các chunks theo token"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())

        return chunks

    def extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa đơn giản"""
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

    def generate_qa_pairs(self, text_chunks: List[str]) -> List[Dict]:
        """Sinh Q2A và A2A từ các chunk"""
        qa_pairs = []
        question_templates = [
            "Luật quy định như thế nào về {}?",
            "Quyền và nghĩa vụ của {} là gì?",
            "Điều kiện {} theo luật là gì?",
            "Thủ tục {} được thực hiện như thế nào?",
            "Hậu quả pháp lý của {} là gì?",
        ]

        for chunk in text_chunks:
            keywords = self.extract_keywords(chunk)
            if not keywords:
                continue

            for keyword in keywords[:2]:
                for template in question_templates[:2]:  # chỉ lấy 2 template để tránh quá dài
                    question = template.format(keyword)

                    # Q2A
                    qa_pairs.append({
                        "instruction": question,
                        "response": chunk
                    })

                    # A2A
                    extended_response = (
                        chunk + " Ngoài ra, bạn nên tham khảo thêm các quy định liên quan trong luật hiện hành để đảm bảo tuân thủ."
                    )
                    qa_pairs.append({
                        "instruction": chunk,
                        "response": extended_response
                    })

        return qa_pairs

    def process_pdfs(self, output_file: str = "training_data.json", chunk_size: int = 512):
        """Xử lý tất cả PDF và tạo dataset"""
        all_qa_pairs = []
        pdf_files = [f for f in os.listdir(self.source_docs_path) if f.lower().endswith('.pdf')]

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            pdf_path = os.path.join(self.source_docs_path, pdf_file)
            text = self.extract_text_from_pdf(pdf_path)

            if text:
                clean_text = self.clean_text(text)
                chunks = self.split_into_chunks_by_tokens(clean_text, chunk_size=chunk_size)
                qa_pairs = self.generate_qa_pairs(chunks)
                all_qa_pairs.extend(qa_pairs)
                print(f"Generated {len(qa_pairs)} QA pairs from {pdf_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

        print(f"✅ Total {len(all_qa_pairs)} QA pairs saved to {output_file}")
        return all_qa_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data for Qwen fine-tuning")
    parser.add_argument("--source_path", default="../source_docs", help="Path to source documents")
    parser.add_argument("--output", default="training_data.json", help="Output JSON file")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in tokens")
    args = parser.parse_args()

    preparator = DataPreparator(args.source_path)
    preparator.process_pdfs(args.output, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
