"""
Script để load và sử dụng mô hình Qwen đã được fine-tune
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os


class QwenInference:
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-0.5B",
                 finetuned_model_path: str = "./qwen_law_assistant"):
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.tokenizer = None
        self.model = None

    def load_finetuned_model(self):
        """Load mô hình đã fine-tune"""
        print("Loading fine-tuned model...")
        if os.path.exists(self.finetuned_model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if os.path.exists(self.finetuned_model_path):
            self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_path)
            print("LoRA weights loaded successfully!")
        else:
            self.model = base_model
            print("Using base model (no fine-tuned weights found)")

    def chunk_text(self, text: str, chunk_size: int = 5):
        """Chia text thành các đoạn chunk_size ký tự"""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_response(self, question: str, max_tokens: int = 250, chunk_size: int = None):
        """Generate response cho câu hỏi, có thể chunk input"""
        if chunk_size:
            chunks = self.chunk_text(question, chunk_size)
            question = " | ".join(chunks)  # ghép lại để model biết là các phần
            print(f"[DEBUG] Chunks ({chunk_size}): {chunks}")

        messages = [{"role": "user", "content": question}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def chat_interface(self):
        """Interactive chat interface"""
        print("=== Qwen Law Assistant ===")
        print("Nhập 'quit' để thoát")
        print("-" * 30)

        while True:
            question = input("\nCâu hỏi: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if question.strip():
                response = self.generate_response(question, chunk_size=5)
                print(f"Trả lời: {response}")

def export_merged_model(base_model_name: str = "Qwen/Qwen2.5-0.5B",
                        finetuned_path: str = "./qwen_law_assistant",
                        output_path: str = "./qwen_merged"):
    """
    Merge LoRA weights với base model và export thành mô hình hoàn chỉnh
    """
    print("Exporting merged model...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load LoRA model
    if os.path.exists(finetuned_path):
        model = PeftModel.from_pretrained(base_model, finetuned_path)

        # Merge weights
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(output_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
        tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")
    else:
        print(f"Fine-tuned model not found at {finetuned_path}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen Fine-tuned Model Inference")
    parser.add_argument("--mode", choices=["chat", "export"], default="chat",
                        help="Mode: chat for interactive chat, export for merging model")
    parser.add_argument("--model_path", default="./qwen_law_assistant",
                        help="Path to fine-tuned model")
    parser.add_argument("--output_path", default="./qwen_merged",
                        help="Output path for merged model")

    args = parser.parse_args()

    if args.mode == "chat":
        # Interactive chat
        inference = QwenInference(finetuned_model_path=args.model_path)
        inference.load_finetuned_model()
        inference.chat_interface()

    elif args.mode == "export":
        # Export merged model
        export_merged_model(
            finetuned_path=args.model_path,
            output_path=args.output_path
        )


if __name__ == "__main__":
    main()
