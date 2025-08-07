"""
Script fine-tune mô hình Qwen 2.5 với LoRA
Tối ưu hóa cho việc fine-tune trên dataset tiếng Việt
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import os
from typing import List, Dict

class QwenFineTuner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", output_dir: str = "./qwen_finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load model và tokenizer"""
        print(f"Loading model and tokenizer from {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model với float16 để tiết kiệm bộ nhớ
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # Không sử dụng quantization nếu bitsandbytes không hỗ trợ GPU
        )
        
        # Chuẩn bị model cho training
        self.model.gradient_checkpointing_enable()
        
        print("Model and tokenizer loaded successfully!")
        
    def setup_lora_config(self):
        """Thiết lập LoRA configuration"""
        lora_config = LoraConfig(
            r=16,  # Rank của LoRA
            lora_alpha=32,  # Alpha parameter
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],  # Target modules cho Qwen
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config
        
    def prepare_dataset(self, data_path: str = None, custom_data: List[Dict] = None):
        """
        Chuẩn bị dataset cho fine-tuning
        
        Args:
            data_path: Đường dẫn đến file JSON chứa data
            custom_data: List các dict chứa 'instruction' và 'response'
        """
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif custom_data:
            data = custom_data
        else:
            # Sample data mẫu cho luật
            data = [
                {
                    "instruction": "Luật doanh nghiệp quy định như thế nào về thành lập công ty?",
                    "response": "Theo Luật Doanh nghiệp 2020, để thành lập công ty cần thực hiện các bước: 1) Chuẩn bị hồ sơ thành lập, 2) Nộp hồ sơ tại cơ quan đăng ký kinh doanh, 3) Nhận Giấy chứng nhận đăng ký doanh nghiệp."
                },
                {
                    "instruction": "Quyền và nghĩa vụ của người lao động trong Bộ luật Lao động?",
                    "response": "Người lao động có quyền được làm việc, được trả lương công bằng, được bảo đảm an toàn lao động, được nghỉ ngơi. Đồng thời có nghĩa vụ thực hiện đúng hợp đồng lao động, tuân thủ nội quy lao động."
                }
            ]
        
        # Chuyển đổi thành format chat template
        def format_data(example):
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            return {"text": text}
        
        # Tạo dataset
        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_data, remove_columns=dataset.column_names)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def train(self, dataset, epochs: int = 3, learning_rate: float = 2e-4):
        """Fine-tune model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,  # Batch size nhỏ để tiết kiệm bộ nhớ
            gradient_accumulation_steps=8,  # Accumulate gradients
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,  # Mixed precision training
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("Fine-tuning completed!")
        
    def test_model(self, test_prompt: str = "Luật doanh nghiệp quy định gì về thành lập công ty?"):
        """Test mô hình đã fine-tune"""
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
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
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        print(f"Question: {test_prompt}")
        print(f"Answer: {response}")
        
        return response

def main():
    """Main function để chạy fine-tuning"""
    
    # Khởi tạo fine-tuner
    fine_tuner = QwenFineTuner(
        model_name="Qwen/Qwen2.5-0.5B",
        output_dir="./qwen_law_assistant"
    )
    
    # Load model và tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Setup LoRA
    fine_tuner.setup_lora_config()
    
    # Chuẩn bị dataset
    # Bạn có thể truyền data_path="path/to/your/data.json" hoặc custom_data=[...]
    dataset = fine_tuner.prepare_dataset()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Fine-tune
    fine_tuner.train(dataset, epochs=3, learning_rate=2e-4)
    
    # Test model
    fine_tuner.test_model("Quyền và nghĩa vụ của người lao động là gì?")

if __name__ == "__main__":
    main()
