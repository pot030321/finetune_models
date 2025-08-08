import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
import json
import os
from typing import List, Dict
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'finetune_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.01):
        self.patience = early_stopping_patience
        self.threshold = early_stopping_threshold
        self.best_loss = float('inf')
        self.patience_counter = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        eval_loss = kwargs.get('metrics', {}).get('eval_loss')
        if eval_loss is None:
            return

        if eval_loss < self.best_loss - self.threshold:
            self.best_loss = eval_loss
            self.patience_counter = 0
            control.should_save = True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered")
                control.should_training_stop = True

class QwenFineTuner:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-0.5B", 
                 output_dir: str = "./qwen_finetuned",
                 max_length: int = 1024):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model and tokenizer from {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.model.gradient_checkpointing_enable()
        logger.info("Model and tokenizer loaded successfully!")
        
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        trainable_params, total_params = self.model.get_nb_trainable_parameters()
        logger.info(f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | "
                   f"Percentage: {trainable_params/total_params*100:.2f}%")
        
        return lora_config
    
    def prepare_dataset(self, 
                       data_path: str = None, 
                       custom_data: List[Dict] = None,
                       train_split: float = 0.9):
        """Prepare dataset for fine-tuning"""
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif custom_data:
            data = custom_data
        else:
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
            
        def format_data(example):
            messages = [
                {"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên nghiệp, am hiểu luật pháp Việt Nam..."},
                {"role": "user", "content": "Hãy trả lời bằng tiếng Việt: " + example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}
            
        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_data, remove_columns=dataset.column_names)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        if len(tokenized_dataset) > 10:  # Only split if dataset is large enough
            dataset_dict = tokenized_dataset.train_test_split(
                train_size=train_split,
                seed=42
            )
            return DatasetDict({
                "train": dataset_dict["train"],
                "eval": dataset_dict["test"]
            })
        return DatasetDict({"train": tokenized_dataset, "eval": tokenized_dataset})
    
    def train(self, 
              dataset: DatasetDict,
              epochs: int = 3,
              learning_rate: float = 2e-4,
              batch_size: int = 8):
        """Fine-tune model"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # Gradient clipping
            dataloader_pin_memory=True,
            report_to="none"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Starting fine-tuning...")
        train_result = trainer.train()
        logger.info(f"Training completed. Final loss: {train_result.metrics['train_loss']:.4f}")
        
        logger.info(f"Saving best model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Log training metrics
        metrics = train_result.metrics
        metrics["eval_loss"] = trainer.evaluate()["eval_loss"]
        logger.info(f"Final metrics: {metrics}")
        
        return metrics
    
    def test_model(self, test_prompt: str = "Luật doanh nghiệp quy định gì về thành lập công ty?"):
        """Test fine-tuned model"""
        messages = [
            {"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên nghiệp, am hiểu luật pháp Việt Nam..."},
            {"role": "user", "content": "Hãy trả lời bằng tiếng Việt: " + test_prompt}
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
        
        logger.info(f"Question: {test_prompt}")
        logger.info(f"Answer: {response}")
        
        return response

def main():
    """Main function to run fine-tuning"""
    fine_tuner = QwenFineTuner(
        model_name="Qwen/Qwen2.5-0.5B",
        output_dir="./qwen_law_assistant",
        max_length=1024
    )
    
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.setup_lora_config()
    
    dataset = fine_tuner.prepare_dataset(data_path="training_data.json")
    logger.info(f"Dataset sizes - Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}")
    
    metrics = fine_tuner.train(dataset, epochs=3, learning_rate=2e-4, batch_size=8)
    
    fine_tuner.test_model("Quyền và nghĩa vụ của người lao động là gì?")
    
    return metrics

if __name__ == "__main__":
    main()