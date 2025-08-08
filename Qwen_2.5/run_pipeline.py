"""
Script chính để chạy toàn bộ pipeline fine-tuning Qwen 2.5
"""

import os
import sys
import argparse
from prepare_data import DataPreparator, create_sample_data
from finetune_qwen import QwenFineTuner
from inference_qwen import QwenInference, export_merged_model

def install_requirements():
    """Cài đặt requirements"""
    print("Installing requirements...")
    os.system("pip install -r requirements_finetune.txt")

def main():
    parser = argparse.ArgumentParser(description="Qwen 2.5 Fine-tuning Pipeline")
    parser.add_argument("--step", choices=["prepare", "train", "inference", "export", "all"], 
                       default="all", help="Pipeline step to run")
    parser.add_argument("--use_sample", action="store_true", 
                       help="Use sample data instead of processing PDFs")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B",
                       help="Base model name")
    parser.add_argument("--output_dir", default="./qwen_law_assistant",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--data_file", default="training_data.json",
                       help="Training data file")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print("=== Qwen 2.5 Fine-tuning Pipeline ===")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print("-" * 40)
    
    # Step 1: Prepare data
    if args.step in ["prepare", "all"]:
        print("\n1. Preparing training data...")
        
        if args.use_sample:
            create_sample_data()
            args.data_file = "sample_training_data.json"
        else:
            preparator = DataPreparator("../source_docs")
            preparator.process_pdfs(args.data_file)
            
        print(f"Training data saved to: {args.data_file}")
    
    # Step 2: Fine-tune model
    if args.step in ["train", "all"]:
        print("\n2. Fine-tuning model...")
        
        if not os.path.exists(args.data_file):
            print(f"Training data file {args.data_file} not found!")
            print("Please run with --step prepare first or use --use_sample")
            return
        
        # Initialize fine-tuner
        fine_tuner = QwenFineTuner(
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        # Load model
        fine_tuner.load_model_and_tokenizer()
        
        # Setup LoRA
        fine_tuner.setup_lora_config()
        
        # Prepare dataset
        dataset = fine_tuner.prepare_dataset(data_path=args.data_file)
        print(f"Dataset size: {len(dataset)}")
        
        # Train
        fine_tuner.train(dataset, epochs=args.epochs, learning_rate=args.learning_rate)
        
        print(f"Fine-tuning completed! Model saved to: {args.output_dir}")
    
    # Step 3: Test inference
    if args.step in ["inference", "all"]:
        print("\n3. Testing inference...")
        
        if not os.path.exists(args.output_dir):
            print(f"Fine-tuned model not found at {args.output_dir}")
            print("Please run training step first")
            return
            
        inference = QwenInference(
            base_model_name=args.model_name,
            finetuned_model_path=args.output_dir
        )
        
        inference.load_finetuned_model()
        
        # Test với một vài câu hỏi
        test_questions = [
            "Luật doanh nghiệp quy định như thế nào về thành lập công ty?",
            "Quyền và nghĩa vụ của người lao động là gì?",
            "Thủ tục ly hôn được thực hiện như thế nào?",
        ]
        
        for question in test_questions:
            print(f"\nCâu hỏi: {question}")
            response = inference.generate_response(question)
            print(f"Trả lời: {response}")
            print("-" * 50)
    
    # Step 4: Export merged model
    if args.step in ["export", "all"]:
        print("\n4. Exporting merged model...")
        
        export_merged_model(
            base_model_name=args.model_name,
            finetuned_path=args.output_dir,
            output_path=f"{args.output_dir}_merged"
        )
    
    print("\n=== Pipeline completed! ===")
    
    if args.step == "all":
        print(f"""
Next steps:
1. Your fine-tuned model is saved at: {args.output_dir}
2. Merged model is saved at: {args.output_dir}_merged
3. To run interactive chat: python inference_qwen.py --mode chat
4. To use in your application: load model from {args.output_dir}_merged
        """)

if __name__ == "__main__":
    main()
