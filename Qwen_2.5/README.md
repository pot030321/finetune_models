# Qwen 2.5 Fine-tuning cho Law Assistant

Hướng dẫn fine-tune mô hình Qwen 2.5 để tạo AI Assistant chuyên về luật Việt Nam.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements_finetune.txt
```

2. Đảm bảo có GPU với ít nhất 8GB VRAM (khuyến nghị 16GB+)

## Sử dụng nhanh

### Chạy toàn bộ pipeline với dữ liệu mẫu:
```bash
python run_pipeline.py --use_sample
```

### Chạy từng bước:

1. **Chuẩn bị dữ liệu từ PDF:**
```bash
python run_pipeline.py --step prepare
```
    
2. **Fine-tune model:**
```bash
python run_pipeline.py --step train
```

3. **Test inference:**
```bash
python run_pipeline.py --step inference
```

4. **Export model đã merge:**
```bash
python run_pipeline.py --step export
```

## Sử dụng chi tiết

### 1. Chuẩn bị dữ liệu

#### Từ sample data:
```bash
python prepare_data.py --sample
```

#### Từ PDF documents:
```bash
python prepare_data.py --source_path ../source_docs --output training_data.json
```

### 2. Fine-tuning

```bash
python finetune_qwen.py
```

Hoặc với custom parameters:
```bash
python run_pipeline.py --step train --epochs 5 --learning_rate 1e-4
```

### 3. Inference

#### Interactive chat:
```bash
python inference_qwen.py --mode chat
```

#### Export merged model:
```bash
python inference_qwen.py --mode export --output_path ./my_merged_model
```

## Cấu trúc files

- `load_models.py`: Script cơ bản để load và test Qwen 2.5
- `finetune_qwen.py`: Script fine-tuning với LoRA
- `inference_qwen.py`: Script inference và export model
- `prepare_data.py`: Script chuẩn bị dữ liệu từ PDF
- `run_pipeline.py`: Script chạy toàn bộ pipeline
- `requirements_finetune.txt`: Dependencies cần thiết

## Cấu hình Fine-tuning

### LoRA Configuration:
- **Rank (r)**: 16 (có thể điều chỉnh 8-64)
- **Alpha**: 32 (thường = 2*rank)
- **Dropout**: 0.1
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Arguments:
- **Batch size**: 1 (với gradient accumulation = 8)
- **Learning rate**: 2e-4
- **Epochs**: 3
- **FP16**: Enabled
- **8-bit loading**: Enabled (tiết kiệm VRAM)

## Tối ưu hóa bộ nhớ

1. **Sử dụng 8-bit quantization**: Giảm VRAM từ 16GB xuống ~8GB
2. **LoRA**: Chỉ fine-tune một phần nhỏ parameters
3. **Gradient accumulation**: Batch size hiệu quả lớn hơn với VRAM hạn chế
4. **FP16**: Giảm bộ nhớ và tăng tốc độ

## Troubleshooting

### CUDA Out of Memory:
- Giảm batch size xuống 1
- Tăng gradient_accumulation_steps
- Giảm max_length từ 512 xuống 256
- Sử dụng 4-bit quantization thay vì 8-bit

### Model không học:
- Tăng learning rate
- Kiểm tra dữ liệu training
- Tăng số epochs
- Điều chỉnh LoRA rank

### Chất lượng output kém:
- Tăng kích thước dataset
- Cải thiện chất lượng dữ liệu
- Fine-tune lâu hơn
- Điều chỉnh generation parameters

## Sử dụng model đã fine-tune

### Load model trong code:
```python
from inference_qwen import QwenInference

# Load model
inference = QwenInference(finetuned_model_path="./qwen_law_assistant")
inference.load_finetuned_model()

# Generate response
response = inference.generate_response("Câu hỏi của bạn")
print(response)
```

### Tích hợp vào RAG pipeline:
Model đã fine-tune có thể thay thế LLM trong RAG chain hiện tại.

## Đánh giá model

Để đánh giá chất lượng model:
1. Tạo test set riêng biệt
2. So sánh response với expected answers
3. Đánh giá manual về độ chính xác pháp lý
4. Test với các câu hỏi edge cases

## Mở rộng

1. **Thêm dữ liệu**: Bổ sung thêm PDF luật, quy định
2. **Cải thiện data preparation**: Sử dụng LLM để tạo QA pairs tốt hơn
3. **Multi-turn conversation**: Fine-tune cho dialogue
4. **Domain-specific**: Fine-tune cho từng lĩnh vực luật cụ thể

## Lưu ý

- Fine-tuning cần GPU mạnh và thời gian dài
- Dữ liệu training phải chất lượng cao
- Model output cần được kiểm tra bởi chuyên gia pháp lý
- Tuân thủ quy định về AI và pháp luật
