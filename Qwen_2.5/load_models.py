# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")
# messages = [
#     {"role": "user", "content": "bạn tên là gì"},
# ]
# pipe(messages)


# # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
messages = [
    {"role": "user", "content": "bạn là ai ?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=400)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))