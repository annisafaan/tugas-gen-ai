from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=== Memuat Tokenizer dan Model (distilgpt2) ===")
# Load a small model for local inference (CPU-friendly)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Memuat arsitektur model khusus untuk melanjutkan teks (Causal LM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, # float32 sangat aman dan stabil untuk CPU
)
model.eval() # Mengunci model ke mode evaluasi (bukan mode pelatihan)

print("\n=== Menghasilkan Teks ===")
prompt = "The key advantage of local LLMs is"
print(f"Prompt asli: {prompt}\n")

# 1. Tokenisasi
inputs = tokenizer(prompt, return_tensors="pt")

# 2. Inference
# torch.no_grad() sangat penting saat inference agar PyTorch tidak memakan 
# banyak RAM untuk merekam riwayat kalkulasi (gradient)
with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

# 3. Decoding
generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Hasil Generasi:")
print(generated)