from transformers import AutoTokenizer

print("Memuat Tokenizer GPT-2...")
# Load tokenizer for any model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# PENYESUAIAN: Menambahkan pad_token agar fungsi padding=True di bawah tidak error
tokenizer.pad_token = tokenizer.eos_token

text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

print("\n=== Proses Tokenisasi ===")
# Encode to token IDs
tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")

# Decode back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Tokenize to see individual tokens (Melihat potongan suku kata)
# Simbol 'Ġ' menandakan spasi (space) sebelum kata tersebut
token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {token_strings}")

print("\n=== Batch Encoding (Untuk Input Model) ===")
# Batch encoding with padding and truncation (for model input)
batch = ["What is RAG?", "Explain vector databases in detail."]
encoded = tokenizer(
    batch,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt", # pt = PyTorch tensors
)

print(f"Batch input_ids shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

print("\n=== Menghitung Jumlah Token ===")
# Count tokens for a prompt before sending to any model
def count_tokens(text: str, model: str = "gpt2") -> int:
    tok = AutoTokenizer.from_pretrained(model)
    return len(tok.encode(text))

prompt = "Summarise the following document: " + "word " * 500
print(f"Prompt token count: {count_tokens(prompt)}")