from transformers import pipeline

# 1. Text classification (Sentiment Analysis)
print("=== 1. Text Classification ===")
classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1, # -1 berarti menggunakan CPU komputermu
)
texts = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks.",
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"[{result['label']} | {result['score']:.3f}] {text[:60]}")


# 2. Named Entity Recognition (Mengenali Nama Orang/Tempat/Organisasi)
print("\n=== 2. Named Entity Recognition ===")
ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)
text = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."
entities = ner(text)
for ent in entities:
    print(f"{ent['word']!r:25s} -> {ent['entity_group']} ({ent['score']:.3f})")


# 3. Text Generation (Melanjutkan teks)
print("\n=== 3. Text Generation ===")
generator = pipeline(
    task="text-generation",
    model="distilgpt2", # model GPT kecil yang ringan untuk CPU
    device=-1,
)
output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256, 
)
for i, seq in enumerate(output):
    print(f"Sequence {i+1}: {seq['generated_text']}")


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 4. Summarisation (Manual Load - Fix untuk Transformers v5)
print("\n=== 4. Summarisation ===")
model_name = "facebook/bart-large-cnn"

# Memuat tokenizer dan model secara langsung tanpa melalui pipeline
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = """
Large language models have fundamentally changed how software is built.
Developers now reach for AI APIs as naturally as they reach for database clients.
The shift from rule-based systems to neural language models has accelerated
since the release of GPT-3 in 2020, and the pace has only increased with
instruction-tuned models like ChatGPT, Claude, and Gemini. These models can
write code, explain concepts, extract information, and reason through complex
problems - tasks that previously required specialised rule-based systems or
significant human effort.
"""

# Proses manual yang tadinya disembunyikan oleh pipeline:
# 1. Tokenisasi (Mengubah teks menjadi angka tensor)
inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)

# 2. Inference (Membuat rangkuman menggunakan model)
summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=20, num_beams=4)

# 3. Decoding (Mengembalikan angka tensor menjadi teks yang bisa dibaca)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)