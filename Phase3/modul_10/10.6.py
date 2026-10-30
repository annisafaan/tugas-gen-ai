from llama_index.core import SummaryIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Pengaturan LLM dan Embedding menggunakan setup lokal Ollama
Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Simulate a long document split into multiple sections
sections = [
    Document(text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."),
    Document(text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."),
    Document(text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."),
    Document(text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."),
    Document(text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."),
]

print("Membangun SummaryIndex...")
index = SummaryIndex.from_documents(sections)
query_engine = index.as_query_engine()

# SummaryIndex membaca SEMUA node, bukan hanya top-k (cocok untuk pertanyaan level dokumen)
print("Menjalankan query engine...")
response = query_engine.query("Which model should I use for a coding project?")

print("\nAnswer:", response)