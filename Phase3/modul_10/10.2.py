from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding # <-- Penyesuaian import
import os
from dotenv import load_dotenv

load_dotenv()

# Configure global settings applied to all indexes and query engines
Settings.llm = OpenAI(
    model="qwen2.5:3b",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.1,
    max_tokens=1024,
)

# Penyesuaian menggunakan modul Ollama secara langsung
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.chunk_size = 512 # tokens per chunk
Settings.chunk_overlap = 64 # token overlap between chunks

print("Setup LlamaIndex berhasil! Pengaturan global telah menggunakan Qwen2.5:3b dan OllamaEmbedding.")