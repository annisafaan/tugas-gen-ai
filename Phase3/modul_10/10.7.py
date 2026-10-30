from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Pengaturan LLM dan Embedding menggunakan setup lokal Ollama
Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Direktori penyimpanan diubah ke path relatif agar mudah dilihat di komputermu
PERSIST_DIR = "./llama_index_store"

# Build and save index
documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

print("Membangun dan menyimpan index ke disk...")
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"Index saved to {PERSIST_DIR}")

# Load index back - no re-embedding needed
print("\nMemuat ulang index dari penyimpanan...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
loaded_index = load_index_from_storage(storage_context)

query_engine = loaded_index.as_query_engine()
response = query_engine.query("How does LlamaIndex handle persistence?")

print("\nAfter reload:", response)