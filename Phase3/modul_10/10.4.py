from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama # <-- Penyesuaian ke LLM Ollama
from llama_index.embeddings.ollama import OllamaEmbedding 
import os
from dotenv import load_dotenv

load_dotenv()

# Pengaturan LLM dan Embedding menggunakan setup lokal Ollama murni
Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = [
    Document(text="RAG combines retrieval with generation to answer questions from external knowledge."),
    Document(text="Fine-tuning trains a model on new data to improve performance on a specific task."),
    Document(text="Prompt engineering optimises the input to an LLM to get better outputs without any training."),
    Document(text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."),
    Document(text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."),
]

# Build index - otomatis memotong dan mengubah dokumen menjadi embedding
print("Membangun index...")
index = VectorStoreIndex.from_documents(documents)

# Query engine - mengambil dokumen yang relevan + menyintesis jawaban dengan LLM
print("Menjalankan query engine...\n")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What is the difference between RAG and fine-tuning?")

print("Answer:", response)
print("\nSource nodes:")
for node in response.source_nodes:
    print(f" Score {node.score:.4f} : {node.text[:60]}...")