from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Pengaturan LLM dan Embedding menggunakan setup lokal Ollama
Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = [
    Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
    Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# Retrieve without synthesising an answer
print("Mencari dokumen yang relevan...")
nodes = retriever.retrieve("open source vector databases")

print("\nRetrieved nodes:")
for node in nodes:
    print(f" [{node.score:.4f}] {node.text}")
    print(f"    tag: {node.metadata.get('tag')}")