import chromadb
from chromadb.config import Settings
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Mengarahkan client OpenAI bawaan langsung ke endpoint Ollama lokalmu
openai_client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"), # mengambil http://localhost:11434/v1 dari .env
    api_key=os.environ.get("OPENAI_API_KEY")    # mengambil "ollama"
)

# Embedding helper - menggunakan model lokal nomic-embed-text
def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="nomic-embed-text", input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

# Persistent client (saves to disk di dalam folder saat ini)
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}, # cosine similarity
)

# Add documents with pre-computed embeddings
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]

ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]
embeddings = embed(documents)

print("Memasukkan dokumen ke dalam ChromaDB...")
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)
print(f"Collection size: {collection.count()}")

# --- Query ---
print("\nMenjalankan pencarian...")
query = "how do I add knowledge to an LLM without retraining?"
q_embed = embed([query])[0]

results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    include=["documents", "distances", "metadatas"],
)

print("\nTop results:")
for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0],
):
    print(f" [{dist:.4f}] {doc}")
    print(f"    category: {meta['category']}")

# --- Metadata filtering ---
print("\nMetadata filtering...")
results_filtered = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}}, # pre-filter
    include=["documents", "distances"],
)
for doc, dist in zip(results_filtered["documents"][0], results_filtered["distances"][0]):
    print(f" [{dist:.4f}] {doc}")

# --- Update a document ---
print("\nMemperbarui dokumen...")
collection.update(
    ids=["doc_0"],
    documents=["RAG retrieves external documents and injects them into the LLM context."],
    embeddings=embed(["RAG retrieves external documents and injects them into the LLM context."]),
)

# --- Delete a document ---
print("Menghapus dokumen...")
collection.delete(ids=["doc_4"])
print(f"After delete, count: {collection.count()}")