from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Mengarahkan client OpenAI bawaan langsung ke endpoint Ollama lokalmu
openai_client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"), # mengambil http://localhost:11434/v1
    api_key=os.environ.get("OPENAI_API_KEY")    # mengambil "ollama"
)

# Embedding helper - menggunakan model lokal nomic-embed-text
def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="nomic-embed-text", input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

# Connect to Pinecone
print("Menghubungkan ke Pinecone...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

INDEX_NAME = "ai-knowledge-base"

# Create index if it doesn't exist
print("Memeriksa dan membuat index (ini mungkin butuh beberapa detik di cloud)...")
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, # Penyesuaian mutlak: nomic-embed-text memiliki dimensi 768
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# --- Upsert vectors ---
print("Memasukkan (Upsert) dokumen ke Pinecone...")
documents = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
    {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
    {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
]

vectors = [
    {
        "id": doc["id"],
        "values": embed([doc["text"]])[0],
        "metadata": {"text": doc["text"], "source": "guide"},
    }
    for doc in documents
]

index.upsert(vectors=vectors)
print("\nStatistik Index:")
print(index.describe_index_stats())

# --- Query ---
print("\nMenjalankan pencarian...")
q_vector = embed(["how to add external knowledge to an LLM?"])[0]

results = index.query(
    vector=q_vector,
    top_k=3,
    include_metadata=True,
)

print("Top Results:")
for match in results["matches"]:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")

# --- Metadata filter ---
print("\nMetadata filter...")
results_filtered = index.query(
    vector=q_vector,
    top_k=3,
    filter={"source": {"$eq": "guide"}},
    include_metadata=True,
)

for match in results_filtered["matches"]:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")