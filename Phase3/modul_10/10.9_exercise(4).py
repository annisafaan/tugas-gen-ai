from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import json, re

Settings.llm = Ollama(model="qwen2.5:3b", temperature=0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
llm = Settings.llm

docs = [Document(text=f"RAG concept detail number {i}.") for i in range(15)]
index = VectorStoreIndex.from_documents(docs)

# Stage 1: Retrieve top-10
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
nodes = retriever.retrieve("What are the details of RAG?")

# Stage 2: Re-rank top-3 menggunakan LLM
scored_nodes = []
for node in nodes:
    prompt = f"Score the relevance of the text to the question from 0 to 10. Output ONLY valid JSON, for example: {{\"score\": 8}}.\nQuestion: What are the details of RAG?\nText: {node.text}"
    
    # Langsung panggil fungsi complete milik LlamaIndex LLM
    resp = llm.complete(prompt).text
    
    # Ambil angka menggunakan Regex agar robust jika model mengeluarkan teks tambahan
    score = 0
    match = re.search(r'"score"\s*:\s*(\d+)', resp)
    if match:
        score = int(match.group(1))
        
    scored_nodes.append((score, node))

# Sortir berdasarkan nilai skor (tertinggi di atas) dan ambil Top-3
scored_nodes.sort(key=lambda x: x[0], reverse=True)
top_3 = scored_nodes[:3]

print("Top 3 Node setelah di-Rerank oleh LLM:")
for score, node in top_3:
    print(f"Skor: {score} | {node.text}")