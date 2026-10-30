from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Pengaturan LLM dan Embedding menggunakan setup lokal Ollama
Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = [
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) unusable at scale."),
]

print("Membangun index...")
index = VectorStoreIndex.from_documents(documents)

# Membuat Retriever manual (k = 2)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

print("\nMenguji Response Modes...")
# Different response modes
for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
    # Membuat Synthesizer manual
    synthesiser = get_response_synthesizer(response_mode=mode)
    
    # Merakit ulang Query Engine
    engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
    
    resp = engine.query("What algorithms are used in vector search?")
    print(f"[{mode}] {str(resp)[:120]}...\n")