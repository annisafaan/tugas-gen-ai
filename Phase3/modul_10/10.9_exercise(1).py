import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Membuat folder dan file dummy untuk simulasi baca folder
os.makedirs("data_teks", exist_ok=True)
with open("data_teks/file1.txt", "w") as f: f.write("LlamaIndex is highly efficient for data indexing.")
with open("data_teks/file2.txt", "w") as f: f.write("Vector databases store embeddings.")

def ask(question: str) -> dict:
    documents = SimpleDirectoryReader("data_teks").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    response = query_engine.query(question)
    
    # Mengekstrak nama file unik dari source_nodes
    source_files = list(set([node.metadata.get('file_name') for node in response.source_nodes]))
    
    return {
        "answer": str(response),
        "source_filenames": source_files
    }

print(ask("What stores embeddings?"))