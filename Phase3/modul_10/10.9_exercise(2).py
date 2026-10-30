from llama_index.core import VectorStoreIndex, SummaryIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen2.5:3b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 10 Paragraf dokumen simulasi
paragraphs = [Document(text=f"Paragraph {i}: The revenue in month {i} was ${i*1000}.") for i in range(1, 11)]

vec_index = VectorStoreIndex.from_documents(paragraphs)
sum_index = SummaryIndex.from_documents(paragraphs)

vec_engine = vec_index.as_query_engine()
sum_engine = sum_index.as_query_engine()

print("=== Pertanyaan Fakta Spesifik ===")
print("VectorStore:", vec_engine.query("What was the revenue in month 7?"))
print("Summary:", sum_engine.query("What was the revenue in month 7?"))

print("\n=== Pertanyaan Dokumen Menyeluruh ===")
print("VectorStore:", vec_engine.query("Calculate the total revenue across all 10 months."))
print("Summary:", sum_engine.query("Calculate the total revenue across all 10 months."))