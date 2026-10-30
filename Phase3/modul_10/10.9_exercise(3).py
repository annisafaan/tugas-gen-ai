from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

docs = [
    Document(text="Target Document 1", metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Target Document 2", metadata={"year": 2024, "topic": "embeddings"}),
    Document(text="Wrong Year", metadata={"year": 2023, "topic": "embeddings"}),
    Document(text="Wrong Topic", metadata={"year": 2024, "topic": "generation"}),
]
index = VectorStoreIndex.from_documents(docs)

# Menerapkan filter
filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="year", value=2024),
        ExactMatchFilter(key="topic", value="embeddings"),
    ]
)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)
nodes = retriever.retrieve("document")

print("Nodes yang lolos filter:")
for node in nodes:
    print(f"- {node.text} | Metadata: {node.metadata}")