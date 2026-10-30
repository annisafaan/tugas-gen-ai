from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os, hashlib
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwen2.5:3b", temperature=0, api_key=os.environ["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(model="nomic-embed-text", api_key=os.environ["OPENAI_API_KEY"], check_embedding_ctx_length=False)

# Mock Data
docs = [Document(page_content="RAG stands for Retrieval-Augmented Generation."), Document(page_content="Fine-tuning bakes knowledge into weights.")]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Chain parafrase
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 paraphrased versions of the given query. Output strictly as a JSON list of strings."),
    ("human", "{query}")
])
query_chain = query_prompt | llm | JsonOutputParser()

def manual_multiquery(query: str):
    # 1. Generate 3 paraphrased queries
    queries = query_chain.invoke({"query": query})
    queries.append(query) # Masukkan kueri asli
    
    # 2. Retrieve untuk setiap kueri
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))
        
    # 3. Deduplikasi menggunakan content hash
    unique_docs = {}
    for d in all_docs:
        doc_hash = hashlib.md5(d.page_content.encode()).hexdigest()
        unique_docs[doc_hash] = d
        
    return list(unique_docs.values())

results = manual_multiquery("What does RAG mean?")
for r in results:
    print("-", r.page_content)