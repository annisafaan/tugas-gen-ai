from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwen2.5:3b", temperature=0, api_key=os.environ["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(model="nomic-embed-text", api_key=os.environ["OPENAI_API_KEY"], check_embedding_ctx_length=False)

# Mock Data
docs = [Document(page_content="RAG stands for Retrieval-Augmented Generation.")]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

def format_docs(docs: list[Document]) -> str:
    return "\n".join(d.page_content for d in docs)

# RAG Dasar
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY context:\n{context}"),
    ("human", "{question}")
])

# RAG Evaluator
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Score the answer's groundedness based on the context from 0 to 10. Output strictly as JSON like {{\"score\": 8}}.\n\nContext: {context}\n\nAnswer: {answer}")
])

# Chain 1: Hasilkan Jawaban
base_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answer=rag_prompt | llm | StrOutputParser())
)

# Chain 2: Hasilkan Skor
eval_chain = eval_prompt | llm | JsonOutputParser()

def rag_with_quality_filter(question: str):
    # Dapatkan jawaban
    base_result = base_chain.invoke(question)
    
    # Nilai jawaban menggunakan LLM kedua
    score_result = eval_chain.invoke({
        "context": base_result["context"],
        "answer": base_result["answer"]
    })
    
    return {
        "answer": base_result["answer"],
        "score": score_result.get("score")
    }

print(rag_with_quality_filter("What does RAG stand for?"))