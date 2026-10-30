from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Tetap menggunakan model qwen2.5:3b ke Ollama lokal
llm = ChatOpenAI(model="qwen2.5:3b", api_key=os.environ["OPENAI_API_KEY"])

# Run two chains in parallel and combine results
summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise this in one sentence."),
    ("human", "{text}"),
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 3 keywords from this text. Return them comma-separated."),
    ("human", "{text}"),
])

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    summary=summarise_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)

result = parallel_chain.invoke({
    "text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."
})

print("=== Parallel LCEL Chain Output ===")
print("Summary: ", result["summary"])
print("Keywords:", result["keywords"])