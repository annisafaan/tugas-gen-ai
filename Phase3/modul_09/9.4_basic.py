from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Menggunakan model qwen2.5:3b agar terhubung ke Ollama lokal
llm = ChatOpenAI(model="qwen2.5:3b", api_key=os.environ["OPENAI_API_KEY"])

# Basic chain: prompt | model | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "sentences": 2,
    "text": """LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, models, memory, and retrieval so developers can
compose complex AI workflows from simple building blocks without writing boilerplate."""
})

print("=== Basic LCEL Chain Output ===")
print(result)