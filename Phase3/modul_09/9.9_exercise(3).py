from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwen2.5:3b", api_key=os.environ["OPENAI_API_KEY"])
parser = StrOutputParser()

map_prompt = ChatPromptTemplate.from_messages([("system", "Summarize this chunk concisely:\n\n{chunk}")])
reduce_prompt = ChatPromptTemplate.from_messages([("system", "Combine these summaries into one final summary:\n\n{summaries}")])

map_chain = map_prompt | llm | parser
reduce_chain = reduce_prompt | llm | parser

def map_reduce_summarize(text: str):
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(text)
    
    # Map step (batching for performance)
    chunk_summaries = map_chain.batch([{"chunk": c} for c in chunks])
    
    # Reduce step
    final_summary = reduce_chain.invoke({"summaries": "\n".join(chunk_summaries)})
    return final_summary

long_text = ("LangChain simplifies AI. " * 30) + ("RAG connects LLMs to data. " * 30)
print(map_reduce_summarize(long_text))