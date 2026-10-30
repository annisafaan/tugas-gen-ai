from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI model (Secara otomatis terhubung ke Ollama berkat .env)
llm_oai = ChatOpenAI(
    model="qwen2.5:3b",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["OPENAI_API_KEY"],
)

# Anthropic model - same interface
# (Tetap ditulis sesuai buku, tapi tidak kita panggil di bawah agar tidak error)
llm_ant = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# Both accept the same message types
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

# Invoke (blocking)
print("=== Invoke ===")
response = llm_oai.invoke(messages)
print(response.content)

# Stream
print("\n=== Stream ===")
for chunk in llm_oai.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# Batch - run multiple inputs in parallel
print("\n=== Batch ===")
responses = llm_oai.batch([
    [HumanMessage(content="What is RAG?")],
    [HumanMessage(content="What is a vector database?")],
])
for r in responses:
    print(r.content[:80])