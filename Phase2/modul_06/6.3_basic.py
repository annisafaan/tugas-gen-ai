from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

response = client.chat.completions.create(
    model="qwen2.5:3b", 
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)

print(response.choices[0].message.content)
print(f"\nTokens used:{response.usage.total_tokens}")