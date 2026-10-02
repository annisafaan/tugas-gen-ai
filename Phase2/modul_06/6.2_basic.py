from openai import OpenAI

# Menggunakan format SDK OpenAI yang diarahkan ke Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

response = client.chat.completions.create(
    model="qwen2.5:3b",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)


print(response.choices[0].message.content)


print(f"\nInput tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")