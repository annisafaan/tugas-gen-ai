from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


stream = client.chat.completions.create(
    model="qwen2.5:3b",
    max_tokens=512,
    messages=[
        {"role": "user", "content": "List 5 use cases for vector databases."}
    ],
    stream=True
)

for chunk in stream:
    
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print() 