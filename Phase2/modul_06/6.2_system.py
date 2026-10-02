from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen2.5:3b",
    max_tokens=512,
    messages=[
        
        {"role": "system", "content": "You are a concise technical writer. Answer in plain English, no jargon."},
        {"role": "user", "content": "Explain what a vector database does."}
    ]
)

print(response.choices[0].message.content)