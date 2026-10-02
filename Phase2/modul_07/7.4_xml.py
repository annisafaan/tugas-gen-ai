from openai import OpenAI
import os, re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Ditambahkan sedikit contoh (few-shot) agar Qwen disiplin menggunakan tag XML
SYSTEM = """Solve problems using this exact format:

<thinking>
Step-by-step reasoning here.
</thinking>

<answer>
The final answer only, no reasoning.
</answer>

Example:
User: "If a box has 2 apples and you add 3, how many?"
<thinking>
Start with 2 apples. Add 3 apples. 2 + 3 = 5.
</thinking>
<answer>
5
</answer>"""

resp = client.chat.completions.create(
    model="qwen2.5:3b",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?"}
    ]
)

text = resp.choices[0].message.content

# Eksekusi regex
thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)

print("Reasoning:", thinking.group(1).strip() if thinking else f"not found\nRaw Output:\n{text}")
print("Answer:   ", answer.group(1).strip() if answer else "not found")