from openai import OpenAI
import os
from dotenv import load_dotenv

# Memuat file .env
load_dotenv()

# Mengarahkan client ke Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue -> Impact -> Fix."""

messages = [
    {
        "role": "user", 
        "content": """Review this function:

def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""
    }
]

# Mengirim system prompt dan pesan user ke model lokal
response = client.chat.completions.create(
    model="qwen2.5:3b",
    max_tokens=1024,
    messages=[{"role": "system", "content": STRONG_SYSTEM}] + messages
)

print(response.choices[0].message.content)