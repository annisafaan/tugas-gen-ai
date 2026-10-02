from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()

# Mengarahkan client ke Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def extract_company_info(text: str) -> dict:
    resp = client.chat.completions.create(
        model="qwen2.5:3b",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    
    # Strip any accidental markdown fences (antisipasi jika model tetap menuliskan ```json)
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

for text in texts:
    info = extract_company_info(text)
    print(json.dumps(info, indent=2))
    print()