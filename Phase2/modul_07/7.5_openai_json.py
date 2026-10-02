from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen2.5:3b",
    response_format={"type": "json_object"},  # enforces valid JSON output
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

result = json.loads(response.choices[0].message.content)
print(result)