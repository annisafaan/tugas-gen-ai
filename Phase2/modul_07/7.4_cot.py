from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?

Think through this step by step before giving the final answer."""

ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX

Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

for label, prompt in [("Direct", DIRECT_PROMPT), ("CoT", COT_PROMPT), ("Zero-shot CoT", ZERO_SHOT_COT)]:
    resp = client.chat.completions.create(
        model="qwen2.5:3b",
        max_tokens=1024,  # Ditingkatkan agar tidak terpotong
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"==={label}===")
    print(resp.choices[0].message.content) # Tampilkan penuh tanpa dipotong [:300] agar terlihat hasilnya
    print()