import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

print("Setup berhasil, semua API Keys berhasil dimuat dari environment.")