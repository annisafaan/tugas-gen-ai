from openai import OpenAI
import os, base64
import urllib.request
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Option A: URL (Gambar di-download dulu oleh Python lalu diubah ke base64 agar Ollama bisa baca)
def describe_image_url(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = response.read()
    
    b64 = base64.b64encode(data).decode('utf-8')
    media_type = "image/jpeg"

    response = client.chat.completions.create(
        model="qwen2.5vl:3b",  
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content


def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode('utf-8')
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"

    response = client.chat.completions.create(
        model="qwen2.5vl:3b",  
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content


print("--- Deskripsi dari URL ---")
try:
    url_result = describe_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg")
    print(url_result)
except Exception as e:
    print(f"Error URL: {e}")

# Eksekusi File Lokal
print("\n--- Deskripsi dari File Lokal ---")
try:
    
    file_result = describe_image_file("contoh_gambar.jpg")
    print(file_result)
except Exception as e:
    print(f"Error membaca file lokal: {e}")