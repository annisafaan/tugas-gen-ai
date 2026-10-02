from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    
    history = [{"role": "system", "content": system}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="qwen2.5:3b",
            max_tokens=1024,
            messages=history
        )

        assistant_text = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_text})

        print(f"Qwen: {assistant_text}\n")

chat(system="You are a helpful Python tutor.")