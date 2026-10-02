from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv()

# Adaptasi base_url ke Ollama agar bisa dieksekusi secara lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"]
            }
        }
    }
]

def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})

messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

response = client.chat.completions.create(
    model="qwen2.5:3b",  
    tools=tools,
    messages=messages
)

if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = get_model_info(**args)
    
    print(f"Tool called: {name}({args})")
    print(f"Tool result: {result}\n")

    
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    final = client.chat.completions.create(model="qwen2.5:3b", messages=messages)
    print("Final answer:")
    print(final.choices[0].message.content)