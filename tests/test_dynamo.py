from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required for local vLLM
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B-FP8", # 
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)

