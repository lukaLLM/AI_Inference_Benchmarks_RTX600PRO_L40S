import requests
import time
import json

print("=== Checking Dynamo TensorRT-LLM Health ===\n")

# 1. Check if frontend is up
print("1. Checking frontend health...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   Status: {response.status_code}")
    health_data = response.json()
    print(f"   Response: {json.dumps(health_data, indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 2. Check available models
print("\n2. Checking available models...")
try:
    response = requests.get("http://localhost:8000/v1/models", timeout=5)
    print(f"   Status: {response.status_code}")
    models = response.json()
    print(f"   Models: {json.dumps(models, indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Try a simple completion (non-streaming)
print("\n3. Testing completion endpoint...")
try:
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "Qwen/Qwen3-32B-FP8",
            "prompt": "Say hello",
            "max_tokens": 10,
            "stream": False
        },
        timeout=60
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success!")
        print(f"   Response: {result}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Try chat completion
print("\n4. Testing chat completion...")
try:
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-32B-FP8",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "stream": False
        },
        timeout=60
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success!")
        print(f"   Response: {result}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n=== Verification Complete ===")