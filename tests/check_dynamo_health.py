import requests
import json

# Check frontend health
try:
    response = requests.get("http://localhost:8000/health")
    print("Frontend health:", response.json())
except Exception as e:
    print(f"Frontend error: {e}")

# Check if backend is registered
try:
    response = requests.get("http://localhost:8000/v1/models")
    print("\nRegistered models:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Models endpoint error: {e}")