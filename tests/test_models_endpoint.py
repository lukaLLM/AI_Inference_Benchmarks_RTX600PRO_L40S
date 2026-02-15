import requests
import time

# Wait for backend to be ready
print("Checking if backend is ready...")
for i in range(30):  # Wait up to 5 minutes
    try:
        response = requests.get("http://localhost:8000/health")
        health = response.json()
        
        # Check if generate endpoint is registered
        endpoints = [inst['endpoint'] for inst in health.get('instances', [])]
        print(f"Attempt {i+1}: Registered endpoints: {endpoints}")
        
        if 'generate' in endpoints or 'chat' in str(endpoints):
            print("✓ Backend is ready!")
            break
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(10)
else:
    print("✗ Backend not ready after 5 minutes")