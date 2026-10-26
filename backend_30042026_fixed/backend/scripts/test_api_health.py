import requests
import json
import time

API_BASE_URL = "http://localhost:8092/api/v1"

def check_endpoint(name: str, url: str, method: str = "GET", expected_status: int = 200):
    print(f"Testing {name} ({method} {url})... ", end="")
    try:
        start_time = time.time()
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json={}, timeout=10)
            
        latency = time.time() - start_time
        
        if response.status_code == expected_status or (method == "POST" and response.status_code in [400, 422]):
            # POST might return 400/422 if body is empty, which still means the endpoint is up
            print(f"[OK] ({latency:.3f}s)")
            return True, response.json() if response.status_code == 200 else None
        else:
            print(f"[FAILED] (Status: {response.status_code})")
            return False, None
    except requests.exceptions.RequestException as e:
        print(f"[FAILED] (Error: {e})")
        return False, None

def run_integration_tests():
    print("=" * 50)
    print("API HEALTH & INTEGRATION TEST SUITE")
    print("=" * 50)
    
    # 1. Health Check
    success, data = check_endpoint("Health Check", f"{API_BASE_URL}/health")
    if success and data:
        print("  System Status:", data.get("status"))
        print("  Services:")
        for service, status in data.get("services", {}).items():
            icon = "[OK]" if status else "[FAILED]"
            print(f"    {icon} {service}: {'UP' if status else 'DOWN'}")
            
    # 2. Stats
    check_endpoint("System Stats", f"{API_BASE_URL}/stats")
    
    # 3. Document Count
    check_endpoint("Document Count", f"{API_BASE_URL}/documents/count")
    
    # 4. Companies List
    check_endpoint("Companies List", f"{API_BASE_URL}/companies")
    
    print("=" * 50)
    print("Integration tests completed.")

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: requests library is required. Install it using: pip install requests")
        exit(1)
        
    run_integration_tests()
