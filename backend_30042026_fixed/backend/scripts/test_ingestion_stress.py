import requests
import json
import time
import argparse
import random
from datetime import datetime, timedelta

def generate_synthetic_announcements(count: int, num_companies: int = 50) -> list:
    """Generate a list of synthetic JSON announcements."""
    companies = [f"COMP{i:03d}" for i in range(1, num_companies + 1)]
    categories = ["Financial Results", "Board Meeting", "Resignation", "Acquisition", "Dividend", "AGM"]
    
    announcements = []
    base_date = datetime.now() - timedelta(days=30)
    
    print(f"Generating {count} synthetic announcements...")
    
    for i in range(count):
        company = random.choice(companies)
        category = random.choice(categories)
        date = base_date + timedelta(hours=random.randint(1, 720))
        
        announcement = {
            "symbol": company,
            "companyName": f"{company} Technologies Limited",
            "subject": f"{category} - Official Announcement ({i})",
            "details": f"This is a detailed synthetic announcement regarding {category.lower()} for {company}. "
                       f"The board met today to discuss the strategic direction and approve the resolutions "
                       f"related to financial performance and compliance. Expected impact is positive. "
                       f"ID: {i}-{int(time.time())}.",
            "broadcastdate": date.strftime("%d-%b-%Y %H:%M"),
            "attachment": "" if random.random() > 0.5 else f"https://example.com/doc_{i}.pdf",
            "pdf_url": "" if random.random() > 0.5 else f"https://example.com/doc_{i}.pdf",
        }
        announcements.append(announcement)
    
    return announcements

def run_ingestion_stress_test(url: str, num_announcements: int):
    """Send synthetic announcements to the JSON ingestion endpoint."""
    print("=" * 50)
    print("INGESTION PIPELINE STRESS TEST")
    print("=" * 50)
    print(f"Target Endpoint: {url}")
    
    payload = generate_synthetic_announcements(num_announcements)
    
    print(f"\nSending payload of {len(payload)} JSON records to API...")
    start_time = time.time()
    
    try:
        response = requests.post(
            url, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # The endpoint should respond quickly and process async
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("\n[OK] API Accepted Request Successfully!")
            print(f"Response Time: {latency:.2f} seconds")
            print("\nResponse Details:")
            print(json.dumps(data, indent=2))
            
            print("\nNOTE: The API accepted the payload. You should now monitor:")
            print("1. Your backend console output (WorkerManager logs)")
            print("2. Your CPU usage (Task Manager / top) to verify ProcessPoolExecutor is working")
            print("3. Qdrant memory usage to verify steady indexing")
        else:
            print(f"\n[FAILED] API Error (Status {response.status_code})")
            print(f"Response Time: {latency:.2f} seconds")
            print(response.text)
            
    except Exception as e:
        print(f"\n[FAILED] Request Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test JSON ingestion pipeline")
    parser.add_argument("--url", type=str, default="http://localhost:8092/api/v1/neaps/ingest-json", help="Target API URL")
    parser.add_argument("-n", "--count", type=int, default=1000, help="Number of announcements to generate and ingest")
    
    args = parser.parse_args()
    
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("Error: requests library is required. Install it using: pip install requests")
        exit(1)
        
    run_ingestion_stress_test(args.url, args.count)
