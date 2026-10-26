import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the backend modules
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from ingestion.job_store import JobStore

async def test_oracle_connection():
    print("=" * 50)
    print("ORACLE DATABASE CONNECTION TEST")
    print("=" * 50)
    
    print(f"Attempting to connect to Oracle DB...")
    print(f"Host/DSN: {settings.oracle_dsn}")
    print(f"User:     {settings.oracle_user}")
    
    store = JobStore()
    pool = store._pool
    
    if pool is None:
        print("\n[FAILED] FAILED: JobStore initialized in NO-OP mode.")
        print("Could not establish a connection pool. Please check your .env credentials.")
        return
        
    print("\n[OK] OK: Connection pool established successfully.")
    
    # Try to execute a simple query
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    print("[OK] OK: Test query (SELECT 1 FROM DUAL) executed successfully.")
                else:
                    print("[FAILED] FAILED: Test query executed but returned unexpected result.")
    except Exception as e:
        print(f"[FAILED] FAILED: Error executing test query: {e}")
        
if __name__ == "__main__":
    try:
        import oracledb
    except ImportError:
        print("Error: oracledb library is required. Install it using: pip install oracledb")
        exit(1)
        
    asyncio.run(test_oracle_connection())
