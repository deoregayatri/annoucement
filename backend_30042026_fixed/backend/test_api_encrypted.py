import requests
import json
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Same key as encryption_middleware.py
ENCRYPTION_KEY = b'U8eR0yD12ak9X!wQ@r5tZ?bY$4L3vP2h'

URL = "http://172.17.177.78:8081/api/v1/search"
TOKEN = "YOUR_TOKEN_HERE"  # Replace with your actual bearer token

def encrypt_aes(plaintext: str) -> str:
    aesgcm = AESGCM(ENCRYPTION_KEY)
    iv = os.urandom(12)
    cipher_text = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    encrypted_data = iv + cipher_text
    return base64.urlsafe_b64encode(encrypted_data).rstrip(b"=").decode("utf-8")

def decrypt_aes(encrypted: str) -> str:
    base64_encoded = encrypted.replace('-', '+').replace('_', '/')
    padding = len(base64_encoded) % 4
    if padding:
        base64_encoded += "=" * (4 - padding)
    encrypted_data = base64.b64decode(base64_encoded)
    iv = encrypted_data[:12]
    cipher_text = encrypted_data[12:]
    aesgcm = AESGCM(ENCRYPTION_KEY)
    return aesgcm.decrypt(iv, cipher_text, None).decode('utf-8')

# Build and encrypt payload
payload = {
    "query": "total operation cost",
    "companies": ["JASH"],
    "top_k": 5
}

encrypted_body = encrypt_aes(json.dumps(payload))
print(f"Sending encrypted payload to: {URL}\n")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "text/plain"
}

response = requests.post(URL, data=f'"{encrypted_body}"', headers=headers)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    try:
        decrypted = decrypt_aes(response.text.strip().strip('"'))
        print(json.dumps(json.loads(decrypted), indent=2))
    except Exception:
        # Response might not be encrypted in some setups
        print(response.text)
else:
    print(response.text)
