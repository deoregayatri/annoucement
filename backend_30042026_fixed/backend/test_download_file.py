import requests
import json
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Same key as encryption_middleware.py
ENCRYPTION_KEY = b'U8eR0yD12ak9X!wQ@r5tZ?bY$4L3vP2h'

BASE_URL = "http://172.17.177.78:8081"
TOKEN = "YOUR_TOKEN_HERE"  # Replace with your actual bearer token

# ---- Filename to download ----
FILENAME = "your_file.pdf"  # Replace with the actual filename


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


def download_file(filename: str):
    url = f"{BASE_URL}/neaps/download/{filename}"
    print(f"Hitting: GET {url}\n")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
    }

    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    # Decrypt the response (middleware encrypts all responses)
    try:
        decrypted_text = decrypt_aes(response.text.strip().strip('"'))
        data = json.loads(decrypted_text)
    except Exception as e:
        # Fallback: maybe response is plain JSON (middleware disabled)
        print(f"Decryption failed ({e}), trying plain JSON...")
        data = response.json()

    file_name = data.get("file_name", filename)
    file_type = data.get("file_type", "bin")
    b64_data   = data.get("base64_data", "")

    if not b64_data:
        print("No file data in response.")
        return

    # Decode base64 and save to current directory
    file_bytes = base64.b64decode(b64_data)
    save_path = os.path.join(os.getcwd(), file_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    print(f"File saved: {save_path}  ({len(file_bytes):,} bytes)")


if __name__ == "__main__":
    download_file(FILENAME)
