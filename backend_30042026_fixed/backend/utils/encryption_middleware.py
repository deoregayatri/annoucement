# import base64
# import json
# import os
# import asyncio
# import logging
# from cryptography.hazmat.primitives.ciphers.aead import AESGCM
# from fastapi.responses import JSONResponse

# # --- Configuration ---
# # 32-byte AES-GCM Key (Ensure this matches your client/frontend)
# # You might want to load this from environment variables in a real project
# ENCRYPTION_KEY = b'U8eR0yD12ak9X!wQ@r5tZ?bY$4L3vP2h'

# # Internal logger
# logger = logging.getLogger("server")

# # --- Helpers ---

# def encrypt_aes(plaintext: str) -> str:
#     """Encrypts a string using AES-GCM and returns a URL-safe base64 string."""
#     try:
#         aesgcm = AESGCM(ENCRYPTION_KEY)
#         iv = os.urandom(12)
#         cipher_text = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
#         encrypted_data = iv + cipher_text
#         return base64.urlsafe_b64encode(encrypted_data).rstrip(b"=").decode("utf-8")
#     except Exception as e:
#         logger.error(f"Encryption failed: {e}")
#         raise

# def decrypt_aes(encrypted: str) -> str:
#     """Decrypts a URL-safe base64 string using AES-GCM."""
#     try:
#         # Fix base64 padding/formatting if needed
#         base64_encoded = encrypted.replace('-', '+').replace('_', '/')
#         padding = len(base64_encoded) % 4
#         if padding:
#             base64_encoded += "=" * (4 - padding)

#         encrypted_data = base64.b64decode(base64_encoded)

#         iv = encrypted_data[:12]
#         cipher_text = encrypted_data[12:]

#         aesgcm = AESGCM(ENCRYPTION_KEY)
#         plaintext = aesgcm.decrypt(iv, cipher_text, None)
#         return plaintext.decode('utf-8')
#     except Exception as e:
#         logger.error(f"Decryption failed: {e}")
#         raise

# # --- Middleware ---

# class EncryptionMiddleware:
#     """
#     ASGI Middleware that:
#     1. Decrypts incoming request bodies (for POST/PUT/PATCH).
#     2. Encrypts outgoing response bodies.
#     """
#     def __init__(self, app):
#         self.app = app

#     async def __call__(self, scope, receive, send):
#         if scope["type"] != "http":
#             await self.app(scope, receive, send)
#             return

#         # ------------------------------------------------------------------
#         # 1. REQUEST DECRYPTION LOGIC (Only for POST/PUT/PATCH)
#         # ------------------------------------------------------------------
#         if scope["method"] in ("POST", "PUT", "PATCH"):
#             # Read the entire body
#             body_chunks = []
#             more_body = True
#             try:
#                 while more_body:
#                     msg = await receive()
#                     body_chunks.append(msg.get("body", b""))
#                     more_body = msg.get("more_body", False)
#             except Exception as e:
#                  logger.error(f"Error reading request body in EncryptionMiddleware: {e}")
#                  await self.app(scope, receive, send)
#                  return

#             full_body = b"".join(body_chunks)
           
#             # Decrypt if body exists
#             if full_body:
#                 try:
#                     # logger.info(f"Processing {len(full_body)} bytes for decryption")
#                     encrypted_payload = full_body.decode("utf-8")
                   
#                     decrypted_text = decrypt_aes(encrypted_payload)
                   
#                     # Validate JSON
#                     decrypted_obj = json.loads(decrypted_text)
#                     if isinstance(decrypted_obj, str):
#                         decrypted_obj = json.loads(decrypted_obj)

#                     new_body_bytes = json.dumps(decrypted_obj).encode("utf-8")

#                     # Update Request Headers (Content-Length / Content-Type)
#                     new_headers = []
#                     for k, v in scope.get("headers", []):
#                         key_lower = k.lower()
#                         if key_lower == b"content-length":
#                             continue
#                         if key_lower == b"content-type":
#                             continue
#                         new_headers.append((k, v))
                   
#                     new_headers.append((b"content-length", str(len(new_body_bytes)).encode()))
#                     new_headers.append((b"content-type", b"application/json"))
                   
#                     scope["headers"] = new_headers
                   
#                     # Define stateful receive for the decrypted body
#                     state = {"consumed": False}

#                     async def new_receive():
#                         if not state["consumed"]:
#                             state["consumed"] = True
#                             return {"type": "http.request", "body": new_body_bytes, "more_body": False}
#                         else:
#                             # Simulate end of stream by waiting forever
#                             await asyncio.Future()

#                     # Replace receive with our decrypted receiver
#                     receive = new_receive

#                 except Exception as e:
#                     logger.error(f"Request Decryption Error: {str(e)}")
#                     response = JSONResponse(
#                         status_code=400,
#                         content={"error": f"Invalid encrypted payload: {str(e)}"},
#                     )
#                     await response(scope, receive, send)
#                     return
       
#         # ------------------------------------------------------------------
#         # 2. RESPONSE ENCRYPTION LOGIC
#         # ------------------------------------------------------------------
       
#         response_body_chunks = []
#         initial_message = {}

#         async def new_send(message):
#             nonlocal initial_message
           
#             if message["type"] == "http.response.start":
#                 # Capture start message, don't send yet
#                 initial_message = message
               
#             elif message["type"] == "http.response.body":
#                 response_body_chunks.append(message.get("body", b""))
               
#                 if not message.get("more_body", False):
#                     # Full body received
#                     full_response_body = b"".join(response_body_chunks)
                   
#                     if full_response_body:
#                         try:
#                             body_str = full_response_body.decode("utf-8")
                           
#                             encrypted_body_str = encrypt_aes(body_str)
#                             encrypted_body_bytes = encrypted_body_str.encode("utf-8")
                           
#                             # Update Content-Length
#                             headers = []
#                             for k, v in initial_message.get("headers", []):
#                                 if k.lower() == b"content-length":
#                                     continue
#                                 headers.append((k, v))
                           
#                             headers.append((b"content-length", str(len(encrypted_body_bytes)).encode()))
#                             initial_message["headers"] = headers
                           
#                             # Send response
#                             await send(initial_message)
#                             await send({
#                                 "type": "http.response.body",
#                                 "body": encrypted_body_bytes,
#                                 "more_body": False
#                             })
                           
#                         except Exception as e:
#                             logger.error(f"Response Encryption Failed: {e}")
#                             # Fallback: send original or error?
#                             # If we allow original, the client (expecting enc) might break.
#                             # For now, we log and suppress to avoid crashing, but client will timeout.
#                             pass
#                     else:
#                         # Empty body, just pass through
#                         await send(initial_message)
#                         await send(message)

#         await self.app(scope, receive, new_send)










import base64
import json
import os
import asyncio
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.responses import JSONResponse

# --- Configuration ---
ENCRYPTION_KEY = b'U8eR0yD12ak9X!wQ@r5tZ?bY$4L3vP2h'

# Internal logger
logger = logging.getLogger("server")

# --- Helpers ---

def encrypt_aes(plaintext: str) -> str:
    """Encrypts a string using AES-GCM and returns a URL-safe base64 string."""
    try:
        aesgcm = AESGCM(ENCRYPTION_KEY)
        iv = os.urandom(12)
        cipher_text = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
        encrypted_data = iv + cipher_text
        return base64.urlsafe_b64encode(encrypted_data).rstrip(b"=").decode("utf-8")
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise

def decrypt_aes(encrypted: str) -> str:
    """Decrypts a URL-safe base64 string using AES-GCM."""
    try:
        # Fix base64 padding/formatting if needed
        base64_encoded = encrypted.replace('-', '+').replace('_', '/')
        padding = len(base64_encoded) % 4
        if padding:
            base64_encoded += "=" * (4 - padding)

        encrypted_data = base64.b64decode(base64_encoded)

        iv = encrypted_data[:12]
        cipher_text = encrypted_data[12:]

        aesgcm = AESGCM(ENCRYPTION_KEY)
        plaintext = aesgcm.decrypt(iv, cipher_text, None)
        return plaintext.decode('utf-8')
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise

# --- Middleware ---

class EncryptionMiddleware:
    """
    ASGI Middleware that:
    1. Decrypts incoming request bodies (for POST/PUT/PATCH).
    2. Encrypts outgoing response bodies.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # ------------------------------------------------------------------
        # REQUEST DECRYPTION LOGIC (Only for POST/PUT/PATCH)
        # ------------------------------------------------------------------
        if scope["method"] in ("POST", "PUT", "PATCH"):
            # Read the entire body
            body_chunks = []
            more_body = True
            try:
                while more_body:
                    msg = await receive()
                    body_chunks.append(msg.get("body", b""))
                    more_body = msg.get("more_body", False)
            except Exception as e:
                 logger.error(f"Error reading request body in EncryptionMiddleware: {e}")
                 await self.app(scope, receive, send)
                 return

            full_body = b"".join(body_chunks)

            logger.info(f"full body {full_body}" )
           
            # Decrypt if body exists
            if full_body:
                try:
                    # logger.info(f"Processing {len(full_body)} bytes for decryption")
                    encrypted_payload = full_body.decode("utf-8").strip('"')
                   
                    decrypted_text = decrypt_aes(encrypted_payload)
                    # logger.info(f"encrypted text from api {decrypted_text}" )
                   
                    # Validate JSON
                    decrypted_obj = json.loads(decrypted_text)
                    if isinstance(decrypted_obj, str):
                        decrypted_obj = json.loads(decrypted_obj)

                    new_body_bytes = json.dumps(decrypted_obj).encode("utf-8")

                    # Update Request Headers (Content-Length / Content-Type)
                    new_headers = []
                    for k, v in scope.get("headers", []):
                        key_lower = k.lower()
                        if key_lower == b"content-length":
                            continue
                        if key_lower == b"content-type":
                            continue
                        new_headers.append((k, v))
                   
                    new_headers.append((b"content-length", str(len(new_body_bytes)).encode()))
                    new_headers.append((b"content-type", b"application/json"))
                   
                    scope["headers"] = new_headers
                   
                    # Define stateful receive for the decrypted body
                    state = {"consumed": False}

                    async def new_receive():
                        if not state["consumed"]:
                            state["consumed"] = True
                            return {"type": "http.request", "body": new_body_bytes, "more_body": False}
                        else:
                            # Simulate end of stream by waiting forever
                            await asyncio.Future()

                    # Replace receive with our decrypted receiver
                    receive = new_receive

                except Exception as e:
                    logger.error(f"Request Decryption Error: {str(e)}")
                    response = JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid encrypted payload: {str(e)}"},
                    )
                    await response(scope, receive, send)
                    return
       
        # ------------------------------------------------------------------
        # RESPONSE ENCRYPTION LOGIC
        # ------------------------------------------------------------------
       
        response_body_chunks = []
        initial_message = {}
        is_streaming = False
        response_started = False

        async def new_send(message):
            nonlocal initial_message, is_streaming, response_started
           
            if message["type"] == "http.response.start":
                # Check for streaming content type
                headers = message.get("headers", [])
                for k, v in headers:
                    if k.lower() == b"content-type" and b"text/event-stream" in v.lower():
                        is_streaming = True
                        break
               
                if is_streaming:
                    # Pass through immediately
                    await send(message)
                    response_started = True
                else:
                    # Buffer for encryption
                    initial_message = message
               
            elif message["type"] == "http.response.body":
                if is_streaming:
                    # Pass through immediately
                    await send(message)
                else:
                    # Buffer
                    response_body_chunks.append(message.get("body", b""))
                   
                    if not message.get("more_body", False):
                        # Full body received
                        full_response_body = b"".join(response_body_chunks)
                       
                        if full_response_body:
                            try:
                                body_str = full_response_body.decode("utf-8")
                               
                                encrypted_body_str = encrypt_aes(body_str)
                                encrypted_body_bytes = encrypted_body_str.encode("utf-8")
                               
                                # Update Content-Length
                                headers = []
                                for k, v in initial_message.get("headers", []):
                                    if k.lower() == b"content-length":
                                        continue
                                    headers.append((k, v))
                               
                                headers.append((b"content-length", str(len(encrypted_body_bytes)).encode()))
                                initial_message["headers"] = headers
                               
                                # Send response
                                await send(initial_message)
                                await send({
                                    "type": "http.response.body",
                                    "body": encrypted_body_bytes,
                                    "more_body": False
                                })
                               
                            except Exception as e:
                                logger.error(f"Response Encryption Failed: {e}")
                                # Fallback: try to send original
                                await send(initial_message)
                                await send({
                                    "type": "http.response.body",
                                    "body": full_response_body,
                                    "more_body": False
                                })
                        else:
                            # Empty body
                            await send(initial_message)
                            await send(message)

        await self.app(scope, receive, new_send)
