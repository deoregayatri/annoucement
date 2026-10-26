#!/usr/bin/env python3
"""
download_sftp_pkey.py
---------------------
Standalone script to download NEAPS PDF files from SFTP server
using SSH Private Key (pkey_path) and optional Passphrase (pkey_passphrase).

Usage:
    # Download all files for a company:
    python download_sftp_pkey.py --company RELIANCE

    # Download a specific file:
    python download_sftp_pkey.py --file 01012024_RELIANCE.pdf --company RELIANCE

    # Override connection details via CLI if needed:
    python download_sftp_pkey.py --company RELIANCE \
        --pkey /path/to/id_rsa \
        --passphrase secret123 \
        --host 172.17.222.97 \
        --user vmuser \
        --remote-base /home/vmuser/neapsdatafiles/file/website \
        --local-dir /mnt/Announcement/neaps_document
"""

import os
import sys
import argparse
import paramiko
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Try loading from config if available, fallback to os.getenv
try:
    from config import settings
    DEFAULT_HOST = settings.neaps_sftp_host
    DEFAULT_PORT = int(settings.neaps_sftp_port)
    DEFAULT_USER = settings.neaps_sftp_user
    DEFAULT_PKEY = getattr(settings, "neaps_sftp_pkey_path", None) or os.getenv("NEAPS_SFTP_PKEY_PATH", "")
    DEFAULT_PASSPHRASE = getattr(settings, "neaps_sftp_pkey_passphrase", None) or os.getenv("NEAPS_SFTP_PKEY_PASSPHRASE", None)
    DEFAULT_REMOTE_BASE = settings.neaps_sftp_remote_base
    DEFAULT_LOCAL_DIR = settings.neaps_local_storage
except Exception:
    DEFAULT_HOST = os.getenv("NEAPS_SFTP_HOST", "172.17.222.97")
    DEFAULT_PORT = int(os.getenv("NEAPS_SFTP_PORT", "22"))
    DEFAULT_USER = os.getenv("NEAPS_SFTP_USER", "vmuser")
    DEFAULT_PKEY = os.getenv("NEAPS_SFTP_PKEY_PATH", "")
    DEFAULT_PASSPHRASE = os.getenv("NEAPS_SFTP_PKEY_PASSPHRASE", None)
    DEFAULT_REMOTE_BASE = os.getenv("NEAPS_SFTP_REMOTE_BASE", "/home/vmuser/neapsdatafiles/file/website")
    DEFAULT_LOCAL_DIR = os.getenv("NEAPS_LOCAL_STORAGE", "/mnt/Announcement/neaps_document")


def load_ssh_private_key(pkey_path: str, passphrase: str = None) -> paramiko.PKey:
    """Load SSH private key supporting RSA, ECDSA, Ed25519, and DSS formats."""
    expanded_path = Path(pkey_path).expanduser().resolve()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"SSH private key file does not exist: {expanded_path}")

    last_error = None
    for key_cls_name in ("RSAKey", "ECDSAKey", "Ed25519Key", "DSSKey"):
        key_cls = getattr(paramiko, key_cls_name, None)
        if not key_cls:
            continue
        try:
            key = key_cls.from_private_key_file(str(expanded_path), password=passphrase)
            return key
        except paramiko.PasswordRequiredException:
            raise ValueError(f"Private key {expanded_path} is encrypted, but no valid passphrase was provided.")
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to parse private key from '{expanded_path}'. "
        f"Ensure key is in OpenSSH/PEM format. Last error: {last_error}"
    )


def download_files(args):
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_base = args.remote_base.rstrip("/") + "/"
    company = args.company.upper() if args.company else ""
    remote_company_dir = f"{remote_base}{company}" if company else remote_base

    print("=" * 65)
    print(" NEAPS SFTP PKEY DOWNLOADER")
    print("=" * 65)
    print(f" SFTP Host         : {args.host}:{args.port}")
    print(f" SFTP User         : {args.user}")
    print(f" PKey Path         : {args.pkey}")
    print(f" Passphrase        : {'***' if args.passphrase else 'None'}")
    print(f" Remote Directory  : {remote_company_dir}")
    print(f" Local Target Dir  : {local_dir}")
    if args.file:
        print(f" Target File       : {args.file}")
    print("=" * 65)

    # 1. Load private key
    if not args.pkey:
        print("[ERROR] Please provide --pkey or set NEAPS_SFTP_PKEY_PATH in .env")
        sys.exit(1)

    print("\n[1/3] Loading SSH Private Key...")
    pkey = load_ssh_private_key(args.pkey, args.passphrase)
    print(f"      [OK] Loaded key: {pkey.get_name()}")

    # 2. Connect via Transport
    print("[2/3] Connecting to SFTP server...")
    transport = paramiko.Transport((args.host, args.port))
    transport.connect(username=args.user, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print("      [OK] SFTP connection established.")

    # 3. Discover and Download
    print("[3/3] Scanning and downloading files...\n")
    downloaded = 0
    cached = 0
    failed = 0

    try:
        # Determine list of files to download
        if args.file:
            # Single file requested
            target_remote_path = f"{remote_company_dir}/{args.file}" if company else f"{remote_base}{args.file}"
            remote_files = [target_remote_path]
        else:
            # List directory for company
            try:
                entries = sftp.listdir(remote_company_dir)
                remote_files = [f"{remote_company_dir}/{e}" for e in entries if e.lower().endswith(".pdf")]
            except IOError:
                # If company folder doesn't exist, try base directory
                print(f"[WARN] Directory '{remote_company_dir}' not found. Trying '{remote_base}'...")
                entries = sftp.listdir(remote_base)
                remote_files = [f"{remote_base}{e}" for e in entries if e.lower().endswith(".pdf")]

        print(f"Found {len(remote_files)} file(s) on remote server.\n")

        for idx, remote_path in enumerate(remote_files, 1):
            filename = Path(remote_path).name
            local_path = local_dir / filename

            # Check if file already exists locally
            if local_path.exists() and local_path.stat().st_size > 0:
                print(f"[{idx}/{len(remote_files)}] [ALREADY LOCAL] {filename} ({local_path.stat().st_size} bytes)")
                cached += 1
                continue

            # Download file
            print(f"[{idx}/{len(remote_files)}] [DOWNLOADING]   {filename} ...", end=" ", flush=True)
            try:
                sftp.get(remote_path, str(local_path))
                if local_path.exists() and local_path.stat().st_size > 0:
                    print(f"DONE ({local_path.stat().st_size} bytes)")
                    downloaded += 1
                else:
                    print("FAILED (0 bytes)")
                    failed += 1
            except Exception as dl_exc:
                print(f"FAILED ({dl_exc})")
                failed += 1

    finally:
        sftp.close()
        transport.close()

    print("\n" + "=" * 65)
    print(f" SUMMARY:")
    print(f"   Total Remote Files : {len(remote_files)}")
    print(f"   Downloaded Now     : {downloaded}")
    print(f"   Already on Disk    : {cached}")
    print(f"   Failed             : {failed}")
    print(f"   Total Ready Local  : {downloaded + cached}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NEAPS PDF files from SFTP using SSH Private Key.")
    parser.add_argument("--company", "-c", default="RELIANCE", help="Company symbol (e.g. RELIANCE, AESTHETIK)")
    parser.add_argument("--file", "-f", default=None, help="Specific filename to download")
    parser.add_argument("--pkey", default=DEFAULT_PKEY, help="Path to SSH private key file")
    parser.add_argument("--passphrase", default=DEFAULT_PASSPHRASE, help="Passphrase for SSH private key")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SFTP server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="SFTP server port")
    parser.add_argument("--user", default=DEFAULT_USER, help="SFTP username")
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE, help="SFTP remote base directory")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Local destination directory")

    args = parser.parse_args()
    download_files(args)
