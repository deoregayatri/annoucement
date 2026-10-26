"""
sync_and_verify.py
Downloads company PDFs using SSH Private Key Authentication (with optional Passphrase)
and settings from config.
"""
import os
import sys
from pathlib import Path
import paramiko
from config import settings

COMPANY_SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "RELIANCE"

def load_private_key(pkey_path_str: str, passphrase: str = None):
    """Load private key supporting RSA, ECDSA, Ed25519, DSS."""
    pkey_path = Path(pkey_path_str).expanduser()
    if not pkey_path.is_file():
        raise FileNotFoundError(f"SFTP private key not found: {pkey_path}")

    last_exc = None
    for key_cls_name in ("RSAKey", "ECDSAKey", "Ed25519Key", "DSSKey"):
        key_cls = getattr(paramiko, key_cls_name, None)
        if key_cls is None:
            continue
        try:
            return key_cls.from_private_key_file(str(pkey_path), password=passphrase)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Could not parse private key from {pkey_path}: {last_exc}")

def main():
    local_dir = Path(settings.neaps_local_storage)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_base = settings.neaps_sftp_remote_base.rstrip("/") + "/"
    remote_company_dir = f"{remote_base}{COMPANY_SYMBOL}"
    pkey_path = getattr(settings, "neaps_sftp_pkey_path", None)
    pkey_passphrase = getattr(settings, "neaps_sftp_pkey_passphrase", None) or None

    print("==================================================")
    print(f"[CONFIG] Local Storage Dir : {local_dir}")
    print(f"[CONFIG] Remote Base       : {remote_base}")
    print(f"[CONFIG] Remote Company Dir: {remote_company_dir}")
    print(f"[CONFIG] SFTP Host         : {settings.neaps_sftp_host}:{settings.neaps_sftp_port}")
    print(f"[CONFIG] SFTP User         : {settings.neaps_sftp_user}")
    print(f"[CONFIG] SFTP PKey Path    : {pkey_path}")
    print(f"[CONFIG] PKey Passphrase   : {'***' if pkey_passphrase else 'None'}")
    print("==================================================")

    # 1. Establish SFTP Connection
    transport = paramiko.Transport((settings.neaps_sftp_host, int(settings.neaps_sftp_port)))

    if pkey_path and Path(pkey_path).expanduser().is_file():
        print(f"[AUTH] Authenticating with SSH Private Key: {pkey_path}")
        pkey = load_private_key(pkey_path, pkey_passphrase)
        transport.connect(username=settings.neaps_sftp_user, pkey=pkey)
    else:
        print(f"[AUTH] Authenticating with Password")
        transport.connect(
            username=settings.neaps_sftp_user,
            password=settings.neaps_sftp_password
        )

    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        # Find all files in remote company folder
        remote_files = []
        try:
            entries = sftp.listdir(remote_company_dir)
            remote_files = [f"{remote_company_dir}/{e}" for e in entries if e.lower().endswith(".pdf")]
        except IOError:
            print(f"[WARN] Direct company directory '{remote_company_dir}' not found. Scanning base folder...")
            try:
                entries = sftp.listdir(remote_base)
                remote_files = [f"{remote_base}{e}" for e in entries if e.lower().endswith(".pdf")]
            except IOError as e:
                print(f"[ERROR] Could not access SFTP remote base: {e}")
                return

        print(f"[INFO] Found {len(remote_files)} PDF(s) on SFTP for {COMPANY_SYMBOL}.\n")

        downloaded_count = 0
        existing_count = 0

        for idx, remote_path in enumerate(remote_files, 1):
            filename = Path(remote_path).name
            local_target = local_dir / filename

            if local_target.exists() and local_target.stat().st_size > 0:
                print(f"[{idx}/{len(remote_files)}] [EXISTS LOCALLY] {filename} ({local_target.stat().st_size} bytes)")
                existing_count += 1
                continue

            try:
                print(f"[{idx}/{len(remote_files)}] [DOWNLOADING] {filename} ...")
                sftp.get(remote_path, str(local_target))
                if local_target.exists() and local_target.stat().st_size > 0:
                    print(f"    -> [SUCCESS] Saved to {local_target} ({local_target.stat().st_size} bytes)")
                    downloaded_count += 1
                else:
                    print(f"    -> [ERROR] Downloaded file was 0 bytes!")
            except Exception as dl_err:
                print(f"    -> [FAILED] {filename}: {dl_err}")

        print("\n==================================================")
        print(f" Summary for {COMPANY_SYMBOL}:")
        print(f"   Total on SFTP: {len(remote_files)}")
        print(f"   Downloaded:    {downloaded_count}")
        print(f"   Already Local: {existing_count}")
        print(f"   Total Ready:   {downloaded_count + existing_count}")
        print("==================================================")

    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    main()
