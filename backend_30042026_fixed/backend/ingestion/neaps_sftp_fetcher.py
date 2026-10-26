"""
neaps_sftp_fetcher.py
=====================
Downloads announcement documents from the NEAPS server via SFTP.

All connection parameters are read from .env via settings:
NEAPS_SFTP_HOST, NEAPS_SFTP_PORT, NEAPS_SFTP_USER, NEAPS_SFTP_PASSWORD
NEAPS_SFTP_REMOTE_BASE, NEAPS_LOCAL_STORAGE, NEAPS_SFTP_WORKERS

Features:
- Skip files that already exist locally (and have size > 0)
- Auto-retry up to 3 times on transient failures
- Parallel downloads via ThreadPoolExecutor
- Recursive fallback search when direct path returns Errno 2 / No such file
- Creates local storage directory if it does not exist
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 2  # seconds between retries


def _is_missing_file_error(exc: Exception) -> bool:
    """
    Return True if the exception means the file was not found on the SFTP server.

    Paramiko on Linux raises IOError(2, 'No such file') which does NOT always
    subclass FileNotFoundError in all Python/paramiko versions.
    We also check errno and the string representation as belt-and-suspenders.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, (IOError, OSError)):
        if getattr(exc, "errno", None) == 2:
            return True
        msg = str(exc).lower()
        if "no such file" in msg:
            return True
    return False


class NeapsSFTPFetcher:
    """
    SFTP-based file downloader for NEAPS documents.
    Each download opens its own paramiko Transport to avoid thread-safety issues.
    """

    def __init__(self) -> None:
        self.host = settings.neaps_sftp_host
        self.port = settings.neaps_sftp_port
        self.user = settings.neaps_sftp_user
        self.password = settings.neaps_sftp_password
        self.pkey_path = getattr(settings, "neaps_sftp_pkey_path", None)
        self.pkey_passphrase = getattr(settings, "neaps_sftp_pkey_passphrase", None) or None
        self.remote_base = settings.neaps_sftp_remote_base.rstrip("/") + "/"
        self.local_base = Path(settings.neaps_local_storage)
        self.workers = settings.neaps_sftp_workers

        # Ensure local storage directory exists
        self.local_base.mkdir(parents=True, exist_ok=True)
        auth_mode = "pkey" if self.pkey_path and Path(self.pkey_path).expanduser().is_file() else "password"
        logger.info(
            f"[NeapsSFTPFetcher] host={self.host}:{self.port} "
            f"auth={auth_mode} remote_base={self.remote_base} local={self.local_base}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sftp(self):
        """Open a new paramiko SFTP connection using private key or password."""
        import paramiko
        transport = paramiko.Transport((self.host, self.port))

        pkey_path = Path(self.pkey_path).expanduser() if self.pkey_path else None
        if pkey_path and pkey_path.is_file():
            pkey = None
            last_exc = None
            for key_cls_name in ("RSAKey", "ECDSAKey", "Ed25519Key", "DSSKey"):
                key_cls = getattr(paramiko, key_cls_name, None)
                if key_cls is None:
                    continue
                try:
                    pkey = key_cls.from_private_key_file(
                        str(pkey_path), password=self.pkey_passphrase
                    )
                    break
                except Exception as exc:
                    last_exc = exc

            if pkey is None:
                raise RuntimeError(
                    f"Could not load private key from {pkey_path}. Last error: {last_exc}"
                )
            transport.connect(username=self.user, pkey=pkey)
        else:
            transport.connect(username=self.user, password=self.password)

        sftp = paramiko.SFTPClient.from_transport(transport)
        return transport, sftp

    def _find_file_recursive(self, sftp, remote_dir: str, filename: str):
        """
        Recursively search for filename under remote_dir via SFTP.

        Returns the full SFTP path if found, None otherwise.
        Skips directories that cause UnicodeDecodeError (e.g. AXISBANK bad filenames)
        so one corrupt folder never blocks the entire search.

        This is called as a fallback when sftp.get(remote_base + filename) fails
        with Errno 2 / No such file — meaning the file lives in a subfolder.

        Example:
            file_ref   = "JASH_25082026001959_RESULTSEARCHABLEMODEJASH.pdf"
            remote_base= "/home/vmuser/neapsdatafiles/file/website/"
            direct     = ".../website/JASH_2508...pdf"  <- NOT FOUND
            recursive  = ".../website/JASH/JASH_2508...pdf"  <- FOUND
        """
        import stat as stat_mod

        try:
            entries = sftp.listdir_attr(remote_dir)
        except UnicodeDecodeError as exc:
            logger.warning(
                f"[SFTP] Encoding issue in directory '{remote_dir}': {exc} — skipping"
            )
            return None
        except Exception as exc:
            logger.warning(f"[SFTP] Cannot list directory '{remote_dir}': {exc}")
            return None

        for entry in entries:
            try:
                entry_path = remote_dir.rstrip("/") + "/" + entry.filename
            except UnicodeDecodeError as exc:
                logger.warning(
                    f"[SFTP] Skipping entry with undecodable filename in '{remote_dir}': {exc}"
                )
                continue

            if stat_mod.S_ISDIR(entry.st_mode):
                found = self._find_file_recursive(sftp, entry_path, filename)
                if found:
                    return found
            else:
                if entry.filename == filename:
                    logger.info(f"[SFTP] Found '{filename}' at: {entry_path}")
                    return entry_path

        return None

    def _download_one(self, file_ref: str) -> Path:
        """
        Download a single file from NEAPS to local storage.
        """
        if file_ref.startswith("/"):
            remote_path = file_ref
        else:
            remote_path = self.remote_base + file_ref

        filename = Path(remote_path).name
        local_path = self.local_base / filename

        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"[SFTP] Already exists locally, skipping: {filename}")
            return local_path

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()
                logger.info(f"[SFTP] Downloading ({attempt}/{MAX_RETRIES}): {remote_path}")
                sftp.get(remote_path, str(local_path))
                logger.info(f"[SFTP] Downloaded: {filename} -> {local_path}")
                return local_path
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[SFTP] Attempt {attempt} failed for '{filename}': {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
            finally:
                if sftp:
                    sftp.close()
                if transport:
                    transport.close()

        raise RuntimeError(
            f"[SFTP] Failed to download '{filename}' after {MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_file(self, file_ref: str) -> Path:
        """Download a single document from the NEAPS server."""
        return self._download_one(file_ref)

    def download_one_to(self, file_ref: str, dest_path: Path) -> Path:
        """
        Download a single file to an explicit destination path.
        Used by the /neaps/download/{filename} API endpoint.

        Strategy:
          1. Try direct path (remote_base + filename  OR  full abs path).
          2. If server returns Errno 2 / No such file  ->  recursive search fallback.
          3. Retry transient errors up to MAX_RETRIES; recursive search is one-shot.
        """
        if file_ref.startswith("/"):
            remote_path = file_ref
        else:
            remote_path = self.remote_base + file_ref

        filename = Path(remote_path).name
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()
                logger.info(
                    f"[SFTP] on-demand download ({attempt}/{MAX_RETRIES}): "
                    f"{remote_path} -> {dest_path}"
                )
                sftp.get(remote_path, str(dest_path))
                logger.info(f"[SFTP] on-demand download done: {filename}")
                return dest_path

            except Exception as exc:
                if _is_missing_file_error(exc):
                    # ── Recursive fallback ───────────────────────────────────
                    logger.info(
                        f"[SFTP] '{filename}' not at direct path "
                        f"(Errno 2 / No such file) — falling back to recursive "
                        f"search under {self.remote_base}"
                    )
                    found = self._find_file_recursive(sftp, self.remote_base, filename)
                    if found:
                        try:
                            sftp.get(found, str(dest_path))
                            logger.info(
                                f"[SFTP] Downloaded via recursive search: "
                                f"{filename} <- {found}"
                            )
                            return dest_path
                        except Exception as dl_exc:
                            last_exc = dl_exc
                            logger.warning(
                                f"[SFTP] Found '{found}' but download failed: {dl_exc}"
                            )
                    else:
                        last_exc = FileNotFoundError(
                            f"'{filename}' not found anywhere under {self.remote_base}"
                        )
                        logger.warning(
                            f"[SFTP] Recursive search found nothing for '{filename}'"
                        )
                    break  # recursive search is one-shot — stop retry loop

                else:
                    # Transient error (network/auth blip) — retry
                    last_exc = exc
                    logger.warning(
                        f"[SFTP] on-demand attempt {attempt} failed for '{filename}': {exc}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_S)

            finally:
                if sftp:
                    try:
                        sftp.close()
                    except Exception:
                        pass
                if transport:
                    try:
                        transport.close()
                    except Exception:
                        pass

        raise RuntimeError(
            f"[SFTP] Failed to download '{filename}' after {MAX_RETRIES} attempts: {last_exc}"
        )

    def download_batch(self, file_list: List[str]) -> Dict[str, Optional[Path]]:
        """
        Download multiple files in parallel using ThreadPoolExecutor.
        Each worker thread opens ONE persistent SFTP connection and downloads
        all its assigned files, reducing handshakes from N to W (workers).

        file_list entries can be:
          - Full absolute SFTP paths  ("/home/vmuser/.../JASH_...pdf")
          - Bare filenames            ("JASH_...pdf")
          - RECURSIVE: prefixed names ("RECURSIVE:JASH_...pdf")  <- ingestion uses this
        """
        results: Dict[str, Optional[Path]] = {}
        self._last_remote_paths: Dict[str, str] = {}

        if not file_list:
            return results

        logger.info(
            f"[SFTP] Starting batch download of {len(file_list)} files "
            f"({self.workers} parallel workers)"
        )

        # Distribute files across worker slots
        chunks: List[List[str]] = [[] for _ in range(self.workers)]
        for i, fname in enumerate(file_list):
            chunks[i % self.workers].append(fname)

        def _download_chunk(file_chunk: List[str]) -> Dict[str, Any]:
            """
            Download a sub-list of files over a single persistent SFTP connection.

            For each file:
              • RECURSIVE: prefix  -> skip direct attempt, go straight to recursive search.
              • Otherwise          -> try direct path; on Errno 2 fall back to recursive.

            Returns:
                {"local_results": {file_ref: Path|None}, "remote_paths": {file_ref: str}}
            """
            chunk_results: Dict[str, Optional[Path]] = {}
            remote_paths: Dict[str, str] = {}

            if not file_chunk:
                return {"local_results": chunk_results, "remote_paths": remote_paths}

            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()

                for file_ref in file_chunk:

                    # ── RECURSIVE-only (bare filename, no abs path known) ──────
                    if file_ref.startswith("RECURSIVE:"):
                        filename = file_ref[10:]
                        local_path = self.local_base / filename

                        if local_path.exists() and local_path.stat().st_size > 0:
                            logger.info(f"[LOCAL] Using cached local file: {filename}")
                            remote_paths[file_ref] = f"{self.remote_base}{filename}"
                            chunk_results[file_ref] = local_path
                            continue

                        logger.info(
                            f"[SFTP] Recursive-only search for '{filename}' "
                            f"under {self.remote_base}"
                        )
                        found = self._find_file_recursive(sftp, self.remote_base, filename)
                        if found:
                            try:
                                sftp.get(found, str(local_path))
                                logger.info(
                                    f"[SFTP] Downloaded via recursive search: "
                                    f"{filename} <- {found}"
                                )
                                remote_paths[file_ref] = found
                                chunk_results[file_ref] = local_path
                            except Exception as dl_exc:
                                logger.error(
                                    f"[SFTP] Found '{found}' but download failed: {dl_exc}"
                                )
                                remote_paths[file_ref] = found
                                chunk_results[file_ref] = None
                        else:
                            logger.warning(
                                f"[SFTP] Recursive search found nothing for '{filename}'"
                            )
                            remote_paths[file_ref] = f"{self.remote_base}{filename}"
                            chunk_results[file_ref] = None
                        continue

                    # ── Normal files (full abs path or relative) ──────────────
                    remote_path = (
                        file_ref if file_ref.startswith("/")
                        else self.remote_base + file_ref
                    )
                    filename = Path(remote_path).name
                    local_path = self.local_base / filename

                    # Skip if cached locally
                    if local_path.exists() and local_path.stat().st_size > 0:
                        logger.info(f"[LOCAL] Using cached local file: {filename}")
                        remote_paths[file_ref] = remote_path
                        chunk_results[file_ref] = local_path
                        continue

                    last_exc: Optional[Exception] = None
                    downloaded = False

                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            logger.info(
                                f"[SFTP] Downloading ({attempt}/{MAX_RETRIES}): {remote_path}"
                            )
                            sftp.get(remote_path, str(local_path))
                            logger.info(f"[SFTP] Downloaded: {filename} -> {local_path}")
                            remote_paths[file_ref] = remote_path
                            chunk_results[file_ref] = local_path
                            downloaded = True
                            last_exc = None
                            break

                        except Exception as exc:
                            if _is_missing_file_error(exc):
                                # ── Recursive fallback ─────────────────────────
                                logger.info(
                                    f"[SFTP] '{filename}' not at direct path "
                                    f"(Errno 2 / No such file) — falling back to "
                                    f"recursive search under {self.remote_base}"
                                )
                                found = self._find_file_recursive(
                                    sftp, self.remote_base, filename
                                )
                                if found:
                                    try:
                                        sftp.get(found, str(local_path))
                                        logger.info(
                                            f"[SFTP] Downloaded via recursive search: "
                                            f"{filename} <- {found}"
                                        )
                                        remote_paths[file_ref] = found
                                        chunk_results[file_ref] = local_path
                                        downloaded = True
                                        last_exc = None
                                    except Exception as dl_exc:
                                        last_exc = dl_exc
                                        logger.warning(
                                            f"[SFTP] Found '{found}' but download "
                                            f"failed: {dl_exc}"
                                        )
                                else:
                                    last_exc = FileNotFoundError(
                                        f"'{filename}' not found anywhere "
                                        f"under {self.remote_base}"
                                    )
                                    logger.warning(
                                        f"[SFTP] Recursive search found nothing "
                                        f"for '{filename}'"
                                    )
                                break  # recursive search is one-shot

                            else:
                                # Transient error — retry
                                last_exc = exc
                                logger.warning(
                                    f"[SFTP] Attempt {attempt} failed for '{filename}': {exc}"
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY_S)

                    if not downloaded:
                        logger.error(f"[SFTP] Failed '{filename}': {last_exc}")
                        remote_paths[file_ref] = remote_path
                        chunk_results[file_ref] = None

            except Exception as conn_exc:
                logger.error(
                    f"[SFTP] Connection failed for chunk of {len(file_chunk)} files: {conn_exc}"
                )
                for file_ref in file_chunk:
                    chunk_results.setdefault(file_ref, None)
            finally:
                if sftp:
                    try:
                        sftp.close()
                    except Exception:
                        pass
                if transport:
                    try:
                        transport.close()
                    except Exception:
                        pass

            return {"local_results": chunk_results, "remote_paths": remote_paths}

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_chunk = {
                executor.submit(_download_chunk, chunk): chunk
                for chunk in chunks
                if chunk
            }
            for future in as_completed(future_to_chunk):
                try:
                    chunk_result = future.result()
                    results.update(chunk_result["local_results"])
                    self._last_remote_paths.update(chunk_result["remote_paths"])
                except Exception as exc:
                    logger.error(f"[SFTP] Worker thread failed: {exc}")

        success = sum(1 for v in results.values() if v is not None)
        logger.info(
            f"[SFTP] Batch complete: {success}/{len(file_list)} files downloaded successfully."
        )
        return results

    def list_remote_files(self) -> List[str]:
        """List all files in the NEAPS remote base directory."""
        transport, sftp = self._make_sftp()
        try:
            files = sftp.listdir(self.remote_base)
            return files
        except Exception as exc:
            logger.error(f"[SFTP] list_remote_files failed: {exc}")
            return []
        finally:
            sftp.close()
            transport.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_fetcher: Optional[NeapsSFTPFetcher] = None


def get_neaps_sftp_fetcher() -> NeapsSFTPFetcher:
    """Get or create the SFTP fetcher singleton."""
    global _fetcher
    if _fetcher is None:
        _fetcher = NeapsSFTPFetcher()
    return _fetcher
