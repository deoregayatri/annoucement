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
  - Recursive remote directory search for bare filenames
  - Creates local storage directory if it does not exist
"""

import logging
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 2  # seconds between retries


class NeapsSFTPFetcher:
    """
    SFTP-based file downloader for NEAPS documents.
    Each download opens its own paramiko Transport to avoid thread-safety issues.
    """

    def __init__(self) -> None:
        self.host = settings.neaps_sftp_host
        self.port = settings.neaps_sftp_port
        self.user = settings.neaps_sftp_user
        self.pkey_path = settings.neaps_sftp_pkey_path
        self.pkey_passphrase = settings.neaps_sftp_pkey_passphrase or None
        self.remote_base = settings.neaps_sftp_remote_base.rstrip("/") + "/"
        self.local_base = Path(settings.neaps_local_storage)
        self.workers = settings.neaps_sftp_workers

        # Ensure local storage directory exists
        self.local_base.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[NeapsSFTPFetcher] host={self.host}:{self.port}  "
            f"remote_base={self.remote_base}  local={self.local_base}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sftp(self):
        """Open a new paramiko SFTP connection using private key auth."""
        import paramiko

        transport = paramiko.Transport((self.host, self.port))

        pkey_path = Path(self.pkey_path).expanduser()
        if not pkey_path.is_file():
            raise FileNotFoundError(f"SFTP private key not found: {pkey_path}")

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
                f"Could not load private key from {pkey_path}. "
                f"Last error: {last_exc}"
            )

        transport.connect(username=self.user, pkey=pkey)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return transport, sftp

    def _find_file_recursive(self, sftp, remote_dir: str, filename: str) -> Optional[str]:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except UnicodeDecodeError as exc:
            logger.warning(f"[SFTP] Encoding issue in directory '{remote_dir}': {exc} — skipping")
            return None
        except Exception as exc:
            logger.warning(f"[SFTP] Cannot list directory '{remote_dir}': {exc}")
            return None

        for entry in entries:
            try:
                entry_path = remote_dir.rstrip("/") + "/" + entry.filename
            except UnicodeDecodeError as exc:
                logger.warning(f"[SFTP] Skipping entry with undecodable filename in '{remote_dir}': {exc}")
                continue

            if stat.S_ISDIR(entry.st_mode):
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
        If file_ref is a bare filename, recursively searches remote_base for it.

        Args:
            file_ref: Either a full absolute SFTP path or a bare filename.

        Returns:
            Local Path of the downloaded file.

        Raises:
            RuntimeError on failure after MAX_RETRIES attempts.
        """
        filename = Path(file_ref).name
        local_path = self.local_base / filename

        # Skip if already downloaded
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"[SFTP] Already exists locally, skipping: {filename}")
            return local_path

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()

                # Resolve remote path
                if file_ref.startswith("/"):
                    remote_path = file_ref
                else:
                    # Recursively search for the file under remote_base
                    logger.info(
                        f"[SFTP] Searching recursively for '{filename}' under {self.remote_base}"
                    )
                    remote_path = self._find_file_recursive(sftp, self.remote_base, filename)
                    if remote_path is None:
                        raise FileNotFoundError(
                            f"File '{filename}' not found anywhere under {self.remote_base}"
                        )

                logger.info(f"[SFTP] Downloading ({attempt}/{MAX_RETRIES}): {remote_path}")
                sftp.get(remote_path, str(local_path))
                logger.info(f"[SFTP] Downloaded: {filename}  {local_path}")
                return local_path

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[SFTP] Attempt {attempt} failed for '{filename}': {exc}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
            finally:
                if sftp:
                    sftp.close()
                if transport:
                    transport.close()

        # All retries exhausted
        raise RuntimeError(
            f"[SFTP] Failed to download '{filename}' after {MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_file(self, file_ref: str) -> Path:
        """
        Download a single document from the NEAPS server.

        Args:
            file_ref: Full SFTP path or bare filename.

        Returns:
            Local Path of the downloaded file.
        """
        return self._download_one(file_ref)

    def download_one_to(self, file_ref: str, dest_path: Path) -> Path:
        """
        Download a single file to an explicit destination path.
        If file_ref is a bare filename, recursively searches remote_base for it.

        Args:
            file_ref: Either a full absolute SFTP path or a bare filename.
            dest_path: Exact local path to write to.

        Returns:
            dest_path after successful download.

        Raises:
            RuntimeError on failure after MAX_RETRIES attempts.
        """
        filename = Path(file_ref).name

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()

                # Resolve remote path
                if file_ref.startswith("/"):
                    remote_path = file_ref
                else:
                    # Recursively search for the file under remote_base
                    logger.info(
                        f"[SFTP] Searching recursively for '{filename}' under {self.remote_base}"
                    )
                    remote_path = self._find_file_recursive(sftp, self.remote_base, filename)
                    if remote_path is None:
                        raise FileNotFoundError(
                            f"File '{filename}' not found anywhere under {self.remote_base}"
                        )

                logger.info(
                    f"[SFTP] on-demand download ({attempt}/{MAX_RETRIES}): "
                    f"{remote_path} -> {dest_path}"
                )
                sftp.get(remote_path, str(dest_path))
                logger.info(f"[SFTP] on-demand download done: {filename}")
                return dest_path

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[SFTP] on-demand attempt {attempt} failed for '{filename}': {exc}"
                )
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

    def download_batch(self, file_list: List[str]) -> Dict[str, Optional[Path]]:
        """
        Download multiple files in parallel using ThreadPoolExecutor.
        Each worker thread opens ONE persistent SFTP connection and recursively
        searches + downloads all files in its chunk.

        Args:
            file_list: List of file refs (absolute paths or bare filenames).

        Returns:
            Dict mapping each file_ref to its local Path (or None on failure).
        """
        results: Dict[str, Optional[Path]] = {}

        if not file_list:
            return results

        logger.info(
            f"[SFTP] Starting persistent-connection batch download of {len(file_list)} files "
            f"({self.workers} parallel workers)"
        )

        # Split file list into per-worker chunks for connection reuse
        chunks: List[List[str]] = [[] for _ in range(self.workers)]
        for i, fname in enumerate(file_list):
            chunks[i % self.workers].append(fname)

        def _download_chunk(file_chunk: List[str]) -> Dict[str, Optional[Path]]:
            """Download a sub-list of files over a single persistent SFTP connection."""
            chunk_results: Dict[str, Optional[Path]] = {}
            if not file_chunk:
                return chunk_results

            transport = None
            sftp = None
            try:
                transport, sftp = self._make_sftp()

                for file_ref in file_chunk:
                    filename = Path(file_ref).name
                    local_path = self.local_base / filename

                    # Skip if already downloaded
                    if local_path.exists() and local_path.stat().st_size > 0:
                        logger.debug(f"[SFTP] Already exists locally, skipping: {filename}")
                        chunk_results[file_ref] = local_path
                        continue

                    # Resolve remote path
                    if file_ref.startswith("/"):
                        remote_path = file_ref
                    else:
                        # Recursively search for the file under remote_base
                        logger.info(
                            f"[SFTP] Searching recursively for '{filename}' under {self.remote_base}"
                        )
                        remote_path = self._find_file_recursive(sftp, self.remote_base, filename)
                        if remote_path is None:
                            logger.error(
                                f"[SFTP] '{filename}' not found anywhere under {self.remote_base}"
                            )
                            chunk_results[file_ref] = None
                            continue

                    last_exc: Optional[Exception] = None
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            logger.info(
                                f"[SFTP] Downloading ({attempt}/{MAX_RETRIES}): {remote_path}"
                            )
                            sftp.get(remote_path, str(local_path))
                            logger.info(f"[SFTP] Downloaded: {filename}")
                            chunk_results[file_ref] = local_path
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                            logger.warning(
                                f"[SFTP] Attempt {attempt} failed for '{filename}': {exc}"
                            )
                            if attempt < MAX_RETRIES:
                                time.sleep(RETRY_DELAY_S)

                    if last_exc is not None:
                        logger.error(
                            f"[SFTP] Failed '{filename}' after {MAX_RETRIES} attempts: {last_exc}"
                        )
                        chunk_results[file_ref] = None

            except Exception as conn_exc:
                # Connection-level failure - mark all files in this chunk as failed
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

            return chunk_results

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_chunk = {
                executor.submit(_download_chunk, chunk): chunk
                for chunk in chunks
                if chunk
            }
            for future in as_completed(future_to_chunk):
                try:
                    chunk_result = future.result()
                    results.update(chunk_result)
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


 