#!/usr/bin/env python3
"""
Parallel downloader for https://vrg.fel.cvut.cz/ilias_data

Variant requested:
- DOES NOT re-check SHA256 for files that already exist:
    * if target file exists and size > 0 => skip download (no checksum verification)
- Prints ONLY ONCE a summary count of "already existing" files (skipped due to existence)
- Keeps everything else the same:
    * parallel downloads for checksum-listed files
    * SHA256 verification still done for newly downloaded files (to catch corruption)
    * retries on checksum mismatch
    * extracts and removes ilias_core.tar
    * global progress: files done/total, MB downloaded, MB/s

Usage:
  python3 download_ilias_parallel_no_sha_skip_count.py [DOWNLOAD_DIR] [MAX_FILES] [MAX_RETRIES] [MAX_WORKERS]
"""

import os
import sys
import time
import hashlib
import tarfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Args & defaults
# ----------------------------
DOWNLOAD_DIR = sys.argv[1] if len(sys.argv) > 1 else "./ilias"
MAX_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_RETRIES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAX_WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8

BASE_URL = "https://vrg.fel.cvut.cz/ilias_data"
CHECKSUMS_URL = f"{BASE_URL}/checksums.txt"
CHECKSUMS_FILE = os.path.join(DOWNLOAD_DIR, "checksums.txt")

PROGRESS_INTERVAL_S = 5.0

# ----------------------------
# Thread-safe printing
# ----------------------------
_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


# ----------------------------
# Global counters for progress
# ----------------------------
_counter_lock = threading.Lock()
_total_bytes_downloaded = 0
_files_done = 0
_files_ok = 0
_files_failed = 0


def add_bytes(n: int) -> None:
    global _total_bytes_downloaded
    with _counter_lock:
        _total_bytes_downloaded += n


def mark_done(ok: bool) -> None:
    global _files_done, _files_ok, _files_failed
    with _counter_lock:
        _files_done += 1
        if ok:
            _files_ok += 1
        else:
            _files_failed += 1


def snapshot_counters():
    with _counter_lock:
        return _files_done, _files_ok, _files_failed, _total_bytes_downloaded


# ----------------------------
# Download + checksum helpers
# ----------------------------
def download_file_counting(url: str, output_path: str) -> None:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            add_bytes(len(chunk))


def verify_sha256(expected_hex: str, filepath: str) -> bool:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_hex


def needs_yfcc_subdir(filename: str) -> bool:
    return "yfcc100m" in filename


def is_ilias_core_tar(filename: str) -> bool:
    return "ilias_core.tar" in filename


def exists_nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


# ----------------------------
# Progress thread
# ----------------------------
def progress_worker(total_files: int, stop_evt: threading.Event) -> None:
    last_t = time.time()
    last_bytes = 0

    while not stop_evt.wait(PROGRESS_INTERVAL_S):
        now = time.time()
        done, ok, failed, total_bytes = snapshot_counters()

        dt = max(1e-6, now - last_t)
        dbytes = total_bytes - last_bytes
        mb = total_bytes / (1024 * 1024)
        mbps = (dbytes / (1024 * 1024)) / dt

        safe_print(
            f"\r[ {done:>5}/{total_files:<5} ] ok={ok:<5} failed={failed:<5} "
            f"downloaded={mb:,.1f} MB  rate={mbps:,.1f} MB/s",
            end="",
        )

        last_t = now
        last_bytes = total_bytes

    done, ok, failed, total_bytes = snapshot_counters()
    mb = total_bytes / (1024 * 1024)
    safe_print(
        f"\r[ {done:>5}/{total_files:<5} ] ok={ok:<5} failed={failed:<5} "
        f"downloaded={mb:,.1f} MB",
        end="",
    )


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    safe_print(f"Downloading files to: {DOWNLOAD_DIR}")
    if MAX_FILES > 0:
        safe_print(f"Limit downloaded YFCC100M shards to {MAX_FILES}.")
    safe_print(f"Using {MAX_RETRIES} max retries per file.")
    safe_print(f"Using {MAX_WORKERS} parallel workers.")
    safe_print(f"Progress updates every {int(PROGRESS_INTERVAL_S)} seconds.")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # ---- checksums.txt (sequential)
    safe_print(f"Downloading checksums.txt from {CHECKSUMS_URL}...")
    try:
        download_file_counting(CHECKSUMS_URL, CHECKSUMS_FILE)
    except Exception as e:
        safe_print(f"\nERROR: Failed to download checksums.txt ({e})")
        return 1
    safe_print("Downloaded checksums.txt successfully.")

    # ---- image id files (sequential; keep bash-like overwrite behavior)
    imageid_urls = [
        f"{BASE_URL}/image_ids/image_query_ids.txt",
        f"{BASE_URL}/image_ids/text_query_ids.txt",
        f"{BASE_URL}/image_ids/positive_ids.txt",
        f"{BASE_URL}/image_ids/distractor_ids.txt.gz",
    ]

    safe_print("Downloading Image ID files...")
    for url in imageid_urls:
        relpath = url.replace(BASE_URL + "/", "")
        local_dir = os.path.join(DOWNLOAD_DIR, os.path.dirname(relpath))
        os.makedirs(local_dir, exist_ok=True)
        outfile = os.path.join(local_dir, os.path.basename(relpath))
        safe_print(f"  - {os.path.basename(relpath)} -> {local_dir}/")
        try:
            download_file_counting(url, outfile)
        except Exception as e:
            safe_print(f"ERROR: Failed to download {url} ({e})")
            return 1
    safe_print("Image ID files lists downloaded.")

    # ---- parse checksums and build worklist
    items = []
    yfcc_added = 0

    with open(CHECKSUMS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            checksum, url = line.split()
            filename = os.path.basename(url)

            if needs_yfcc_subdir(filename):
                if MAX_FILES > 0 and yfcc_added >= MAX_FILES:
                    continue
                yfcc_added += 1

            subdir = os.path.join(DOWNLOAD_DIR, "yfcc100m") if needs_yfcc_subdir(filename) else DOWNLOAD_DIR
            os.makedirs(subdir, exist_ok=True)
            filepath = os.path.join(subdir, filename)

            items.append((checksum, url, filename, subdir, filepath))

    total_files = len(items)

    # ---- count already-existing files ONCE (for checksum-listed items only)
    already_existing = sum(1 for (_, _, _, _, path) in items if exists_nonempty(path))
    safe_print(f"Queue size: {total_files} file(s). Already present: {already_existing} file(s).")

    # ---- worker function
    def fetch_one(checksum: str, url: str, filename: str, subdir: str, filepath: str) -> bool:
        # Skip download if file already exists (NO SHA check)
        if exists_nonempty(filepath):
            return True

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                download_file_counting(url, filepath)

                if verify_sha256(checksum, filepath):
                    if is_ilias_core_tar(filename):
                        safe_print(f"\nExtracting {filepath}...")
                        with tarfile.open(filepath, "r") as tar:
                            tar.extractall(path=subdir)
                        os.remove(filepath)
                        safe_print(f"Extracted and removed: {filepath}")
                    return True

                raise ValueError("Checksum mismatch")

            except Exception:
                attempt += 1
                safe_print(f"\nChecksum failed for {filepath} (attempt {attempt})")
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    pass

        safe_print(f"\nERROR: Failed to download {filepath} correctly after {MAX_RETRIES} attempts.")
        return False

    # ---- start progress thread
    stop_evt = threading.Event()
    prog_thread = threading.Thread(target=progress_worker, args=(total_files, stop_evt), daemon=True)
    prog_thread.start()

    # ---- parallel execution
    safe_print("Downloading shard files in parallel...")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_one, *it) for it in items]
            for fut in as_completed(futures):
                ok = False
                try:
                    ok = bool(fut.result())
                except Exception:
                    ok = False
                mark_done(ok)
    finally:
        stop_evt.set()
        prog_thread.join(timeout=2.0)
        safe_print()  # newline after progress line

    done, ok, failed, total_bytes = snapshot_counters()
    mb = total_bytes / (1024 * 1024)
    safe_print(f"Done. Successful: {ok}, Failed: {failed}, Downloaded: {mb:,.1f} MB")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
