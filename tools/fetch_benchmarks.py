"""Benchmark dataset downloader and integrity verifier for Menipy.

Usage:
    uv run python tools/fetch_benchmarks.py               # Download all benchmark suites
    uv run python tools/fetch_benchmarks.py --all         # Explicit download all
    uv run python tools/fetch_benchmarks.py --category pendant
    uv run python tools/fetch_benchmarks.py --verify-only # Check local file SHA-256 hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_benchmarks")

MANIFEST_PATH = Path("data/MANIFEST.json")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Download file from URL to local destination with timeout and header."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {dest.name} from {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MenipyBenchmarkFetcher/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out:
            downloaded = 0
            while True:
                buf = resp.read(65536)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
        logger.info(f"Downloaded {dest.name} ({downloaded} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed downloading {dest.name}: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def run_benchmark_fetcher(
    category: str = "all",
    verify_only: bool = False,
    manifest_path: Path = MANIFEST_PATH,
) -> int:
    """Fetch and verify benchmark datasets against manifest."""
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    datasets: list[dict[str, Any]] = manifest.get("datasets", [])
    logger.info(f"Loaded manifest with {len(datasets)} dataset entries")

    # Filter by category
    selected: list[dict[str, Any]] = []
    for d in datasets:
        cat = d.get("category", "")
        if category == "all" or cat == category:
            selected.append(d)

    logger.info(f"Processing {len(selected)} datasets for category='{category}' (verify_only={verify_only})")

    success_count = 0
    failure_count = 0

    for d in selected:
        dataset_id = d.get("id", "unknown")
        filename = d.get("filename", "")
        dest_dir = Path(d.get("destination_dir", "data/benchmarks"))
        target_path = dest_dir / filename
        source_url = d.get("source_url")
        expected_sha = d.get("sha256")

        if target_path.exists():
            actual_sha = compute_sha256(target_path)
            if expected_sha and actual_sha != expected_sha:
                logger.warning(
                    f"[{dataset_id}] Checksum mismatch: expected {expected_sha[:8]}..., got {actual_sha[:8]}..."
                )
                if not verify_only and source_url:
                    logger.info(f"[{dataset_id}] Re-downloading to fix corrupt file...")
                    if download_file(source_url, target_path):
                        actual_sha = compute_sha256(target_path)

            if not expected_sha or actual_sha == expected_sha:
                logger.info(f"[{dataset_id}] OK: {target_path} verified")
                success_count += 1
            else:
                logger.error(f"[{dataset_id}] FAILED: Checksum verification failed")
                failure_count += 1
        else:
            if verify_only:
                logger.warning(f"[{dataset_id}] Missing local file: {target_path}")
                failure_count += 1
                continue

            if not source_url:
                logger.info(f"[{dataset_id}] Local/legacy fixture missing: {target_path}")
                failure_count += 1
                continue

            if download_file(source_url, target_path):
                actual_sha = compute_sha256(target_path)
                if expected_sha and actual_sha != expected_sha:
                    logger.error(
                        f"[{dataset_id}] Checksum mismatch after download: {actual_sha} != {expected_sha}"
                    )
                    failure_count += 1
                else:
                    logger.info(f"[{dataset_id}] Successfully downloaded and verified {target_path}")
                    success_count += 1
            else:
                failure_count += 1

    logger.info(f"Benchmark fetch complete: {success_count} succeeded, {failure_count} failed")
    return 0 if failure_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify Menipy benchmark datasets.")
    parser.add_argument(
        "--category",
        choices=["all", "pendant", "sessile", "dynamic", "legacy_lab"],
        default="all",
        help="Dataset category to download (default: all)",
    )
    parser.add_argument(
        "--all",
        dest="all_flag",
        action="store_true",
        help="Download all categories (default behaviour)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify checksums of existing files without downloading missing ones",
    )
    args = parser.parse_args()
    chosen_category = "all" if args.all_flag else args.category
    return run_benchmark_fetcher(category=chosen_category, verify_only=args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
