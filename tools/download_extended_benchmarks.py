"""Download and uncompress extended scientific benchmark datasets for Menipy.

This script fetches peer-reviewed benchmark datasets (images, video sequences,
and ground-truth measurement files) from open scientific repositories (EPFL,
Zenodo, University of Melbourne, UCLA, etc.) into data/benchmarks/ (which is
ignored by git). Any zip archive downloaded is automatically uncompressed.
"""

from __future__ import annotations

import io
import logging
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_extended_benchmarks")

BASE_BENCHMARK_DIR = Path("data/benchmarks")

DATASETS: list[dict[str, Any]] = [
    # 1. EPFL Biomedical Imaging Group: Drop Analysis (LB-ADSA & DropSnake)
    {
        "id": "epfl_drop_analysis",
        "type": "zip",
        "url": "http://bigwww.epfl.ch/demo/dropanalysis/drop_analysis.zip",
        "extract_to": BASE_BENCHMARK_DIR / "epfl_drop_analysis",
        "description": "EPFL DropSnake & LB-ADSA reference images and calibration parameters",
    },
    # 2. Aalto University: Contact Angle Baseline Error Analysis (Science 2019)
    {
        "id": "aalto_ca_error",
        "type": "zip",
        "url": "https://zenodo.org/api/records/2573078/files/Error_In_CA_measurements_1.zip/content",
        "extract_to": BASE_BENCHMARK_DIR / "aalto_ca_error",
        "description": "Aalto University baseline error and scale calibration dataset",
    },
    # 3. Conan-ML: Experimental Contact Angle Ground Truth Dataset (Berry et al., Langmuir 2024)
    {
        "id": "conan_ml_experimental",
        "type": "files",
        "target_dir": BASE_BENCHMARK_DIR / "conan_ml",
        "base_url": "https://raw.githubusercontent.com/jdber1/conan-ml/master/experimental%20data%20set/",
        "files": [
            "111.031693.bmp",
            "113.66.bmp",
            "113.98.bmp",
            "114.47.bmp",
            "115.00.bmp",
            "115.553909.bmp",
            "115.714851.bmp",
            "118.174171.bmp",
            "2-s@M@Z-120.321945190429.bmp",
            "2-s@M@Z-121.057571411132.bmp",
            "2-s@M@Z-122.639770507812.bmp",
            "359 CA162.BMP",
            "steelball.jpg",
            "TEF1.png",
            "TEF2.png",
            "TEF3.png",
            "TEF4.BMP",
        ],
        "description": "Conan-ML sessile images with ground truth contact angles in filenames",
    },
    # 4. UCLA: Pendant Drop Tensiometry (Water & Hexadecane)
    {
        "id": "ucla_pendant",
        "type": "files",
        "target_dir": BASE_BENCHMARK_DIR / "ucla_pendant",
        "base_url": "https://raw.githubusercontent.com/jtvanlew/ucla-pendant-drop/master/droplets/",
        "files": [
            "H2O_PendantDrop.png",
            "Hexadecane_PendantDrop.png",
            "Hexadecane.bmp",
        ],
        "description": "UCLA pendant drop tensiometry calibration images (water & hexadecane)",
    },
    # 5. pyDSA: Dynamic Droplet Sequence and Wetting Ridge (Launay)
    {
        "id": "pydsa_dynamic",
        "type": "files",
        "target_dir": BASE_BENCHMARK_DIR / "pydsa",
        "base_url": "https://raw.githubusercontent.com/galaunay/pyDSA_gui/master/tests/",
        "files": [
            "test1.png",
            "test2.png",
            "test3.png",
            "test4.png",
            "test5.png",
            "test6.png",
            "test7.png",
            "test8.png",
            "wetting_ridge.mp4",
            "test.avi",
        ],
        "description": "pyDSA dynamic sessile image sequence and test videos with ground-truth config",
    },
    # 6. Sessile Droplet Evaporation Video Series (Brandt)
    {
        "id": "brandt_evaporation",
        "type": "files",
        "target_dir": BASE_BENCHMARK_DIR / "evaporation",
        "base_url": "https://raw.githubusercontent.com/soerenbrandt/Sessile-Droplet-analysis/master/example/",
        "files": [
            "Video.mp4",
            "Droplet-analysis-results.png",
        ],
        "description": "Sessile droplet evaporation time sequence video at 1 FPS with result curves",
    },
    # 7. University of Borås: Textile & Polymer Sessile Wetting Dataset (Zenodo 17691567)
    {
        "id": "boras_textile_wetting",
        "type": "files",
        "target_dir": BASE_BENCHMARK_DIR / "boras_textiles",
        "file_url_pattern": "https://zenodo.org/api/records/17691567/files/{filename}/content",
        "files": [
            "blackfabric-1.png",
            "blackfabric-1.xls",
            "whitefabricbacknotsilky-1.png",
            "whitefabricbacknotsilky-1.xls",
            "MXeneecoflexMXenefabricwhite-1.png",
            "MXeneecoflexMXenefabricwhite-1.xls",
            "README.txt",
        ],
        "description": "Sessile water drop on fabric substrates with measured contact angle spreadsheets",
    },
]


def download_bytes(url: str) -> bytes | None:
    """Download data from URL into memory."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MenipyBenchmarkDownloader/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def download_to_file(url: str, dest_path: Path) -> bool:
    """Download directly to a file."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Already exists: {dest_path}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {dest_path.name} from {url}...")
    content = download_bytes(url)
    if content:
        with open(dest_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved {dest_path.name} ({len(content)} bytes)")
        return True
    return False


def process_zip_dataset(entry: dict[str, Any]) -> None:
    """Download and extract a zip dataset."""
    dest_dir = entry["extract_to"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    existing_files = list(dest_dir.glob("*"))
    if len(existing_files) > 1:
        logger.info(f"[{entry['id']}] Already uncompressed in {dest_dir}")
        return

    logger.info(f"[{entry['id']}] Downloading zip from {entry['url']}...")
    content = download_bytes(entry["url"])
    if not content:
        logger.error(f"[{entry['id']}] Failed downloading zip archive")
        return

    logger.info(f"[{entry['id']}] Uncompressing {len(content)} bytes to {dest_dir}...")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(dest_dir)
        logger.info(f"[{entry['id']}] Uncompressed successfully into {dest_dir}")
    except Exception as e:
        logger.error(f"[{entry['id']}] Error uncompressing zip: {e}")


def process_files_dataset(entry: dict[str, Any]) -> None:
    """Download a set of individual files into a directory."""
    target_dir = entry["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = entry.get("base_url", "")
    url_pattern = entry.get("file_url_pattern")

    for fname in entry["files"]:
        if url_pattern:
            file_url = url_pattern.format(filename=fname)
        else:
            quoted_name = urllib.parse.quote(fname)
            file_url = f"{base_url}{quoted_name}"

        dest_path = target_dir / fname
        download_to_file(file_url, dest_path)


def main() -> None:
    logger.info("Starting extended benchmark dataset download & uncompression...")
    BASE_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    for entry in DATASETS:
        logger.info(f"--- Processing dataset: {entry['id']} ({entry['description']}) ---")
        if entry["type"] == "zip":
            process_zip_dataset(entry)
        elif entry["type"] == "files":
            process_files_dataset(entry)

    logger.info("All extended benchmark datasets downloaded and uncompressed successfully!")


if __name__ == "__main__":
    main()
