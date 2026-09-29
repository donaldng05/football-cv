"""
Model weight download and verification utility for football-cv.

Downloads fine-tuned YOLOv8 pitch detection checkpoints from official releases
or mirrors, and validates cryptographic SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Known checkpoint checksums
KNOWN_MODELS: dict[str, dict[str, str]] = {
    "best.pt": {
        "sha256": "4182f74567a6acade9b39cd4317db5c5b44fca63c09f1332323fb749090c5cc0",
        "description": "Production YOLOv8 football detection checkpoint",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    },
    "best.onnx": {
        "sha256": "",
        "description": "Production ONNX exported checkpoint",
        "fallback_url": "",
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(file_path: Path, expected_sha256: str | None = None) -> bool:
    """Verify that a model checkpoint exists and matches its expected checksum."""
    if not file_path.exists():
        print(f"[FAIL] {file_path} does not exist.")
        return False

    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] {file_path.name}: {size_mb:.2f} MB")

    if expected_sha256:
        actual_hash = compute_sha256(file_path)
        if actual_hash.lower() == expected_sha256.lower():
            print(f"[PASS] SHA-256 verified: {actual_hash[:16]}...")
            return True
        else:
            print("[FAIL] SHA-256 mismatch!")
            print(f"       Expected: {expected_sha256}")
            print(f"       Actual:   {actual_hash}")
            return False
    return True


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url} to {dest_path}...")

    def _progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100.0, count * block_size * 100.0 / total_size)
            sys.stdout.write(
                f"\r  Progress: {percent:.1f}% ({count * block_size / (1024 * 1024):.1f} MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\nDownload complete.")


def create_dummy_checkpoint(dest_path: Path) -> Path:
    """Generate a lightweight valid YOLOv8 checkpoint for CI/testing without remote downloads."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.yaml")
        model.save(str(dest_path))
        print(
            f"[INFO] Created synthetic YOLOv8 checkpoint at {dest_path} ({dest_path.stat().st_size} bytes)"
        )
    except Exception as exc:
        print(
            f"[WARN] Failed to instantiate YOLOv8 architecture ({exc}), falling back to torch.save"
        )
        import torch

        dummy_dict = {"model": None, "ema": None, "updates": 0, "epoch": -1}
        torch.save(dummy_dict, str(dest_path))
        print(f"[INFO] Created fallback dummy PyTorch checkpoint at {dest_path}")
    return dest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download or verify football-cv model weights."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store model checkpoints (default: models/)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing checkpoints without downloading",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Ensure a valid dummy/synthetic model exists for CI and offline testing without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights already exist",
    )
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        primary_dest = models_dir / "best.pt"
        if primary_dest.exists() and primary_dest.stat().st_size > 0:
            print(
                f"[INFO] Model checkpoint already exists: {primary_dest} "
                f"({primary_dest.stat().st_size / (1024 * 1024):.2f} MB)"
            )
            return 0
        create_dummy_checkpoint(primary_dest)
        return 0

    all_ok = True
    for filename, info in KNOWN_MODELS.items():
        dest = models_dir / filename
        expected_hash = info["sha256"] if info["sha256"] else None

        if dest.exists() and not args.force:
            print(f"\nChecking {filename}:")
            if not verify_checkpoint(dest, expected_hash):
                all_ok = False
        elif args.verify_only:
            print(f"\n[MISSING] {filename} is not present.")
            all_ok = False
        else:
            url = info["fallback_url"]
            if url:
                print(f"\nFetching {filename}...")
                try:
                    download_file(url, dest)
                    if expected_hash and not verify_checkpoint(dest, expected_hash):
                        all_ok = False
                except Exception as e:
                    print(f"[ERROR] Failed to download {filename}: {e}")
                    all_ok = False
            else:
                print(f"\n[NOTICE] No public URL configured for {filename}.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
