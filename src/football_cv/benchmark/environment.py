"""
Environment and hardware metadata collection for reproducible benchmarking.
"""

import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EnvironmentCollector:
    """
    Collects complete hardware, OS, Python runtime, and package environment metadata
    to ensure benchmark reproducibility across hardware platforms.
    """

    @staticmethod
    def get_git_info() -> tuple[str, str]:
        """Retrieve current Git commit short SHA and active branch name."""
        commit_sha = "unknown"
        branch_name = "unknown"
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            branch_name = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            pass
        return commit_sha, branch_name

    @classmethod
    def collect(cls) -> dict[str, Any]:
        """
        Gather system specifications and library versions.

        Returns:
            Dictionary containing hardware and software manifest.
        """
        # RAM via psutil with fallback
        ram_total_gb = None
        ram_avail_gb = None
        cpu_phys = None
        try:
            import psutil

            vm = psutil.virtual_memory()
            ram_total_gb = round(vm.total / (1024**3), 2)
            ram_avail_gb = round(vm.available / (1024**3), 2)
            cpu_phys = psutil.cpu_count(logical=False)
        except Exception:
            pass

        # PyTorch & Accelerator info
        torch_ver = "unknown"
        cuda_avail = False
        cuda_ver = None
        gpu_count = 0
        gpu_name = None

        try:
            import torch

            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                cuda_ver = getattr(torch.version, "cuda", None)
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        # OpenCV version
        cv_ver = "unknown"
        try:
            import cv2

            cv_ver = cv2.__version__
        except Exception:
            pass

        # Ultralytics version
        ultra_ver = "unknown"
        try:
            import ultralytics

            ultra_ver = ultralytics.__version__
        except Exception:
            pass

        # ONNX & ONNX Runtime version
        onnx_ver = "unknown"
        ort_ver = "unknown"
        try:
            import onnx

            onnx_ver = onnx.__version__
        except Exception:
            pass
        try:
            import onnxruntime

            ort_ver = onnxruntime.__version__
        except Exception:
            pass

        commit_sha, branch_name = cls.get_git_info()

        manifest: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git": {
                "commit": commit_sha,
                "branch": branch_name,
            },
            "system": {
                "os": platform.system(),
                "os_release": platform.release(),
                "platform": platform.platform(),
                "architecture": platform.machine(),
            },
            "cpu": {
                "model": platform.processor() or "unknown",
                "physical_cores": cpu_phys,
                "logical_cores": os.cpu_count(),
            },
            "memory": {
                "total_gb": ram_total_gb,
                "available_gb": ram_avail_gb,
            },
            "accelerator": {
                "cuda_available": cuda_avail,
                "cuda_version": cuda_ver,
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
            },
            "packages": {
                "python": platform.python_version(),
                "torch": torch_ver,
                "opencv": cv_ver,
                "ultralytics": ultra_ver,
                "onnx": onnx_ver,
                "onnxruntime": ort_ver,
            },
        }

        return manifest

    @classmethod
    def export_json(cls, output_path: str | Path) -> Path:
        """Collect metadata and write to output JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = cls.collect()

        with open(target, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved benchmark environment manifest to {target}")
        return target
