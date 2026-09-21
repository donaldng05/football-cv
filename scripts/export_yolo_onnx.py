"""
Export Ultralytics YOLOv8 PyTorch model to optimized ONNX format and validate graph integrity.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_yolo_onnx")


def export_model(
    model_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    device: str = "cpu",
    half: bool = False,
    simplify: bool = True,
) -> Path:
    """
    Export YOLO checkpoint to ONNX format.

    Args:
        model_path: Path to PyTorch .pt weights file.
        output_path: Target path for the .onnx file (defaults to same name as model_path with .onnx).
        imgsz: Image resolution dimension (square 640x640).
        opset: Target ONNX operator set version (recommended: 17).
        device: Device to export on ('cpu' or 'cuda').
        half: Export FP16 half precision.
        simplify: Run onnx-simplifier if available.

    Returns:
        Path to exported and verified ONNX model file.
    """
    model_p = Path(model_path)
    if not model_p.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_p}")

    logger.info(f"Loading YOLO model from: {model_p}")
    from ultralytics import YOLO

    model = YOLO(str(model_p))
    logger.info(f"Loaded YOLO task='{model.task}', classes={model.names}")

    logger.info(
        f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, device='{device}', simplify={simplify})..."
    )
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        device=device,
        half=half,
        simplify=simplify,
        dynamic=False,
    )

    exported_p = Path(exported_file)
    logger.info(f"Ultralytics exported to: {exported_p}")

    # If specific output path was requested, copy/move there
    if output_path is not None:
        target_p = Path(output_path)
        if target_p.resolve() != exported_p.resolve():
            target_p.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copyfile(exported_p, target_p)
            exported_p = target_p
            logger.info(f"Copied ONNX model to target location: {exported_p}")

    # Validate ONNX graph integrity
    logger.info("Validating ONNX graph integrity with onnx.checker...")
    import onnx

    onnx_model = onnx.load(str(exported_p))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph topology and operator verification PASSED.")

    # Inspect inputs and outputs
    graph = onnx_model.graph
    logger.info(f"ONNX Graph Name: {graph.name}")
    for inp in graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        logger.info(
            f"  Input:  '{inp.name}' (shape={shape}, type={inp.type.tensor_type.elem_type})"
        )
    for out in graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        logger.info(
            f"  Output: '{out.name}' (shape={shape}, type={out.type.tensor_type.elem_type})"
        )

    # Validate with onnxruntime session
    logger.info("Testing test inference with onnxruntime...")
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported_p), providers=["CPUExecutionProvider"])
    dummy_input = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})
    logger.info(f"Inference smoke test passed. Output tensor shape: {outputs[0].shape}")

    return exported_p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 PyTorch model to ONNX format with graph validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to YOLO .pt weights file (default: models/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best.onnx",
        help="Target output ONNX path (default: models/best.onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size dimension for model input (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to export with ('cpu' or 'cuda', default: 'cpu')",
    )

    args = parser.parse_args()

    try:
        out_path = export_model(
            model_path=args.model,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
        )
        print(f"\n[PASS] Model successfully exported and verified: {out_path}\n")
        return 0
    except Exception as exc:
        logger.error(f"Export failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
