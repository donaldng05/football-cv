"""
Command-Line Interface (CLI) for football_cv.
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .exceptions import ConfigurationError, FootballCVError, ValidationError
from .logging_config import setup_logging
from .utils.validation import run_preflight_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-cv",
        description="football-cv: Computer Vision & Advanced Football Analytics Pipeline",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Subcommand to execute"
    )

    # --------------------------------------------------------------------------
    # Subcommand: validate
    # --------------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run pre-flight sanity checks on config, model weights, and video decodability",
    )
    validate_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )

    # --------------------------------------------------------------------------
    # Subcommand: analyze
    # --------------------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run detection, tracking, and analytics pipeline on video"
    )
    analyze_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    analyze_parser.add_argument(
        "-i", "--input", dest="input_path", type=str, help="Path to input video"
    )
    analyze_parser.add_argument(
        "-o", "--output", dest="output_path", type=str, help="Path to output video"
    )
    analyze_parser.add_argument(
        "--confidence",
        type=float,
        help="Model detection confidence threshold (0.0 - 1.0)",
    )
    analyze_parser.add_argument(
        "--device", type=str, help="Compute device ('auto', 'cpu', 'cuda')"
    )
    analyze_parser.add_argument(
        "--batch-size", type=int, help="Batch size for object detector"
    )
    analyze_parser.add_argument(
        "--start-frame", type=int, help="Starting frame index for analysis"
    )
    analyze_parser.add_argument(
        "--end-frame", type=int, help="Ending frame index for analysis"
    )
    analyze_parser.add_argument(
        "--use-stubs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cached tracking and camera stubs if available",
    )
    analyze_parser.add_argument(
        "--stub-path", type=str, help="Path to cached track stubs pickle file"
    )
    analyze_parser.add_argument(
        "--export-dir", type=str, help="Directory to save structured data exports"
    )

    # --------------------------------------------------------------------------
    # Subcommand: report
    # --------------------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="Generate analytics reports (heatmaps, pass networks) from tracking data",
    )
    report_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    report_parser.add_argument(
        "--tracks",
        type=str,
        required=True,
        help="Path to exported tracks or events JSON file",
    )
    report_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/report",
        help="Directory to save generated figures and report cards",
    )
    report_parser.add_argument(
        "--team",
        type=int,
        default=None,
        help="Filter report generation to a specific team ID (e.g. 1 or 2)",
    )
    report_parser.add_argument(
        "--player",
        type=int,
        default=None,
        help="Filter report generation to a specific player track ID",
    )
    report_parser.add_argument(
        "--theme",
        type=str,
        choices=["tactical_dark", "classic_turf", "light"],
        default=None,
        help="Pitch visual theme (tactical_dark, classic_turf, light)",
    )

    # --------------------------------------------------------------------------
    # Subcommand: benchmark
    # --------------------------------------------------------------------------
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Profile stage-by-stage pipeline latency and throughput"
    )
    benchmark_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/fast.yaml",
        help="Path to YAML configuration file",
    )
    benchmark_parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to benchmark",
    )
    benchmark_parser.add_argument(
        "--device", type=str, help="Override benchmark device ('cpu', 'cuda')"
    )

    return parser


def _extract_overrides(args: argparse.Namespace) -> dict:
    """Extract non-None CLI arguments into nested override dictionary."""
    overrides: dict = {}

    model_overrides = {}
    if getattr(args, "confidence", None) is not None:
        model_overrides["confidence"] = args.confidence
    if getattr(args, "device", None) is not None:
        model_overrides["device"] = args.device
    if getattr(args, "batch_size", None) is not None:
        model_overrides["batch_size"] = args.batch_size
    if model_overrides:
        overrides["model"] = model_overrides

    video_overrides = {}
    if getattr(args, "input_path", None) is not None:
        video_overrides["input_path"] = args.input_path
    if getattr(args, "output_path", None) is not None:
        video_overrides["output_path"] = args.output_path
    if getattr(args, "start_frame", None) is not None:
        video_overrides["start_frame"] = args.start_frame
    if getattr(args, "end_frame", None) is not None:
        video_overrides["end_frame"] = args.end_frame
    if video_overrides:
        overrides["video"] = video_overrides

    tracking_overrides = {}
    if getattr(args, "use_stubs", None) is not None:
        tracking_overrides["use_cached_tracks"] = args.use_stubs
    if getattr(args, "stub_path", None) is not None:
        tracking_overrides["cache_path"] = args.stub_path
    if tracking_overrides:
        overrides["tracking"] = tracking_overrides

    analytics_overrides = {}
    if getattr(args, "export_dir", None) is not None:
        analytics_overrides["export_dir"] = args.export_dir
    if analytics_overrides:
        overrides["analytics"] = analytics_overrides

    return overrides


def handle_validate(args: argparse.Namespace) -> int:
    config_path = args.config
    print("\n=======================================================")
    print("  football-cv: Pre-Flight Configuration & Asset Check  ")
    print("=======================================================")
    print(f"Config File: {config_path}")

    try:
        config = load_config(config_path)
        setup_logging(level=config.logging.level, log_file=config.logging.log_file)
        results = run_preflight_checks(config)

        print("\n[PASS] Configuration syntax and boundary parameters valid")
        print(
            f"[PASS] Model weights:  {results['model']['path']} ({results['model']['size_mb']} MB)"
        )
        print(
            f"[PASS] Input video:    {results['video']['path']} ({results['video']['width']}x{results['video']['height']} @ {results['video']['fps']} FPS, {results['video']['frame_count']} frames)"
        )
        print(
            f"[PASS] Compute device: {results['device']['requested']} -> {results['device']['resolved']} ({results['device']['device_name']})"
        )
        print(
            f"[PASS] Output storage: Video: {results['output_dirs']['video_dir']}, Analytics: {results['output_dirs']['analytics_dir']}"
        )
        print("\n>>> PRE-FLIGHT CHECKS PASSED: Environment ready for execution. <<<\n")
        return 0

    except (ConfigurationError, ValidationError) as exc:
        print(f"\n[FAIL] Validation Failed: {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error during validation: {exc}\n", file=sys.stderr)
        return 2


def handle_analyze(args: argparse.Namespace) -> int:
    overrides = _extract_overrides(args)
    try:
        config = load_config(args.config, overrides=overrides)
        logger = setup_logging(
            level=config.logging.level, log_file=config.logging.log_file
        )
        logger.info("Starting football-cv analysis pipeline")
        logger.info(
            f"Input: {config.video.input_path} -> Output: {config.video.output_path}"
        )
        logger.info(
            f"Model: {config.model.path} (confidence: {config.model.confidence}, device: {config.model.device})"
        )
        logger.info(
            f"Frame range: [{config.video.start_frame} : {config.video.end_frame or 'END'}]"
        )
        logger.info(f"Tracking cache enabled: {config.tracking.use_cached_tracks}")

        run_preflight_checks(config)
        logger.info("Pre-flight asset validation passed.")

        from .pipeline import MatchPipeline

        pipeline = MatchPipeline(config)
        results = pipeline.run()

        print(
            f"\n>>> Video analysis complete! Annotated output saved to: {config.video.output_path} <<<\n"
        )
        if "export_paths" in results:
            print(
                f"[INFO] Structured analytics exported to: {config.analytics.export_dir}"
            )
            print(
                f"       - Possession intervals: {len(results.get('possession_intervals', []))}"
            )
            print(f"       - Candidate events:    {len(results.get('events', []))}")
        return 0

    except FootballCVError as exc:
        print(f"[FAIL] Pipeline Configuration Error: {exc}", file=sys.stderr)
        return 1


def handle_report(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        if getattr(args, "theme", None):
            config.analytics.heatmap.theme = args.theme
        logger = setup_logging(level=config.logging.level)
        tracks_path = Path(args.tracks)
        if not tracks_path.exists():
            print(f"[FAIL] Tracking file not found: {tracks_path}", file=sys.stderr)
            return 1
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Analytics report engine initialized for {tracks_path}")

        from .analytics.heatmap import HeatmapGenerator

        generator = HeatmapGenerator(config=config, output_dir=output_dir)
        artifacts = generator.generate_from_file(
            file_path=tracks_path,
            team_filter=getattr(args, "team", None),
            player_filter=getattr(args, "player", None),
        )

        print(
            f"[PASS] Successfully generated {len(artifacts)} reporting artifacts in {output_dir}"
        )
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")

        return 0
    except (FootballCVError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Report error: {exc}", file=sys.stderr)
        return 1


def handle_benchmark(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        logger = setup_logging(level=config.logging.level)
        logger.info(f"Benchmarking runner initialized ({args.num_frames} frames)")
        print(
            f"[INFO] Benchmark configured for {args.num_frames} frames using {config.model.path}"
        )
        return 0
    except FootballCVError as exc:
        print(f"[FAIL] Benchmark error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return handle_validate(args)
    elif args.command == "analyze":
        return handle_analyze(args)
    elif args.command == "report":
        return handle_report(args)
    elif args.command == "benchmark":
        return handle_benchmark(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
