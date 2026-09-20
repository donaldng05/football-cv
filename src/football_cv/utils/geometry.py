"""
Geometric and bounding-box utility functions.
"""

from collections.abc import Sequence

Point2D = Sequence[float] | tuple[float, float]
BBox = Sequence[float] | tuple[float, float, float, float]


def get_center_of_bbox(bbox: BBox) -> tuple[int, int]:
    """
    Calculate the center coordinate (x, y) of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (center_x, center_y) as integers.
    """
    x1, y1, x2, y2 = bbox[:4]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return int(center_x), int(center_y)


def get_bbox_width(bbox: BBox) -> float:
    """Calculate the horizontal width of a bounding box."""
    return float(bbox[2] - bbox[0])


def get_bbox_height(bbox: BBox) -> float:
    """Calculate the vertical height of a bounding box."""
    return float(bbox[3] - bbox[1])


def get_foot_position(bbox: BBox) -> tuple[int, int]:
    """
    Calculate the bottom-center coordinate representing player foot contact.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (foot_x, foot_y) as integers.
    """
    x1, _, x2, y2 = bbox[:4]
    foot_x = (x1 + x2) / 2.0
    foot_y = y2
    return int(foot_x), int(foot_y)


def measure_distance(p1: Point2D, p2: Point2D) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def measure_xy_distance(p1: Point2D, p2: Point2D) -> tuple[float, float]:
    """Calculate signed (dx, dy) displacement from p2 to p1."""
    return float(p1[0] - p2[0]), float(p1[1] - p2[1])


def extract_centers(bboxes: Sequence[BBox]) -> list[tuple[int, int]]:
    """Extract center points for a collection of bounding boxes."""
    return [get_center_of_bbox(b) for b in bboxes]


def extract_foot_positions(bboxes: Sequence[BBox]) -> list[tuple[int, int]]:
    """Extract foot contact points for a collection of bounding boxes."""
    return [get_foot_position(b) for b in bboxes]


def filter_valid_bboxes(bboxes: Sequence[BBox]) -> list[BBox]:
    """Filter out bounding boxes with non-finite coordinates or negative dimensions."""
    import math

    valid: list[BBox] = []
    for b in bboxes:
        if len(b) < 4:
            continue
        x1, y1, x2, y2 = b[:4]
        if not (
            math.isfinite(x1)
            and math.isfinite(y1)
            and math.isfinite(x2)
            and math.isfinite(y2)
        ):
            continue
        if x2 >= x1 and y2 >= y1:
            valid.append(b)
    return valid


def find_nearest_point(
    target: Point2D,
    candidates: Sequence[Point2D],
    max_distance: float = float("inf"),
) -> int | None:
    """Find the index of the nearest candidate point within max_distance."""
    best_idx: int | None = None
    min_dist = max_distance

    for idx, cand in enumerate(candidates):
        dist = measure_distance(target, cand)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx

    return best_idx
