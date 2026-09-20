#pragma once

#include "types.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace football_cv {

/**
 * @brief Calculate the center point of a bounding box.
 */
Point2D get_center_of_bbox(const BoundingBox& bbox) noexcept;

/**
 * @brief Calculate the bottom-center foot contact point of a bounding box.
 */
Point2D get_foot_position(const BoundingBox& bbox) noexcept;

/**
 * @brief Get the horizontal width of a bounding box.
 */
double get_bbox_width(const BoundingBox& bbox) noexcept;

/**
 * @brief Get the vertical height of a bounding box.
 */
double get_bbox_height(const BoundingBox& bbox) noexcept;

/**
 * @brief Calculate Euclidean distance between two 2D points.
 */
double measure_distance(const Point2D& p1, const Point2D& p2) noexcept;

/**
 * @brief Calculate signed (dx, dy) displacement from p2 to p1 (p1 - p2).
 */
std::pair<double, double> measure_xy_distance(const Point2D& p1, const Point2D& p2) noexcept;

/**
 * @brief Extract center points for a collection of bounding boxes.
 */
std::vector<Point2D> extract_centers(const std::vector<BoundingBox>& bboxes);

/**
 * @brief Extract foot contact points for a collection of bounding boxes.
 */
std::vector<Point2D> extract_foot_positions(const std::vector<BoundingBox>& bboxes);

/**
 * @brief Filter out bounding boxes with invalid coordinates, negative size, or NaNs.
 */
std::vector<BoundingBox> filter_valid_bboxes(const std::vector<BoundingBox>& bboxes);

/**
 * @brief Find the index of the nearest point within an optional maximum distance.
 *
 * @param target Reference point.
 * @param candidates Collection of candidate points.
 * @param max_distance Maximum Euclidean distance threshold.
 * @return Index of nearest candidate in candidates, or std::nullopt if none within threshold.
 */
std::optional<size_t> find_nearest_point(
    const Point2D& target,
    const std::vector<Point2D>& candidates,
    double max_distance = std::numeric_limits<double>::infinity()
);

} // namespace football_cv
