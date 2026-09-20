#include "football_cv/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace football_cv {

Point2D get_center_of_bbox(const BoundingBox& bbox) noexcept {
    return bbox.center();
}

Point2D get_foot_position(const BoundingBox& bbox) noexcept {
    return bbox.foot_position();
}

double get_bbox_width(const BoundingBox& bbox) noexcept {
    return bbox.width();
}

double get_bbox_height(const BoundingBox& bbox) noexcept {
    return bbox.height();
}

double measure_distance(const Point2D& p1, const Point2D& p2) noexcept {
    return p1.distance_to(p2);
}

std::pair<double, double> measure_xy_distance(const Point2D& p1, const Point2D& p2) noexcept {
    // Matches Python measure_xy_distance(p1, p2) -> (p1[0] - p2[0], p1[1] - p2[1])
    return {p1.x - p2.x, p1.y - p2.y};
}

std::vector<Point2D> extract_centers(const std::vector<BoundingBox>& bboxes) {
    std::vector<Point2D> centers;
    centers.reserve(bboxes.size());
    for (const auto& box : bboxes) {
        centers.push_back(box.center());
    }
    return centers;
}

std::vector<Point2D> extract_foot_positions(const std::vector<BoundingBox>& bboxes) {
    std::vector<Point2D> feet;
    feet.reserve(bboxes.size());
    for (const auto& box : bboxes) {
        feet.push_back(box.foot_position());
    }
    return feet;
}

std::vector<BoundingBox> filter_valid_bboxes(const std::vector<BoundingBox>& bboxes) {
    std::vector<BoundingBox> valid;
    valid.reserve(bboxes.size());
    for (const auto& box : bboxes) {
        if (box.is_valid()) {
            valid.push_back(box);
        }
    }
    return valid;
}

std::optional<size_t> find_nearest_point(
    const Point2D& target,
    const std::vector<Point2D>& candidates,
    double max_distance
) {
    if (!target.is_finite() || candidates.empty() || max_distance < 0.0) {
        return std::nullopt;
    }

    std::optional<size_t> best_idx = std::nullopt;
    double min_dist = max_distance;

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& pt = candidates[i];
        if (!pt.is_finite()) {
            continue;
        }
        const double d = target.distance_to(pt);
        if (d <= min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }

    return best_idx;
}

} // namespace football_cv
