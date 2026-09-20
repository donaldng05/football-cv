#include "football_cv/camera_motion.hpp"

#include <algorithm>
#include <cmath>

namespace football_cv {

CameraMotionEstimator::CameraMotionEstimator(
    double minimum_distance,
    double scene_cut_threshold
) noexcept
    : minimum_distance_(minimum_distance),
      scene_cut_threshold_(scene_cut_threshold) {}

CameraMotion CameraMotionEstimator::estimate_from_features(
    const std::vector<Point2D>& old_features,
    const std::vector<Point2D>& new_features
) const noexcept {
    if (old_features.empty() || new_features.empty() || old_features.size() != new_features.size()) {
        return CameraMotion{0.0, 0.0, false};
    }

    double max_distance = 0.0;
    double cam_dx = 0.0;
    double cam_dy = 0.0;

    for (size_t i = 0; i < old_features.size(); ++i) {
        const auto& old_pt = old_features[i];
        const auto& new_pt = new_features[i];

        if (!old_pt.is_finite() || !new_pt.is_finite()) {
            continue;
        }

        const double dist = old_pt.distance_to(new_pt);
        if (dist > max_distance) {
            max_distance = dist;
            // In Python: cam_dx, cam_dy = measure_xy_distance(old.ravel(), new.ravel())
            // which computes (old[0] - new[0], old[1] - new[1])
            cam_dx = old_pt.x - new_pt.x;
            cam_dy = old_pt.y - new_pt.y;
        }
    }

    if (max_distance > scene_cut_threshold_) {
        // Sudden flow magnitude jump indicates a broadcast scene cut
        return CameraMotion{0.0, 0.0, true};
    }

    if (max_distance > minimum_distance_) {
        return CameraMotion{cam_dx, cam_dy, false};
    }

    return CameraMotion{0.0, 0.0, false};
}

std::vector<Point2D> CameraMotionEstimator::filter_margin_points(
    const std::vector<Point2D>& points,
    double frame_width,
    double left_margin_width,
    double right_margin_start,
    double right_margin_end
) {
    (void)frame_width;
    std::vector<Point2D> filtered;
    filtered.reserve(points.size());

    for (const auto& pt : points) {
        if (!pt.is_finite()) {
            continue;
        }
        const bool in_left_margin = (pt.x >= 0.0 && pt.x <= left_margin_width);
        const bool in_right_margin = (pt.x >= right_margin_start && pt.x <= right_margin_end);

        if (in_left_margin || in_right_margin) {
            filtered.push_back(pt);
        }
    }

    return filtered;
}

std::vector<Point2D> CameraMotionEstimator::accumulate_motion(
    const std::vector<CameraMotion>& motion_steps
) {
    std::vector<Point2D> cumulative;
    cumulative.reserve(motion_steps.size());

    double total_dx = 0.0;
    double total_dy = 0.0;

    for (const auto& step : motion_steps) {
        if (!step.is_scene_cut) {
            total_dx += step.dx;
            total_dy += step.dy;
        }
        cumulative.emplace_back(total_dx, total_dy);
    }

    return cumulative;
}

} // namespace football_cv
