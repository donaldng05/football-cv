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

    std::vector<double> dxs;
    std::vector<double> dys;
    std::vector<double> dists;
    dxs.reserve(old_features.size());
    dys.reserve(old_features.size());
    dists.reserve(old_features.size());

    for (size_t i = 0; i < old_features.size(); ++i) {
        const auto& old_pt = old_features[i];
        const auto& new_pt = new_features[i];

        if (!old_pt.is_finite() || !new_pt.is_finite()) {
            continue;
        }

        const double dist = old_pt.distance_to(new_pt);
        dxs.push_back(old_pt.x - new_pt.x);
        dys.push_back(old_pt.y - new_pt.y);
        dists.push_back(dist);
    }

    if (dxs.empty()) {
        return CameraMotion{0.0, 0.0, false};
    }

    auto compute_median = [](std::vector<double>& v) -> double {
        const size_t n = v.size();
        const size_t mid = n / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        if (n % 2 != 0) {
            return v[mid];
        }
        auto max_lower = *std::max_element(v.begin(), v.begin() + mid);
        return (v[mid] + max_lower) * 0.5;
    };

    double med_dist = compute_median(dists);
    if (med_dist > scene_cut_threshold_) {
        // Sudden flow magnitude jump indicates a broadcast scene cut
        return CameraMotion{0.0, 0.0, true};
    }

    double cam_dx = compute_median(dxs);
    double cam_dy = compute_median(dys);
    double displacement_mag = std::sqrt(cam_dx * cam_dx + cam_dy * cam_dy);

    if (displacement_mag > minimum_distance_) {
        return CameraMotion{cam_dx, cam_dy, false};
    }

    if (med_dist > minimum_distance_) {
        // Opposing flow vectors canceled out, but feature points underwent significant motion
        // Find point with distance closest to median distance
        double best_diff = std::numeric_limits<double>::infinity();
        size_t best_idx = 0;
        for (size_t i = 0; i < dists.size(); ++i) {
            double diff = std::abs(dists[i] - med_dist);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        return CameraMotion{dxs[best_idx], dys[best_idx], false};
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
