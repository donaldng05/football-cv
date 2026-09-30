#include "football_cv/camera_motion.hpp"

#include <algorithm>
#include <cmath>
#include <random>

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

AffineResult CameraMotionEstimator::estimate_affine_partial_ransac(
    const std::vector<Point2D>& from_features,
    const std::vector<Point2D>& to_features,
    double reprojection_threshold,
    int max_iterations
) const noexcept {
    AffineResult result;
    result.matrix = {1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 1.0f};
    result.inlier_count = 0;
    result.success = false;

    if (from_features.empty() || to_features.empty() || from_features.size() != to_features.size()) {
        return result;
    }

    // Filter finite pairs
    std::vector<Point2D> pts_from;
    std::vector<Point2D> pts_to;
    pts_from.reserve(from_features.size());
    pts_to.reserve(to_features.size());

    for (size_t i = 0; i < from_features.size(); ++i) {
        if (from_features[i].is_finite() && to_features[i].is_finite()) {
            pts_from.push_back(from_features[i]);
            pts_to.push_back(to_features[i]);
        }
    }

    const size_t N = pts_from.size();
    if (N < 2) {
        if (N == 1) {
            float dx = static_cast<float>(pts_to[0].x - pts_from[0].x);
            float dy = static_cast<float>(pts_to[0].y - pts_from[0].y);
            result.matrix[2] = dx;
            result.matrix[5] = dy;
        }
        return result;
    }

    const double thresh_sq = reprojection_threshold * reprojection_threshold;

    // Minimal 2-point solver for 2D similarity transform: to = [a, -b; b, a] * from + [tx; ty]
    auto solve_2pts = [](const Point2D& p1, const Point2D& p2,
                         const Point2D& q1, const Point2D& q2,
                         double& a, double& b, double& tx, double& ty) noexcept -> bool {
        const double dx = p1.x - p2.x;
        const double dy = p1.y - p2.y;
        const double D = dx * dx + dy * dy;
        if (D < 1e-6) {
            return false;
        }
        const double du = q1.x - q2.x;
        const double dv = q1.y - q2.y;
        a = (dx * du + dy * dv) / D;
        b = (dx * dv - dy * du) / D;
        tx = q1.x - (a * p1.x - b * p1.y);
        ty = q1.y - (b * p1.x + a * p1.y);
        return true;
    };

    // If exactly 2 points: direct exact solution
    if (N == 2) {
        double a = 1.0, b = 0.0, tx = 0.0, ty = 0.0;
        if (solve_2pts(pts_from[0], pts_from[1], pts_to[0], pts_to[1], a, b, tx, ty)) {
            result.matrix = {
                static_cast<float>(a), static_cast<float>(-b), static_cast<float>(tx),
                static_cast<float>(b), static_cast<float>(a),  static_cast<float>(ty),
                0.0f,                  0.0f,                   1.0f
            };
            result.inlier_count = 2;
            result.success = true;
        }
        return result;
    }

    // RANSAC consensus loop
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist_idx(0, N - 1);

    int best_inlier_count = 0;
    std::vector<uint8_t> best_inlier_mask(N, 0);
    std::vector<uint8_t> current_mask(N, 0);

    const int iters = std::max(10, max_iterations);
    for (int iter = 0; iter < iters; ++iter) {
        size_t idx1 = dist_idx(rng);
        size_t idx2 = dist_idx(rng);
        int retries = 0;
        while (idx1 == idx2 && retries < 10) {
            idx2 = dist_idx(rng);
            retries++;
        }
        if (idx1 == idx2) continue;

        double a = 1.0, b = 0.0, tx = 0.0, ty = 0.0;
        if (!solve_2pts(pts_from[idx1], pts_from[idx2], pts_to[idx1], pts_to[idx2], a, b, tx, ty)) {
            continue;
        }

        int inliers = 0;
        for (size_t i = 0; i < N; ++i) {
            const double pred_u = a * pts_from[i].x - b * pts_from[i].y + tx;
            const double pred_v = b * pts_from[i].x + a * pts_from[i].y + ty;
            const double err_u = pred_u - pts_to[i].x;
            const double err_v = pred_v - pts_to[i].y;
            const double err_sq = err_u * err_u + err_v * err_v;

            if (err_sq <= thresh_sq) {
                current_mask[i] = 1;
                inliers++;
            } else {
                current_mask[i] = 0;
            }
        }

        if (inliers > best_inlier_count) {
            best_inlier_count = inliers;
            best_inlier_mask = current_mask;
            if (inliers >= static_cast<int>(N * 0.90) && inliers >= 10) {
                break;
            }
        }
    }

    if (best_inlier_count < 3) {
        // Fallback to median translation
        CameraMotion motion = estimate_from_features(pts_from, pts_to);
        result.matrix[2] = static_cast<float>(-motion.dx); // Note: motion is old - new, affine maps from -> to
        result.matrix[5] = static_cast<float>(-motion.dy);
        result.inlier_count = best_inlier_count;
        result.success = false;
        return result;
    }

    // Umeyama closed-form least-squares refinement over all consensus inliers
    double sum_px = 0.0, sum_py = 0.0;
    double sum_qx = 0.0, sum_qy = 0.0;
    for (size_t i = 0; i < N; ++i) {
        if (best_inlier_mask[i]) {
            sum_px += pts_from[i].x;
            sum_py += pts_from[i].y;
            sum_qx += pts_to[i].x;
            sum_qy += pts_to[i].y;
        }
    }

    const double K = static_cast<double>(best_inlier_count);
    const double mean_px = sum_px / K;
    const double mean_py = sum_py / K;
    const double mean_qx = sum_qx / K;
    const double mean_qy = sum_qy / K;

    double var_p = 0.0;
    double S_xx = 0.0;
    double S_yx = 0.0;

    for (size_t i = 0; i < N; ++i) {
        if (best_inlier_mask[i]) {
            const double cpx = pts_from[i].x - mean_px;
            const double cpy = pts_from[i].y - mean_py;
            const double cqx = pts_to[i].x - mean_qx;
            const double cqy = pts_to[i].y - mean_qy;

            var_p += (cpx * cpx + cpy * cpy);
            S_xx  += (cqx * cpx + cqy * cpy);
            S_yx  += (cqy * cpx - cqx * cpy);
        }
    }

    if (var_p > 1e-6) {
        const double a = S_xx / var_p;
        const double b = S_yx / var_p;
        const double tx = mean_qx - (a * mean_px - b * mean_py);
        const double ty = mean_qy - (b * mean_px + a * mean_py);

        result.matrix = {
            static_cast<float>(a), static_cast<float>(-b), static_cast<float>(tx),
            static_cast<float>(b), static_cast<float>(a),  static_cast<float>(ty),
            0.0f,                  0.0f,                   1.0f
        };
        result.inlier_count = best_inlier_count;
        result.success = true;
    } else {
        result.matrix[2] = static_cast<float>(mean_qx - mean_px);
        result.matrix[5] = static_cast<float>(mean_qy - mean_py);
        result.inlier_count = best_inlier_count;
        result.success = false;
    }

    return result;
}

} // namespace football_cv
