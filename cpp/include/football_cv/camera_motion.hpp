#pragma once

#include "types.hpp"

#include <array>
#include <vector>

namespace football_cv {

/**
 * @brief Represents the result of an affine / similarity 2D RANSAC estimation.
 */
struct AffineResult {
    std::array<float, 9> matrix{1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f}; ///< Row-major 3x3 Sim(2) transformation matrix
    int inlier_count{0};                           ///< Number of consensus inliers
    bool success{false};                           ///< True if consensus met minimum inlier threshold
};

/**
 * @brief Compensates for camera pan and tilt movements by tracking feature displacements
 *        and isolating camera motion from moving pitch objects.
 */
class CameraMotionEstimator {
public:
    /**
     * @brief Construct CameraMotionEstimator with sensitivity thresholds.
     *
     * @param minimum_distance Minimum feature displacement (pixels) to register camera pan/tilt.
     * @param scene_cut_threshold Displacement magnitude above which a broadcast cut is declared.
     */
    explicit CameraMotionEstimator(
        double minimum_distance = 5.0,
        double scene_cut_threshold = 80.0
    ) noexcept;

    /**
     * @brief Estimate camera displacement between tracked feature correspondences.
     *
     * @param old_features Feature positions in previous frame.
     * @param new_features Corresponding feature positions in current frame.
     * @return CameraMotion containing (dx, dy) translation and scene cut flag.
     */
    CameraMotion estimate_from_features(
        const std::vector<Point2D>& old_features,
        const std::vector<Point2D>& new_features
    ) const noexcept;

    /**
     * @brief Estimate 2D Affine Partial (Sim(2) 4-DoF: scale, rotation, translation) matrix via RANSAC.
     *
     * Finds 3x3 transformation matrix mapping from_features to to_features:
     *   to_pt ~ [a, -b, tx; b, a, ty; 0, 0, 1] * from_pt
     *
     * @param from_features Source feature coordinates (e.g. current frame).
     * @param to_features Target feature coordinates (e.g. previous frame).
     * @param reprojection_threshold Maximum Euclidean distance in pixels to qualify as inlier (default: 3.0).
     * @param max_iterations Maximum RANSAC trials (default: 100).
     * @return AffineResult with 3x3 row-major transformation matrix and consensus metrics.
     */
    AffineResult estimate_affine_partial_ransac(
        const std::vector<Point2D>& from_features,
        const std::vector<Point2D>& to_features,
        double reprojection_threshold = 3.0,
        int max_iterations = 100
    ) const noexcept;

    /**
     * @brief Filter points to isolate vertical broadcast margins (excluding players on pitch).
     *
     * @param points Candidate points.
     * @param frame_width Width of the video frame.
     * @param left_margin_width Width of left margin band (default: 20px).
     * @param right_margin_start X coordinate where right margin starts (default: 900px).
     * @param right_margin_end X coordinate where right margin ends (default: 1050px).
     * @return Points lying within the valid margin strips.
     */
    static std::vector<Point2D> filter_margin_points(
        const std::vector<Point2D>& points,
        double frame_width,
        double left_margin_width = 20.0,
        double right_margin_start = 900.0,
        double right_margin_end = 1050.0
    );

    /**
     * @brief Accumulate cumulative camera displacement across a sequence of frames.
     */
    static std::vector<Point2D> accumulate_motion(
        const std::vector<CameraMotion>& motion_steps
    );

    double minimum_distance() const noexcept { return minimum_distance_; }
    double scene_cut_threshold() const noexcept { return scene_cut_threshold_; }

private:
    double minimum_distance_;
    double scene_cut_threshold_;
};

} // namespace football_cv
