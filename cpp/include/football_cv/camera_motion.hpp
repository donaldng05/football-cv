#pragma once

#include "types.hpp"

#include <vector>

namespace football_cv {

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
