#pragma once

#include "types.hpp"

#include <array>
#include <optional>
#include <vector>

namespace football_cv {

/**
 * @brief Projects pixel broadcast coordinates to metric pitch coordinates (meters)
 *        via 4-point planar homography with calibrated polygon boundary tests.
 */
class PerspectiveTransformer {
public:
    /**
     * @brief Construct transformer with pixel vertices and pitch dimensions.
     *
     * @param pixel_vertices Exactly 4 boundary vertices in image coordinates [top-left, bottom-left, bottom-right, top-right].
     * @param court_width Real-world pitch width in meters (default: 68.0).
     * @param court_length Real-world pitch length in meters (default: 23.32).
     */
    explicit PerspectiveTransformer(
        const std::vector<Point2D>& pixel_vertices = default_pixel_vertices(),
        double court_width = 68.0,
        double court_length = 23.32
    );

    /**
     * @brief Compute effective 3x3 homography compounded with an optional camera matrix.
     *        H_eff = H_{0->pitch} @ camera_matrix.
     *
     * @param camera_matrix Optional 3x3 row-major inter-frame camera transformation H_{t -> 0}.
     * @return 3x3 row-major effective homography matrix.
     */
    std::array<double, 9> get_effective_homography(
        const std::optional<std::array<double, 9>>& camera_matrix = std::nullopt
    ) const noexcept;

    /**
     * @brief Project a 2D image coordinate to pitch metric coordinates with policy and camera matrix.
     *
     * @param point Image coordinate (pixel x, y).
     * @param policy Out of bounds policy (Strict, Clip, Extrapolate).
     * @param camera_matrix Optional 3x3 row-major camera matrix H_{t -> 0}.
     * @return Point2D in pitch meters, or std::nullopt if point is rejected or singularity occurs.
     */
    std::optional<Point2D> transform_point(
        const Point2D& point,
        OutOfBoundsPolicy policy,
        const std::optional<std::array<double, 9>>& camera_matrix = std::nullopt
    ) const noexcept;

    /**
     * @brief Backwards-compatible overload using boolean check_boundary.
     *
     * @param point Image coordinate (pixel x, y).
     * @param check_boundary If true, uses OutOfBoundsPolicy::Strict; otherwise Extrapolate.
     * @return Point2D in pitch meters, or std::nullopt if point is outside polygon.
     */
    std::optional<Point2D> transform_point(const Point2D& point, bool check_boundary = true) const noexcept;

    /**
     * @brief Batch project multiple image coordinates with policy and camera matrix.
     */
    std::vector<std::optional<Point2D>> transform_points(
        const std::vector<Point2D>& points,
        OutOfBoundsPolicy policy = OutOfBoundsPolicy::Strict,
        const std::optional<std::array<double, 9>>& camera_matrix = std::nullopt
    ) const;

    /**
     * @brief High-performance contiguous batch transformation for (N, 2) double buffers.
     *
     * @param points_in Pointer to contiguous (N, 2) row-major pixel coordinates [x, y].
     * @param count Number of points N.
     * @param points_out Pointer to preallocated (N, 2) output buffer for metric coordinates.
     * @param policy Out of bounds policy.
     * @param camera_matrix Optional 3x3 camera matrix.
     * @param valid_mask Optional output buffer of size N (true if valid / inside boundary).
     */
    void transform_points_batch_raw(
        const double* points_in,
        size_t count,
        double* points_out,
        OutOfBoundsPolicy policy = OutOfBoundsPolicy::Extrapolate,
        const std::optional<std::array<double, 9>>& camera_matrix = std::nullopt,
        bool* valid_mask = nullptr
    ) const noexcept;

    /**
     * @brief Check if a point lies within or on the calibrated boundary polygon.
     */
    bool is_point_inside(const Point2D& point) const noexcept;

    double court_width() const noexcept { return court_width_; }
    double court_length() const noexcept { return court_length_; }
    const std::vector<Point2D>& pixel_vertices() const noexcept { return pixel_vertices_; }
    const std::array<double, 9>& homography_matrix() const noexcept { return h_; }

    static std::vector<Point2D> default_pixel_vertices();

private:
    double court_width_;
    double court_length_;
    std::vector<Point2D> pixel_vertices_;
    std::vector<Point2D> target_vertices_;
    std::array<double, 9> h_{}; // 3x3 row-major homography matrix

    void compute_homography();
};

} // namespace football_cv
