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
     * @brief Project a 2D image coordinate to pitch metric coordinates.
     *
     * @param point Image coordinate (pixel x, y).
     * @return Point2D in pitch meters, or std::nullopt if point is outside polygon boundary.
     */
    std::optional<Point2D> transform_point(const Point2D& point) const noexcept;

    /**
     * @brief Batch project multiple image coordinates.
     */
    std::vector<std::optional<Point2D>> transform_points(const std::vector<Point2D>& points) const;

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
