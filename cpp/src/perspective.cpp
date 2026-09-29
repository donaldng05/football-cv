#include "football_cv/perspective.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace football_cv {

std::vector<Point2D> PerspectiveTransformer::default_pixel_vertices() {
    return {
        Point2D{110.0, 1035.0},
        Point2D{265.0, 275.0},
        Point2D{910.0, 260.0},
        Point2D{1640.0, 915.0}
    };
}

PerspectiveTransformer::PerspectiveTransformer(
    const std::vector<Point2D>& pixel_vertices,
    double court_width,
    double court_length
)
    : court_width_(court_width),
      court_length_(court_length),
      pixel_vertices_(pixel_vertices.empty() ? default_pixel_vertices() : pixel_vertices),
      target_vertices_({
          Point2D{0.0, 0.0},
          Point2D{0.0, court_length},
          Point2D{court_width, court_length},
          Point2D{court_width, 0.0}
      })
{
    if (pixel_vertices_.size() != 4) {
        throw std::invalid_argument("PerspectiveTransformer requires exactly 4 pixel vertices.");
    }
    compute_homography();
}

void PerspectiveTransformer::compute_homography() {
    // Set up 8x8 linear system A * h = b to find 3x3 homography matrix with h[8] = 1.0
    constexpr int N = 8;
    std::array<std::array<double, N + 1>, N> m{};

    for (int i = 0; i < 4; ++i) {
        const double x = pixel_vertices_[i].x;
        const double y = pixel_vertices_[i].y;
        const double u = target_vertices_[i].x;
        const double v = target_vertices_[i].y;

        // Equation 1: x*h0 + y*h1 + h2 - u*x*h6 - u*y*h7 = u
        m[2 * i][0] = x;
        m[2 * i][1] = y;
        m[2 * i][2] = 1.0;
        m[2 * i][3] = 0.0;
        m[2 * i][4] = 0.0;
        m[2 * i][5] = 0.0;
        m[2 * i][6] = -u * x;
        m[2 * i][7] = -u * y;
        m[2 * i][8] = u;

        // Equation 2: x*h3 + y*h4 + h5 - v*x*h6 - v*y*h7 = v
        m[2 * i + 1][0] = 0.0;
        m[2 * i + 1][1] = 0.0;
        m[2 * i + 1][2] = 0.0;
        m[2 * i + 1][3] = x;
        m[2 * i + 1][4] = y;
        m[2 * i + 1][5] = 1.0;
        m[2 * i + 1][6] = -v * x;
        m[2 * i + 1][7] = -v * y;
        m[2 * i + 1][8] = v;
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < N; ++col) {
        int max_row = col;
        double max_val = std::abs(m[col][col]);
        for (int r = col + 1; r < N; ++r) {
            if (std::abs(m[r][col]) > max_val) {
                max_val = std::abs(m[r][col]);
                max_row = r;
            }
        }

        if (max_val < 1e-12) {
            throw std::runtime_error("Degenerate vertices: cannot compute perspective homography.");
        }

        if (max_row != col) {
            std::swap(m[col], m[max_row]);
        }

        // Eliminate below and above
        for (int r = 0; r < N; ++r) {
            if (r != col) {
                const double factor = m[r][col] / m[col][col];
                for (int c = col; c <= N; ++c) {
                    m[r][c] -= factor * m[col][c];
                }
            }
        }
    }

    // Extract solutions
    for (int i = 0; i < N; ++i) {
        h_[i] = m[i][N] / m[i][i];
    }
    h_[8] = 1.0;
}

std::array<double, 9> PerspectiveTransformer::get_effective_homography(
    const std::optional<std::array<double, 9>>& camera_matrix
) const noexcept {
    if (!camera_matrix.has_value()) {
        return h_;
    }
    const auto& cam = *camera_matrix;
    std::array<double, 9> h_eff{};
    // Matrix multiplication: H_eff = H @ H_cam (both 3x3 row-major)
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            h_eff[r * 3 + c] = h_[r * 3 + 0] * cam[0 * 3 + c] +
                               h_[r * 3 + 1] * cam[1 * 3 + c] +
                               h_[r * 3 + 2] * cam[2 * 3 + c];
        }
    }
    return h_eff;
}

bool PerspectiveTransformer::is_point_inside(const Point2D& pt) const noexcept {
    if (!pt.is_finite()) {
        return false;
    }

    constexpr double eps = 1e-5;
    const size_t count = pixel_vertices_.size();

    // Check if point is on any vertex or boundary edge (matches cv2.pointPolygonTest >= 0)
    for (size_t i = 0, j = count - 1; i < count; j = i++) {
        const double xi = pixel_vertices_[i].x;
        const double yi = pixel_vertices_[i].y;
        const double xj = pixel_vertices_[j].x;
        const double yj = pixel_vertices_[j].y;

        // Check if point matches vertex
        if (std::abs(pt.x - xi) < eps && std::abs(pt.y - yi) < eps) {
            return true;
        }

        // Check if point lies on segment between (xi, yi) and (xj, yj)
        const double cross = (pt.y - yi) * (xj - xi) - (pt.x - xi) * (yj - yi);
        if (std::abs(cross) < 1e-3) {
            const double dot = (pt.x - xi) * (xj - xi) + (pt.y - yi) * (yj - yi);
            const double len_sq = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi);
            if (dot >= -eps && dot <= len_sq + eps) {
                return true;
            }
        }
    }

    // Ray-casting algorithm to test if point is inside calibrated 4-vertex polygon
    bool inside = false;
    for (size_t i = 0, j = count - 1; i < count; j = i++) {
        const double xi = pixel_vertices_[i].x;
        const double yi = pixel_vertices_[i].y;
        const double xj = pixel_vertices_[j].x;
        const double yj = pixel_vertices_[j].y;

        const bool intersect = ((yi > pt.y) != (yj > pt.y)) &&
            (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

std::optional<Point2D> PerspectiveTransformer::transform_point(
    const Point2D& point,
    OutOfBoundsPolicy policy,
    const std::optional<std::array<double, 9>>& camera_matrix
) const noexcept {
    if (!point.is_finite()) {
        return std::nullopt;
    }

    const auto h_eff = get_effective_homography(camera_matrix);
    const double x = point.x;
    const double y = point.y;

    // Check projective horizon singularity (w' = h31*x + h32*y + h33)
    const double w_prime = h_eff[6] * x + h_eff[7] * y + h_eff[8];
    if (w_prime <= 1e-6) {
        return std::nullopt;
    }

    double tx = (h_eff[0] * x + h_eff[1] * y + h_eff[2]) / w_prime;
    double ty = (h_eff[3] * x + h_eff[4] * y + h_eff[5]) / w_prime;

    bool is_inside = false;
    if (!camera_matrix.has_value()) {
        is_inside = is_point_inside(point);
    } else {
        // When camera motion is compensated, evaluate against invariant pitch metric bounds
        is_inside = (tx >= 0.0 && tx <= court_width_ && ty >= 0.0 && ty <= court_length_);
    }

    if (!is_inside && policy == OutOfBoundsPolicy::Strict) {
        return std::nullopt;
    }

    if (!is_inside && policy == OutOfBoundsPolicy::Clip) {
        tx = std::clamp(tx, 0.0, court_width_);
        ty = std::clamp(ty, 0.0, court_length_);
    }

    return Point2D{tx, ty};
}

std::optional<Point2D> PerspectiveTransformer::transform_point(
    const Point2D& point,
    bool check_boundary
) const noexcept {
    return transform_point(
        point,
        check_boundary ? OutOfBoundsPolicy::Strict : OutOfBoundsPolicy::Extrapolate,
        std::nullopt
    );
}

std::vector<std::optional<Point2D>> PerspectiveTransformer::transform_points(
    const std::vector<Point2D>& points,
    OutOfBoundsPolicy policy,
    const std::optional<std::array<double, 9>>& camera_matrix
) const {
    std::vector<std::optional<Point2D>> results;
    results.reserve(points.size());
    for (const auto& pt : points) {
        results.push_back(transform_point(pt, policy, camera_matrix));
    }
    return results;
}

void PerspectiveTransformer::transform_points_batch_raw(
    const double* points_in,
    size_t count,
    double* points_out,
    OutOfBoundsPolicy policy,
    const std::optional<std::array<double, 9>>& camera_matrix,
    bool* valid_mask
) const noexcept {
    if (!points_in || !points_out || count == 0) {
        return;
    }

    const auto h_eff = get_effective_homography(camera_matrix);
    const bool has_cam = camera_matrix.has_value();

    for (size_t i = 0; i < count; ++i) {
        const double x = points_in[2 * i + 0];
        const double y = points_in[2 * i + 1];

        if (!std::isfinite(x) || !std::isfinite(y)) {
            points_out[2 * i + 0] = std::numeric_limits<double>::quiet_NaN();
            points_out[2 * i + 1] = std::numeric_limits<double>::quiet_NaN();
            if (valid_mask) valid_mask[i] = false;
            continue;
        }

        const double w_prime = h_eff[6] * x + h_eff[7] * y + h_eff[8];
        if (w_prime <= 1e-6) {
            points_out[2 * i + 0] = std::numeric_limits<double>::quiet_NaN();
            points_out[2 * i + 1] = std::numeric_limits<double>::quiet_NaN();
            if (valid_mask) valid_mask[i] = false;
            continue;
        }

        double tx = (h_eff[0] * x + h_eff[1] * y + h_eff[2]) / w_prime;
        double ty = (h_eff[3] * x + h_eff[4] * y + h_eff[5]) / w_prime;

        bool is_inside = false;
        if (!has_cam) {
            is_inside = is_point_inside(Point2D{x, y});
        } else {
            is_inside = (tx >= 0.0 && tx <= court_width_ && ty >= 0.0 && ty <= court_length_);
        }

        if (!is_inside && policy == OutOfBoundsPolicy::Strict) {
            points_out[2 * i + 0] = std::numeric_limits<double>::quiet_NaN();
            points_out[2 * i + 1] = std::numeric_limits<double>::quiet_NaN();
            if (valid_mask) valid_mask[i] = false;
            continue;
        }

        if (!is_inside && policy == OutOfBoundsPolicy::Clip) {
            tx = std::clamp(tx, 0.0, court_width_);
            ty = std::clamp(ty, 0.0, court_length_);
        }

        points_out[2 * i + 0] = tx;
        points_out[2 * i + 1] = ty;
        if (valid_mask) valid_mask[i] = true;
    }
}

} // namespace football_cv
