#pragma once

#include <cmath>
#include <limits>
#include <utility>

namespace football_cv {

/**
 * @brief Represents a 2D coordinate point with floating-point precision.
 */
struct Point2D {
    double x{0.0};
    double y{0.0};

    constexpr Point2D() noexcept = default;
    constexpr Point2D(double x_coord, double y_coord) noexcept : x(x_coord), y(y_coord) {}

    bool operator==(const Point2D& other) const noexcept {
        constexpr double epsilon = 1e-9;
        return std::abs(x - other.x) < epsilon && std::abs(y - other.y) < epsilon;
    }

    bool operator!=(const Point2D& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Calculate Euclidean distance to another 2D point.
     */
    double distance_to(const Point2D& other) const noexcept {
        const double dx = x - other.x;
        const double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    /**
     * @brief Calculate signed (dx, dy) displacement from other to this point.
     */
    std::pair<double, double> displacement_from(const Point2D& other) const noexcept {
        return {x - other.x, y - other.y};
    }

    /**
     * @brief Check whether both coordinates are finite numbers (not NaN or Inf).
     */
    bool is_finite() const noexcept {
        return std::isfinite(x) && std::isfinite(y);
    }
};

/**
 * @brief Axis-aligned 2D bounding box defined by [x1, y1, x2, y2].
 */
struct BoundingBox {
    double x1{0.0};
    double y1{0.0};
    double x2{0.0};
    double y2{0.0};

    constexpr BoundingBox() noexcept = default;
    constexpr BoundingBox(double left, double top, double right, double bottom) noexcept
        : x1(left), y1(top), x2(right), y2(bottom) {}

    bool operator==(const BoundingBox& other) const noexcept {
        constexpr double epsilon = 1e-9;
        return std::abs(x1 - other.x1) < epsilon &&
               std::abs(y1 - other.y1) < epsilon &&
               std::abs(x2 - other.x2) < epsilon &&
               std::abs(y2 - other.y2) < epsilon;
    }

    bool operator!=(const BoundingBox& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Calculate the geometric center of the bounding box.
     */
    Point2D center() const noexcept {
        return Point2D{(x1 + x2) * 0.5, (y1 + y2) * 0.5};
    }

    /**
     * @brief Calculate the bottom-center point representing player pitch contact.
     */
    Point2D foot_position() const noexcept {
        return Point2D{(x1 + x2) * 0.5, y2};
    }

    /**
     * @brief Horizontal width of the bounding box.
     */
    double width() const noexcept {
        return x2 - x1;
    }

    /**
     * @brief Vertical height of the bounding box.
     */
    double height() const noexcept {
        return y2 - y1;
    }

    /**
     * @brief Surface area of the bounding box. Returns 0 if invalid.
     */
    double area() const noexcept {
        if (!is_valid()) {
            return 0.0;
        }
        return width() * height();
    }

    /**
     * @brief Check if the box has valid positive dimensions and finite coordinates.
     */
    bool is_valid() const noexcept {
        return is_finite() && (x2 >= x1) && (y2 >= y1);
    }

    /**
     * @brief Check whether all bounding box coordinates are finite numbers.
     */
    bool is_finite() const noexcept {
        return std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2);
    }

    /**
     * @brief Check if a 2D point is contained within the bounding box (inclusive).
     */
    bool contains(const Point2D& pt) const noexcept {
        if (!is_valid() || !pt.is_finite()) {
            return false;
        }
        return (pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2);
    }
};

/**
 * @brief Represents estimated inter-frame camera translation and cut status.
 */
struct CameraMotion {
    double dx{0.0};
    double dy{0.0};
    bool is_scene_cut{false};
};

/**
 * @brief Represents an object detection with bounding box, confidence score, and class label.
 */
struct Detection {
    BoundingBox bbox;
    float confidence{0.0f};
    int class_id{-1};

    constexpr Detection() noexcept = default;
    constexpr Detection(const BoundingBox& b, float conf, int cls) noexcept
        : bbox(b), confidence(conf), class_id(cls) {}

    bool operator==(const Detection& other) const noexcept {
        constexpr float epsilon = 1e-5f;
        return bbox == other.bbox &&
               std::abs(confidence - other.confidence) < epsilon &&
               class_id == other.class_id;
    }

    bool operator!=(const Detection& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace football_cv
