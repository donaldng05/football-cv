#include "football_cv/perspective.hpp"

#include <gtest/gtest.h>

using namespace football_cv;

class PerspectiveTest : public ::testing::Test {};

TEST_F(PerspectiveTest, DefaultInitialization) {
    PerspectiveTransformer transformer;
    EXPECT_DOUBLE_EQ(transformer.court_width(), 68.0);
    EXPECT_DOUBLE_EQ(transformer.court_length(), 23.32);
    EXPECT_EQ(transformer.pixel_vertices().size(), 4u);
}

TEST_F(PerspectiveTest, BoundaryPolygonContainment) {
    PerspectiveTransformer transformer;

    // Point (500, 500) lies comfortably inside the default broadcast pitch polygon
    Point2D inside_pt{500.0, 500.0};
    EXPECT_TRUE(transformer.is_point_inside(inside_pt));

    // Points outside the calibrated camera broadcast bounds
    Point2D outside_pt1{0.0, 0.0};
    EXPECT_FALSE(transformer.is_point_inside(outside_pt1));

    Point2D outside_pt2{2000.0, 2000.0};
    EXPECT_FALSE(transformer.is_point_inside(outside_pt2));
}

TEST_F(PerspectiveTest, PointTransformationInsidePolygon) {
    PerspectiveTransformer transformer;

    Point2D pt{500.0, 500.0};
    auto transformed = transformer.transform_point(pt);

    ASSERT_TRUE(transformed.has_value());
    // Transformed pitch coordinate must be positive and within pitch bounds
    EXPECT_GE(transformed->x, 0.0);
    EXPECT_LE(transformed->x, 68.0);
    EXPECT_GE(transformed->y, 0.0);
    EXPECT_LE(transformed->y, 23.32);
}

TEST_F(PerspectiveTest, PointOutsidePolygonReturnsNullopt) {
    PerspectiveTransformer transformer;

    Point2D outside_pt{0.0, 0.0};
    auto transformed = transformer.transform_point(outside_pt);

    EXPECT_FALSE(transformed.has_value());
}

TEST_F(PerspectiveTest, BatchTransformation) {
    PerspectiveTransformer transformer;

    std::vector<Point2D> pts = {
        Point2D{500.0, 500.0}, // Inside
        Point2D{0.0, 0.0},     // Outside
        Point2D{600.0, 400.0}  // Inside
    };

    auto results = transformer.transform_points(pts);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].has_value());
    EXPECT_FALSE(results[1].has_value());
    EXPECT_TRUE(results[2].has_value());
}

TEST_F(PerspectiveTest, InvalidVerticesCountThrows) {
    std::vector<Point2D> three_vertices = {
        Point2D{0.0, 0.0},
        Point2D{10.0, 0.0},
        Point2D{10.0, 10.0}
    };

    EXPECT_THROW(
        (void)PerspectiveTransformer{three_vertices},
        std::invalid_argument
    );
}
