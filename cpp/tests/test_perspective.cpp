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

TEST_F(PerspectiveTest, EffectiveHomographyCalculation) {
    PerspectiveTransformer transformer;
    auto h_static = transformer.get_effective_homography();
    EXPECT_DOUBLE_EQ(h_static[8], 1.0);

    // Identity camera matrix yields identical homography
    std::array<double, 9> eye = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    auto h_eye = transformer.get_effective_homography(eye);
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(h_eye[i], h_static[i], 1e-9);
    }
}

TEST_F(PerspectiveTest, DynamicCameraMatrixCompounding) {
    PerspectiveTransformer transformer;
    Point2D pt{500.0, 500.0};

    // Camera pan translation matrix (dx = 50, dy = 30)
    std::array<double, 9> cam_trans = {
        1.0, 0.0, 50.0,
        0.0, 1.0, 30.0,
        0.0, 0.0, 1.0
    };

    auto res_static = transformer.transform_point(Point2D{550.0, 530.0}, OutOfBoundsPolicy::Extrapolate);
    auto res_compounded = transformer.transform_point(pt, OutOfBoundsPolicy::Extrapolate, cam_trans);

    ASSERT_TRUE(res_static.has_value());
    ASSERT_TRUE(res_compounded.has_value());
    EXPECT_NEAR(res_compounded->x, res_static->x, 1e-5);
    EXPECT_NEAR(res_compounded->y, res_static->y, 1e-5);
}

TEST_F(PerspectiveTest, HorizonSingularityRejection) {
    PerspectiveTransformer transformer;

    // Extreme camera transform that drives denominator w' <= 0
    std::array<double, 9> extreme_cam = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        -10.0, -10.0, -1000.0
    };

    Point2D pt{500.0, 500.0};
    auto res = transformer.transform_point(pt, OutOfBoundsPolicy::Extrapolate, extreme_cam);
    EXPECT_FALSE(res.has_value());
}

TEST_F(PerspectiveTest, OutOfBoundsPoliciesHandling) {
    PerspectiveTransformer transformer;
    // (0, 0) projects outside the default calibrated pitch boundary
    Point2D outside_pt{0.0, 0.0};

    // 1. Strict rejects
    auto res_strict = transformer.transform_point(outside_pt, OutOfBoundsPolicy::Strict);
    EXPECT_FALSE(res_strict.has_value());

    // 2. Extrapolate returns coordinates directly
    auto res_extrap = transformer.transform_point(outside_pt, OutOfBoundsPolicy::Extrapolate);
    ASSERT_TRUE(res_extrap.has_value());

    // 3. Clip clamps to pitch court limits [0, 68] x [0, 23.32]
    auto res_clip = transformer.transform_point(outside_pt, OutOfBoundsPolicy::Clip);
    ASSERT_TRUE(res_clip.has_value());
    EXPECT_GE(res_clip->x, 0.0);
    EXPECT_LE(res_clip->x, 68.0);
    EXPECT_GE(res_clip->y, 0.0);
    EXPECT_LE(res_clip->y, 23.32);
}

TEST_F(PerspectiveTest, BatchRawBufferTransformation) {
    PerspectiveTransformer transformer;

    std::vector<double> in_pts = {
        500.0, 500.0,
        0.0, 0.0,
        600.0, 400.0
    };
    std::vector<double> out_pts(6, 0.0);
    bool mask[3] = {false, false, false};

    transformer.transform_points_batch_raw(
        in_pts.data(), 3, out_pts.data(),
        OutOfBoundsPolicy::Strict, std::nullopt, mask
    );

    EXPECT_TRUE(mask[0]);
    EXPECT_FALSE(mask[1]);
    EXPECT_TRUE(mask[2]);

    EXPECT_TRUE(std::isfinite(out_pts[0]));
    EXPECT_TRUE(std::isnan(out_pts[2]));
    EXPECT_TRUE(std::isfinite(out_pts[4]));
}
