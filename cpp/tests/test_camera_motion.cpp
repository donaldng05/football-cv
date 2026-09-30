#include "football_cv/camera_motion.hpp"

#include <gtest/gtest.h>

using namespace football_cv;

class CameraMotionTest : public ::testing::Test {};

TEST_F(CameraMotionTest, DefaultParameters) {
    CameraMotionEstimator estimator;
    EXPECT_DOUBLE_EQ(estimator.minimum_distance(), 5.0);
    EXPECT_DOUBLE_EQ(estimator.scene_cut_threshold(), 80.0);
}

TEST_F(CameraMotionTest, EmptyOrMismatchedFeatures) {
    CameraMotionEstimator estimator;
    std::vector<Point2D> pts1 = {Point2D{10.0, 10.0}};
    std::vector<Point2D> pts2 = {};

    CameraMotion motion1 = estimator.estimate_from_features({}, {});
    EXPECT_DOUBLE_EQ(motion1.dx, 0.0);
    EXPECT_DOUBLE_EQ(motion1.dy, 0.0);
    EXPECT_FALSE(motion1.is_scene_cut);

    CameraMotion motion2 = estimator.estimate_from_features(pts1, pts2);
    EXPECT_DOUBLE_EQ(motion2.dx, 0.0);
    EXPECT_DOUBLE_EQ(motion2.dy, 0.0);
    EXPECT_FALSE(motion2.is_scene_cut);
}

TEST_F(CameraMotionTest, SubThresholdMotionFiltered) {
    CameraMotionEstimator estimator(5.0, 80.0);

    // Displacement of 2.0 pixels is below 5.0px minimum threshold
    std::vector<Point2D> old_pts = {Point2D{100.0, 100.0}};
    std::vector<Point2D> new_pts = {Point2D{102.0, 100.0}};

    CameraMotion motion = estimator.estimate_from_features(old_pts, new_pts);
    EXPECT_DOUBLE_EQ(motion.dx, 0.0);
    EXPECT_DOUBLE_EQ(motion.dy, 0.0);
    EXPECT_FALSE(motion.is_scene_cut);
}

TEST_F(CameraMotionTest, ValidCameraPanEstimated) {
    CameraMotionEstimator estimator(5.0, 80.0);

    // Displacement of 10px horizontal and 5px vertical (magnitude ~11.18px)
    std::vector<Point2D> old_pts = {Point2D{100.0, 100.0}};
    std::vector<Point2D> new_pts = {Point2D{90.0, 95.0}};

    CameraMotion motion = estimator.estimate_from_features(old_pts, new_pts);
    EXPECT_DOUBLE_EQ(motion.dx, 10.0);
    EXPECT_DOUBLE_EQ(motion.dy, 5.0);
    EXPECT_FALSE(motion.is_scene_cut);
}

TEST_F(CameraMotionTest, SceneCutDetected) {
    CameraMotionEstimator estimator(5.0, 80.0);

    // Large displacement > 80px indicates a broadcast camera switch/cut
    std::vector<Point2D> old_pts = {Point2D{100.0, 100.0}};
    std::vector<Point2D> new_pts = {Point2D{200.0, 200.0}};

    CameraMotion motion = estimator.estimate_from_features(old_pts, new_pts);
    EXPECT_DOUBLE_EQ(motion.dx, 0.0);
    EXPECT_DOUBLE_EQ(motion.dy, 0.0);
    EXPECT_TRUE(motion.is_scene_cut);
}

TEST_F(CameraMotionTest, MarginFilteringIsolatesBorders) {
    std::vector<Point2D> points = {
        Point2D{10.0, 500.0},  // Left margin [0, 20]
        Point2D{500.0, 500.0}, // Center pitch (player running) -> should be excluded
        Point2D{950.0, 400.0}  // Right margin [900, 1050]
    };

    auto filtered = CameraMotionEstimator::filter_margin_points(points, 1920.0);
    ASSERT_EQ(filtered.size(), 2u);
    EXPECT_DOUBLE_EQ(filtered[0].x, 10.0);
    EXPECT_DOUBLE_EQ(filtered[1].x, 950.0);
}

TEST_F(CameraMotionTest, MotionAccumulation) {
    std::vector<CameraMotion> steps = {
        CameraMotion{10.0, 2.0, false},
        CameraMotion{5.0, 3.0, false},
        CameraMotion{0.0, 0.0, true},  // Scene cut
        CameraMotion{2.0, 1.0, false}
    };

    auto accumulated = CameraMotionEstimator::accumulate_motion(steps);
    ASSERT_EQ(accumulated.size(), 4u);
    EXPECT_EQ(accumulated[0], Point2D(10.0, 2.0));
    EXPECT_EQ(accumulated[1], Point2D(15.0, 5.0));
    EXPECT_EQ(accumulated[2], Point2D(15.0, 5.0)); // Maintained on cut
    EXPECT_EQ(accumulated[3], Point2D(17.0, 6.0));
}

TEST_F(CameraMotionTest, EstimateAffinePartialRansacExactFit) {
    CameraMotionEstimator estimator;

    // Known Sim(2) parameters: scale=1.02, angle=0.03 rad (~1.72 deg), tx=15.0, ty=-8.0
    const double s = 1.02;
    const double theta = 0.03;
    const double a_true = s * std::cos(theta);
    const double b_true = s * std::sin(theta);
    const double tx_true = 15.0;
    const double ty_true = -8.0;

    std::vector<Point2D> from_pts = {
        Point2D{100.0, 150.0},
        Point2D{250.0, 300.0},
        Point2D{400.0, 120.0},
        Point2D{550.0, 480.0},
        Point2D{700.0, 210.0},
        Point2D{850.0, 600.0},
        Point2D{920.0, 350.0},
        Point2D{150.0, 700.0}
    };

    std::vector<Point2D> to_pts;
    for (const auto& p : from_pts) {
        double u = a_true * p.x - b_true * p.y + tx_true;
        double v = b_true * p.x + a_true * p.y + ty_true;
        to_pts.emplace_back(u, v);
    }

    AffineResult res = estimator.estimate_affine_partial_ransac(from_pts, to_pts, 3.0, 100);
    EXPECT_TRUE(res.success);
    EXPECT_EQ(res.inlier_count, static_cast<int>(from_pts.size()));

    // Verify matrix values: [a, -b, tx,  b, a, ty,  0, 0, 1]
    EXPECT_NEAR(res.matrix[0], static_cast<float>(a_true), 1e-3f);
    EXPECT_NEAR(res.matrix[1], static_cast<float>(-b_true), 1e-3f);
    EXPECT_NEAR(res.matrix[2], static_cast<float>(tx_true), 1e-2f);
    EXPECT_NEAR(res.matrix[3], static_cast<float>(b_true), 1e-3f);
    EXPECT_NEAR(res.matrix[4], static_cast<float>(a_true), 1e-3f);
    EXPECT_NEAR(res.matrix[5], static_cast<float>(ty_true), 1e-2f);
    EXPECT_FLOAT_EQ(res.matrix[6], 0.0f);
    EXPECT_FLOAT_EQ(res.matrix[7], 0.0f);
    EXPECT_FLOAT_EQ(res.matrix[8], 1.0f);
}

TEST_F(CameraMotionTest, EstimateAffinePartialRansacOutlierRejection) {
    CameraMotionEstimator estimator;

    // Pure translation + small scale: a=1.0, b=0.0, tx=10.0, ty=5.0
    const double tx_true = 10.0;
    const double ty_true = 5.0;

    std::vector<Point2D> from_pts;
    std::vector<Point2D> to_pts;

    // 15 clean inliers
    for (int i = 0; i < 15; ++i) {
        double x = 100.0 + i * 50.0;
        double y = 200.0 + (i % 3) * 80.0;
        from_pts.emplace_back(x, y);
        to_pts.emplace_back(x + tx_true, y + ty_true);
    }

    // 5 strong outliers
    for (int i = 0; i < 5; ++i) {
        from_pts.emplace_back(200.0 + i * 30.0, 300.0 + i * 20.0);
        to_pts.emplace_back(999.0 + i * 100.0, -888.0 - i * 50.0);
    }

    AffineResult res = estimator.estimate_affine_partial_ransac(from_pts, to_pts, 3.0, 150);
    EXPECT_TRUE(res.success);
    EXPECT_GE(res.inlier_count, 15);
    EXPECT_NEAR(res.matrix[0], 1.0f, 1e-2f);
    EXPECT_NEAR(res.matrix[1], 0.0f, 1e-2f);
    EXPECT_NEAR(res.matrix[2], static_cast<float>(tx_true), 0.2f);
    EXPECT_NEAR(res.matrix[3], 0.0f, 1e-2f);
    EXPECT_NEAR(res.matrix[4], 1.0f, 1e-2f);
    EXPECT_NEAR(res.matrix[5], static_cast<float>(ty_true), 0.2f);
}

TEST_F(CameraMotionTest, EstimateAffinePartialRansacDegenerate) {
    CameraMotionEstimator estimator;

    // Empty inputs
    AffineResult res_empty = estimator.estimate_affine_partial_ransac({}, {});
    EXPECT_FALSE(res_empty.success);
    EXPECT_EQ(res_empty.inlier_count, 0);

    // Single point
    std::vector<Point2D> p1 = {Point2D{10.0, 10.0}};
    std::vector<Point2D> q1 = {Point2D{20.0, 15.0}};
    AffineResult res_single = estimator.estimate_affine_partial_ransac(p1, q1);
    EXPECT_FALSE(res_single.success);

    // Identical duplicate points (D == 0)
    std::vector<Point2D> p_dup = {Point2D{50.0, 50.0}, Point2D{50.0, 50.0}};
    std::vector<Point2D> q_dup = {Point2D{60.0, 60.0}, Point2D{60.0, 60.0}};
    AffineResult res_dup = estimator.estimate_affine_partial_ransac(p_dup, q_dup);
    EXPECT_FALSE(res_dup.success);
}
