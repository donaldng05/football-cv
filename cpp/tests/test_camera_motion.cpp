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
