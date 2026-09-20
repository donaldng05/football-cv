#include "football_cv/geometry.hpp"

#include <gtest/gtest.h>
#include <limits>

using namespace football_cv;

class GeometryTest : public ::testing::Test {};

TEST_F(GeometryTest, Point2DProperties) {
    Point2D p1{0.0, 0.0};
    Point2D p2{3.0, 4.0};

    EXPECT_DOUBLE_EQ(p1.distance_to(p2), 5.0);
    EXPECT_TRUE(p1 == Point2D(0.0, 0.0));
    EXPECT_TRUE(p1 != p2);

    auto [dx, dy] = p2.displacement_from(p1);
    EXPECT_DOUBLE_EQ(dx, 3.0);
    EXPECT_DOUBLE_EQ(dy, 4.0);

    EXPECT_TRUE(p1.is_finite());
    Point2D nan_pt{std::numeric_limits<double>::quiet_NaN(), 1.0};
    EXPECT_FALSE(nan_pt.is_finite());
}

TEST_F(GeometryTest, BoundingBoxCalculations) {
    BoundingBox bbox{100.0, 200.0, 300.0, 400.0};

    Point2D center = bbox.center();
    EXPECT_DOUBLE_EQ(center.x, 200.0);
    EXPECT_DOUBLE_EQ(center.y, 300.0);

    Point2D foot = bbox.foot_position();
    EXPECT_DOUBLE_EQ(foot.x, 200.0);
    EXPECT_DOUBLE_EQ(foot.y, 400.0);

    EXPECT_DOUBLE_EQ(bbox.width(), 200.0);
    EXPECT_DOUBLE_EQ(bbox.height(), 200.0);
    EXPECT_DOUBLE_EQ(bbox.area(), 40000.0);
    EXPECT_TRUE(bbox.is_valid());

    EXPECT_TRUE(bbox.contains(Point2D{200.0, 300.0}));
    EXPECT_FALSE(bbox.contains(Point2D{50.0, 50.0}));
}

TEST_F(GeometryTest, InvalidBoundingBoxes) {
    BoundingBox inverted{300.0, 400.0, 100.0, 200.0};
    EXPECT_FALSE(inverted.is_valid());
    EXPECT_DOUBLE_EQ(inverted.area(), 0.0);

    BoundingBox nan_box{std::numeric_limits<double>::quiet_NaN(), 0.0, 10.0, 10.0};
    EXPECT_FALSE(nan_box.is_valid());
    EXPECT_FALSE(nan_box.is_finite());
}

TEST_F(GeometryTest, StandaloneFunctionsMatchPython) {
    // Matches tests/unit/test_geometry.py
    BoundingBox bbox1{100.0, 200.0, 300.0, 400.0};
    EXPECT_EQ(get_center_of_bbox(bbox1), Point2D(200.0, 300.0));

    BoundingBox bbox_dim{50.0, 80.0, 150.0, 200.0};
    EXPECT_DOUBLE_EQ(get_bbox_width(bbox_dim), 100.0);
    EXPECT_DOUBLE_EQ(get_bbox_height(bbox_dim), 120.0);

    BoundingBox bbox_foot{100.0, 200.0, 200.0, 400.0};
    EXPECT_EQ(get_foot_position(bbox_foot), Point2D(150.0, 400.0));

    Point2D p1{0.0, 0.0};
    Point2D p2{3.0, 4.0};
    EXPECT_DOUBLE_EQ(measure_distance(p1, p2), 5.0);

    Point2D p_a{10.0, 20.0};
    Point2D p_b{4.0, 8.0};
    auto [dx, dy] = measure_xy_distance(p_a, p_b);
    EXPECT_DOUBLE_EQ(dx, 6.0);
    EXPECT_DOUBLE_EQ(dy, 12.0);
}

TEST_F(GeometryTest, BatchExtractionAndFiltering) {
    std::vector<BoundingBox> boxes = {
        BoundingBox{10.0, 20.0, 30.0, 40.0},
        BoundingBox{50.0, 60.0, 70.0, 80.0},
        BoundingBox{100.0, 200.0, 50.0, 100.0} // Inverted
    };

    auto centers = extract_centers(boxes);
    ASSERT_EQ(centers.size(), 3u);
    EXPECT_EQ(centers[0], Point2D(20.0, 30.0));
    EXPECT_EQ(centers[1], Point2D(60.0, 70.0));

    auto valid_boxes = filter_valid_bboxes(boxes);
    ASSERT_EQ(valid_boxes.size(), 2u);
    EXPECT_EQ(valid_boxes[0].x1, 10.0);
    EXPECT_EQ(valid_boxes[1].x1, 50.0);
}

TEST_F(GeometryTest, FindNearestPoint) {
    Point2D ball{50.0, 50.0};
    std::vector<Point2D> players = {
        Point2D{100.0, 100.0},
        Point2D{52.0, 51.0},
        Point2D{20.0, 20.0}
    };

    auto nearest = find_nearest_point(ball, players, 10.0);
    ASSERT_TRUE(nearest.has_value());
    EXPECT_EQ(*nearest, 1u);

    auto outside = find_nearest_point(ball, players, 1.0);
    EXPECT_FALSE(outside.has_value());
}
