#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "football_cv/inference/onnx_detector.hpp"
#include "football_cv/types.hpp"

namespace fs = std::filesystem;
using namespace football_cv;

class OnnxDetectorTest : public ::testing::Test {
protected:
    std::string model_path;

    void SetUp() override {
        // Resolve model path relative to repo root
        std::vector<std::string> candidates = {
            "../../models/best.onnx",
            "../models/best.onnx",
            "models/best.onnx",
            "D:/football-cv/models/best.onnx"
        };
        for (const auto& p : candidates) {
            if (fs::exists(p)) {
                model_path = fs::canonical(p).string();
                break;
            }
        }
    }
};

TEST_F(OnnxDetectorTest, ModelInitialization) {
    if (model_path.empty()) {
        GTEST_SKIP() << "models/best.onnx not found, skipping ONNX detector tests.";
    }

    OnnxDetector detector(model_path, 2);
    EXPECT_EQ(detector.input_width(), 640);
    EXPECT_EQ(detector.input_height(), 640);
    EXPECT_EQ(detector.num_classes(), 4);
    EXPECT_FALSE(detector.model_path().empty());
}

TEST_F(OnnxDetectorTest, NullOrEmptyInputHandling) {
    if (model_path.empty()) {
        GTEST_SKIP() << "models/best.onnx not found, skipping test.";
    }

    OnnxDetector detector(model_path, 2);
    auto results_null = detector.detect(nullptr, 640, 640);
    EXPECT_TRUE(results_null.empty());

    std::vector<uint8_t> dummy(640 * 640 * 3, 0);
    auto results_zero_w = detector.detect(dummy.data(), 0, 640);
    EXPECT_TRUE(results_zero_w.empty());

    auto results_zero_h = detector.detect(dummy.data(), 640, 0);
    EXPECT_TRUE(results_zero_h.empty());
}

TEST_F(OnnxDetectorTest, SyntheticInferenceExecution) {
    if (model_path.empty()) {
        GTEST_SKIP() << "models/best.onnx not found, skipping test.";
    }

    OnnxDetector detector(model_path, 2);

    // 1080p blank synthetic frame (1920x1080 BGR)
    std::vector<uint8_t> frame(1920 * 1080 * 3, 114);
    auto detections = detector.detect(frame.data(), 1920, 1080, 0.50f, 0.45f);

    // On a solid gray synthetic frame, detections should be empty or have valid coordinates
    for (const auto& det : detections) {
        EXPECT_GE(det.confidence, 0.50f);
        EXPECT_GE(det.class_id, 0);
        EXPECT_LT(det.class_id, 4);
        EXPECT_TRUE(det.bbox.is_valid());
        EXPECT_GE(det.bbox.x1, 0.0);
        EXPECT_LE(det.bbox.x2, 1920.0);
        EXPECT_GE(det.bbox.y1, 0.0);
        EXPECT_LE(det.bbox.y2, 1080.0);
    }
}

TEST_F(OnnxDetectorTest, ClassSpecificThresholdsFiltering) {
    if (model_path.empty()) {
        GTEST_SKIP() << "models/best.onnx not found, skipping test.";
    }

    OnnxDetector detector(model_path, 2);
    std::vector<uint8_t> frame(1920 * 1080 * 3, 114);

    // Class thresholds: Ball (0)=0.10f, Goalkeeper (1)=0.30f, Player (2)=0.30f, Referee (3)=0.30f
    std::vector<float> class_thresh = {0.10f, 0.30f, 0.30f, 0.30f};
    auto detections = detector.detect(frame.data(), 1920, 1080, 0.20f, 0.45f, class_thresh);

    for (const auto& det : detections) {
        ASSERT_GE(det.class_id, 0);
        ASSERT_LT(det.class_id, static_cast<int>(class_thresh.size()));
        EXPECT_GE(det.confidence, class_thresh[det.class_id]);
    }

    // Suppress all classes except ball
    std::vector<float> suppress_others = {0.05f, 1.0f, 1.0f, 1.0f};
    auto detections_ball_only = detector.detect(frame.data(), 1920, 1080, 0.20f, 0.45f, suppress_others);
    for (const auto& det : detections_ball_only) {
        EXPECT_EQ(det.class_id, 0);
        EXPECT_GE(det.confidence, 0.05f);
    }
}

TEST_F(OnnxDetectorTest, DetectBatchWithClassThresholds) {
    if (model_path.empty()) {
        GTEST_SKIP() << "models/best.onnx not found, skipping test.";
    }

    OnnxDetector detector(model_path, 2);
    std::vector<uint8_t> frame(640 * 640 * 3, 114);
    std::vector<const uint8_t*> frames = {frame.data(), frame.data()};

    std::vector<float> class_thresh = {0.12f, 0.25f, 0.25f, 0.25f};
    auto batch_results = detector.detect_batch(frames, 640, 640, 0.20f, 0.45f, class_thresh);

    ASSERT_EQ(batch_results.size(), 2);
    for (const auto& detections : batch_results) {
        for (const auto& det : detections) {
            ASSERT_GE(det.class_id, 0);
            ASSERT_LT(det.class_id, static_cast<int>(class_thresh.size()));
            EXPECT_GE(det.confidence, class_thresh[det.class_id]);
        }
    }
}
