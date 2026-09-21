#pragma once

#include <memory>
#include <string>
#include <vector>

#include "football_cv/inference/detector.hpp"

// Forward-declare Ort types to avoid leaking heavy ONNX Runtime headers in the public header
namespace Ort {
struct Env;
struct Session;
struct SessionOptions;
} // namespace Ort

namespace football_cv {

/**
 * @brief High-performance C++ YOLOv8 detector using ONNX Runtime.
 *
 * Implements:
 * 1. Aspect-ratio preserving letterbox preprocessing and RGB normalization.
 * 2. Multi-threaded ONNX Runtime session execution.
 * 3. Native Non-Maximum Suppression (NMS) and coordinate rescaling.
 */
class OnnxDetector : public Detector {
public:
    /**
     * @brief Construct an ONNX Runtime detector instance.
     *
     * @param model_path Path to the exported .onnx model file.
     * @param num_threads Intra-op thread count (0 = auto-detect CPU cores).
     */
    explicit OnnxDetector(const std::string& model_path, int num_threads = 0);
    ~OnnxDetector() override;

    // Non-copyable, movable
    OnnxDetector(const OnnxDetector&) = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;
    OnnxDetector(OnnxDetector&&) noexcept;
    OnnxDetector& operator=(OnnxDetector&&) noexcept;

    /**
     * @brief Run inference on a contiguous BGR frame.
     */
    std::vector<Detection> detect(
        const uint8_t* bgr_data,
        int width,
        int height,
        float confidence_threshold = 0.10f,
        float nms_threshold = 0.50f
    ) override;

    /**
     * @brief Get model input tensor width (default: 640).
     */
    int input_width() const noexcept { return input_width_; }

    /**
     * @brief Get model input tensor height (default: 640).
     */
    int input_height() const noexcept { return input_height_; }

    /**
     * @brief Number of detected classes (default: 4 for ball, goalkeeper, player, referee).
     */
    int num_classes() const noexcept { return num_classes_; }

    /**
     * @brief Model file path.
     */
    const std::string& model_path() const noexcept { return model_path_; }

private:
    std::string model_path_;
    int input_width_{640};
    int input_height_{640};
    int num_classes_{4};

    // PIMPL / opaque pointers for ONNX Runtime internals
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;

    // Preallocated buffers for performance
    std::vector<float> input_tensor_values_;

    void preprocess(
        const uint8_t* bgr_data,
        int width,
        int height,
        std::vector<float>& chw_tensor,
        float& scale,
        float& pad_x,
        float& pad_y
    );

    std::vector<Detection> postprocess(
        const float* output_data,
        int64_t num_channels,
        int64_t num_anchors,
        float scale,
        float pad_x,
        float pad_y,
        int orig_w,
        int orig_h,
        float confidence_threshold,
        float nms_threshold
    );
};

} // namespace football_cv
