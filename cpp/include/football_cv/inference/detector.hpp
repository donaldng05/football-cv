#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "football_cv/types.hpp"

namespace football_cv {

/**
 * @brief Abstract detector interface decoupling the pipeline from concrete inference runtimes.
 */
class Detector {
public:
    virtual ~Detector() = default;

    /**
     * @brief Run object detection on a single BGR image.
     *
     * @param bgr_data Pointer to contiguous BGR image buffer (height * width * 3 bytes).
     * @param width Image width in pixels.
     * @param height Image height in pixels.
     * @param confidence_threshold Minimum score threshold to retain detections.
     * @param nms_threshold Intersection-over-Union (IoU) threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Filtered object detections with bounding boxes.
     */
    virtual std::vector<Detection> detect(
        const uint8_t* bgr_data,
        int width,
        int height,
        float confidence_threshold = 0.10f,
        float nms_threshold = 0.50f
    ) = 0;

    /**
     * @brief Run object detection across a batch of BGR images.
     */
    virtual std::vector<std::vector<Detection>> detect_batch(
        const std::vector<const uint8_t*>& frames_data,
        int width,
        int height,
        float confidence_threshold = 0.10f,
        float nms_threshold = 0.50f
    ) {
        std::vector<std::vector<Detection>> results;
        results.reserve(frames_data.size());
        for (const auto* frame : frames_data) {
            results.push_back(detect(frame, width, height, confidence_threshold, nms_threshold));
        }
        return results;
    }
};

} // namespace football_cv
