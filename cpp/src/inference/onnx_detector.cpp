#include "football_cv/inference/onnx_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

#include "onnxruntime_cxx_api.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static std::wstring string_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (size <= 0) return std::wstring();
    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
    if (!wstr.empty() && wstr.back() == L'\0') {
        wstr.pop_back();
    }
    return wstr;
}
#endif

namespace football_cv {

static float calculate_iou(const BoundingBox& b1, const BoundingBox& b2) noexcept {
    const double inter_x1 = std::max(b1.x1, b2.x1);
    const double inter_y1 = std::max(b1.y1, b2.y1);
    const double inter_x2 = std::min(b1.x2, b2.x2);
    const double inter_y2 = std::min(b1.y2, b2.y2);

    const double inter_w = std::max(0.0, inter_x2 - inter_x1);
    const double inter_h = std::max(0.0, inter_y2 - inter_y1);
    const double inter_area = inter_w * inter_h;

    const double area1 = b1.area();
    const double area2 = b2.area();
    const double union_area = area1 + area2 - inter_area;

    if (union_area <= 0.0) return 0.0f;
    return static_cast<float>(inter_area / union_area);
}

OnnxDetector::OnnxDetector(const std::string& model_path, int num_threads)
    : model_path_(model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "football_cv_onnx");
    session_options_ = std::make_unique<Ort::SessionOptions>();

    const int threads = (num_threads > 0) ? num_threads : static_cast<int>(std::thread::hardware_concurrency());
    session_options_->SetIntraOpNumThreads(std::max(1, threads));
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    const std::wstring w_path = string_to_wstring(model_path_);
    session_ = std::make_unique<Ort::Session>(*env_, w_path.c_str(), *session_options_);
#else
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_options_);
#endif

    // Retrieve input & output tensor names and dimensions
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name_alloc.get();

    auto output_name_alloc = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = output_name_alloc.get();

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto in_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape.size() == 4) {
        is_dynamic_batch_ = (in_shape[0] < 0);
        if (in_shape[2] > 0) input_height_ = static_cast<int>(in_shape[2]);
        if (in_shape[3] > 0) input_width_ = static_cast<int>(in_shape[3]);
    }

    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto out_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() == 3 && out_shape[1] > 4) {
        num_classes_ = static_cast<int>(out_shape[1] - 4);
    }

    input_tensor_values_.resize(1 * 3 * input_height_ * input_width_);
}

OnnxDetector::~OnnxDetector() = default;

OnnxDetector::OnnxDetector(OnnxDetector&&) noexcept = default;
OnnxDetector& OnnxDetector::operator=(OnnxDetector&&) noexcept = default;

void OnnxDetector::preprocess(
    const uint8_t* bgr_data,
    int width,
    int height,
    std::vector<float>& chw_tensor,
    float& scale,
    float& pad_x,
    float& pad_y
) {
    chw_tensor.resize(1 * 3 * input_height_ * input_width_);
    preprocess_to_ptr(bgr_data, width, height, chw_tensor.data(), scale, pad_x, pad_y);
}

void OnnxDetector::preprocess_to_ptr(
    const uint8_t* bgr_data,
    int width,
    int height,
    float* chw_dst,
    float& scale,
    float& pad_x,
    float& pad_y
) {
    const float r = std::min(
        static_cast<float>(input_width_) / static_cast<float>(width),
        static_cast<float>(input_height_) / static_cast<float>(height)
    );
    scale = r;

    const int unpad_w = std::min(input_width_, static_cast<int>(std::round(width * r)));
    const int unpad_h = std::min(input_height_, static_cast<int>(std::round(height * r)));

    pad_x = (input_width_ - unpad_w) * 0.5f;
    pad_y = (input_height_ - unpad_h) * 0.5f;
    const int pad_x_int = static_cast<int>(std::round(pad_x));
    const int pad_y_int = static_cast<int>(std::round(pad_y));

    // Fill with gray background (114 / 255.0f = 0.4470588f)
    constexpr float gray_val = 114.0f / 255.0f;
    std::fill(chw_dst, chw_dst + (3 * input_width_ * input_height_), gray_val);

    const int plane_stride = input_width_ * input_height_;
    float* r_plane = chw_dst;
    float* g_plane = chw_dst + plane_stride;
    float* b_plane = chw_dst + plane_stride * 2;

    constexpr float inv_255 = 1.0f / 255.0f;

    // Precompute horizontal interpolation weights and coordinate indices
    struct XCoord {
        int x0;
        int x1;
        float wx0;
        float wx1;
    };
    std::vector<XCoord> x_coords(unpad_w);
    for (int tx = 0; tx < unpad_w; ++tx) {
        const float src_x = (tx + 0.5f) / r - 0.5f;
        const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(src_x))));
        const int x1 = std::min(width - 1, x0 + 1);
        const float wx = src_x - std::floor(src_x);
        x_coords[tx] = {x0, x1, 1.0f - wx, wx};
    }

    // Bilinear resize and BGR -> RGB normalization directly into CHW layout
    for (int ty = 0; ty < unpad_h; ++ty) {
        const float src_y = (ty + 0.5f) / r - 0.5f;
        const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(src_y))));
        const int y1 = std::min(height - 1, y0 + 1);
        const float wy = src_y - std::floor(src_y);
        const float wy0 = 1.0f - wy;
        const float wy1 = wy;

        const int target_y = ty + pad_y_int;
        const int row_offset = target_y * input_width_ + pad_x_int;
        const int row0_offset = y0 * width;
        const int row1_offset = y1 * width;

        for (int tx = 0; tx < unpad_w; ++tx) {
            const auto& xc = x_coords[tx];
            const int idx00 = (row0_offset + xc.x0) * 3;
            const int idx01 = (row0_offset + xc.x1) * 3;
            const int idx10 = (row1_offset + xc.x0) * 3;
            const int idx11 = (row1_offset + xc.x1) * 3;

            // BGR channels
            const float b_val = (wy0 * (xc.wx0 * bgr_data[idx00 + 0] + xc.wx1 * bgr_data[idx01 + 0]) +
                                 wy1 * (xc.wx0 * bgr_data[idx10 + 0] + xc.wx1 * bgr_data[idx11 + 0])) * inv_255;
            const float g_val = (wy0 * (xc.wx0 * bgr_data[idx00 + 1] + xc.wx1 * bgr_data[idx01 + 1]) +
                                 wy1 * (xc.wx0 * bgr_data[idx10 + 1] + xc.wx1 * bgr_data[idx11 + 1])) * inv_255;
            const float r_val = (wy0 * (xc.wx0 * bgr_data[idx00 + 2] + xc.wx1 * bgr_data[idx01 + 2]) +
                                 wy1 * (xc.wx0 * bgr_data[idx10 + 2] + xc.wx1 * bgr_data[idx11 + 2])) * inv_255;

            const int target_idx = row_offset + tx;
            r_plane[target_idx] = r_val;
            g_plane[target_idx] = g_val;
            b_plane[target_idx] = b_val;
        }
    }
}

std::vector<Detection> OnnxDetector::postprocess(
    const float* output_data,
    int64_t num_channels,
    int64_t num_anchors,
    float scale,
    float pad_x,
    float pad_y,
    int orig_w,
    int orig_h,
    float confidence_threshold,
    float nms_threshold,
    const std::vector<float>& class_thresholds
) {
    std::vector<Detection> candidates;
    candidates.reserve(128);

    const int classes = static_cast<int>(num_channels - 4);
    const float inv_scale = 1.0f / scale;

    const bool has_class_thresholds = !class_thresholds.empty();
    const float min_global_threshold = has_class_thresholds
        ? *std::min_element(class_thresholds.begin(), class_thresholds.end())
        : confidence_threshold;

    // Output tensor shape is [1, 8, 8400]
    // output_data[channel * num_anchors + anchor_idx]
    for (int64_t i = 0; i < num_anchors; ++i) {
        float max_score = -1.0f;
        int best_cls = -1;

        for (int c = 0; c < classes; ++c) {
            const float score = output_data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_cls = c;
            }
        }

        if (max_score < min_global_threshold) {
            continue;
        }

        const float thresh = (has_class_thresholds && best_cls >= 0 && static_cast<size_t>(best_cls) < class_thresholds.size())
            ? class_thresholds[best_cls]
            : confidence_threshold;

        if (max_score >= thresh) {
            const float cx = output_data[0 * num_anchors + i];
            const float cy = output_data[1 * num_anchors + i];
            const float w = output_data[2 * num_anchors + i];
            const float h = output_data[3 * num_anchors + i];

            double x1 = (cx - w * 0.5f - pad_x) * inv_scale;
            double y1 = (cy - h * 0.5f - pad_y) * inv_scale;
            double x2 = (cx + w * 0.5f - pad_x) * inv_scale;
            double y2 = (cy + h * 0.5f - pad_y) * inv_scale;

            // Clip coordinates to frame boundary
            x1 = std::max(0.0, std::min(static_cast<double>(orig_w), x1));
            y1 = std::max(0.0, std::min(static_cast<double>(orig_h), y1));
            x2 = std::max(0.0, std::min(static_cast<double>(orig_w), x2));
            y2 = std::max(0.0, std::min(static_cast<double>(orig_h), y2));

            if (x2 > x1 && y2 > y1) {
                candidates.emplace_back(BoundingBox(x1, y1, x2, y2), max_score, best_cls);
            }
        }
    }

    if (candidates.empty()) {
        return {};
    }

    // Sort descending by confidence
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    // Class-specific greedy NMS
    std::vector<Detection> kept;
    kept.reserve(candidates.size());
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id == candidates[j].class_id) {
                if (calculate_iou(candidates[i].bbox, candidates[j].bbox) > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return kept;
}

std::vector<Detection> OnnxDetector::detect(
    const uint8_t* bgr_data,
    int width,
    int height,
    float confidence_threshold,
    float nms_threshold,
    const std::vector<float>& class_thresholds
) {
    if (!bgr_data || width <= 0 || height <= 0) {
        return {};
    }

    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;

    preprocess_to_ptr(
        bgr_data, width, height, input_tensor_values_.data(), scale, pad_x, pad_y
    );

    const std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    const float* output_data = output_tensors[0].GetTensorData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // shape: [1, num_channels, num_anchors]
    const int64_t num_channels = shape[1];
    const int64_t num_anchors = shape[2];

    return postprocess(
        output_data,
        num_channels,
        num_anchors,
        scale,
        pad_x,
        pad_y,
        width,
        height,
        confidence_threshold,
        nms_threshold,
        class_thresholds
    );
}

std::vector<Detection> OnnxDetector::detect(
    const uint8_t* bgr_data,
    int width,
    int height,
    const std::vector<float>& class_thresholds,
    float nms_threshold
) {
    return detect(bgr_data, width, height, 0.10f, nms_threshold, class_thresholds);
}

std::vector<std::vector<Detection>> OnnxDetector::detect_batch(
    const std::vector<const uint8_t*>& frames_data,
    int width,
    int height,
    float confidence_threshold,
    float nms_threshold,
    const std::vector<float>& class_thresholds
) {
    if (frames_data.empty() || width <= 0 || height <= 0) {
        return {};
    }

    const size_t batch_size = frames_data.size();
    if (!is_dynamic_batch_ && batch_size > 1) {
        // Fall back to sequential execution if model has fixed batch size 1
        std::vector<std::vector<Detection>> results;
        results.reserve(batch_size);
        for (const auto* frame : frames_data) {
            results.push_back(detect(frame, width, height, confidence_threshold, nms_threshold, class_thresholds));
        }
        return results;
    }

    // Allocate / resize contiguous batch input buffer
    const size_t frame_tensor_size = 3 * input_height_ * input_width_;
    batch_tensor_values_.resize(batch_size * frame_tensor_size);

    std::vector<float> scales(batch_size);
    std::vector<float> pads_x(batch_size);
    std::vector<float> pads_y(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        float* dst = batch_tensor_values_.data() + i * frame_tensor_size;
        preprocess_to_ptr(frames_data[i], width, height, dst, scales[i], pads_x[i], pads_y[i]);
    }

    const std::vector<int64_t> batch_shape = {
        static_cast<int64_t>(batch_size), 3, input_height_, input_width_
    };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value batch_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        batch_tensor_values_.data(),
        batch_tensor_values_.size(),
        batch_shape.data(),
        batch_shape.size()
    );

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &batch_input_tensor,
        1,
        output_names,
        1
    );

    const float* output_data = output_tensors[0].GetTensorData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // shape: [B, num_channels, num_anchors]
    const int64_t num_channels = shape[1];
    const int64_t num_anchors = shape[2];
    const int64_t frame_output_stride = num_channels * num_anchors;

    std::vector<std::vector<Detection>> batch_results;
    batch_results.reserve(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* frame_output = output_data + b * frame_output_stride;
        batch_results.push_back(postprocess(
            frame_output,
            num_channels,
            num_anchors,
            scales[b],
            pads_x[b],
            pads_y[b],
            width,
            height,
            confidence_threshold,
            nms_threshold,
            class_thresholds
        ));
    }

    return batch_results;
}

std::vector<std::vector<Detection>> OnnxDetector::detect_batch(
    const std::vector<const uint8_t*>& frames_data,
    int width,
    int height,
    const std::vector<float>& class_thresholds,
    float nms_threshold
) {
    return detect_batch(frames_data, width, height, 0.10f, nms_threshold, class_thresholds);
}

} // namespace football_cv
