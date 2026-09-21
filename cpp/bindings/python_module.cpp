#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "football_cv/camera_motion.hpp"
#include "football_cv/geometry.hpp"
#include "football_cv/inference/onnx_detector.hpp"
#include "football_cv/perspective.hpp"
#include "football_cv/types.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace football_cv;
using py::ssize_t;

namespace {

/**
 * @brief Convert arbitrary Python coordinate object (Point2D, sequence, or numpy array) to Point2D.
 */
Point2D parse_point(const py::handle& obj) {
    if (py::isinstance<Point2D>(obj)) {
        return obj.cast<Point2D>();
    }
    if (py::isinstance<py::sequence>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (seq.size() == 2) {
            return Point2D{seq[0].cast<double>(), seq[1].cast<double>()};
        }
    }
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
        if (arr && arr.size() == 2) {
            auto r = arr.data();
            return Point2D{r[0], r[1]};
        }
    }
    throw std::invalid_argument("Expected Point2D, 2-element sequence, or 2-element numpy array");
}

/**
 * @brief Parse a collection of points from None, list of points, or 2D/3D numpy array.
 */
std::vector<Point2D> parse_points(const py::object& obj) {
    if (obj.is_none()) {
        return PerspectiveTransformer::default_pixel_vertices();
    }

    if (py::isinstance<py::array>(obj)) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
        if (arr) {
            auto info = arr.request();
            if (info.ndim == 2 && info.shape[1] == 2) {
                std::vector<Point2D> pts;
                pts.reserve(info.shape[0]);
                auto r = arr.unchecked<2>();
                for (ssize_t i = 0; i < info.shape[0]; ++i) {
                    pts.emplace_back(r(i, 0), r(i, 1));
                }
                return pts;
            }
            if (info.ndim == 3 && info.shape[1] == 1 && info.shape[2] == 2) {
                // OpenCV cv2.goodFeaturesToTrack format: (N, 1, 2)
                std::vector<Point2D> pts;
                pts.reserve(info.shape[0]);
                auto r = arr.unchecked<3>();
                for (ssize_t i = 0; i < info.shape[0]; ++i) {
                    pts.emplace_back(r(i, 0, 0), r(i, 0, 1));
                }
                return pts;
            }
        }
    }

    if (py::isinstance<py::sequence>(obj)) {
        auto seq = obj.cast<py::sequence>();
        std::vector<Point2D> pts;
        pts.reserve(seq.size());
        for (auto item : seq) {
            pts.push_back(parse_point(item));
        }
        return pts;
    }

    throw std::invalid_argument(
        "Expected None, a sequence of 2D points, or a numpy array of shape (N, 2) or (N, 1, 2)"
    );
}

} // anonymous namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native C++ vision core extension for football-cv";

    // ------------------------------------------------------------------------
    // Point2D
    // ------------------------------------------------------------------------
    py::class_<Point2D>(m, "Point2D", "Represents a 2D coordinate point with floating-point precision.")
        .def(py::init<double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
        .def_readwrite("x", &Point2D::x)
        .def_readwrite("y", &Point2D::y)
        .def("distance_to", &Point2D::distance_to, py::arg("other"),
             "Calculate Euclidean distance to another 2D point.")
        .def("displacement_from", &Point2D::displacement_from, py::arg("other"),
             "Calculate signed (dx, dy) displacement from other to this point.")
        .def("is_finite", &Point2D::is_finite,
             "Check whether both coordinates are finite numbers (not NaN or Inf).")
        .def("to_tuple", [](const Point2D& p) {
            return py::make_tuple(p.x, p.y);
        }, "Convert Point2D to a 2-tuple (x, y).")
        .def("to_numpy", [](const Point2D& p) {
            py::array_t<double> arr(2);
            auto r = arr.mutable_unchecked<1>();
            r(0) = p.x;
            r(1) = p.y;
            return arr;
        }, "Convert Point2D to a 1D numpy array [x, y].")
        .def("__len__", [](const Point2D&) { return 2; })
        .def("__getitem__", [](const Point2D& p, ssize_t idx) {
            if (idx == 0 || idx == -2) return p.x;
            if (idx == 1 || idx == -1) return p.y;
            throw py::index_error("Point2D index out of range (must be 0 or 1)");
        })
        .def("__eq__", &Point2D::operator==)
        .def("__repr__", [](const Point2D& p) {
            std::ostringstream ss;
            ss << "Point2D(x=" << p.x << ", y=" << p.y << ")";
            return ss.str();
        });

    // ------------------------------------------------------------------------
    // BoundingBox
    // ------------------------------------------------------------------------
    py::class_<BoundingBox>(m, "BoundingBox", "Axis-aligned 2D bounding box defined by [x1, y1, x2, y2].")
        .def(py::init<double, double, double, double>(),
             py::arg("x1") = 0.0, py::arg("y1") = 0.0,
             py::arg("x2") = 0.0, py::arg("y2") = 0.0)
        .def_readwrite("x1", &BoundingBox::x1)
        .def_readwrite("y1", &BoundingBox::y1)
        .def_readwrite("x2", &BoundingBox::x2)
        .def_readwrite("y2", &BoundingBox::y2)
        .def("center", &BoundingBox::center,
             "Calculate the geometric center of the bounding box.")
        .def("foot_position", &BoundingBox::foot_position,
             "Calculate the bottom-center point representing player pitch contact.")
        .def("width", &BoundingBox::width, "Horizontal width of the bounding box.")
        .def("height", &BoundingBox::height, "Vertical height of the bounding box.")
        .def("area", &BoundingBox::area, "Surface area of the bounding box.")
        .def("is_valid", &BoundingBox::is_valid,
             "Check if the box has valid positive dimensions and finite coordinates.")
        .def("is_finite", &BoundingBox::is_finite,
             "Check whether all bounding box coordinates are finite numbers.")
        .def("contains", &BoundingBox::contains, py::arg("pt"),
             "Check if a 2D point is contained within the bounding box.")
        .def("to_tuple", [](const BoundingBox& b) {
            return py::make_tuple(b.x1, b.y1, b.x2, b.y2);
        }, "Convert BoundingBox to a 4-tuple (x1, y1, x2, y2).")
        .def("__len__", [](const BoundingBox&) { return 4; })
        .def("__getitem__", [](const BoundingBox& b, ssize_t idx) {
            if (idx < 0) idx += 4;
            switch (idx) {
                case 0: return b.x1;
                case 1: return b.y1;
                case 2: return b.x2;
                case 3: return b.y2;
                default: throw py::index_error("BoundingBox index out of range [0..3]");
            }
        })
        .def("__eq__", [](const BoundingBox& a, const BoundingBox& b) {
            constexpr double eps = 1e-9;
            return std::abs(a.x1 - b.x1) < eps && std::abs(a.y1 - b.y1) < eps &&
                   std::abs(a.x2 - b.x2) < eps && std::abs(a.y2 - b.y2) < eps;
        })
        .def("__repr__", [](const BoundingBox& b) {
            std::ostringstream ss;
            ss << "BoundingBox(x1=" << b.x1 << ", y1=" << b.y1
               << ", x2=" << b.x2 << ", y2=" << b.y2 << ")";
            return ss.str();
        });

    // ------------------------------------------------------------------------
    // CameraMotion
    // ------------------------------------------------------------------------
    py::class_<CameraMotion>(m, "CameraMotion", "Estimated inter-frame camera translation and cut status.")
        .def(py::init<double, double, bool>(),
             py::arg("dx") = 0.0, py::arg("dy") = 0.0, py::arg("is_scene_cut") = false)
        .def_readwrite("dx", &CameraMotion::dx)
        .def_readwrite("dy", &CameraMotion::dy)
        .def_readwrite("is_scene_cut", &CameraMotion::is_scene_cut)
        .def("to_tuple", [](const CameraMotion& cm) {
            return py::make_tuple(cm.dx, cm.dy);
        }, "Convert motion displacement to a 2-tuple (dx, dy).")
        .def("__eq__", [](const CameraMotion& a, const CameraMotion& b) {
            constexpr double eps = 1e-9;
            return std::abs(a.dx - b.dx) < eps && std::abs(a.dy - b.dy) < eps &&
                   a.is_scene_cut == b.is_scene_cut;
        })
        .def("__repr__", [](const CameraMotion& cm) {
            std::ostringstream ss;
            ss << "CameraMotion(dx=" << cm.dx << ", dy=" << cm.dy
               << ", is_scene_cut=" << (cm.is_scene_cut ? "True" : "False") << ")";
            return ss.str();
        });

    // ------------------------------------------------------------------------
    // Standalone Geometry Functions
    // ------------------------------------------------------------------------
    m.def("get_center_of_bbox", &get_center_of_bbox, py::arg("bbox"),
          "Calculate the center point of a bounding box.");
    m.def("get_foot_position", &get_foot_position, py::arg("bbox"),
          "Calculate the bottom-center foot contact point of a bounding box.");
    m.def("get_bbox_width", &get_bbox_width, py::arg("bbox"),
          "Get the horizontal width of a bounding box.");
    m.def("get_bbox_height", &get_bbox_height, py::arg("bbox"),
          "Get the vertical height of a bounding box.");
    m.def("measure_distance", &measure_distance, py::arg("p1"), py::arg("p2"),
          "Calculate Euclidean distance between two 2D points.");
    m.def("measure_xy_distance", &measure_xy_distance, py::arg("p1"), py::arg("p2"),
          "Calculate signed (dx, dy) displacement from p2 to p1.");
    m.def("extract_centers", &extract_centers, py::arg("bboxes"),
          "Extract center points for a collection of bounding boxes.");
    m.def("extract_foot_positions", &extract_foot_positions, py::arg("bboxes"),
          "Extract foot contact points for a collection of bounding boxes.");
    m.def("filter_valid_bboxes", &filter_valid_bboxes, py::arg("bboxes"),
          "Filter out bounding boxes with invalid coordinates or negative dimensions.");
    m.def("find_nearest_point", &find_nearest_point,
          py::arg("target"), py::arg("candidates"),
          py::arg("max_distance") = std::numeric_limits<double>::infinity(),
          "Find the index of the nearest point within an optional maximum distance.");

    // ------------------------------------------------------------------------
    // PerspectiveTransformer
    // ------------------------------------------------------------------------
    py::class_<PerspectiveTransformer>(m, "PerspectiveTransformer",
        "Projects pixel broadcast coordinates to metric pitch coordinates (meters) via 4-point planar homography.")
        .def(py::init([](py::object pixel_vertices, double court_width, double court_length) {
            std::vector<Point2D> vertices = parse_points(pixel_vertices);
            return std::make_unique<PerspectiveTransformer>(vertices, court_width, court_length);
        }), py::arg("pixel_vertices") = py::none(),
            py::arg("court_width") = 68.0,
            py::arg("court_length") = 23.32)
        .def("transform_point", [](const PerspectiveTransformer& self, py::object point) -> std::optional<Point2D> {
            Point2D pt = parse_point(point);
            return self.transform_point(pt);
        }, py::arg("point"),
           "Transform a 2D point from broadcast pixel coordinates to metric pitch coordinates.")
        .def("transform_point", [](const PerspectiveTransformer& self, double x, double y) -> std::optional<Point2D> {
            return self.transform_point(Point2D{x, y});
        }, py::arg("x"), py::arg("y"),
           "Transform (x, y) coordinates to metric pitch coordinates.")
        .def("transform_points", [](const PerspectiveTransformer& self, py::object points) {
            std::vector<Point2D> pts = parse_points(points);
            return self.transform_points(pts);
        }, py::arg("points"), "Batch project multiple image coordinates.")
        .def("is_point_inside", [](const PerspectiveTransformer& self, py::object point) {
            Point2D pt = parse_point(point);
            return self.is_point_inside(pt);
        }, py::arg("point"), "Check if a point lies within or on the calibrated boundary polygon.")
        .def_property_readonly("court_width", &PerspectiveTransformer::court_width)
        .def_property_readonly("court_length", &PerspectiveTransformer::court_length)
        .def_property_readonly("pixel_vertices", &PerspectiveTransformer::pixel_vertices)
        .def_property_readonly("homography_matrix", [](const PerspectiveTransformer& self) {
            py::array_t<double> arr({3, 3});
            auto r = arr.mutable_unchecked<2>();
            const auto& h = self.homography_matrix();
            for (ssize_t i = 0; i < 3; ++i) {
                for (ssize_t j = 0; j < 3; ++j) {
                    r(i, j) = h[i * 3 + j];
                }
            }
            return arr;
        }, "3x3 planar homography transformation matrix.")
        .def_static("default_pixel_vertices", &PerspectiveTransformer::default_pixel_vertices,
                    "Return default broadcast pitch boundary polygon vertices.");

    // ------------------------------------------------------------------------
    // CameraMotionEstimator
    // ------------------------------------------------------------------------
    py::class_<CameraMotionEstimator>(m, "CameraMotionEstimator",
        "Compensates for camera pan and tilt movements by tracking feature displacements.")
        .def(py::init<double, double>(),
             py::arg("minimum_distance") = 5.0,
             py::arg("scene_cut_threshold") = 80.0)
        .def("estimate_from_features", [](const CameraMotionEstimator& self,
                                          py::object old_features,
                                          py::object new_features) {
            std::vector<Point2D> old_pts = parse_points(old_features);
            std::vector<Point2D> new_pts = parse_points(new_features);
            return self.estimate_from_features(old_pts, new_pts);
        }, py::arg("old_features"), py::arg("new_features"),
           "Estimate camera displacement between tracked feature correspondences.")
        .def_static("filter_margin_points", [](py::object points,
                                               double frame_width,
                                               double left_margin_width,
                                               double right_margin_start,
                                               double right_margin_end) {
            std::vector<Point2D> pts = parse_points(points);
            return CameraMotionEstimator::filter_margin_points(
                pts, frame_width, left_margin_width, right_margin_start, right_margin_end
            );
        }, py::arg("points"), py::arg("frame_width"),
           py::arg("left_margin_width") = 20.0,
           py::arg("right_margin_start") = 900.0,
           py::arg("right_margin_end") = 1050.0,
           "Filter points to isolate vertical broadcast margins (excluding players on pitch).")
        .def_static("accumulate_motion", &CameraMotionEstimator::accumulate_motion,
                    py::arg("motion_steps"),
                    "Accumulate cumulative camera displacement across a sequence of frames.")
        .def_property_readonly("minimum_distance", &CameraMotionEstimator::minimum_distance)
        .def_property_readonly("scene_cut_threshold", &CameraMotionEstimator::scene_cut_threshold);

    // ------------------------------------------------------------------------
    // Detection
    // ------------------------------------------------------------------------
    py::class_<Detection>(m, "Detection", "Object detection result with bounding box, confidence, and class ID.")
        .def(py::init<BoundingBox, float, int>(),
             py::arg("bbox"), py::arg("confidence"), py::arg("class_id"))
        .def_readwrite("bbox", &Detection::bbox)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("class_id", &Detection::class_id)
        .def("__repr__", [](const Detection& d) {
            std::ostringstream oss;
            oss << "Detection(bbox=BoundingBox("
                << d.bbox.x1 << ", " << d.bbox.y1 << ", "
                << d.bbox.x2 << ", " << d.bbox.y2 << "), conf="
                << d.confidence << ", class_id=" << d.class_id << ")";
            return oss.str();
        })
        .def("__eq__", &Detection::operator==);

#ifdef FOOTBALL_CV_HAS_ONNX
    // ------------------------------------------------------------------------
    // OnnxDetector
    // ------------------------------------------------------------------------
    py::class_<OnnxDetector>(m, "OnnxDetector", "C++ ONNX Runtime YOLOv8 detector with native NMS.")
        .def(py::init<const std::string&, int>(),
             py::arg("model_path"),
             py::arg("num_threads") = 0)
        .def("detect", [](OnnxDetector& self,
                          py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
                          float confidence_threshold,
                          float nms_threshold) {
            auto info = image.request();
            if (info.ndim != 3 || info.shape[2] != 3) {
                throw std::invalid_argument("Expected 3D BGR image array with shape (H, W, 3)");
            }
            int height = static_cast<int>(info.shape[0]);
            int width = static_cast<int>(info.shape[1]);
            const uint8_t* ptr = static_cast<const uint8_t*>(info.ptr);

            py::gil_scoped_release release;
            return self.detect(ptr, width, height, confidence_threshold, nms_threshold);
        }, py::arg("frame"), py::arg("confidence_threshold") = 0.10f, py::arg("nms_threshold") = 0.50f,
           "Run object detection on a BGR NumPy array (H, W, 3).")
        .def("detect_batch", [](OnnxDetector& self,
                                py::sequence frames,
                                float confidence_threshold,
                                float nms_threshold) {
            std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>> arrays;
            std::vector<const uint8_t*> ptrs;
            arrays.reserve(frames.size());
            ptrs.reserve(frames.size());

            int width = -1;
            int height = -1;

            for (auto item : frames) {
                auto arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(item);
                if (!arr) {
                    throw std::invalid_argument("All batch items must be NumPy image arrays");
                }
                auto info = arr.request();
                if (info.ndim != 3 || info.shape[2] != 3) {
                    throw std::invalid_argument("Batch image arrays must have shape (H, W, 3)");
                }
                if (width == -1) {
                    height = static_cast<int>(info.shape[0]);
                    width = static_cast<int>(info.shape[1]);
                } else if (height != static_cast<int>(info.shape[0]) || width != static_cast<int>(info.shape[1])) {
                    throw std::invalid_argument("All images in a batch must have identical dimensions");
                }
                arrays.push_back(arr);
                ptrs.push_back(static_cast<const uint8_t*>(info.ptr));
            }

            py::gil_scoped_release release;
            return self.detect_batch(ptrs, width, height, confidence_threshold, nms_threshold);
        }, py::arg("frames"), py::arg("confidence_threshold") = 0.10f, py::arg("nms_threshold") = 0.50f,
           "Run object detection on a batch of BGR NumPy arrays.")
        .def_property_readonly("input_width", &OnnxDetector::input_width)
        .def_property_readonly("input_height", &OnnxDetector::input_height)
        .def_property_readonly("num_classes", &OnnxDetector::num_classes)
        .def_property_readonly("model_path", &OnnxDetector::model_path);
#endif
}
