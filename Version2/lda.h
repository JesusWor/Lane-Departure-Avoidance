// =============================================================================
// Lane Departure Avoidance & Lane-Change Validation  --  port a C++17 (v4)
// Dependencia unica: OpenCV (core, imgproc, dnn, videoio, imgcodecs)
// Todo el codigo vive en el namespace lda (header-only).
// =============================================================================
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lda {

constexpr double PI = 3.14159265358979323846;
inline double deg2rad(double d) { return d * PI / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / PI; }
inline double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }
// Equivalente a (a + 180) % 360 - 180 de Python (modulo siempre no negativo)
inline double wrap180(double a) {
    a = std::fmod(a + 180.0, 360.0);
    if (a < 0) a += 360.0;
    return a - 180.0;
}

// ----------------------------------------------------------------------------
// 2. Parametros del sistema
// ----------------------------------------------------------------------------
struct VehicleParams {
    double length = 4.963;     // m (Porsche Taycan 4S)
    double wheelbase = 2.9;    // m
    double width = 1.966;      // m
    double mass = 2170;        // kg
};
struct ControlParams {
    double k_lateral = 0.8;
    double k_heading = 1.2;
    double max_steering_angle = deg2rad(30);
};
struct SafetyParams {
    double min_safe_distance = 10.0;
    double min_safe_ttc = 3.0;
    double lane_safety_margin = 0.30;
};
struct SimulationParams {
    double dt = 0.05;
    double total_time = 10.0;
};
struct LaneState {
    double center = 0.0;
    double heading = 0.0;
    double width = 3.5;
};
struct VehicleState {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double speed = 15.0;
};

// ----------------------------------------------------------------------------
// 3. Modelo cinematico (bicycle model)
// ----------------------------------------------------------------------------
class KinematicBicycleModel {
public:
    explicit KinematicBicycleModel(VehicleParams v) : vehicle_(v) {}

    void derivatives(const VehicleState& s, double steering, double& x_dot, double& y_dot,
                     double& yaw_dot) const {
        x_dot = s.speed * std::cos(s.yaw);
        y_dot = s.speed * std::sin(s.yaw);
        yaw_dot = (s.speed / vehicle_.wheelbase) * std::tan(steering);
    }
    VehicleState step(const VehicleState& s, double steering, double dt) const {
        double xd, yd, yawd;
        derivatives(s, steering, xd, yd, yawd);
        VehicleState n;
        n.x = s.x + xd * dt;
        n.y = s.y + yd * dt;
        n.yaw = s.yaw + yawd * dt;
        n.speed = s.speed;
        return n;
    }
private:
    VehicleParams vehicle_;
};

// ----------------------------------------------------------------------------
// 4. Geometria del carril
// ----------------------------------------------------------------------------
struct LaneGeometry {
    static std::pair<double, double> boundaries(const LaneState& l) {
        return {l.center + l.width / 2, l.center - l.width / 2};  // {left, right}
    }
    static double lateral_error(double vehicle_y, const LaneState& l) { return vehicle_y - l.center; }
    static double heading_error(double vehicle_yaw, const LaneState& l) { return vehicle_yaw - l.heading; }
    static bool is_departing(double vehicle_y, const LaneState& l, const VehicleParams& v,
                             const SafetyParams& s) {
        auto [left, right] = boundaries(l);
        double vehicle_left = vehicle_y + v.width / 2;
        double vehicle_right = vehicle_y - v.width / 2;
        double safe_left = left - s.lane_safety_margin;
        double safe_right = right + s.lane_safety_margin;
        return vehicle_left > safe_left || vehicle_right < safe_right;
    }
};

// ----------------------------------------------------------------------------
// 5. Controlador de Lane Keeping
// ----------------------------------------------------------------------------
class LaneKeepingController {
public:
    explicit LaneKeepingController(ControlParams c) : c_(c) {}
    double compute_steering(double lateral_error, double heading_error) const {
        double steering = -(c_.k_lateral * lateral_error + c_.k_heading * heading_error);
        return clip(steering, -c_.max_steering_angle, c_.max_steering_angle);
    }
private:
    ControlParams c_;
};

// ----------------------------------------------------------------------------
// 6. Deteccion de carril clasica (OpenCV)
// ----------------------------------------------------------------------------
struct LineFit { double slope, intercept; };
struct LaneDetection {
    std::optional<LineFit> left_fit, right_fit;
    std::optional<double> center_px;
    std::vector<cv::Vec4i> lines;
};

class ClassicalLaneDetector {
public:
    explicit ClassicalLaneDetector(int canny_low = 50, int canny_high = 150)
        : canny_low_(canny_low), canny_high_(canny_high) {}

    cv::Mat preprocess(const cv::Mat& frame) const {
        cv::Mat gray, blur;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        return blur;
    }
    cv::Mat edges(const cv::Mat& gb) const {
        cv::Mat e;
        cv::Canny(gb, e, canny_low_, canny_high_);
        return e;
    }
    cv::Mat region_of_interest(const cv::Mat& e) const {
        int h = e.rows, w = e.cols;
        std::vector<std::vector<cv::Point>> poly = {{
            cv::Point(int(0.05 * w), h), cv::Point(int(0.45 * w), int(0.6 * h)),
            cv::Point(int(0.55 * w), int(0.6 * h)), cv::Point(int(0.95 * w), h)}};
        cv::Mat mask = cv::Mat::zeros(e.size(), e.type());
        cv::fillPoly(mask, poly, cv::Scalar(255));
        cv::Mat out;
        cv::bitwise_and(e, mask, out);
        return out;
    }
    std::vector<cv::Vec4i> hough_lines(const cv::Mat& roi) const {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(roi, lines, 2, PI / 180, 50, 40, 100);
        return lines;
    }
    static void fit_lane_lines(const std::vector<cv::Vec4i>& lines, std::optional<LineFit>& left,
                               std::optional<LineFit>& right) {
        left.reset(); right.reset();
        if (lines.empty()) return;
        double ls = 0, li = 0, rs = 0, ri = 0; int ln = 0, rn = 0;
        for (const auto& l : lines) {
            double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            if (x2 == x1) continue;
            double slope = (y2 - y1) / (x2 - x1);
            double intercept = y1 - slope * x1;
            if (slope < -0.3) { ls += slope; li += intercept; ++ln; }
            else if (slope > 0.3) { rs += slope; ri += intercept; ++rn; }
        }
        if (ln) left = LineFit{ls / ln, li / ln};
        if (rn) right = LineFit{rs / rn, ri / rn};
    }
    static std::optional<double> estimate_lane_center_px(const std::optional<LineFit>& l,
                                                         const std::optional<LineFit>& r,
                                                         double y_eval) {
        auto x_at_y = [&](const LineFit& f) -> std::optional<double> {
            if (f.slope == 0) return std::nullopt;
            return (y_eval - f.intercept) / f.slope;
        };
        if (l && r) {
            auto xl = x_at_y(*l), xr = x_at_y(*r);
            if (xl && xr) return (*xl + *xr) / 2;
        }
        return std::nullopt;
    }
    LaneDetection process(const cv::Mat& frame) const {
        cv::Mat gb = preprocess(frame);
        cv::Mat roi = region_of_interest(edges(gb));
        LaneDetection d;
        d.lines = hough_lines(roi);
        fit_lane_lines(d.lines, d.left_fit, d.right_fit);
        d.center_px = estimate_lane_center_px(d.left_fit, d.right_fit, frame.rows * 0.95);
        return d;
    }
private:
    int canny_low_, canny_high_;
};

// ----------------------------------------------------------------------------
// 7. Deteccion de vehiculos con red preentrenada
//    NOTA: en C++ no existe `ultralytics`. Se usa cv::dnn con un modelo YOLOv8
//    exportado a ONNX (salida [1, 4+nc, N]). YOLOPv2 (TorchScript) requeriria LibTorch.
// ----------------------------------------------------------------------------
struct Detection {
    std::array<double, 4> bbox;  // x1, y1, x2, y2 (px)
    std::string class_name;
    double confidence;
};

class VehicleDetector {
public:
    explicit VehicleDetector(const std::string& onnx_path = "", double conf_thr = 0.25,
                             double nms_thr = 0.45, int input_size = 640)
        : conf_thr_(conf_thr), nms_thr_(nms_thr), input_(input_size) {
        if (!onnx_path.empty()) load(onnx_path);
    }
    void load(const std::string& onnx_path) {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        loaded_ = true;
    }
    std::vector<Detection> detect(const cv::Mat& frame) {
        if (!loaded_) throw std::runtime_error("Modelo no cargado. Pasa la ruta .onnx a VehicleDetector.");
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(input_, input_),
                                              cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat out = net_.forward();
        int dims = out.size[1], n = out.size[2];
        cv::Mat m = cv::Mat(dims, n, CV_32F, out.ptr<float>()).t();  // n x dims
        double sx = frame.cols / double(input_), sy = frame.rows / double(input_);
        std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> ids;
        for (int i = 0; i < n; ++i) {
            const float* r = m.ptr<float>(i);
            cv::Mat cls = m.row(i).colRange(4, dims);
            double mx; cv::Point p;
            cv::minMaxLoc(cls, nullptr, &mx, nullptr, &p);
            if (mx < conf_thr_) continue;
            double w = r[2] * sx, h = r[3] * sy, x1 = r[0] * sx - w / 2, y1 = r[1] * sy - h / 2;
            boxes.emplace_back(int(x1), int(y1), int(w), int(h));
            scores.push_back(float(mx)); ids.push_back(p.x);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, float(conf_thr_), float(nms_thr_), keep);
        std::vector<Detection> dets;
        for (int k : keep) {
            const auto& b = boxes[k];
            dets.push_back({{double(b.x), double(b.y), double(b.x + b.width), double(b.y + b.height)},
                            class_name(ids[k]), double(scores[k])});
        }
        return dets;
    }
    static std::string class_name(int id) {
        static const char* coco[] = {
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
            "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"};
        return (id >= 0 && id < 80) ? coco[id] : "class_" + std::to_string(id);
    }
private:
    cv::dnn::Net net_;
    bool loaded_ = false;
    double conf_thr_, nms_thr_;
    int input_;
};

// ----------------------------------------------------------------------------
// 8. Tipos de datos de objetos
// ----------------------------------------------------------------------------
struct TrackedObject {
    int object_id;
    double x, y, speed;
    int lane;
    bool data_is_valid = true;
};

// ----------------------------------------------------------------------------
// 7.3 LDA-32: distancia y posicion lateral desde camara (piso plano)
// ----------------------------------------------------------------------------
struct CameraCalibration {
    double focal_length_px = 700.0;
    double camera_height = 1.4;
    int image_width = 960;
    int image_height = 540;
};

class CameraObjectLocalizer {
public:
    explicit CameraObjectLocalizer(CameraCalibration c) : calib_(c) {}
    // {forward_distance, lateral_offset}
    std::optional<std::pair<double, double>> estimate_position(const Detection& d) const {
        double u = (d.bbox[0] + d.bbox[2]) / 2, v = d.bbox[3];
        double cx = calib_.image_width / 2.0, cy = calib_.image_height / 2.0;
        if (v <= cy) return std::nullopt;
        double fwd = calib_.focal_length_px * calib_.camera_height / (v - cy);
        double lat = (u - cx) * fwd / calib_.focal_length_px;
        return std::make_pair(fwd, lat);
    }
    std::optional<TrackedObject> to_tracked_object(const Detection& d, const VehicleState& veh,
                                                   int object_id, double speed_estimate = 0.0,
                                                   int lane = 0) const {
        auto p = estimate_position(d);
        if (!p) return std::nullopt;
        return TrackedObject{object_id, veh.x + p->first, veh.y + p->second, speed_estimate, lane};
    }
private:
    CameraCalibration calib_;
};

// ----------------------------------------------------------------------------
// 9. Tracking: Kalman filter simple (estado [x y vx vy])
// ----------------------------------------------------------------------------
class SimpleKalmanTracker {
public:
    explicit SimpleKalmanTracker(double dt = 0.05) : dt_(dt) {
        x_ = cv::Matx41d(0, 0, 0, 0);
        P_ = cv::Matx44d::eye() * 10.0;
        F_ = cv::Matx44d(1, 0, dt, 0,  0, 1, 0, dt,  0, 0, 1, 0,  0, 0, 0, 1);
        H_ = cv::Matx24d(1, 0, 0, 0,  0, 1, 0, 0);
        Q_ = cv::Matx44d::eye() * 0.01;
        R_ = cv::Matx22d::eye() * 1.0;
    }
    cv::Matx41d predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.t() + Q_;
        return x_;
    }
    cv::Matx41d update(double mx, double my) {
        cv::Matx21d z(mx, my);
        cv::Matx21d y = z - H_ * x_;
        cv::Matx22d S = H_ * P_ * H_.t() + R_;
        cv::Matx42d K = P_ * H_.t() * S.inv();
        x_ = x_ + K * y;
        P_ = (cv::Matx44d::eye() - K * H_) * P_;
        return x_;
    }
    std::pair<double, double> velocity() const { return {x_(2, 0), x_(3, 0)}; }
private:
    double dt_;
    cv::Matx41d x_;
    cv::Matx44d P_, F_, Q_;
    cv::Matx24d H_;
    cv::Matx22d R_;
};

// ----------------------------------------------------------------------------
// 10. Sensor Fusion (promedio ponderado por confianza)
// ----------------------------------------------------------------------------
struct SensorObs { double x = 0.0, y = 0.0, speed = 0.0, confidence = 0.5; };
struct FusedObject {
    double x, y, speed, confidence;
    std::vector<std::string> sources;
};

class SensorFusionEKF {
public:
    std::optional<FusedObject> fuse(const std::optional<SensorObs>& camera,
                                    const std::optional<SensorObs>& radar,
                                    const std::optional<SensorObs>& lidar) const {
        const std::optional<SensorObs>* objs[3] = {&camera, &radar, &lidar};
        const char* names[3] = {"camera", "radar", "lidar"};
        std::vector<double> xs, ys, sp, w;
        FusedObject f{};
        for (int i = 0; i < 3; ++i) {
            if (!*objs[i]) continue;
            f.sources.push_back(names[i]);
            xs.push_back((*objs[i])->x); ys.push_back((*objs[i])->y);
            sp.push_back((*objs[i])->speed); w.push_back((*objs[i])->confidence);
        }
        if (xs.empty()) return std::nullopt;
        double sum = std::accumulate(w.begin(), w.end(), 0.0);
        for (auto& v : w) v /= sum;
        auto dot = [&](const std::vector<double>& a) {
            double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * w[i]; return s; };
        f.x = dot(xs); f.y = dot(ys); f.speed = dot(sp);
        f.confidence = *std::max_element(w.begin(), w.end());
        return f;
    }
};

// ----------------------------------------------------------------------------
// RNG auxiliar (reemplaza numpy.random.default_rng)
// ----------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(unsigned long long seed = std::random_device{}()) : gen_(seed) {}
    double uniform(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(gen_); }
    double normal(double mean, double sd) { return sd <= 0 ? mean : std::normal_distribution<double>(mean, sd)(gen_); }
    int randint(int n) { return std::uniform_int_distribution<int>(0, n - 1)(gen_); }
    std::mt19937_64& engine() { return gen_; }
private:
    std::mt19937_64 gen_;
};

// ----------------------------------------------------------------------------
// 11. LiDAR: nube de puntos (suite real del proyecto)
// ----------------------------------------------------------------------------
struct LidarUnitSpec {
    std::string name;
    std::array<double, 3> mount_position;
    double fov_deg;
    double max_range;
    std::string scan_type;  // mechanical_360 | solid_state_long_range | solid_state_blind_spot
    int points_per_object = 20;
};

inline const std::vector<LidarUnitSpec> LIDAR_SUITE = {
    {"ouster_os1_roof",          {0.0, 0.0, 1.6},   360.0, 120.0, "mechanical_360",        30},
    {"robosense_em4_front",      {2.4, 0.0, 0.5},    30.0, 200.0, "solid_state_long_range", 15},
    {"robosense_e1r_front_left", {2.3, 0.9, 0.4},   120.0,  30.0, "solid_state_blind_spot", 10},
    {"robosense_e1r_front_right",{2.3, -0.9, 0.4},  120.0,  30.0, "solid_state_blind_spot", 10},
    {"robosense_e1r_rear_left",  {-2.3, 0.9, 0.4},  120.0,  30.0, "solid_state_blind_spot", 10},
    {"robosense_e1r_rear_right", {-2.3, -0.9, 0.4}, 120.0,  30.0, "solid_state_blind_spot", 10},
};

struct LidarPoint { double x, y, z, intensity, reflectivity; };

class LidarPointCloudSimulator {
public:
    explicit LidarPointCloudSimulator(Rng rng = Rng()) : rng_(rng) {}

    std::map<int, std::vector<LidarPoint>> scan_unit(const VehicleState& v, const LidarUnitSpec& u,
                                                     const std::vector<TrackedObject>& objs,
                                                     const VehicleParams& vp) {
        std::map<int, std::vector<LidarPoint>> result;
        for (const auto& o : objs) {
            double dist, ang;
            distance_and_angle(v, u, o, dist, ang);
            if (dist > u.max_range || std::fabs(ang) > u.fov_deg / 2) continue;
            std::vector<LidarPoint> pts;
            double noise = (u.scan_type.find("solid_state") != std::string::npos) ? 0.02 : 0.04;
            for (int i = 0; i < u.points_per_object; ++i) {
                double px = o.x + rng_.uniform(-vp.length / 2, vp.length / 2) + rng_.normal(0, noise);
                double py = o.y + rng_.uniform(-vp.width / 2, vp.width / 2) + rng_.normal(0, noise);
                double pz = rng_.uniform(0.0, 1.6);
                double inten = clip(rng_.normal(120, 30), 0, 255);
                double refl = clip(rng_.normal(40, 15), 0, 100);
                pts.push_back({px, py, pz, inten, refl});
            }
            result[o.object_id] = std::move(pts);
        }
        return result;
    }
    std::map<int, std::vector<LidarPoint>> scan_all(const VehicleState& v,
                                                    const std::vector<TrackedObject>& objs,
                                                    const VehicleParams& vp) {
        std::map<int, std::vector<LidarPoint>> combined;
        for (const auto& o : objs) combined[o.object_id];
        for (const auto& u : LIDAR_SUITE)
            for (auto& [id, pts] : scan_unit(v, u, objs, vp))
                combined[id].insert(combined[id].end(), pts.begin(), pts.end());
        return combined;
    }
private:
    static void distance_and_angle(const VehicleState& v, const LidarUnitSpec& u, const TrackedObject& o,
                                   double& dist, double& ang) {
        double sx = v.x + u.mount_position[0] * std::cos(v.yaw) - u.mount_position[1] * std::sin(v.yaw);
        double sy = v.y + u.mount_position[0] * std::sin(v.yaw) + u.mount_position[1] * std::cos(v.yaw);
        double dx = o.x - sx, dy = o.y - sy;
        dist = std::hypot(dx, dy);
        ang = wrap180(rad2deg(std::atan2(dy, dx) - v.yaw));
    }
    Rng rng_;
};

// ----------------------------------------------------------------------------
// 11.1 Ground plane segmentation (RANSAC): ax + by + cz + d = 0
// ----------------------------------------------------------------------------
struct GroundPlaneResult {
    std::optional<std::array<double, 4>> plane;  // a, b, c, d
    std::vector<bool> ground_mask;
};

inline GroundPlaneResult fit_ground_plane_ransac(const std::vector<cv::Vec3d>& pts,
                                                 double epsilon_ground = 0.15, int n_iter = 100,
                                                 Rng* rng_in = nullptr) {
    Rng local; Rng& rng = rng_in ? *rng_in : local;
    size_t n = pts.size();
    GroundPlaneResult res; res.ground_mask.assign(n, false);
    if (n < 3) return res;
    int best = 0;
    std::vector<size_t> idx(n);
    for (int it = 0; it < n_iter; ++it) {
        std::iota(idx.begin(), idx.end(), 0);
        for (int k = 0; k < 3; ++k) std::swap(idx[k], idx[k + rng.randint(int(n) - k)]);
        const cv::Vec3d &p1 = pts[idx[0]], &p2 = pts[idx[1]], &p3 = pts[idx[2]];
        cv::Vec3d normal = (p2 - p1).cross(p3 - p1);
        double norm = cv::norm(normal);
        if (norm < 1e-6) continue;
        cv::Vec3d abc = normal / norm;
        double d = -abc.dot(p1);
        std::vector<bool> mask(n); int inl = 0;
        for (size_t i = 0; i < n; ++i) {
            mask[i] = std::fabs(pts[i].dot(abc) + d) <= epsilon_ground;
            inl += mask[i];
        }
        if (inl > best) { best = inl; res.plane = {abc[0], abc[1], abc[2], d}; res.ground_mask = mask; }
    }
    return res;
}

// ----------------------------------------------------------------------------
// 12. Radar 4D (con elevacion)
// ----------------------------------------------------------------------------
struct RadarUnitSpec {
    std::string name;
    std::array<double, 3> mount_position;
    double facing_deg;
    double fov_deg = 120.0;
    double max_range = 250.0;
};
inline const std::vector<RadarUnitSpec> RADAR_SUITE = {
    {"radar_4d_front", {2.4, 0.0, 0.5}, 0.0},
    {"radar_4d_rear", {-2.4, 0.0, 0.5}, 180.0},
};
struct RadarDetection { double range_m, azimuth_deg, elevation_deg, relative_velocity; };

class Radar4DSimulator {
public:
    explicit Radar4DSimulator(Rng rng = Rng()) : rng_(rng) {}
    std::map<int, RadarDetection> scan_unit(const VehicleState& v, const RadarUnitSpec& u,
                                            const std::vector<TrackedObject>& objs) {
        std::map<int, RadarDetection> result;
        for (const auto& o : objs) {
            double dx = o.x - v.x, dy = o.y - v.y;
            double dist = std::hypot(dx, dy);
            double ang = wrap180(rad2deg(std::atan2(dy, dx) - v.yaw) - u.facing_deg);
            if (dist > u.max_range || std::fabs(ang) > u.fov_deg / 2) continue;
            RadarDetection d;
            d.range_m = dist + rng_.normal(0, 0.1);
            d.azimuth_deg = ang + rng_.normal(0, 0.3);
            d.elevation_deg = rng_.normal(0.0, 0.5);
            d.relative_velocity = (v.speed - o.speed) + rng_.normal(0, 0.1);
            result[o.object_id] = d;
        }
        return result;
    }
    std::map<int, std::vector<RadarDetection>> scan_all(const VehicleState& v,
                                                        const std::vector<TrackedObject>& objs) {
        std::map<int, std::vector<RadarDetection>> combined;
        for (const auto& u : RADAR_SUITE)
            for (auto& [id, d] : scan_unit(v, u, objs)) combined[id].push_back(d);
        return combined;
    }
private:
    Rng rng_;
};

// ----------------------------------------------------------------------------
// 13. Camara sintetica (pinhole) desde ground truth
// ----------------------------------------------------------------------------
struct PinholeCameraParams {
    int image_width = 960;
    int image_height = 540;
    double focal_length_px = 700.0;
    double camera_height = 1.4;
};

class SyntheticCameraRenderer {
public:
    explicit SyntheticCameraRenderer(PinholeCameraParams c) : cam_(c) {}

    std::optional<cv::Point> world_to_pixel(double forward, double lateral) const {
        if (forward <= 1.0) return std::nullopt;
        double cx = cam_.image_width / 2.0, cy = cam_.image_height / 2.0;
        double u = cx + cam_.focal_length_px * (lateral / forward);
        double v = cy + cam_.focal_length_px * (cam_.camera_height / forward);
        return cv::Point(int(u), int(v));
    }

    cv::Mat render(const VehicleState& veh, const LaneState& lane, const std::vector<TrackedObject>& objs,
                   const VehicleParams& vp) const {
        cv::Mat img(cam_.image_height, cam_.image_width, CV_8UC3, cv::Scalar(90, 90, 90));
        cv::rectangle(img, cv::Point(0, cam_.image_height / 2), cv::Point(cam_.image_width, cam_.image_height),
                      cv::Scalar(60, 60, 60), -1);

        struct L { double off; cv::Scalar col; };
        for (const L& l : {L{0.0, {0, 255, 255}}, L{lane.width / 2, {255, 255, 255}},
                           L{-lane.width / 2, {255, 255, 255}}}) {
            std::vector<cv::Point> pts;
            for (int i = 0; i < 40; ++i) {
                double d = 2.0 + (100.0 - 2.0) * i / 39.0;  // linspace(2, 100, 40)
                if (auto p = world_to_pixel(d, lane.center + l.off - veh.y)) pts.push_back(*p);
            }
            if (pts.size() >= 2) cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, false, l.col, 2);
        }
        for (const auto& o : objs) {
            double fwd = o.x - veh.x, lat = o.y - veh.y;
            auto c = world_to_pixel(fwd, lat);
            if (!c) continue;
            int wpx = std::max(4, int(cam_.focal_length_px * vp.width / fwd));
            int hpx = std::max(3, int(cam_.focal_length_px * 1.5 / fwd));
            cv::Point tl(c->x - wpx / 2, c->y - hpx), br(c->x + wpx / 2, c->y);
            cv::rectangle(img, tl, br, cv::Scalar(0, 140, 255), -1);
            cv::rectangle(img, tl, br, cv::Scalar(0, 0, 0), 1);
        }
        return img;
    }
private:
    PinholeCameraParams cam_;
};

// ----------------------------------------------------------------------------
// 14. Validacion de cambio de carril
// ----------------------------------------------------------------------------
class LaneChangeValidator {
public:
    explicit LaneChangeValidator(SafetyParams s) : safety(s) {}
    SafetyParams safety;

    static std::optional<int> target_lane(int current, const std::string& dir) {
        if (dir == "LEFT") return current + 1;
        if (dir == "RIGHT") return current - 1;
        return std::nullopt;
    }
    static std::vector<TrackedObject> objects_in_lane(int lane, const std::vector<TrackedObject>& objs) {
        std::vector<TrackedObject> r;
        for (const auto& o : objs) if (o.lane == lane) r.push_back(o);
        return r;
    }
    static double distance(double vx, double vy, double ox, double oy) {
        return std::sqrt((ox - vx) * (ox - vx) + (oy - vy) * (oy - vy));
    }
    static double relative_speed(double vs, double os) { return vs - os; }
    static double ttc(double dist, double rel) {
        if (rel <= 0) return std::numeric_limits<double>::infinity();
        return dist / rel;
    }
    bool is_safe_distance(double d) const { return d >= safety.min_safe_distance; }
    bool is_safe_ttc(double t) const { return std::isinf(t) ? true : t >= safety.min_safe_ttc; }

    bool can_change_lane(const std::string& dir, const VehicleState& v, int current_lane,
                         const std::vector<TrackedObject>& objs) const {
        auto target = target_lane(current_lane, dir);
        if (!target) return false;
        auto tobjs = objects_in_lane(*target, objs);
        if (tobjs.empty()) return true;
        for (const auto& o : tobjs) {
            if (!o.data_is_valid) return false;
            double d = distance(v.x, v.y, o.x, o.y);
            double t = ttc(d, relative_speed(v.speed, o.speed));
            if (!is_safe_distance(d)) return false;
            if (!is_safe_ttc(t)) return false;
        }
        return true;
    }
};

// ----------------------------------------------------------------------------
// 15. Simulador integrado
// ----------------------------------------------------------------------------
struct History {
    std::vector<double> time, x, y, yaw, lateral_error, heading_error, steering;
    std::vector<bool> departure;
};

class Simulator {
public:
    Simulator(VehicleParams v, ControlParams c, SafetyParams s, SimulationParams sim, LaneState lane)
        : vehicle_params(v), sim_params(sim), lane(lane), bicycle_model(v), controller(c), validator(s) {
        state.speed = 15.0;
    }
    void run() {
        double t = 0.0;
        while (t <= sim_params.total_time) {
            double ey = LaneGeometry::lateral_error(state.y, lane);
            double ep = LaneGeometry::heading_error(state.yaw, lane);
            double steer = controller.compute_steering(ey, ep);
            bool dep = LaneGeometry::is_departing(state.y, lane, vehicle_params, validator.safety);
            history.time.push_back(t);
            history.x.push_back(state.x);
            history.y.push_back(state.y);
            history.yaw.push_back(state.yaw);
            history.lateral_error.push_back(ey);
            history.heading_error.push_back(ep);
            history.steering.push_back(steer);
            history.departure.push_back(dep);
            state = bicycle_model.step(state, steer, sim_params.dt);
            t += sim_params.dt;
        }
    }
    std::pair<bool, bool> evaluate_lane_change(int current_lane, const std::vector<TrackedObject>& objs) const {
        return {validator.can_change_lane("LEFT", state, current_lane, objs),
                validator.can_change_lane("RIGHT", state, current_lane, objs)};
    }

    VehicleParams vehicle_params;
    SimulationParams sim_params;
    LaneState lane;
    KinematicBicycleModel bicycle_model;
    LaneKeepingController controller;
    LaneChangeValidator validator;
    VehicleState state;
    History history;
};

// ----------------------------------------------------------------------------
// 16.1 Vista de dron (bird's-eye) de la escena
// ----------------------------------------------------------------------------
inline cv::Mat render_topdown(const VehicleState& veh, const LaneState& lane,
                              const std::vector<TrackedObject>& objs, const VehicleParams& vp,
                              bool can_change_left, int W = 960, int H = 260) {
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(50, 110, 50));
    double x_min = veh.x - 15, x_max = veh.x + 85;
    double y_min = -3.0, y_max = lane.width * 2.0;
    auto w2c = [&](double x, double y) {
        return cv::Point(int((x - x_min) / (x_max - x_min) * W),
                         int(H - (y - y_min) / (y_max - y_min) * H));
    };
    cv::rectangle(canvas, w2c(x_min, lane.width * 1.5 + lane.width / 2), w2c(x_max, -lane.width / 2),
                  cv::Scalar(70, 70, 70), -1);
    for (double yl : {-lane.width / 2, lane.width / 2, lane.width * 1.5 + lane.width / 2})
        cv::line(canvas, w2c(x_min, yl), w2c(x_max, yl), cv::Scalar(255, 255, 255), 2);

    auto draw_vehicle = [&](double x, double y, cv::Scalar color, const std::string& label) {
        cv::Point c = w2c(x, y);
        int hl = std::max(6, int(vp.length / (x_max - x_min) * W / 2));
        int hw = std::max(4, int(vp.width / (y_max - y_min) * H / 2));
        cv::rectangle(canvas, cv::Point(c.x - hl, c.y - hw), cv::Point(c.x + hl, c.y + hw), color, -1);
        cv::putText(canvas, label, cv::Point(c.x - hl, c.y - hw - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    };
    draw_vehicle(veh.x, veh.y, cv::Scalar(0, 255, 255), "EGO");
    for (const auto& o : objs)
        draw_vehicle(o.x, o.y, can_change_left ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255),
                     "obj" + std::to_string(o.object_id));
    cv::putText(canvas, can_change_left ? "LEFT: OK" : "LEFT: NO", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                can_change_left ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255), 2);
    return canvas;
}

}  // namespace lda
