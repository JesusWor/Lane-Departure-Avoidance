// Equivalente a ejecutar el notebook LDA_LCV_v4 de principio a fin.
// Uso:  ./lda_demo [--video ruta.mp4] [--weights yolov8n.onnx]
#include "lda.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>

using namespace lda;

static int g_fail = 0;
static void run_case(const std::string& name, bool cond, const std::string& detail = "") {
    std::printf("[%s] %s %s\n", cond ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    if (!cond) ++g_fail;
}

// ---------------------------------------------------------------- 6. detector
static void demo_lane_detector(const std::string& video_path) {
    ClassicalLaneDetector det;
    cv::Mat synth = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::line(synth, {150, 480}, {280, 280}, {255, 255, 255}, 8);
    cv::line(synth, {500, 480}, {360, 280}, {255, 255, 255}, 8);
    auto r = det.process(synth);
    std::printf("=== Prueba con imagen sintetica ===\n");
    if (r.left_fit)  std::printf("left_fit: (%f, %f)\n", r.left_fit->slope, r.left_fit->intercept);
    if (r.right_fit) std::printf("right_fit: (%f, %f)\n", r.right_fit->slope, r.right_fit->intercept);
    if (r.center_px) std::printf("center_px estimado: %f\n", *r.center_px);

    if (!video_path.empty()) {
        cv::VideoCapture cap(video_path);
        cv::Mat frame;
        if (cap.read(frame)) {
            auto rr = det.process(frame);
            std::printf("=== Prueba con frame real ===\n");
            if (rr.center_px) std::printf("center_px estimado: %f\n", *rr.center_px);
            cv::Mat overlay = frame.clone();
            for (auto& l : rr.lines) cv::line(overlay, {l[0], l[1]}, {l[2], l[3]}, {0, 255, 0}, 3);
            cv::imwrite("lane_lines_real_frame.png", overlay);
        } else {
            std::printf("No se pudo leer el video.\n");
        }
    }
}

// ------------------------------------------------------- 7.3 LDA-32 round trip
static void demo_localizer() {
    CameraCalibration calib;
    CameraObjectLocalizer loc(calib);
    std::printf("=== LDA-32: prueba round-trip ===\n");
    const double cases[][2] = {{25.0, 1.8}, {55.0, -0.2}, {10.0, 0.0}, {80.0, 3.0}};
    for (auto& c : cases) {
        double cx = calib.image_width / 2.0, cy = calib.image_height / 2.0;
        double u = cx + calib.focal_length_px * (c[1] / c[0]);
        double v = cy + calib.focal_length_px * (calib.camera_height / c[0]);
        Detection d{{u - 20, v - 60, u + 20, v}, "car", 0.9};
        auto [de, le] = *loc.estimate_position(d);
        bool ok = std::fabs(c[0] - de) < 1e-6 && std::fabs(c[1] - le) < 1e-6;
        std::printf("[%s] distancia real=%.1fm estimada=%.3fm | lateral real=%.1fm estimado=%.3fm\n",
                    ok ? "PASS" : "FAIL", c[0], de, c[1], le);
    }
}

// ---------------------------------------------------------- 11 / 12 sensores
static void demo_sensors() {
    VehicleState ego{0.0, 0.0, 0.0, 15.0};
    std::vector<TrackedObject> objs = {{1, 40.0, 3.5, 13.0, 1}, {2, 15.0, -1.0, 15.0, 0}};
    LidarPointCloudSimulator lidar;
    for (auto& [id, pts] : lidar.scan_all(ego, objs, VehicleParams()))
        std::printf("Objeto %d: %zu puntos de LiDAR detectados\n", id, pts.size());
    Radar4DSimulator radar;
    for (auto& [id, dets] : radar.scan_all(ego, objs))
        for (auto& d : dets)
            std::printf("Objeto %d: rango=%.1fm azimuth=%.1f° elevacion=%.1f° vel_relativa=%.1fm/s\n", id,
                        d.range_m, d.azimuth_deg, d.elevation_deg, d.relative_velocity);
}

// ----------------------------------------------------------- 13 camara sintetica
static void demo_synthetic_camera() {
    PinholeCameraParams cp;
    SyntheticCameraRenderer renderer(cp);
    LaneState lane{0.0, 0.0, 3.5};
    VehicleState ego{0.0, 0.3, 0.0, 15.0};
    std::vector<TrackedObject> objs = {{1, 25.0, 1.8, 13.0, 1}, {2, 55.0, -0.2, 15.0, 0}};
    cv::Mat frame = renderer.render(ego, lane, objs, VehicleParams());
    cv::imwrite("synthetic_camera.png", frame);
    ClassicalLaneDetector det;
    auto r = det.process(frame);
    if (r.center_px) std::printf("center_px detectado por ClassicalLaneDetector: %f\n", *r.center_px);
    std::printf("Centro real de la imagen (referencia): %f\n", cp.image_width / 2.0);
}

// ------------------------------------------------------------ 16 simulacion
static void run_simulation() {
    VehicleParams vp; ControlParams cp; SafetyParams sp; SimulationParams sim; LaneState lane;
    Simulator simulator(vp, cp, sp, sim, lane);
    simulator.run();
    std::vector<TrackedObject> objs = {{1, 40.0, 3.5, 13.0, 1}, {2, 70.0, -3.5, 15.0, -1}};
    auto [cl, cr] = simulator.evaluate_lane_change(0, objs);
    auto& h = simulator.history;
    std::printf("==========================================\nLANE CHANGE VALIDATION\n"
                "==========================================\n");
    std::printf("Can change LEFT: %s\nCan change RIGHT: %s\n\n", cl ? "True" : "False", cr ? "True" : "False");
    std::printf("==========================================\nSIMULATION RESULTS\n"
                "==========================================\n");
    std::printf("Final X: %.3f m\nFinal Y: %.3f m\nFinal Yaw: %.3f degrees\n", simulator.state.x,
                simulator.state.y, rad2deg(simulator.state.yaw));
    std::printf("Final lateral error: %.3f m\nFinal steering: %.3f degrees\n\n", h.lateral_error.back(),
                rad2deg(h.steering.back()));
    int steps = 0; for (bool d : h.departure) steps += d;
    std::printf("==========================================\nLANE DEPARTURE\n"
                "==========================================\n");
    std::printf("Departure detected: %s\nNumber of detected steps: %d\n", steps > 0 ? "True" : "False", steps);

    // Reemplazo de las graficas de matplotlib: CSV (abrir con Excel/Python/gnuplot)
    std::ofstream csv("history.csv");
    csv << "time,x,y,yaw,lateral_error,heading_error,steering,departure\n";
    for (size_t i = 0; i < h.time.size(); ++i)
        csv << h.time[i] << ',' << h.x[i] << ',' << h.y[i] << ',' << h.yaw[i] << ',' << h.lateral_error[i]
            << ',' << h.heading_error[i] << ',' << h.steering[i] << ',' << int(h.departure[i]) << '\n';
}

// ------------------------------------------ 16.1 animacion con alerta + correccion
static void run_animation() {
    VehicleParams vp; ControlParams cp; SafetyParams sp;
    LaneState lane{0.0, 0.0, 3.5};
    KinematicBicycleModel model(vp);
    LaneKeepingController controller(cp);
    LaneChangeValidator validator(sp);
    SyntheticCameraRenderer renderer{PinholeCameraParams()};

    VehicleState st{0.0, 0.0, 0.0, 15.0};
    TrackedObject other{1, 55.0, lane.center + lane.width, 8.0, 1};
    const double DISTURBANCE = deg2rad(0.05);
    const double dt = 0.05;
    const int n_steps = 200;

    std::vector<cv::Mat> frames;
    bool alerted = false; int trigger_step = -1, settle_step = -1;

    for (int step = 0; step < n_steps; ++step) {
        double t = step * dt;
        double ey = LaneGeometry::lateral_error(st.y, lane);
        double ep = LaneGeometry::heading_error(st.yaw, lane);
        bool departing = LaneGeometry::is_departing(st.y, lane, vp, sp);

        if (departing && !alerted) {
            alerted = true; trigger_step = step;
            std::printf("[t=%.2fs] ALERTA: saliendo del carril hacia la %s -- iniciando correccion\n", t,
                        st.y > lane.center ? "DERECHA" : "IZQUIERDA");
        }
        double steering;
        if (alerted) {
            steering = controller.compute_steering(ey, ep);
            if (settle_step < 0 && std::fabs(ey) < 0.05 && std::fabs(ep) < deg2rad(1)) {
                settle_step = step;
                std::printf("[t=%.2fs] Centrado de nuevo en el carril\n", t);
            }
        } else {
            steering = DISturbanceFix:
            steering = DISTURBANCE;
        }
        bool can_left = validator.can_change_lane("LEFT", st, 0, {other});

        cv::Mat cam = renderer.render(st, lane, {other}, vp);
        if (alerted && trigger_step >= 0 && (t - trigger_step * dt) < 1.5) {
            cv::rectangle(cam, {0, 0}, {cam.cols, 50}, {0, 0, 220}, -1);
            cv::putText(cam, "ALERTA: SALIENDO DEL CARRIL", {20, 33}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {255, 255, 255}, 2);
        } else if (alerted && settle_step < 0) {
            cv::putText(cam, "Corrigiendo rumbo...", {20, 33}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 200, 255}, 2);
        } else if (settle_step >= 0) {
            cv::putText(cam, "Centrado en el carril", {20, 33}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        }
        char buf[64]; std::snprintf(buf, sizeof buf, "t=%4.1fs  err_lat=%+.2fm", t, ey);
        cv::putText(cam, buf, {20, cam.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1);

        cv::Mat top = render_topdown(st, lane, {other}, vp, can_left);
        cv::Mat top_r;
        cv::resize(top, top_r, cv::Size(cam.cols, int(double(cam.cols) * top.rows / top.cols)));
        cv::Mat combined;
        cv::vconcat(cam, top_r, combined);
        frames.push_back(combined);

        st = model.step(st, steering, dt);
        other.x += other.speed * dt;
    }
    std::printf("\nFrames generados: %zu\ny final: %.4f m (deberia ser ~0, centrado y estable)\n", frames.size(),
                st.y);

    cv::VideoWriter writer("animation.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           int(std::lround(1.0 / dt)), frames[0].size());
    for (auto& f : frames) writer.write(f);
    std::printf("Video guardado en animation.mp4 (codec mp4v; convertir con ffmpeg a H.264 si hace falta)\n");
}

// ------------------------------------------------------- 18 pruebas unitarias
static void run_unit_tests() {
    std::printf("=== CASOS DE LANE KEEPING ===\n\n");
    LaneState lane{0.0, 0.0, 3.5};
    LaneKeepingController control{ControlParams()};

    double ey = LaneGeometry::lateral_error(0.0, lane), ep = LaneGeometry::heading_error(0.0, lane);
    run_case("Vehiculo centrado en carril", ey == 0.0 && ep == 0.0 && control.compute_steering(ey, ep) == 0.0);

    ey = LaneGeometry::lateral_error(1.5, lane);
    double s = control.compute_steering(ey, 0.0);
    char b[64]; std::snprintf(b, sizeof b, "(steering=%.3f)", s);
    run_case("Acercandose al limite izquierdo -> corrige a la derecha", s < 0, b);

    ey = LaneGeometry::lateral_error(-1.5, lane);
    s = control.compute_steering(ey, 0.0);
    std::snprintf(b, sizeof b, "(steering=%.3f)", s);
    run_case("Acercandose al limite derecho -> corrige a la izquierda", s > 0, b);

    VehicleParams vp; SafetyParams sf;
    run_case("Vehiculo muy desviado -> detecta riesgo de salida", LaneGeometry::is_departing(2.0, lane, vp, sf));
    run_case("Vehiculo centrado -> NO detecta riesgo de salida", !LaneGeometry::is_departing(0.0, lane, vp, sf));

    std::printf("\n=== CASOS DE LANE-CHANGE VALIDATION ===\n\n");
    LaneChangeValidator v{SafetyParams()};
    VehicleState ego{0.0, 0.0, 0.0, 15.0};
    run_case("Carril objetivo vacio -> permite cambio", v.can_change_lane("LEFT", ego, 0, {}));
    run_case("Vehiculo lejano delante -> permite cambio",
             v.can_change_lane("LEFT", ego, 0, {{1, 60.0, 0.0, 14.0, 1}}));
    run_case("Vehiculo cercano delante -> rechaza cambio",
             !v.can_change_lane("LEFT", ego, 0, {{1, 5.0, 0.0, 14.0, 1}}));
    run_case("Dos vehiculos, uno inseguro -> rechaza cambio",
             !v.can_change_lane("LEFT", ego, 0, {{1, 60.0, 0.0, 14.0, 1}, {2, 5.0, 0.0, 14.0, 1}}));
    run_case("Dato invalido en carril objetivo -> rechaza (conservador)",
             !v.can_change_lane("LEFT", ego, 0, {{1, 60.0, 0.0, 14.0, 1, false}}));
    std::printf("\n%s\n", g_fail == 0 ? "Todos los casos PASS." : "Hay casos FAIL.");
}

int main(int argc, char** argv) {
    std::string video, weights;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--video") video = argv[i + 1];
        else if (a == "--weights") weights = argv[i + 1];
    }
    demo_lane_detector(video);
    demo_localizer();
    demo_sensors();
    demo_synthetic_camera();
    run_simulation();
    run_animation();
    run_unit_tests();

    if (!weights.empty() && !video.empty()) {  // opcional: YOLO ONNX sobre un frame real
        VehicleDetector detector(weights);
        cv::VideoCapture cap(video); cv::Mat frame;
        if (cap.read(frame))
            for (auto& d : detector.detect(frame))
                std::printf("%s %.2f [%.0f %.0f %.0f %.0f]\n", d.class_name.c_str(), d.confidence, d.bbox[0],
                            d.bbox[1], d.bbox[2], d.bbox[3]);
    }
    return g_fail == 0 ? 0 : 1;
}
