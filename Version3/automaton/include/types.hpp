#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace lda
{

    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kPi = 3.14159265358979323846; // M_PI no existe por defecto en MSVC

    struct Params
    {
        // Vehiculo (Porsche Taycan 4S, seccion 2 del notebook)
        double vehicle_length = 4.963;
        double vehicle_width = 1.966;
        double wheelbase = 2.9;
        // Control (seccion 11 del documento)
        double k_lateral = 0.8;
        double k_heading = 1.2;
        double max_steering = 30.0 * kPi / 180.0;
        double ff_gain = 1.0; // feedforward de curvatura (0 = desactivado, igual que el documento)
        // Seguridad (parametros de SIMULACION, no certificados)
        double min_safe_distance = 10.0;  // m
        double min_safe_ttc = 3.0;        // s
        double lane_safety_margin = 0.30; // m
        double target_lane_tol = 0.5;     // m
        // Tiempos
        double max_age_s = 0.30;
        double hz = 50.0;
        // Modo asistencia: solo corrige cuando hay riesgo de salida (como la animacion 16.1)
        bool assist_only = false;
        double recover_ey = 0.15; // m
    };

    struct LaneObs
    {
        bool valid = false;
        double lateral_error = 0.0; // y_vehiculo - y_centro_carril  (m, izquierda +)
        double heading_error = 0.0; // yaw_vehiculo - yaw_carril     (rad, CCW +)
        double curvature = 0.0;     // 1/m, izquierda +
        double lane_width = 3.5;    // m
        double confidence = 0.0;
    };

    struct ObjectObs
    {
        int id = -1;
        double x = 0.0;      // m, adelante +, marco del ego (centro del objeto)
        double y = 0.0;      // m, izquierda +
        double vx_rel = 0.0; // m/s, (v_obj - v_ego) a lo largo del carril
        double length = 4.5;
        bool valid = true;
        double confidence = 1.0;
        std::string source;
    };

    enum class Request
    {
        NONE,
        LEFT,
        RIGHT
    };

    struct Perception
    {
        int64_t ts_ns = 0;
        int64_t frame_id = -1;
        LaneObs lane;
        std::vector<ObjectObs> objects;
        Request request = Request::NONE;
        double ego_speed = 0.0;
        bool ego_speed_valid = false;
    };

    enum class State
    {
        LANE_KEEPING,
        CORRECTING,
        DEGRADED
    };

    inline const char *to_str(State s)
    {
        switch (s)
        {
        case State::LANE_KEEPING:
            return "LANE_KEEPING";
        case State::CORRECTING:
            return "CORRECTING";
        default:
            return "DEGRADED";
        }
    }

    struct Decision
    {
        State state = State::DEGRADED;
        double steer_rad = 0.0; // delta, izquierda +
        bool departing = false;
        bool fresh = false;
        bool lane_change_asked = false;
        bool lane_change_ok = false; // solo significativo si lane_change_asked
        std::string reason;
        int64_t frame_id = -1;
    };

} // namespace lda