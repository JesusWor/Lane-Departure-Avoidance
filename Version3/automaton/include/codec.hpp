#pragma once
// JSON <-> tipos. Esquema documentado en el README (seccion "Protocolo").
#include <nlohmann/json.hpp>
#include "types.hpp"

namespace lda {
using json = nlohmann::json;

inline Perception parse_perception(const std::string& s) {
    json j = json::parse(s);
    Perception p;
    p.ts_ns = j.value("ts_ns", int64_t{0});
    p.frame_id = j.value("frame_id", int64_t{-1});
    const std::string req = j.value("request", "NONE");
    p.request = req == "LEFT" ? Request::LEFT : req == "RIGHT" ? Request::RIGHT : Request::NONE;
    if (j.contains("ego")) {
        p.ego_speed = j["ego"].value("speed", 0.0);
        p.ego_speed_valid = j["ego"].value("speed_valid", false);
    }
    if (j.contains("lane")) {
        const auto& l = j["lane"];
        p.lane.valid = l.value("valid", false);
        p.lane.lateral_error = l.value("lateral_error_m", 0.0);
        p.lane.heading_error = l.value("heading_error_rad", 0.0);
        p.lane.curvature = l.value("curvature_1pm", 0.0);
        p.lane.lane_width = l.value("lane_width_m", 3.5);
        p.lane.confidence = l.value("confidence", 0.0);
    }
    for (const auto& o : j.value("objects", json::array())) {
        ObjectObs ob;
        ob.id = o.value("id", -1);
        ob.x = o.value("x_m", 0.0);
        ob.y = o.value("y_m", 0.0);
        ob.vx_rel = o.value("vx_rel_mps", 0.0);
        ob.length = o.value("length_m", 4.5);
        ob.valid = o.value("valid", true);
        ob.confidence = o.value("conf", 1.0);
        ob.source = o.value("source", "");
        p.objects.push_back(ob);
    }
    return p;
}

inline std::string encode_decision(const Decision& d, int64_t now_ns) {
    json j;
    j["ts_ns"] = now_ns;
    j["frame_id"] = d.frame_id;            // eco: permite al puente hacer lockstep
    j["state"] = to_str(d.state);
    j["steer_rad"] = d.steer_rad;          // delta, izquierda +
    j["departing"] = d.departing;
    j["fresh"] = d.fresh;
    j["lane_change_asked"] = d.lane_change_asked;
    j["lane_change_ok"] = d.lane_change_ok;
    j["reason"] = d.reason;
    return j.dump();
}
}  // namespace lda
