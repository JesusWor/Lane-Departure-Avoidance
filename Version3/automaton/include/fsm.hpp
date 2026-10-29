#pragma once
#include "types.hpp"
#include "lane_keeping.hpp"
#include "lane_change.hpp"

namespace lda {

// Automata determinista. step() solo guarda el estado de histeresis.
class Automaton {
public:
    explicit Automaton(const Params& p) : p_(p), ctl_(p), val_(p) {}

    Decision step(const Perception& in, int64_t now_ns) {
        Decision d;
        d.frame_id = in.frame_id;
        d.lane_change_asked = (in.request != Request::NONE);

        const double age_s = (now_ns - in.ts_ns) * 1e-9;
        d.fresh = (in.frame_id >= 0) && age_s <= p_.max_age_s && age_s > -0.5;

        // 1) Dato viejo o carril invalido -> DEGRADED: no se adivina (RNF-05).
        if (!d.fresh || !in.lane.valid) {
            state_ = State::DEGRADED;
            d.state = state_;
            d.steer_rad = 0.0;
            d.lane_change_ok = false;
            d.reason = !d.fresh ? "dato de percepcion viejo/ausente" : "carril invalido";
            return d;
        }

        // 2) Riesgo de salida con histeresis.
        d.departing = LaneGeometry::is_departing(in.lane.lateral_error, in.lane.lane_width, p_);
        if (d.departing) state_ = State::CORRECTING;
        else if (state_ == State::CORRECTING &&
                 std::fabs(in.lane.lateral_error) < p_.recover_ey) state_ = State::LANE_KEEPING;
        else if (state_ == State::DEGRADED) state_ = State::LANE_KEEPING;
        d.state = state_;

        // 3) Direccion deseada.
        const bool steer_on = !p_.assist_only || state_ == State::CORRECTING;
        d.steer_rad = steer_on
            ? ctl_.compute_steering(in.lane.lateral_error, in.lane.heading_error, in.lane.curvature)
            : 0.0;

        // 4) Validacion de cambio de carril (TRUE/FALSE).
        if (d.lane_change_asked) {
            auto r = val_.can_change_lane(in.request, in.lane, in.objects);
            d.lane_change_ok = r.ok;
            d.reason = r.reason;
        }
        return d;
    }

private:
    Params p_;
    LaneKeepingController ctl_;
    LaneChangeValidator val_;
    State state_ = State::DEGRADED;
};

}  // namespace lda
