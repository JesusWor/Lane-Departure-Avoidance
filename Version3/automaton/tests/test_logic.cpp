// Port de la seccion 18 del notebook (casos de la seccion 26 del documento).
#include <cstdio>
#include "fsm.hpp"
using namespace lda;
static int fails = 0;
static void check(const char* n, bool c) { std::printf("[%s] %s\n", c ? "PASS" : "FAIL", n); fails += !c; }

static LaneObs lane(double ey = 0, double epsi = 0) {
    LaneObs l; l.valid = true; l.lateral_error = ey; l.heading_error = epsi; return l;
}
static ObjectObs obj(int id, double x, double y, double vx = 0, bool valid = true) {
    ObjectObs o; o.id = id; o.x = x; o.y = y; o.vx_rel = vx; o.valid = valid; return o;
}

int main() {
    Params p; p.ff_gain = 0.0;
    LaneKeepingController c(p);
    check("centrado -> steering 0", c.compute_steering(0, 0) == 0.0);
    check("izquierda -> corrige a la derecha (delta<0)", c.compute_steering(1.5, 0) < 0);
    check("derecha -> corrige a la izquierda (delta>0)", c.compute_steering(-1.5, 0) > 0);
    check("saturacion a +-30 deg", std::fabs(c.compute_steering(50, 0)) <= p.max_steering + 1e-12);
    check("muy desviado -> riesgo de salida", LaneGeometry::is_departing(2.0, 3.5, p));
    check("centrado -> sin riesgo", !LaneGeometry::is_departing(0.0, 3.5, p));

    LaneChangeValidator v(p);
    auto L = lane();
    check("carril objetivo vacio -> TRUE", v.can_change_lane(Request::LEFT, L, {}).ok);
    check("lejano delante, mas lento (vx=-1) -> TRUE", v.can_change_lane(Request::LEFT, L, {obj(1, 60, 3.5, -1)}).ok);
    check("cercano delante -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, 12, 3.5, 0)}).ok);
    check("TTC corto delante -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, 25, 3.5, -10)}).ok);
    check("detras acercandose rapido -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, -25, 3.5, +10)}).ok);
    check("detras lejano y mas lento -> TRUE", v.can_change_lane(Request::LEFT, L, {obj(1, -40, 3.5, -2)}).ok);
    check("blind spot (x~0) -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, 0.5, 3.5, 0)}).ok);
    check("dos objetos, uno inseguro -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, 60, 3.5), obj(2, 5, 3.5)}).ok);
    check("dato invalido -> FALSE", !v.can_change_lane(Request::LEFT, L, {obj(1, 60, 3.5, 0, false)}).ok);
    check("objeto en el OTRO carril no afecta", v.can_change_lane(Request::LEFT, L, {obj(1, 5, -3.5)}).ok);
    check("carril invalido -> FALSE", !v.can_change_lane(Request::LEFT, LaneObs{}, {}).ok);
    check("RIGHT usa y negativo", !v.can_change_lane(Request::RIGHT, L, {obj(1, 5, -3.5)}).ok);

    Automaton a(p);
    Perception in; in.frame_id = 1; in.ts_ns = 1'000'000'000; in.lane = lane(0.2);
    auto d = a.step(in, in.ts_ns + 10'000'000);
    check("FSM: dato fresco -> LANE_KEEPING", d.state == State::LANE_KEEPING && d.fresh);
    d = a.step(in, in.ts_ns + 2'000'000'000);
    check("FSM: dato viejo -> DEGRADED, steer 0", d.state == State::DEGRADED && d.steer_rad == 0 && !d.lane_change_ok);
    in.lane = lane(1.2); d = a.step(in, in.ts_ns + 1'000'000);
    check("FSM: desviado -> CORRECTING", d.state == State::CORRECTING && d.departing);
    in.lane = lane(0.4); d = a.step(in, in.ts_ns + 1'000'000);
    check("FSM: histeresis, sigue CORRECTING a 0.4 m", d.state == State::CORRECTING);
    in.lane = lane(0.05); d = a.step(in, in.ts_ns + 1'000'000);
    check("FSM: centrado -> LANE_KEEPING", d.state == State::LANE_KEEPING);

    std::printf("\n%s\n", fails ? "HAY FALLOS" : "TODO OK");
    return fails ? 1 : 0;
}
