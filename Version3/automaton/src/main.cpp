#include <zmq.hpp>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include "codec.hpp"
#include "fsm.hpp"

using namespace lda;
static volatile std::sig_atomic_t g_run = 1;
static void on_sig(int) { g_run = 0; }

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string sub_ep = "tcp://127.0.0.1:5555", pub_ep = "tcp://127.0.0.1:5556";
    bool verbose = false;
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "falta valor para " << a << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if      (a == "--sub") sub_ep = next();
        else if (a == "--pub") pub_ep = next();
        else if (a == "--hz") p.hz = std::stod(next());
        else if (a == "--max-age-ms") p.max_age_s = std::stod(next()) * 1e-3;
        else if (a == "--kl") p.k_lateral = std::stod(next());
        else if (a == "--kh") p.k_heading = std::stod(next());
        else if (a == "--ff") p.ff_gain = std::stod(next());
        else if (a == "--min-dist") p.min_safe_distance = std::stod(next());
        else if (a == "--min-ttc") p.min_safe_ttc = std::stod(next());
        else if (a == "--assist-only") p.assist_only = true;
        else if (a == "-v") verbose = true;
        else { std::cerr << "arg desconocido: " << a << "\n"; return 1; }
    }
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::conflate, 1);        // solo el ULTIMO mensaje: nunca se procesa dato atrasado
    sub.set(zmq::sockopt::subscribe, "");
    sub.connect(sub_ep);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.set(zmq::sockopt::conflate, 1);
    pub.bind(pub_ep);                           // ej. tcp://127.0.0.1:5556 o tcp://*:5556

    Automaton fsm(p);
    Perception latest;                           // frame_id = -1 => "sin dato"
    const auto period = std::chrono::milliseconds(static_cast<int>(1000.0 / p.hz));
    std::cerr << "[automata] sub=" << sub_ep << " pub=" << pub_ep << " hz=" << p.hz
              << " max_age=" << p.max_age_s * 1e3 << "ms" << (p.assist_only ? " assist-only" : "") << "\n";

    zmq::pollitem_t items[] = {{static_cast<void*>(sub), 0, ZMQ_POLLIN, 0}};
    State last_state = State::DEGRADED;
    std::string last_reason;
    while (g_run) {
        zmq::poll(items, 1, period);
        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t m;
            if (sub.recv(m, zmq::recv_flags::dontwait)) {
                try { latest = parse_perception(std::string(static_cast<char*>(m.data()), m.size())); }
                catch (const std::exception& e) { std::cerr << "[automata] JSON invalido: " << e.what() << "\n"; }
            }
        }
        const int64_t t = now_ns();
        Decision d = fsm.step(latest, t);
        const std::string out = encode_decision(d, t);
        pub.send(zmq::buffer(out), zmq::send_flags::dontwait);
        if (verbose && (d.state != last_state || d.reason != last_reason)) {
            std::cerr << "[automata] " << to_str(d.state) << " steer=" << d.steer_rad
                      << " ok=" << d.lane_change_ok << " (" << d.reason << ")\n";
            last_state = d.state;
            last_reason = d.reason;
        }
    }
    return 0;
}