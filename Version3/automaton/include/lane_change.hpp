    #pragma once
    #include <algorithm>
    #include <string>
    #include "types.hpp"

    namespace lda {

    struct LaneChangeResult {
        bool ok = false;
        std::string reason;
    };

    // Port de LaneChangeValidator, ya con posicion delante/detras (la seccion 15 del
    // documento indicaba que la version real debia ser mas precisa en eso).
    class LaneChangeValidator {
    public:
        explicit LaneChangeValidator(const Params& p) : p_(p) {}

        static double ttc(double gap, double closing) {
            return closing > 0.0 ? gap / closing : kInf;
        }

        LaneChangeResult can_change_lane(Request dir, const LaneObs& lane,
                                        const std::vector<ObjectObs>& objects) const {
            if (dir == Request::NONE) return {false, "sin solicitud"};
            // Conservador: sin geometria de carril valida no se puede definir el carril objetivo.
            if (!lane.valid) return {false, "carril invalido"};

            const double side = (dir == Request::LEFT) ? +1.0 : -1.0;
            const double target_y = side * lane.lane_width;
            const double tol = lane.lane_width / 2.0 + p_.target_lane_tol;

            bool any = false;
            for (const auto& o : objects) {
                if (std::fabs(o.y - target_y) > tol) continue;   // no esta en el carril objetivo
                any = true;
                if (!o.valid)
                    return {false, "dato invalido en carril objetivo (id " + std::to_string(o.id) + ")"};

                const bool ahead = o.x >= 0.0;
                const double gap = std::max(0.0, std::fabs(o.x) - (p_.vehicle_length + o.length) / 2.0);
                // velocidad de cierre: delante -> ego mas rapido; detras -> objeto mas rapido
                const double closing = ahead ? -o.vx_rel : o.vx_rel;
                const double t = ttc(gap, closing);

                if (gap < p_.min_safe_distance)
                    return {false, "distancia insegura " + std::to_string(gap) + " m (id " +
                                std::to_string(o.id) + (ahead ? ", delante)" : ", detras/lateral)")};
                if (t < p_.min_safe_ttc)
                    return {false, "TTC inseguro " + std::to_string(t) + " s (id " + std::to_string(o.id) + ")"};
            }
            return {true, any ? "objetos del carril objetivo seguros" : "carril objetivo vacio"};
        }
    private:
        Params p_;
    };

    }  // namespace lda
