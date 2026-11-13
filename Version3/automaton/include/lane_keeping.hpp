#pragma once
#include <algorithm>
#include "types.hpp"

namespace lda
{

    // Port de LaneGeometry + LaneKeepingController del notebook.
    struct LaneGeometry
    {
        static bool is_departing(double lateral_error, double lane_width, const Params &p)
        {
            const double half_free = lane_width / 2.0 - p.lane_safety_margin;
            return (std::fabs(lateral_error) + p.vehicle_width / 2.0) > half_free;
        }
    };

    class LaneKeepingController
    {
    public:
        explicit LaneKeepingController(const Params &p) : p_(p) {}

        // delta = -(K_lat*e_y + K_head*e_psi) [+ feedforward de curvatura]
        double compute_steering(double e_y, double e_psi, double curvature = 0.0) const
        {
            double delta = -(p_.k_lateral * e_y + p_.k_heading * e_psi);
            delta += p_.ff_gain * std::atan(p_.wheelbase * curvature);
            return std::clamp(delta, -p_.max_steering, p_.max_steering);
        }

    private:
        Params p_;
    };

}
