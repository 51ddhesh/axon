#pragma once

namespace axon {
    class GradMode {
    public:
        static bool enabled;
        static bool is_enabled() {
            return enabled;
        }

        static void set_enable(bool b) {
            enabled = b;
        }
    };

    struct NoGradGuard {
        bool prev_state;
        NoGradGuard() {
            prev_state = GradMode::is_enabled();
            GradMode::set_enable(false);
        }

        ~NoGradGuard() {
            GradMode::set_enable(prev_state);
        }
    };
} // namespace axon
