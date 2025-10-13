// utils/random_.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_UTILS_RANDOM
#define AXON_UTILS_RANDOM

#include <random>
#include <chrono>

namespace axon_random {
    inline std::mt19937& get_engine() {
        static std::mt19937 engine(
            static_cast<unsigned int> (std::chrono::steady_clock::now().time_since_epoch().count())
        );
        return engine;
    }

    inline double random_double(double min_ = 0.0, double max_ = 1.0) {
        std::uniform_real_distribution<double> dist(min_, max_);
        return dist(get_engine());
    }

    inline double random_int(double min_ = 0, double max_ = 1000) {
        std::uniform_int_distribution<int> dist(min_, max_);
        double random_int = static_cast<double> (dist(get_engine()));
        return random_int;
    }

} // namespace axon_random

#endif // AXON_UTILS_RANDOM
