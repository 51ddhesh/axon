// include/axon/ShapeUtils.hpp
// github.com/51ddhesh/axon
// MIT License

#ifndef AXON_SHAPE_UTILS_HPP
#define AXON_SHAPE_UTILS_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace axon {
namespace shape_utils {

    inline std::vector<size_t> broadcast_shapes(const std::vector<size_t>& s1, const std::vector<size_t>& s2) {
        size_t l1 = s1.size();
        size_t l2 = s2.size();
        size_t max_len = std::max(l1, l2);
        
        std::vector<size_t> out_shape(max_len);

        for (size_t i = 0; i < max_len; i++) {
            size_t dim1 = (i < l1) ? s1[l1 - 1 - i] : 1;
            size_t dim2 = (i < l2) ? s2[l2 - 1 - i] : 1;

            if (dim1 == dim2) {
                out_shape[max_len - 1 - i] = dim1;
            } else if (dim1 == 1) {
                out_shape[max_len - 1 - i] = dim2;
            } else if (dim2 == 1) {
                out_shape[max_len - 1 - i] = dim1;
            } else {
                throw std::runtime_error("Shapes mismatch: Incompatible dimensions for broadcasting...");
            }
        }

        return out_shape;
    }
} // namespace shape_utils
} // namespace axon


#endif // AXON_SHAPE_UTILS_HPP

