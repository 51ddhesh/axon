#pragma once

#include <memory>
#include <cstring>

namespace axon {
    struct Storage {
        float* data;
        size_t size;
        bool owns_memory;

        explicit Storage(size_t num_elements) 
            : size(num_elements), owns_memory(true) {
                data = new float[num_elements];
        }

        Storage(float* external_ptr, size_t num_elements) 
            : size(num_elements), data(external_ptr), owns_memory(false) {

        }

        ~Storage() {
            if (owns_memory && data) {
                delete[] data;
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator= (const Storage&) = delete;

        float* ptr() {
            return data;
        }

        const float* ptr() const {
            return data;
        }
    };
}
