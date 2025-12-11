#pragma once

#include "device.hpp"
#include "allocator.hpp"
#include <memory>
#include <cstring>

namespace axon {
    struct Storage {
        void* data;
        size_t nbytes;
        Device device;
        Allocator* allocator;
        bool owns_memory;


        Storage(size_t num_bytes, Device dev = Device(DeviceType::CPU)) :
            nbytes(num_bytes), device(dev), owns_memory(true) {

            allocator = get_allocator(dev.type);
            data = allocator -> allocate(nbytes);
        }

        Storage(void* external_ptr, size_t num_bytes, Device dev) :
            data(external_ptr), nbytes(num_bytes), 
            device(dev), allocator(nullptr), 
            owns_memory(false) {}

        ~Storage() {
            if (owns_memory && data && allocator) {
                allocator -> deallocate(data);
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator= (const Storage&) = delete;

        template <typename T> 
        T* ptr() {
            return static_cast<T*>(data);
        }

        template <typename T>
        const T* ptr() const {
            return static_cast<T*>(data);
        }
    };
}
