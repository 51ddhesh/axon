#pragma once 

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include "device.hpp"

namespace axon {

    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void* allocate(size_t nbytes) = 0;
        virtual void deallocate(void* ptr) = 0;
    };

    class CPUAllocator : public Allocator {
    public:
        void* allocate(size_t nbytes) override {
            void* ptr = std::malloc(nbytes);
            if (!ptr) {
                throw std::runtime_error("CPU out of memory");
            }

            return ptr;
        }

        void deallocate(void* ptr) override {
            std::free(ptr);
        }
    };

    inline Allocator* get_allocator(DeviceType device) {
        static CPUAllocator cpu_alloc;
        if (device == DeviceType::CPU) {
            return &cpu_alloc;
        }
    }

} // namespace axon
