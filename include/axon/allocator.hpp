#pragma once 

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include "device.hpp"

extern "C" {
    using cudaError_t = int;
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaFree(void* devPtr);
}

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
            // use _mm_malloc or aligned_alloc for AVX optimization
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

    class CUDAAllocator : public Allocator {
    public:
        void* allocate(size_t nbytes) override {
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, nbytes) != 0) {
                throw std::runtime_error("GPU out of memory");
            }

            return ptr;
        }

        void deallocate(void* ptr) override {
            cudaFree(ptr);
        }
    };

    inline Allocator* get_allocator(DeviceType device) {
        static CPUAllocator cpu_alloc;
        static CUDAAllocator cuda_alloc;

        if (device == DeviceType::CPU) {
            return &cpu_alloc;
        } else if (device == DeviceType::CUDA) {
            return &cuda_alloc;
        } else {
            throw std::runtime_error("Unknown allocator");
        }
    }

} // namespace axon
