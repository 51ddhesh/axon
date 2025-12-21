#pragma once 

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include "device.hpp"

extern "C" {
    using cudaError_t = int;
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaFree(void* devPtr);
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind);
    cudaError_t cudaMemset(void* devPtr, int value, size_t count);
}


namespace axon {

    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void* allocate(size_t nbytes) = 0;
        virtual void deallocate(void* ptr) = 0;
        virtual void set_zero(void* ptr, size_t nbytes) = 0;
    };

    class CPUAllocator : public Allocator {
    public:
        static constexpr size_t ALIGNMENT = 32;
        
        void* allocate(size_t nbytes) override {
            // use std::aligned_alloc 
            // requires size to be a multiple of alignment
            size_t padded_size = (nbytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1); 
            void* ptr = std::aligned_alloc(ALIGNMENT, padded_size);

            if (!ptr) {
                throw std::runtime_error("[ALLOCATOR] Error: CPU out of memory");
            }

            return ptr;
        }

        void deallocate(void* ptr) override {
            std::free(ptr);
        }

        void set_zero(void* ptr, size_t nbytes) override {
            std::memset(ptr, 0, nbytes);
        }
    };

    class CUDAAllocator : public Allocator {
    public:
        void* allocate(size_t nbytes) override {
            void* ptr = nullptr;
            // CUDA allocation is 256 byte aligned by default
            if (cudaMalloc(&ptr, nbytes) != 0) {
                throw std::runtime_error("GPU out of memory");
            }

            return ptr;
        }

        void deallocate(void* ptr) override {
            cudaFree(ptr);
        }

        void set_zero(void* ptr, size_t nbytes) override {
            cudaMemset(ptr, 0, nbytes);
        }
    };

    inline Allocator* get_allocator(DeviceType device) {
        static CPUAllocator cpu_alloc;
        static CUDAAllocator cuda_alloc;

        if (device == DeviceType::CPU) {
            return &cpu_alloc;
        } else if (device == DeviceType::CUDA) {
            return &cuda_alloc;
        } 
        
        throw std::runtime_error("[ALLOCATOR] Error: Unknown allocator");
    }

} // namespace axon
