#include "axon/tensor.hpp"
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

namespace axon {
    
    const uint32_t MAGIC_NUMBER = 0x41584F4E;
    void save_model(const std::vector<Tensor>& params, const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open the file to save weights, file: " + filepath);
        }

        // write the header
        file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));

        // write the number of tensors
        uint32_t num_tensors = static_cast<uint32_t>(params.size());
        file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

        // write the data
        for (const auto& t : params) {
            uint32_t rank = static_cast<uint32_t>(t.get_shape().size());
            file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

            // Write Shape Dims
            for (int s : t.get_shape()) {
                uint32_t dim = static_cast<uint32_t>(s);
                file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }

            // Write Data
            // We force contiguous before saving to ensure byte-stream is clean
            Tensor t_c = t.is_contiguous() ? t : t.contiguous();
            file.write(reinterpret_cast<const char*>(t_c.data_ptr()), t_c.numel() * sizeof(float));
        }

        file.close();
        std::cout << "[Axon] Saved " << num_tensors << " tensors to " << filepath << std::endl;
    }

    void load_model(std::vector<Tensor>& params, const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for loading: " + filepath);
        }

        // 1. Check Magic Number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != MAGIC_NUMBER) {
            throw std::runtime_error("Invalid file format: Magic number mismatch.");
        }

        // 2. Check Count
        uint32_t num_tensors;
        file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

        if (num_tensors != params.size()) {
            std::cerr << "[Warning] Model has " << params.size() << " params but file contains " << num_tensors << ".\n";
            // We continue, but this usually indicates a mismatch.
        }

        // 3. Read Tensors
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor& t = params[i];

            // Read Rank
            uint32_t rank;
            file.read(reinterpret_cast<char*>(&rank), sizeof(rank));

            // Read Shape
            std::vector<int> file_shape;
            uint32_t dim_val;
            size_t file_numel = 1;

            for (uint32_t r = 0; r < rank; ++r) {
                file.read(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
                file_shape.push_back(static_cast<int>(dim_val));
                file_numel *= dim_val;
            }

            // Verify Shape
            if (file_numel != t.numel()) {
                throw std::runtime_error("Shape mismatch loading parameter " + std::to_string(i));
            }

            // Read Data directly into the Tensor's memory
            // Note: We assume the target tensor is contiguous for loading.
            // If it's a parameter in a module, it usually is.
            if (!t.is_contiguous()) {
                // If this happens, we must load into a buffer and copy element-wise.
                // For simplified parameters, we can assume contiguity or throw.
                throw std::runtime_error("Cannot load into non-contiguous parameter " + std::to_string(i));
            }
            
            file.read(reinterpret_cast<char*>(t.data_ptr()), file_numel * sizeof(float));
        }

        file.close();
        std::cout << "[Axon] Loaded weights from " << filepath << "\n";
    }
} // namespace axon