#pragma once

#include <iostream>

namespace axon {

    enum class DeviceType {
        CPU,
        CUDA
    };

    struct Device {
        DeviceType type;
        int index;

        Device(DeviceType t, int i = -1) : type(t), index(i) {}
        
        Device() : type(DeviceType::CPU), index(-1) {}

        bool operator== (const Device& other) const {
            return type == other.type && index == other.index;
        }

        bool operator!= (const Device& other) const {
            return !(*this == other);
        }

        std::string str() const {
            if (type == DeviceType::CPU) {
                return "cpu";
            }
            return "cuda:" + std::to_string(index);
        }
    };

} // namespace axon 
