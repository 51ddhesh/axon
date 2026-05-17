#pragma once

namespace axon::core {

enum class DeviceType {
    CPU,
    CUDA
};

struct Device {
    DeviceType type;
    int id;

    Device() : type(DeviceType::CPU), id(0) {}
    Device(DeviceType t, int i = 0) : type(t), id(i) {}

    bool operator==(const Device& other) const {
        return type == other.type && id == other.id;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    bool is_cpu() const { return type == DeviceType::CPU; }
    bool is_cuda() const { return type == DeviceType::CUDA; }
};

inline Device CPU() { return Device(DeviceType::CPU, 0); }
inline Device CUDA(int id = 0) { return Device(DeviceType::CUDA, id); }

} // namespace axon::core