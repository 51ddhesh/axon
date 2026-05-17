#pragma once

#include "device.hpp"
#include "storage_body.hpp"
#include <memory>

namespace axon::core {

class Storage {
private:
    std::shared_ptr<StorageBody> body_;

public:
    Storage() = default;
    Storage(size_t size, Device device);
    explicit Storage(StorageBody* body);

    Storage(const Storage& other);
    Storage(Storage&& other) noexcept;
    Storage& operator=(const Storage& other);
    Storage& operator=(Storage&& other) noexcept;

    // Access
    float* data() { return body_ ? body_->data() : nullptr; }
    const float* data() const { return body_ ? body_->data() : nullptr; }
    size_t size() const { return body_ ? body_->size() : 0; }
    Device device() const { return body_ ? body_->device() : CPU(); }

    bool is_valid() const { return body_ != nullptr; }
    bool is_unique() const { return body_ && body_->ref_count() == 1; }

    // Operations
    void zero();
    void fill(float value);
};

inline Storage::Storage(size_t size, Device device) {
    body_ = std::make_shared<StorageBody>(size, device);
}

inline Storage::Storage(StorageBody* body) : body_(body) {}

inline Storage::Storage(const Storage& other) : body_(other.body_) {
    if (body_) body_->inc_ref();
}

inline Storage::Storage(Storage&& other) noexcept : body_(std::move(other.body_)) {
    other.body_ = nullptr;
}

inline Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        if (body_) body_->dec_ref();
        body_ = other.body_;
        if (body_) body_->inc_ref();
    }
    return *this;
}

inline Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (body_) body_->dec_ref();
        body_ = std::move(other.body_);
        other.body_ = nullptr;
    }
    return *this;
}

inline void Storage::zero() {
    if (body_) body_->zero();
}

inline void Storage::fill(float value) {
    if (body_) body_->fill(value);
}

} // namespace axon::core