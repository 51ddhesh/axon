#pragma once

#include <cstddef>
#include <stdexcept>
#include <cstdlib>

namespace axon::core {

class Arena {
public:
    Arena() = default;
    explicit Arena(size_t bytes);
    ~Arena();

    // Non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Moveable
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    void* allocate(size_t bytes);
    void clear();

    size_t size() const { return capacity_; }
    size_t used() const { return cursor_; }
    size_t remaining() const { return capacity_ - cursor_; }
    bool is_valid() const { return buffer_ != nullptr; }

private:
    static constexpr size_t alignment = 256;

    void* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t cursor_ = 0;

    static size_t align_up(size_t x) {
        return (x + alignment - 1) & ~(alignment - 1);
    }
};

inline Arena::Arena(size_t bytes) : capacity_(bytes), cursor_(0) {
    buffer_ = aligned_alloc(alignment, capacity_);
    if (!buffer_) {
        throw std::bad_alloc();
    }
}

inline Arena::~Arena() {
    if (buffer_) {
        std::free(buffer_);
    }
}

inline Arena::Arena(Arena&& other) noexcept
    : buffer_(other.buffer_), capacity_(other.capacity_), cursor_(other.cursor_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.cursor_ = 0;
}

inline Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            std::free(buffer_);
        }
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        cursor_ = other.cursor_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.cursor_ = 0;
    }
    return *this;
}

inline void* Arena::allocate(size_t bytes) {
    bytes = align_up(bytes);
    if (cursor_ + bytes > capacity_) {
        throw std::bad_alloc();
    }
    void* ptr = static_cast<char*>(buffer_) + cursor_;
    cursor_ += bytes;
    return ptr;
}

inline void Arena::clear() {
    cursor_ = 0;
}

} // namespace axon::core