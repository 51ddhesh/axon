#include <gtest/gtest.h>
#include "axon/core/core.hpp"

class StorageTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(StorageTest, CreateStorage) {
    axon::core::Storage s(10, axon::core::CPU());
    EXPECT_EQ(s.size(), 10);
    EXPECT_TRUE(s.is_valid());
    EXPECT_NE(s.data(), nullptr);
}

TEST_F(StorageTest, ZeroStorage) {
    axon::core::Storage s(5, axon::core::CPU());
    s.data()[0] = 1.0f;
    s.data()[4] = 2.0f;

    s.zero();

    EXPECT_FLOAT_EQ(s.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(s.data()[4], 0.0f);
}

TEST_F(StorageTest, FillStorage) {
    axon::core::Storage s(3, axon::core::CPU());

    s.fill(3.14f);

    EXPECT_FLOAT_EQ(s.data()[0], 3.14f);
    EXPECT_FLOAT_EQ(s.data()[1], 3.14f);
    EXPECT_FLOAT_EQ(s.data()[2], 3.14f);
}

TEST_F(StorageTest, CopySharesData) {
    axon::core::Storage s1(4, axon::core::CPU());
    s1.fill(1.0f);

    // Before copy - should be unique
    EXPECT_TRUE(s1.is_unique());

    axon::core::Storage s2 = s1;

    // After copy - should share (not unique)
    EXPECT_FALSE(s1.is_unique());
    EXPECT_FALSE(s2.is_unique());
}

TEST_F(StorageTest, MoveStorage) {
    axon::core::Storage s1(4, axon::core::CPU());
    s1.fill(2.0f);
    float* data_ptr = s1.data();

    axon::core::Storage s2 = std::move(s1);

    // s1 should be empty after move
    EXPECT_FALSE(s1.is_valid());
    EXPECT_TRUE(s2.is_valid());
    EXPECT_EQ(s2.data(), data_ptr);
}

TEST_F(StorageTest, CopyOnWriteData) {
    axon::core::Storage s1(4, axon::core::CPU());
    s1.fill(1.0f);

    // Copy creates shared storage
    axon::core::Storage s2 = s1;

    // Modify through s2 - triggers copy-on-write
    s2.data()[0] = 99.0f;
    
    // Data should be properly isolated
    EXPECT_FLOAT_EQ(s1.data()[0], 1.0f);  // s1 unchanged
    EXPECT_FLOAT_EQ(s2.data()[0], 99.0f); // s2 modified
}

TEST_F(StorageTest, CopyOnWriteZero) {
    axon::core::Storage s1(4, axon::core::CPU());
    s1.fill(5.0f);

    axon::core::Storage s2 = s1;
    s2.zero();

    EXPECT_FLOAT_EQ(s1.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(s2.data()[0], 0.0f);
}

TEST_F(StorageTest, CopyOnWriteFill) {
    axon::core::Storage s1(3, axon::core::CPU());
    s1.fill(1.0f);

    axon::core::Storage s2 = s1;
    s2.fill(9.0f);

    EXPECT_FLOAT_EQ(s1.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(s2.data()[0], 9.0f);
}

TEST_F(StorageTest, ArenaAllocate) {
    axon::core::Arena arena(1024);

    EXPECT_TRUE(arena.is_valid());
    EXPECT_EQ(arena.size(), 1024);
    EXPECT_EQ(arena.used(), 0);
    EXPECT_EQ(arena.remaining(), 1024);
}

TEST_F(StorageTest, ArenaAllocateAndClear) {
    axon::core::Arena arena(1024);

    void* p1 = arena.allocate(256);
    EXPECT_EQ(arena.used(), 256);

    arena.clear();
    EXPECT_EQ(arena.used(), 0);
    EXPECT_EQ(arena.remaining(), 1024);

    void* p2 = arena.allocate(256);
    // Should reuse the same memory
    EXPECT_EQ(p1, p2);
}

TEST_F(StorageTest, ArenaAllocateMultiple) {
    axon::core::Arena arena(1024);

    void* p1 = arena.allocate(256);
    void* p2 = arena.allocate(512);

    EXPECT_EQ(arena.used(), 768);
    EXPECT_NE(p1, p2);
}

TEST_F(StorageTest, ArenaOutOfMemory) {
    axon::core::Arena arena(256);

    EXPECT_THROW(arena.allocate(512), std::bad_alloc);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}