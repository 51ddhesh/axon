#include <gtest/gtest.h>
#include "axon/tensor/tensor.hpp"

using namespace axon;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TensorTest, ConstructorShapeAndStrides) {
    // 2x3x4 tensor
    tensor::Tensor t({2, 3, 4}, core::CPU());
    
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.dim(), 3);
    EXPECT_EQ(t.numel(), 24);
    
    // Check shapes
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 4);
    
    // Check contiguous strides
    EXPECT_EQ(t.stride()[0], 12); // 3 * 4
    EXPECT_EQ(t.stride()[1], 4);  // 4 * 1
    EXPECT_EQ(t.stride()[2], 1);  // 1
    
    EXPECT_TRUE(t.is_contiguous());
}

TEST_F(TensorTest, ConstructorFromStorage) {
    core::Storage storage(10, core::CPU());
    storage.fill(5.0f);
    
    // View as 2x2 starting from offset 2, with strides [2, 1]
    tensor::Tensor view(storage, 2, {2, 2}, {2, 1});
    
    EXPECT_EQ(view.dim(), 2);
    EXPECT_EQ(view.numel(), 4);
    EXPECT_EQ(view.storage_offset(), 2);
    
    // Access data through offset
    EXPECT_FLOAT_EQ(view.data()[0], 5.0f); // this is storage[2]
}

TEST_F(TensorTest, ValueSemantics) {
    tensor::Tensor t1({2, 2}, core::CPU());
    t1.storage().fill(1.0f);
    
    // Shallow copy
    tensor::Tensor t2 = t1;
    
    EXPECT_TRUE(t1.is_same(t2));
    
    // Modifying storage through t2 modifies t1
    t2.data()[0] = 99.0f;
    EXPECT_FLOAT_EQ(t1.data()[0], 99.0f);
}

TEST_F(TensorTest, NullTensor) {
    tensor::Tensor t;
    EXPECT_FALSE(t.is_valid());
    EXPECT_EQ(t.dim(), 0);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.data(), nullptr);
    EXPECT_FALSE(t.is_contiguous());
    
    EXPECT_THROW(t.shape(), std::runtime_error);
    EXPECT_THROW(t.stride(), std::runtime_error);
}

TEST_F(TensorTest, NonContiguousCheck) {
    core::Storage storage(20, core::CPU());
    
    // A transposed 2x3 -> 3x2 tensor
    // Original shape {2, 3}, strides {3, 1}
    // Transposed shape {3, 2}, strides {1, 3}
    tensor::Tensor t(storage, 0, {3, 2}, {1, 3});
    
    EXPECT_FALSE(t.is_contiguous());
}


