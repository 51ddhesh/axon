add_executable(axon_test
    core/storage_test.cpp
)

target_link_libraries(axon_test PRIVATE
    axon
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(axon_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(axon_test)