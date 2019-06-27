cmake_minimum_required(VERSION 3.4.3)
project(MyTest)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(FindComputeCpp)

include_directories(/usr/local/computecpp/include)

set(CMAKE_CXX_STANDARD 17)

message(${CMAKE_CURRENT_BINARY_DIR})
add_executable(MyTest samples/simple_add.cpp)
add_sycl_to_target(TARGET MyTest SOURCES samples/simple_add.cpp)


install(TARGETS MyTest RUNTIME DESTINATION bin)
