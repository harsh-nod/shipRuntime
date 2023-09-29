cmake_minimum_required(VERSION 3.20...3.27)
project(shipRuntime VERSION 1.0 LANGUAGES CXX)

add_executable(shipRuntime hip_benchmark.cc)
target_link_libraries(shipRuntime PRIVATE)

