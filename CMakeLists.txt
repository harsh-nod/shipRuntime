cmake_minimum_required(VERSION 3.20...3.27)
project(shipRuntime VERSION 1.0 LANGUAGES CXX)

add_executable(shipRuntime hip_benchmark.cc)
find_package(hip REQUIRED )
find_package(OpenMP REQUIRED )

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

set(HIP_LIBS hip::host)
set(OMP_LIBS OpenMP::OpenMP_CXX)
add_definitions(-D__HIP_PLATFORM_AMD__)
#add_definitions(-DNDEBUG)
target_link_directories(shipRuntime PUBLIC ${HIP_LIBS} ${OMP_LIBS})
target_link_options(shipRuntime PUBLIC -fopenmp)

