cmake_minimum_required(VERSION 3.5)

project(cipher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(Multi-Robot-OMPL EXCLUDE_FROM_ALL)

add_executable(geometric_decoupled_rrt
  src/geometric/decoupled_rrt.cpp
)
target_include_directories(geometric_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(geometric_decoupled_rrt PRIVATE ompl)

add_executable(kinodynamic_decoupled_rrt
  src/kinodynamic/decoupled_rrt.cpp
)
target_include_directories(kinodynamic_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(kinodynamic_decoupled_rrt PRIVATE ompl)
