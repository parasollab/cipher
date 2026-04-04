cmake_minimum_required(VERSION 3.5)

project(cipher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(fcl REQUIRED)

add_subdirectory(Multi-Robot-OMPL EXCLUDE_FROM_ALL)
find_package(crocoddyl REQUIRED)
set(crocoddyl_DIR /usr/local/lib/cmake/crocoddyl)
add_subdirectory(db-CBS EXCLUDE_FROM_ALL)

add_executable(geometric_cipher
  src/geometric/cipher.cpp
  src/utils/decomposition.cpp
)
target_include_directories(geometric_cipher PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(geometric_cipher PRIVATE ompl Eigen3::Eigen Boost::headers)

add_executable(geometric_decoupled_rrt
  src/geometric/decoupled_rrt.cpp
)
target_include_directories(geometric_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(geometric_decoupled_rrt PRIVATE ompl Eigen3::Eigen Boost::headers)

add_executable(kinodynamic_decoupled_rrt
  src/kinodynamic/decoupled_rrt.cpp
)
target_include_directories(kinodynamic_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(kinodynamic_decoupled_rrt PRIVATE ompl Eigen3::Eigen Boost::headers)

add_executable(kinodynamic_cipher
  src/kinodynamic/cipher_kino.cpp
)
target_include_directories(kinodynamic_cipher PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
)
target_link_libraries(kinodynamic_cipher PRIVATE
  ompl
  Eigen3::Eigen
  Boost::headers
  idbastar::optimization
  ${FCL_LIBRARIES}
  -L/opt/homebrew/lib
  ipopt
)
