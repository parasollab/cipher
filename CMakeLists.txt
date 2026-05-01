cmake_minimum_required(VERSION 3.5)

project(cipher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework program_options)
find_package(fcl REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YAML REQUIRED yaml-cpp)

set(crocoddyl_DIR /usr/local/lib/cmake/crocoddyl)
find_package(crocoddyl REQUIRED)

add_subdirectory(Multi-Robot-OMPL EXCLUDE_FROM_ALL)
add_subdirectory(db-CBS EXCLUDE_FROM_ALL)
set_target_properties(main_ompl main_kcbs db_cbs PROPERTIES EXCLUDE_FROM_ALL FALSE)

include_directories(
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/include
  ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/dynobench/include
)

add_executable(geometric_cipher
  src/geometric/cipher.cpp
  src/utils/grid_decomposition.cpp
  src/mapf/cbs.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(geometric_cipher PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(geometric_cipher PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(geometric_cipher PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(geometric_decoupled_rrt
  src/geometric/decoupled_rrt.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(geometric_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(geometric_decoupled_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(geometric_decoupled_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(geometric_coupled_rrt
  src/geometric/coupled_rrt.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(geometric_coupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(geometric_coupled_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(geometric_coupled_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(srrt
  src/geometric/srrt.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(srrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(srrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(srrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(drrt
  src/geometric/drrt.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(drrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(drrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(drrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(arc
  src/geometric/arc.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(arc PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(arc PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(arc PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(kinodynamic_decoupled_rrt
  src/kinodynamic/decoupled_rrt.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(kinodynamic_decoupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(kinodynamic_decoupled_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(kinodynamic_decoupled_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::headers
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(kinodynamic_coupled_rrt
  src/kinodynamic/coupled_rrt.cpp
  db-CBS/src/robots.cpp
)

add_executable(k_arc
  src/kinodynamic/k_arc.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(kinodynamic_coupled_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(kinodynamic_coupled_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(kinodynamic_coupled_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

target_include_directories(k_arc PRIVATE
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(k_arc PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(k_arc PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
  crocoddyl::crocoddyl
)

enable_testing()

add_executable(test_grid_decomposition
  test/test_grid_decomposition.cpp
  src/utils/grid_decomposition.cpp
)
target_include_directories(test_grid_decomposition PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(test_grid_decomposition PRIVATE
  ompl
  Eigen3::Eigen
  Boost::unit_test_framework
)
add_test(NAME GridDecompositionTests COMMAND test_grid_decomposition)

add_executable(test_cbs
  test/test_cbs.cpp
  src/mapf/cbs.cpp
  src/utils/grid_decomposition.cpp
)
target_include_directories(test_cbs PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
)
target_link_libraries(test_cbs PRIVATE
  ompl
  Eigen3::Eigen
  Boost::unit_test_framework
)
add_test(NAME CBSTests COMMAND test_cbs)

# add_executable(kinodynamic_db_cbs
#   src/kinodynamic/db_cbs.cpp
#   db-CBS/src/robots.cpp
# )
# target_include_directories(kinodynamic_db_cbs PRIVATE
#   ${CMAKE_SOURCE_DIR}/db-CBS/src
#   ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/include
#   ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/dynobench/include
#   ${YAML_INCLUDE_DIRS}
#   ${FCL_INCLUDE_DIRS}
# )
# target_link_directories(kinodynamic_db_cbs PRIVATE ${YAML_LIBRARY_DIRS})
# target_link_libraries(kinodynamic_db_cbs PRIVATE
#   ompl
#   Eigen3::Eigen
#   Boost::program_options
#   ${YAML_LIBRARIES}
#   ${FCL_LIBRARIES}
#   idbastar::optimization
#   -L/opt/homebrew/lib
#   ipopt
# )

set(guided_dbrrt_SOURCES
  src/guided/guided_dbrrt.cpp
  db-CBS/src/robots.cpp
)

set(guided_dbrrt_INCLUDES
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/include
  ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/dynobench/include
  ${CMAKE_SOURCE_DIR}/db-CBS/dynoplan/deps/nigh/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)

set(guided_dbrrt_LIBS
  ompl
  Eigen3::Eigen
  Boost::program_options
  Boost::serialization
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
  idbastar::dbrrt
  idbastar::dbastar
  idbastar::optimization
  -L/opt/homebrew/lib
  ipopt
)

add_executable(guided_dbrrt
  ${guided_dbrrt_SOURCES}
  src/guided/main_guided_dbrrt.cpp
  src/mapf/cbs.cpp
  src/utils/grid_decomposition.cpp
)
target_include_directories(guided_dbrrt PRIVATE ${guided_dbrrt_INCLUDES})
target_link_directories(guided_dbrrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(guided_dbrrt PRIVATE ${guided_dbrrt_LIBS})

add_executable(main_guided_dbrrt
  ${guided_dbrrt_SOURCES}
  src/guided/main_guided_dbrrt.cpp
  src/mapf/cbs.cpp
  src/utils/grid_decomposition.cpp
)
target_include_directories(main_guided_dbrrt PRIVATE ${guided_dbrrt_INCLUDES})
target_link_directories(main_guided_dbrrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(main_guided_dbrrt PRIVATE ${guided_dbrrt_LIBS})

add_executable(guided_geometric_rrt
  src/guided/main_guided_geometric_rrt.cpp
  src/mapf/cbs.cpp
  src/utils/grid_decomposition.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(guided_geometric_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/guided
  ${CMAKE_SOURCE_DIR}/src/mapf
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(guided_geometric_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(guided_geometric_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

add_executable(guided_control_rrt
  src/guided/main_guided_control_rrt.cpp
  src/mapf/cbs.cpp
  src/utils/grid_decomposition.cpp
  db-CBS/src/robots.cpp
)
target_include_directories(guided_control_rrt PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/guided
  ${CMAKE_SOURCE_DIR}/src/mapf
  ${CMAKE_SOURCE_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_BINARY_DIR}/Multi-Robot-OMPL/src
  ${CMAKE_SOURCE_DIR}/db-CBS/src
  ${YAML_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
target_link_directories(guided_control_rrt PRIVATE ${YAML_LIBRARY_DIRS})
target_link_libraries(guided_control_rrt PRIVATE
  ompl
  Eigen3::Eigen
  Boost::program_options
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
)

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
