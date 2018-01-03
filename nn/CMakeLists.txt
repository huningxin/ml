cmake_minimum_required(VERSION 2.8.9)
project(nn)
set(CMAKE_BUILD_TYPE Release)

include_directories(
  common/include
  common/
  external/eigen
  external/gemmlowp/fixedpoint
  external/gemmlowp/public)
 
set(SOURCES
    bind/src/binding.cpp
    common/operations/Activation.cpp
    common/operations/Concatenation.cpp
    common/operations/Conv2D.cpp
    common/operations/DepthwiseConv2D.cpp
    common/operations/FullyConnected.cpp
    common/operations/Normalization.cpp
    common/operations/Pooling.cpp
    common/operations/Reshape.cpp
    common/operations/SimpleMath.cpp
    common/OperationsUtils.cpp
    common/Utils.cpp)

add_executable(nn ${SOURCES})
set_property(TARGET nn PROPERTY CXX_STANDARD 11)
set_target_properties(nn PROPERTIES LINK_FLAGS "-s WASM=1 --memory-init-file 0 --bind")

