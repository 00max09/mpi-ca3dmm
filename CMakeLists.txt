cmake_minimum_required(VERSION 3.5)
project (ca3dmm)

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14  -Wall -O3")
include(FindBLAS)

add_executable(ca3dmm 
        densematgen.cpp
        densematgen.h
        ca3dmm.cpp)

if (BLAS_FOUND)
        target_compile_options(${PROJECT_NAME} PRIVATE ${BLAS_LINKER_FLAGS})
        target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
endif ()
