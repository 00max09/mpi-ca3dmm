cmake_minimum_required(VERSION 3.5)
project (MIMUW-matgen)

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14  -g -Wall -O0")
LINK_LIBRARIES("blas")

add_executable(generator
        densematgen.cpp
        densematgen.h
        generator_main.cpp)

add_executable(generator_all
        densematgen.cpp
        densematgen.h
        generator_all.cpp)

add_executable(ca3dmm 
        densematgen.cpp
        densematgen.h
        ca3dmm.cpp)
