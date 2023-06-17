cmake_minimum_required(VERSION 3.5)
project (ca3dmm)

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3 -I/net/software/local/openblas/0.3.7/include/ -L/net/software/local/openblas/0.3.7/lib -Wl,-rpath,/net/software/local/openblas/0.3.7/lib -lopenblas")

add_executable(ca3dmm 
        densematgen.cpp
        densematgen.h
        ca3dmm.cpp)

