cmake_minimum_required(VERSION 3.14)
project(neural_network)

set(CMAKE_CXX_STANDARD 14)

add_executable(neural_network main.cpp
                               Neuron.h
                               Neuron.cpp
                               neuralNetwork.h
                               neuralNetwork.cpp)