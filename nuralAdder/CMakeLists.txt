cmake_minimum_required(VERSION 2.8)

project(neuralAdder)

set(SRC_LIST main.cpp Neuron.cpp Layer.cpp Net.cpp)
add_executable(${PROJECT_NAME} ${SRC_LIST})
