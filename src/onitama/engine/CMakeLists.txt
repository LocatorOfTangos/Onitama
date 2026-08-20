cmake_minimum_required(VERSION 3.15)
project(onitama_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(onitama_engine
    onitama_engine.cpp
)

target_include_directories(onitama_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(onitama_engine PRIVATE -O3 -march=native -Wall -Wextra)

# Copy the built .so into the package directory so Python picks it up immediately
# without a separate install step.
add_custom_command(TARGET onitama_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onitama_engine>
        ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:onitama_engine>
    COMMENT "Deploying onitama_engine to engine/ package directory"
)
