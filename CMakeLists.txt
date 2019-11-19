cmake_minimum_required (VERSION 2.8.11)
project (hanabi_learning_environment)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -fPIC")

# automatically add __declspec(dllexport) to all the functions for windows
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory (hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

add_executable (game_example game_example.cc)
target_link_libraries (game_example LINK_PUBLIC hanabi)
