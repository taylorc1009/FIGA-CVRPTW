cmake_minimum_required(VERSION 3.10)

enable_language(C)

project(objective_function)

add_executable(objective_function list.c data.c objective_function.c)