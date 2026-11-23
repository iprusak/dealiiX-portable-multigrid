cmake_minimum_required(VERSION 3.13.4)

# Diagnostic: memory of device vectors after they are destroyed, see program.cc.
#   cmake -S . -B build -DDEAL_II_DIR=<deal.II install> -DCMAKE_BUILD_TYPE=Release

set(TARGET "program")
set(TARGET_SRC ${TARGET}.cc)

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET} CXX)
deal_ii_invoke_autopilot()
