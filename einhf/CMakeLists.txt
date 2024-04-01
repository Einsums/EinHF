#
# @BEGIN LICENSE
#
# einhf by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2023 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(einhf CXX)

find_package(psi4 REQUIRED)
find_package(Einsums REQUIRED)

option(BUILD_GPU "Whether to build the GPU plugin." OFF)

if(BUILD_GPU)

enable_language(HIP)

        if(NOT HIP_INSTALL_DIR)
                message(FATAL_ERROR "Please set HIP_INSTALL_DIR.")
        endif()

        cmake_path(APPEND HIP_INSTALL_DIR "lib" "cmake" OUTPUT_VARIABLE __hip_cmake_dir)

        cmake_path(APPEND __hip_cmake_dir "AMDDeviceLibs" OUTPUT_VARIABLE AMDDeviceLibs_DIR)
        set(ENV{AMDDeviceLibs_DIR} ${AMDDeviceLibs_DIR})

        cmake_path(APPEND __hip_cmake_dir "amd_comgr" OUTPUT_VARIABLE amd_comgr_DIR)
        set(ENV{amd_comgr_DIR} ${amd_comgr_DIR})

        cmake_path(APPEND __hip_cmake_dir "hipblas" OUTPUT_VARIABLE hipblas_DIR)
        set(ENV{hipblas_DIR} ${hipblas_DIR})

        cmake_path(APPEND __hip_cmake_dir "hip" OUTPUT_VARIABLE hip_DIR)
        set(ENV{hip_DIR} ${hipblas_DIR})

        cmake_path(APPEND __hip_cmake_dir "hsa-runtime64" OUTPUT_VARIABLE hsa-runtime64_DIR)
        set(ENV{hsa-runtime64_DIR} ${hsa-runtime64_DIR})

        cmake_path(APPEND __hip_cmake_dir "hipsolver" OUTPUT_VARIABLE hipsolver_DIR)
        set(ENV{hipsolver_DIR} ${hipsolver_DIR})

        enable_language(HIP)
        find_package(hipblas REQUIRED)
        find_package(hipsolver REQUIRED)
        set(CURSES_NEED_NCURSES True)
        find_package(Curses)

add_psi4_plugin(einhf plugin.cc scf.cc scf-gpu.cc scf-uhf.cc)
else()
add_psi4_plugin(einhf plugin.cc scf.cc scf-uhf.cc)
endif()

target_compile_options(einhf BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-gdwarf-2 -O0 -g3 -ggdb>>)
target_compile_options(einhf BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Og -g3 -ggdb>>)

target_link_libraries(einhf PUBLIC Einsums::einsums)
target_include_directories(einhf PUBLIC ${Einsums_INCLUDE_DIRS})
