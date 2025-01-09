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

include(FetchContent)

# FetchContent_Declare(Einsums
#         GIT_REPOSITORY https://github.com/jturney/Einsums.git
#         GIT_TAG new_structure
#         FIND_PACKAGE_ARGS 1)

fetchcontent_declare(
        fmt
        URL https://github.com/fmtlib/fmt/archive/refs/tags/11.0.2.tar.gz
        URL_HASH SHA256=6cb1e6d37bdcb756dbbe59be438790db409cdb4868c66e888d5df9f13f7c027f
        FIND_PACKAGE_ARGS 11
)
        

FetchContent_MakeAvailable(fmt)

find_package(psi4 REQUIRED)
find_package(Einsums REQUIRED)

if(EINSUMS_WITH_GPU_SUPPORT)

add_psi4_plugin(einhf plugin.cc rhf.cc rhf-gpu.hip uhf.cc uhf-gpu.hip rmp2.cc ump2.cc rmp2-gpu.hip ump2-gpu.hip)
target_compile_definitions(einhf PUBLIC -DBUILD_GPU)
else()
add_psi4_plugin(einhf plugin.cc rhf.cc uhf.cc rmp2.cc ump2.cc)
endif()

target_compile_options(einhf BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-O0 -gdwarf-4 -ggdb -g3>>)
target_compile_options(einhf BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:HIP,Clang>:-O0 -gdwarf-4 -g3 -ggdb>>)
target_compile_options(einhf BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Og -g3>>)

target_link_libraries(einhf PUBLIC Einsums::Einsums)
target_include_directories(einhf PUBLIC ${Einsums_INCLUDE_DIRS})

get_property(fmt_INCLUDE_DIRS TARGET fmt::fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

message("fmt_INCLUDE_DIRS=${fmt_INCLUDE_DIRS}")

target_include_directories(einhf BEFORE PUBLIC ${fmt_INCLUDE_DIRS})
