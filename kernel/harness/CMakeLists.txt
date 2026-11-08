cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(gmm_harness LANGUAGES ASC CXX)

add_executable(driver driver.asc)
target_include_directories(driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(driver PRIVATE tiling_api register platform m dl)

# dav-2201 is this chip's architecture; see hw.md.
target_compile_options(driver PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
