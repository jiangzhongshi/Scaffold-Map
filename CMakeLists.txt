cmake_minimum_required(VERSION 3.0)
project(scaffold-map)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(LIBIGL_USE_STATIC_LIBRARY    "Use libigl as static library" OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use menu"   ON)
option(LIBIGL_WITH_PNG              "Use PNG"            ON)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)

option(NOGUI        "Just for Profiling"           OFF)

if(NOGUI)
  list(APPEND LIBIGL_DEFINITIONS "-DNOGUI")
endif()

include(DownloadProject)

download_project(
  PROJ         libigl
  SOURCE_DIR   ${CMAKE_SOURCE_DIR}/external/libigl
  DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/external/.cache/libigl
  QUIET
  GIT_REPOSITORY https://github.com/jiangzhongshi/libigl.git
)

download_project(
  PROJ cli11
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/cli11
  DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/external/.cache/cli11
  URL     https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  URL_MD5 13a8872af1137c8139cb5b77b1e50b39
)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/cli11)

find_package(LIBIGL REQUIRED QUIET)


# Add your project files
add_library(scaffold_map
src/ReWeightedARAP.cpp
src/ReWeightedARAP_solve.cpp
src/ScafData.cpp
src/StateManager.cpp
)

target_link_libraries(scaffold_map PUBLIC igl::triangle igl::core igl::tetgen igl::opengl_glfw igl::png)


FILE(GLOB SRCFILES
    "src/Newton/*.cpp")

add_library(scaf_utils
    src/UI/DeformGUI.cpp
    src/UI/DeformGUI_key_press.cpp
    src/util/triangle_utils.cpp
    src/demo/flow_demo.cpp
    src/demo/parameterization_init.cpp
    src/demo/tight_packing_demo.cpp
    src/util/tet_utils.cpp
    src/util/tetgenio_parser.cpp
    ${SRCFILES})

add_executable(scaf_pack_bin src/main_texture.cpp
  src/UI/TextureGUI.cpp
  src/UI/TextureGUI_extended_menu.cpp
  src/UI/TextureGUI_mouse_control.cpp)

add_executable(scaf_param_bin src/main.cpp)
add_executable(scaf_flow_bin src/main_flow.cpp)

target_link_libraries(scaf_utils PUBLIC igl::png igl::triangle igl::tetgen igl::opengl_glfw igl::opengl_glfw_imgui)
target_link_libraries(scaf_pack_bin
  igl::png scaffold_map scaf_utils CLI11::CLI11)
target_link_libraries(scaf_param_bin
  igl::png scaffold_map scaf_utils CLI11::CLI11)
target_link_libraries(scaf_flow_bin
  igl::png scaffold_map scaf_utils CLI11::CLI11 igl::tetgen)
