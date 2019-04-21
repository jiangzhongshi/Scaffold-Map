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

find_package(LIBIGL REQUIRED QUIET)


# Add your project files
add_library(scaffold_map
src/ReWeightedARAP.cpp
src/ReWeightedARAP_solve.cpp
src/ScafData.cpp
src/StateManager.cpp
)


FILE(GLOB SRCFILES
    "src/UI/*.cpp"
    "src/util/*.cpp"
    "src/demo/*.cpp"
    "src/Newton/*.cpp")

add_library(scaf_utils ${SRCFILES})

add_executable(${PROJECT_NAME}_bin src/main.cpp)
target_link_libraries(scaffold_map PUBLIC igl::triangle igl::core igl::tetgen igl::opengl_glfw igl::png)
target_link_libraries(scaf_utils PUBLIC igl::png igl::triangle igl::tetgen igl::opengl_glfw igl::opengl_glfw_imgui)
target_link_libraries(${PROJECT_NAME}_bin igl::png scaffold_map scaf_utils)

