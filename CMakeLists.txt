cmake_minimum_required(VERSION 2.6)
project(scaffold-map)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBIGL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libigl/include)
set(LIBIGL_INCLUDE_DIRS ${LIBIGL_INCLUDE_DIR}  ${LIBIGL_INCLUDE_DIR}/../external/Singular_Value_Decomposition)

# Compilation flags: adapt to your needs
if(MSVC)
  # Enable parallel compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj") 
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR} )
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} )
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "." )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# libigl options: choose between header only and compiled static library
# Header-only is preferred for small projects. For larger projects the static build
# considerably reduces the compilation times
option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)

# add a customizable menu bar
option(LIBIGL_WITH_NANOGUI     "Use Nanogui menu"   ON)

# libigl options: choose your dependencies (by default everything is OFF except opengl) 
option(LIBIGL_WITH_VIEWER      "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL      "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW"           ON)
option(LIBIGL_WITH_BBW         "Use BBW"            ON)
option(LIBIGL_WITH_EMBREE      "Use Embree"         OFF)
option(LIBIGL_WITH_PNG         "Use PNG"            ON)
option(LIBIGL_WITH_TETGEN      "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE    "Use Triangle"       ON)
option(LIBIGL_WITH_XML         "Use XML"            OFF)
option(LIBIGL_WITH_LIM         "Use LIM"            OFF)
option(LIBIGL_WITH_COMISO      "Use CoMiso"         OFF)
option(LIBIGL_WITH_MATLAB      "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK       "Use MOSEK"          OFF)
option(LIBIGL_WITH_CGAL        "Use CGAL"           OFF)
if(LIBIGL_WITH_CGAL) # Do not remove or move this block, the cgal build system fails without it
  find_package(CGAL REQUIRED)
  set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "CGAL's CMAKE Setup is super annoying ")
  include(${CGAL_USE_FILE})
endif()

option(NOGUI        "Just for Profiling"           ON)

# Adding libigl: choose the path to your local copy libigl
# This is going to compile everything you requested 
add_subdirectory("${LIBIGL_INCLUDE_DIR}/../shared/cmake" "libigl")

# libigl information 
message("libigl includes: ${LIBIGL_INCLUDE_DIRS}")
message("libigl libraries: ${LIBIGL_LIBRARIES}")
message("libigl extra sources: ${LIBIGL_EXTRA_SOURCES}")
message("libigl extra libraries: ${LIBIGL_EXTRA_LIBRARIES}")
message("libigl definitions: ${LIBIGL_DEFINITIONS}")

# Prepare the build environment
include_directories(${LIBIGL_INCLUDE_DIRS})
add_definitions(${LIBIGL_DEFINITIONS})

# Add your project files
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

FILE(GLOB SRCFILES
    "src/*.cpp"
    "src/UI/*.cpp"
    "src/util/*.cpp"
    "src/demo/*.cpp"
    "src/Newton/*.cpp"
    "src/igl_dev/*.cpp")

add_executable(${PROJECT_NAME}_bin ${SRCFILES} ${LIBIGL_EXTRA_SOURCES})
target_link_libraries(${PROJECT_NAME}_bin ${LIBIGL_LIBRARIES} ${LIBIGL_EXTRA_LIBRARIES})

