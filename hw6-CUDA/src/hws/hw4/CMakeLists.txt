cmake_minimum_required(VERSION 3.2)

SET(project_LIBS ${LIBPNG_STATIC} ${ZLIB_STATIC})

SET(project_BIN ${PROJECT_NAME})
SET(project_LIB ${PROJECT_NAME}_lib CACHE INTERNAL "Project Library")

SET(project_SRCS
        main.cpp
        Image.cpp
        Scene.cpp
        Shape.cpp
        Sphere.cpp
)

ADD_LIBRARY(${project_LIB} ${project_SRCS})

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

ADD_EXECUTABLE(${project_BIN} main.cpp)

SET_TARGET_PROPERTIES(
        ${project_BIN}
        PROPERTIES
        OUTPUT_NAME ${APPLICATION_NAME}
        CXX_STANDARD 14
        )

SET_TARGET_PROPERTIES(
        ${project_LIB}
        PROPERTIES
        OUTPUT_NAME ${APPLICATION_NAME}
        CXX_STANDARD 14
        )

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/pngzlib/libpng/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib/pngzlib/libpng/
)

TARGET_LINK_LIBRARIES(${project_LIB} ${project_LIBS})
TARGET_LINK_LIBRARIES(${project_BIN} ${project_LIB} ${project_LIBS})

SET_TARGET_PROPERTIES(${project_BIN} PROPERTIES
        OUTPUT_NAME ${project_BIN}
        CLEAN_DIRECT_OUTPUT 1)