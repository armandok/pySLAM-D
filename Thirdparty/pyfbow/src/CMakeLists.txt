# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.1)
project(pyfbow)

# Uncomment for debugging CMake / linking issues
#------------------------------------------------------------------------------
#set( CMAKE_VERBOSE_MAKEFILE on)

# Project version
#------------------------------------------------------------------------------
set(pyfbow_MAJOR_VERSION 0)
set(pyfbow_MINOR_VERSION 1)
set(pyfbow_PATCH_VERSION 0)
set(pyfbow_VERSION ${pyfbow_MAJOR_VERSION}.${pyfbow_MINOR_VERSION}.${pyfbow_PATCH_VERSION})
set(pyfbow_VERSION_STR ${pyfbow_VERSION})
set(pyfbow_VERSION_ID ${pyfbow_MAJOR_VERSION}${pyfbow_MINOR_VERSION}${pyfbow_PATCH_VERSION})


# Building options
#-------------------------------------------------------------------------------
option(BUILD_PYTHON3 "Build fbow Python3 binding instead of default Python2." ON)
OPTION(BUILD_STATICALLY_LINKED  "Set to OFF to link with opencv and fbow dynamically" OFF)

SET(Boost_DEBUG OFF)

if(BUILD_PYTHON3)
    set(PYTHON_VERSION 3)
else()
    set(PYTHON_VERSION 2)
endif()

message( STATUS "BUILD_PYTHON3 = ${BUILD_PYTHON3}")
message( STATUS "PYTHON_VERSION = ${PYTHON_VERSION}")


# CMake options
#-----------------------------------------------------------------------------------------------------------------------
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall  -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wall -O3")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0042 NEW)
endif (COMMAND cmake_policy)

# Add cmake_modules to module
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# Set the release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX)
    add_definitions("-fPIC")
endif ()


#-----------------------------------------------------------------------------------------------------------------------

#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

IF(NOT UNIX)
  SET(OpenCV_USE_STATIC_LIBS ON)
ENDIF()
find_package(OpenCV REQUIRED)

SET(fbow_USE_STATIC_LIBS ON)
find_package(fbow REQUIRED)

#-----------------------------------------------------------------------------------------------------------------------
# Python binding
#-----------------------------------------------------------------------------------------------------------------------

if (BUILD_PYTHON3)
  FIND_PACKAGE(PythonInterp 3 REQUIRED)
  FIND_PACKAGE(PythonLibs 3 REQUIRED)
else()
  SET(PythonLibs_FIND_VERSION 2.7)
  FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
  FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
endif()

if(PYTHON_VERSION_STRING VERSION_LESS 2.7)
  message(WARNING "Python version less than 2.7: \"${PYTHON_VERSION_STRING}\".")
endif()

FIND_PACKAGE(NumPy REQUIRED)

IF (NOT UNIX)
  SET(Boost_USE_STATIC_LIBS ON)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
ENDIF()
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(BOOST_ALL_NO_LIB ON)

FIND_PACKAGE(Boost 1.54)

if (NOT UNIX)
  FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} log REQUIRED)
else()
  FIND_PACKAGE(Boost 1.54 COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} log REQUIRED)
endif()


if (UNIX)
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  ${fbow_INCLUDE_DIRS}/fbow
  lib/)
else()
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  ${fbow_INCLUDE_DIRS}/fbow
  lib/)
endif()

if (BUILD_STATICALLY_LINKED)
    ADD_DEFINITIONS(-DBOOST_PYTHON_STATIC_LIB)
else()
    ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
endif ()

# Build np<=>opencv converter library
ADD_LIBRARY(np_opencv_converter lib/conversion.cpp lib/np_opencv_converter.cpp)
TARGET_COMPILE_OPTIONS(np_opencv_converter PRIVATE -DBOOST_ALL_NO_LIB)
if (NOT UNIX)
elseif (APPLE)
  TARGET_LINK_LIBRARIES(np_opencv_converter ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
  set_target_properties(np_opencv_converter PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  TARGET_LINK_LIBRARIES(np_opencv_converter ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif()


# Build python extension
ADD_LIBRARY(pyfbow SHARED fbow_wrapper.cpp)
TARGET_COMPILE_OPTIONS(pyfbow PRIVATE -DBOOST_ALL_NO_LIB)

if (NOT UNIX)
  TARGET_LINK_LIBRARIES(pyfbow np_opencv_converter ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${fbow_LIBS} "-static")
  SET_TARGET_PROPERTIES(pyfbow PROPERTIES PREFIX "" SUFFIX ".pyd")
elseif (APPLE)
  TARGET_LINK_LIBRARIES(pyfbow np_opencv_converter ${fbow_LIBS})
  SET_TARGET_PROPERTIES(pyfbow PROPERTIES PREFIX "" LINK_FLAGS "-undefined dynamic_lookup")
else ()
  TARGET_LINK_LIBRARIES(pyfbow np_opencv_converter ${fbow_LIBS})
  SET_TARGET_PROPERTIES(pyfbow PROPERTIES PREFIX "")
endif ()


# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message("    Built as static libs?:" ${BUILD_STATICALLY_LINKED})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")

message( STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "C++ flags (Relase+Debug):         ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "NUMPY_LIBRARIES = ${NUMPY_LIBRARIES}")
message(STATUS "NUMPY_INCLUDE_DIRS = ${NUMPY_INCLUDES}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
message(STATUS "fbow_INCLUDE_DIRS = ${fbow_INCLUDE_DIRS}")
message(STATUS "fbow_LIB_DIR = ${fbow_LIB_DIR}")
message(STATUS "fbow_LIBS = ${fbow_LIBS}")
message(STATUS "PROJECT LIBRARIES = ${PROJECT_LIBRARIES}")
