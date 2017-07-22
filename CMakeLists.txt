cmake_minimum_required(VERSION 2.6)

set(PROJECT_NAME CudaKinect)
project(${PROJECT_NAME})

set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)
set(${PROJECT_NAME}_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${${PROJECT_NAME}_VERSION_PATCH}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# ================================================================================
# Project
# ================================================================================

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif(NOT CMAKE_BUILD_TYPE)

# Windows' math include does not define constants by default.
# Set this definition so it does.
# Also set NOMINMAX so the min and max functions are not overwritten with macros.
IF(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
ENDIF()

# ================================================================================
# GL
# ================================================================================
find_package(OpenGL REQUIRED SYSTEM)
if (OpenGL_FOUND)
    message(STATUS "OpenGL found")
	include_directories(${OPENGL_INCLUDE_DIR})
else()
	message(ERROR " OpenGL not found!")
endif()

# ================================================================================
# GLUT
# ================================================================================
find_package(FREEGLUT REQUIRED SYSTEM)
if (FREEGLUT_FOUND)
    message(STATUS "FreeGlut found " ${FREEGLUT_LIBRARIES})
	include_directories(${FREEGLUT_INCLUDE_DIR})
else()
	message(ERROR " FreeGlut not found!")
endif()
	
# ================================================================================
# GLEW
# ================================================================================
find_package(GLEW REQUIRED SYSTEM)
if (GLEW_FOUND)
    message(STATUS "Glew found " ${GLEW_LIBRARIES})
	include_directories(${GLEW_INCLUDE_DIR})
else()
	message(ERROR " GLEW not found!")
endif()
	
# ================================================================================
# KINECT
# ================================================================================
find_package(KINECT REQUIRED SYSTEM)
if (KINECT_FOUND)
    message(STATUS "Kinect found " ${KINECT_LIBRARIES})
	include_directories(${KINECT_INCLUDE_DIR})
else()
	message(ERROR " KINECT not found!")
endif()
	
# ================================================================================
# CUDA
# ================================================================================
find_package(CUDA REQUIRED SYSTEM)
if (CUDA_FOUND)
list(APPEND FIND_PACKAGES_DEFINES USE_CUDA)
message(STATUS "CUDA found and selected for build")
include_directories(${CUDA_TOOLKIT_INCLUDE})
else()
message(ERROR " CUDA not found. Project will not be built with that technology")
endif()

set(CUDA_ARCHITECTURES "sm_30")
if (WIN32)
set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS")
list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCHITECTURES}")
endif()
if (UNIX)
if (APPLE)
  set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
  list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCHITECTURES}")
else()
  set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -std=c++0x ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCHITECTURES}")
endif()
endif()

# ================================================================================
# Application
# ================================================================================
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_FORCE_INLINES --maxrregcount=32)
CUDA_COMPILE(RAY_O Ray.cu)
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp ${RAY_O} Ray.cu)
TARGET_LINK_LIBRARIES(
	${PROJECT_NAME} 
	${CUDA_LIBRARIES}
	${FREEGLUT_LIBRARIES} 
	${GLEW_LIBRARIES}
	${KINECT_LIBRARIES}
	)
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION bin)
