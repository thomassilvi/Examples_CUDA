cmake_minimum_required (VERSION 2.6)
project (BumpMappingWithCuda)

#Searching CUDA

FIND_PACKAGE(CUDA REQUIRED)
IF(CUDA_FOUND)
        MESSAGE(STATUS "Looking for CUDA : Cuda ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} found")
ELSE()
        MESSAGE(SEND_ERROR "Looking for CUDA : not found!")
        MESSAGE(FATAL_ERROR "Please get it from www.nvidia.com")
ENDIF()

INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )

#Searching SDL and SDL image

FIND_PACKAGE(SDL REQUIRED)
IF(SDL_FOUND)
        MESSAGE(STATUS "Looking for SDL: found")
ELSE()
        MESSAGE(SEND_ERROR "Looking for SDL: not found!")
        MESSAGE(FATAL_ERROR "Please get it from www.libsdl.org")
ENDIF()

INCLUDE_DIRECTORIES( ${SDL_INCLUDE_DIR} )

FIND_PACKAGE(SDL_image REQUIRED)
IF(SDLIMAGE_FOUND)
        MESSAGE(STATUS "Looking for SDL image: found")
ELSE()
        MESSAGE(SEND_ERROR "Looking for SDL image: not found!")
        MESSAGE(FATAL_ERROR "Please get it from www.libsdl.org")
ENDIF()


CUDA_ADD_EXECUTABLE (
        BumpMappingWithCuda
        bump_kernel.cu
        main.cpp
)


target_link_libraries (BumpMappingWithCuda ${CUDA_LIBRARIES} ${SDL_LIBRARY} m ${SDLIMAGE_LIBRARY})


