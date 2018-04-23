# Find the Protobuf package. This includes the libraries and include
# files.
#
# This code defines the following variables:
#
#  PROTOBUF_FOUND          - TRUE if all components are found.
#  PROTOBUF_INCLUDE_DIRS   - Full paths to all include dirs.
#  PROTOBUF_LIBRARIES      - Full paths to all libraries.
#
# Example Usages:
#  find_package(PROTOBUF)
#
# The location of PROTOBUF can be specified using the environment variable
# or cmake parameter PATH_PROTOBUF. If different installations
# for 32-/64-bit versions and release/debug versions are available,
# they can be specified with
#   PATH_PROTOBUF32
#   PATH_PROTOBUF64
#   PATH_PROTOBUF_RELEASE32
#   PATH_PROTOBUF_RELEASE64
#   PATH_PROTOBUF_DEBUG32
#   PATH_PROTOBUF_DEBUG64
# More specific paths are preferred over less specific ones when searching
# for libraries and 64 bit over 32 bit.
#
# Note that the standard FIND_PACKAGE features are supported
# (QUIET, REQUIRED, etc.).

set(PROTOBUF_SUPPRESS_WARNINGS "FLAG_SUPPRESS_WARNINGS")

foreach(BITWIDTH 32 64)
    foreach(BUILDMODE "RELEASE" "DEBUG")
        set(PROTOBUF_HINT_PATHS_${BUILDMODE}${BITWIDTH}
            ${PATH_PROTOBUF_${BUILDMODE}${BITWIDTH}}
            $ENV{PATH_PROTOBUF_${BUILDMODE}${BITWIDTH}}
            ${PATH_PROTOBUF${BITWIDTH}}
            $ENV{PATH_PROTOBUF${BITWIDTH}}
            ${PATH_PROTOBUF}
            $ENV{PATH_PROTOBUF}
        )
    endforeach()
endforeach()

if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
    set(PROTOBUF_HINT_PATHS_RELEASE ${PROTOBUF_HINT_PATHS_RELEASE32})
    set(PROTOBUF_HINT_PATHS_DEBUG ${PROTOBUF_HINT_PATHS_DEBUG32})
elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(PROTOBUF_HINT_PATHS_RELEASE ${PROTOBUF_HINT_PATHS_RELEASE64})
    set(PROTOBUF_HINT_PATHS_DEBUG ${PROTOBUF_HINT_PATHS_DEBUG64})
else()
    message(WARNING "Bitwidth could not be detected, preferring 64-bit version of PROTOBUF")
    set(PROTOBUF_HINT_PATHS_RELEASE
        ${PROTOBUF_HINT_PATHS_RELEASE64}
        ${PROTOBUF_HINT_PATHS_RELEASE32}
    )
    set(PROTOBUF_HINT_PATHS_DEBUG
        ${PROTOBUF_HINT_PATHS_DEBUG64}
        ${PROTOBUF_HINT_PATHS_DEBUG32}
    )
endif()


find_path(PROTOBUF_INCLUDE_DIRS
    NAMES google
    HINTS ${PROTOBUF_HINT_PATHS_RELEASE} ${PROTOBUF_HINT_PATHS_DEBUG}
    PATH_SUFFIXES include
)

find_library(PROTOBUF_PROTOBUF_LIBRARY_RELEASE
    NAMES protobuf
    HINTS ${PROTOBUF_HINT_PATHS_RELEASE}
    PATH_SUFFIXES lib
)

find_library(PROTOBUF_PROTOC_LIBRARY_RELEASE
    NAMES protoc
    HINTS ${PROTOBUF_HINT_PATHS_RELEASE}
    PATH_SUFFIXES lib
)

find_library(PROTOBUF_PROTOBUF_LIBRARY_DEBUG
    NAMES protobuf
    HINTS ${PROTOBUF_HINT_PATHS_DEBUG}
    PATH_SUFFIXES lib
)

find_library(PROTOBUF_PROTOC_LIBRARY_DEBUG
    NAMES protoc
    HINTS ${PROTOBUF_HINT_PATHS_DEBUG}
    PATH_SUFFIXES lib
)


set(PROTOBUF_LIBRARIES
    optimized ${PROTOBUF_PROTOBUF_LIBRARY_RELEASE}
    optimized ${PROTOBUF_PROTOC_LIBRARY_RELEASE}
    debug ${PROTOBUF_PROTOBUF_LIBRARY_DEBUG}
    debug ${PROTOBUF_PROTOC_LIBRARY_DEBUG})


# Check for consistency and handle arguments like QUIET, REQUIRED, etc.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    PROTOBUF
    REQUIRED_VARS PROTOBUF_INCLUDE_DIRS PROTOBUF_LIBRARIES
)

# Do not show internal variables in cmake GUIs like ccmake.
mark_as_advanced(PROTOBUF_INCLUDE_DIRS
                 PROTOBUF_LIBRARY_RELEASE PROTOBUF_LIBRARY_DEBUG
                 PROTOBUF_LIBRARIES
                 PROTOBUF_HINT_PATHS_RELEASE PROTOBUF_HINT_PATHS_DEBUG)


