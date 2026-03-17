cmake_minimum_required(VERSION 3.20)

project(smallsat-cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0 -g")
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(SOFA_DIR "${CMAKE_SOURCE_DIR}/unit_tests/sofa")
if(NOT EXISTS "${SOFA_DIR}")
    set(SOFA_TAR_URL "https://www.iausofa.org/s/sofa_c-20231011tar.gz")
    set(SOFA_TAR_PATH "${CMAKE_BINARY_DIR}/sofa_temp.tar.gz")
    set(SOFA_EXTRACT_DIR "${CMAKE_BINARY_DIR}/sofa_extract")

    message(STATUS "Downloading IAU SOFA library...")
    file(DOWNLOAD "${SOFA_TAR_URL}" "${SOFA_TAR_PATH}" STATUS DL_STATUS)
    
    # Check if download was successful
    list(GET DL_STATUS 0 DL_STATUS_CODE)
    if(NOT DL_STATUS_CODE EQUAL 0)
        list(GET DL_STATUS 1 DL_ERROR_MSG)
        message(FATAL_ERROR "Failed to download SOFA: ${DL_ERROR_MSG}")
    endif()

    message(STATUS "Extracting IAU SOFA library...")
    file(MAKE_DIRECTORY "${SOFA_EXTRACT_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${SOFA_TAR_PATH}" DESTINATION "${SOFA_EXTRACT_DIR}")

    # Move the nested 'sofa/20231011/c' folder directly to our intended SOFA_DIR
    message(STATUS "Placing SOFA contents into ${SOFA_DIR}...")
    file(RENAME "${SOFA_EXTRACT_DIR}/sofa/20231011/c" "${SOFA_DIR}")

    # Cleanup temporary files
    file(REMOVE "${SOFA_TAR_PATH}")
    file(REMOVE_RECURSE "${SOFA_EXTRACT_DIR}")

    # Run make inside sofa/src
    message(STATUS "Building SOFA library via make...")
    execute_process(
        COMMAND make
        WORKING_DIRECTORY "${SOFA_DIR}/src"
        RESULT_VARIABLE MAKE_RESULT
    )
    
    if(NOT MAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build SOFA library (make returned ${MAKE_RESULT}).")
    endif()

    execute_process(
        COMMAND make check
        WORKING_DIRECTORY "${SOFA_DIR}/src"
        RESULT_VARIABLE MAKE_RESULT
    )
    
    if(NOT MAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to test SOFA library (make check returned ${MAKE_RESULT}).")
    endif()
    message(STATUS "SOFA library built and validated successfully.")
endif()

add_library(sofa STATIC IMPORTED)
set_target_properties(sofa PROPERTIES 
    IMPORTED_LOCATION "${SOFA_DIR}/src/libsofa_c.a"
)
target_include_directories(sofa INTERFACE "${SOFA_DIR}/src")

# =====================================================================

set(VENV_DIR "${CMAKE_SOURCE_DIR}/.venv")
if (NOT EXISTS "${VENV_DIR}/bin/activate")
    message(STATUS "Creating Python virtual environment at ${VENV_DIR}")
    execute_process(
            COMMAND python3 -m venv ${VENV_DIR}
            RESULT_VARIABLE VENV_RESULT
    )
    if (NOT VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create Python virtual environment.")
    endif ()

    set(PIP_EXECUTABLE "${VENV_DIR}/bin/pip")
    message(STATUS "Installing Python dependencies from requirements.txt")
    execute_process(
            COMMAND "${PIP_EXECUTABLE}" install --upgrade pip
            COMMAND "${PIP_EXECUTABLE}" install -r "${CMAKE_SOURCE_DIR}/requirements.txt"
            RESULT_VARIABLE PIP_RESULT
    )
    if (NOT PIP_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install Python dependencies.")
    endif ()
endif ()

add_subdirectory(exec)
add_subdirectory(lib)
add_subdirectory(unit_tests)
add_subdirectory(accuracy_tests)