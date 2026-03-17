set(NAME gpu_utils)

add_library(${NAME} INTERFACE)

target_include_directories(${NAME}
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
    INTERFACE
    CUDA::cublas
)