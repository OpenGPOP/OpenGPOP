set(NAME gpu_startup)

add_library(${NAME} STATIC)

set_target_properties(${NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

target_sources(${NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_startup.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/initialize_memory.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/reset_temp_vectors.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/copy_prev_state.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/calculate_all_s.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/move_state_to_temp.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/device_functions.cu
)

target_include_directories(${NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
        CUDA::cublas
        CUDA::curand
        gpu_utils
        gpu_constants
        force_model
)