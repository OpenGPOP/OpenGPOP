set(NAME gpu_propagator)

add_library(${NAME} STATIC)

set_target_properties(${NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

target_sources(${NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_propagator.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/copy_s.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/dev_functions.cu
)

target_include_directories(${NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
        CUDA::cublas
        gpu_startup
        gpu_integrator
        gpu_utils
        gpu_constants
        force_model
)