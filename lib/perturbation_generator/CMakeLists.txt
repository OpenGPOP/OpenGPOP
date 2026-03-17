set(NAME perturbation_generator)

add_library(${NAME} STATIC)

# set_target_properties(${NAME} PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   CUDA_RESOLVE_DEVICE_SYMBOLS    ON
# )
# target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

target_sources(${NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/perturbation_generator.cu
)

target_include_directories(${NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
    CUDA::cublas
    CUDA::curand
    gpu_utils
)