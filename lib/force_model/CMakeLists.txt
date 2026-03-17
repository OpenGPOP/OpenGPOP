set(NAME force_model)

add_library(${NAME} STATIC)

set_target_properties(${NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
        # I needed this to shut up the compiler in time_point_params_kernel.cu
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_sources(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/force_model.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/time_points.cu

        ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/ref_sys_trans.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/dataset_loaders.cu

        ${CMAKE_CURRENT_SOURCE_DIR}/src/acceleration/gravity_source.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/acceleration/gravity_source_device.cu
)

target_include_directories(${NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
        CUDA::cublas
        types
        gpu_utils
)