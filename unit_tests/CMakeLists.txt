set(NAME unit_tests)

add_executable(${NAME})

add_custom_target(${NAME}.run
        COMMAND "${CMAKE_BINARY_DIR}/${NAME}/${NAME}"
        DEPENDS ${NAME}
        USES_TERMINAL
        COMMENT "Running gpu tests..."
)

add_custom_target(${NAME}.memcheck
        COMMAND compute-sanitizer --leak-check full "${CMAKE_BINARY_DIR}/${NAME}/${NAME}"
        DEPENDS ${NAME}
        USES_TERMINAL
        COMMENT "Running gpu tests with compute sanitizer..."
)

target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Werror
        -Xcompiler -Wall
        # -Xcompiler -Wextra
        >)


target_sources(${NAME} PRIVATE
        rk8_test.cu
        fm_gravity_source_test.cu
        integrator_prediction_test.cu
        integrator_test.cu
        startup_test.cu
        startup_correction_test.cu
        fm_ref_sys_trans_test.cu
        fm_time_points_test.cu
        main.cu
)

target_include_directories(${NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${NAME}
        GTest::gtest
        GTest::gmock
        CUDA::cublas
        gpu_startup
        perturbation_generator
        gpu_propagator
        gpu_utils
        sofa
)