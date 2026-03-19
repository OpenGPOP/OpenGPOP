set(NAME accuracy_tests)

add_executable(${NAME})

add_custom_target(${NAME}.run
        COMMAND "${CMAKE_BINARY_DIR}/${NAME}/${NAME}"
        DEPENDS ${NAME}
        USES_TERMINAL
        COMMENT "Running accuracy tests..."
)

target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Werror
        -Xcompiler -Wall
        # -Xcompiler -Wextra
        >)


target_sources(${NAME} PRIVATE
        main.cu
)

target_link_libraries(${NAME}
        gpu_propagator
        force_model
        progressbar
)

