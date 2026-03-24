set(NAME exec)

add_executable(${NAME})

add_custom_target(${NAME}.run
        COMMAND "${CMAKE_BINARY_DIR}/${NAME}/${NAME}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${NAME}/"
        DEPENDS ${NAME}
        USES_TERMINAL
        COMMENT "Running GPU executable..."
)

add_custom_target(${NAME}.memcheck
        COMMAND compute-sanitizer --leak-check full "${CMAKE_BINARY_DIR}/${NAME}/${NAME}"
        DEPENDS ${NAME}
        USES_TERMINAL
        COMMENT "Running GPU executable with compute sanitizer..."
)

target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Werror
        -Xcompiler -Wall
        -rdynamic -G
        -Xcompiler -Wextra
        >)

target_sources(${NAME} PRIVATE
        main.cpp
        runner.cu
)

target_link_libraries(${NAME}
        force_model
        perturbation_generator
        gpu_startup
        gpu_propagator
        closest_approach
        nlohmann_json::nlohmann_json
        CLI11::CLI11
        indicators::indicators
)
