set(NAME progressbar)

add_library(${NAME} INTERFACE)

target_include_directories(${NAME}
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${NAME} INTERFACE
)
