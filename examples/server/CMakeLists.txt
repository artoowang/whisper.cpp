set(TARGET server)

option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${TARGET} server.cpp json.hpp httplib.h)

target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_11)