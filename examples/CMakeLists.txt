add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE autogradpp::autogradpp)

add_executable(and and.cpp)
target_link_libraries(and PRIVATE autogradpp::autogradpp)

add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE autogradpp::autogradpp)

find_package(OpenMP REQUIRED)

add_executable(handwriting handwriting.cpp)
target_link_libraries(handwriting PRIVATE autogradpp::autogradpp OpenMP::OpenMP_CXX)
target_compile_options(handwriting PRIVATE
    -O3
    -g
    -fno-omit-frame-pointer
    -march=native
)

add_executable(cora cora.cpp)
target_link_libraries(cora PRIVATE autogradpp::autogradpp OpenMP::OpenMP_CXX)

