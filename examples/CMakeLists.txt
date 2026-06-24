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
    -O3          # maximum optimization
    -march=native # use all CPU features available on the build machine (enables AVX2 if supported)
    -ffast-math  # allow floating point optimizations that may violate IEEE strictness
)
