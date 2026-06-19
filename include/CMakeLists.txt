add_library(autogradpp INTERFACE)
add_library(autogradpp::autogradpp ALIAS autogradpp)

target_include_directories(autogradpp INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(autogradpp INTERFACE cxx_std_17)
