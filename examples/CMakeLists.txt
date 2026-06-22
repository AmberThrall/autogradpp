add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE autogradpp::autogradpp)

add_executable(and and.cpp)
target_link_libraries(and PRIVATE autogradpp::autogradpp)
