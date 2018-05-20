# We have some multithreaded tests, so let's activate threading support.
include(YACMAThreadingSetup)
include(YACMACompilerLinkerSettings)

list(APPEND RAKAU_CXX_FLAGS_DEBUG ${YACMA_THREADING_CXX_FLAGS})
list(APPEND RAKAU_CXX_FLAGS_RELEASE ${YACMA_THREADING_CXX_FLAGS})

function(ADD_RAKAU_TESTCASE arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} rakau Threads::Threads)
  target_compile_options(${arg1} PRIVATE "$<$<CONFIG:DEBUG>:${RAKAU_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${RAKAU_CXX_FLAGS_RELEASE}>")
  set_property(TARGET ${arg1} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${arg1} PROPERTY CXX_STANDARD_REQUIRED YES)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
  add_test(${arg1} ${arg1})
endfunction()

ADD_RAKAU_TESTCASE(basic)
