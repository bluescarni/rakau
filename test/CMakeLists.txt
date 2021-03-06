function(ADD_RAKAU_TESTCASE arg1)
  if(RAKAU_TEST_NSPLIT)
    # Increase by one the _RAKAU_TEST_NUM variable in the parent scope.
    # NOTE: we do it here (rather than at the end of the function) because we may exit the
    # function earlier in the return() below.
    # NOTE: even though we updated the _RAKAU_TEST_NUM variable in the parent scope,
    # the _RAKAU_TEST_NUM in the *current* scope still contains the old value.
    math(EXPR __RAKAU_TEST_NUM "(${_RAKAU_TEST_NUM} + 1) % ${RAKAU_TEST_NSPLIT}")
    set(_RAKAU_TEST_NUM ${__RAKAU_TEST_NUM} PARENT_SCOPE)
  endif()
  if(RAKAU_TEST_NSPLIT AND NOT "${RAKAU_TEST_SPLIT_NUM}" STREQUAL "${_RAKAU_TEST_NUM}")
    return()
  endif()

  add_executable(${arg1} ${arg1}.cpp)
  # NOTE: explicitly link to tbb as we are using
  # it to parallelise some tests.
  target_link_libraries(${arg1} rakau TBB::tbb)
  target_compile_options(${arg1} PRIVATE "$<$<CONFIG:DEBUG>:${RAKAU_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${RAKAU_CXX_FLAGS_RELEASE}>")
  set_target_properties(${arg1} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(${arg1} PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
  set_property(TARGET ${arg1} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${arg1} PROPERTY CXX_STANDARD_REQUIRED YES)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
  add_test(${arg1} ${arg1})
endfunction()

ADD_RAKAU_TESTCASE(accuracy_acc)
ADD_RAKAU_TESTCASE(accuracy_acc_pot)
ADD_RAKAU_TESTCASE(accuracy_pot)
ADD_RAKAU_TESTCASE(auto_box_size)
ADD_RAKAU_TESTCASE(basic)
ADD_RAKAU_TESTCASE(coll)
ADD_RAKAU_TESTCASE(g_constant_acc)
ADD_RAKAU_TESTCASE(g_constant_acc_pot)
ADD_RAKAU_TESTCASE(g_constant_pot)
ADD_RAKAU_TESTCASE(median_error_acc)
ADD_RAKAU_TESTCASE(morton)
ADD_RAKAU_TESTCASE(node_centre)
ADD_RAKAU_TESTCASE(ordering_acc)
ADD_RAKAU_TESTCASE(ordering_acc_pot)
ADD_RAKAU_TESTCASE(ordering_pot)
ADD_RAKAU_TESTCASE(readme_example)
ADD_RAKAU_TESTCASE(softening_acc)
ADD_RAKAU_TESTCASE(softening_acc_pot)
ADD_RAKAU_TESTCASE(softening_pot)
ADD_RAKAU_TESTCASE(update)
ADD_RAKAU_TESTCASE(update_masses)
ADD_RAKAU_TESTCASE(zero_masses)
