find_package(Boost 1.58.0 REQUIRED COMPONENTS ${_RAKAU_BOOST_COMPONENTS})

# Might need to recreate targets if they are missing (e.g., older CMake versions).
if(NOT TARGET Boost::boost)
    message(STATUS "The 'Boost::boost' target is missing, creating it.")
    add_library(Boost::boost INTERFACE IMPORTED)
    set_target_properties(Boost::boost PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()
if(NOT TARGET Boost::disable_autolinking)
    message(STATUS "The 'Boost::disable_autolinking' target is missing, creating it.")
    add_library(Boost::disable_autolinking INTERFACE IMPORTED)
    if(WIN32)
        set_target_properties(Boost::disable_autolinking PROPERTIES INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")
    endif()
endif()

foreach(_RAKAU_BOOST_COMPONENT ${_RAKAU_BOOST_COMPONENTS})
    if(NOT TARGET Boost::${_RAKAU_BOOST_COMPONENT})
        message(STATUS "The 'Boost::${_RAKAU_BOOST_COMPONENT}' imported target is missing, creating it.")
        string(TOUPPER ${_RAKAU_BOOST_COMPONENT} _RAKAU_BOOST_UPPER_COMPONENT)
        if(Boost_USE_STATIC_LIBS)
            add_library(Boost::${_RAKAU_BOOST_COMPONENT} STATIC IMPORTED)
        else()
            add_library(Boost::${_RAKAU_BOOST_COMPONENT} UNKNOWN IMPORTED)
        endif()
        set_target_properties(Boost::${_RAKAU_BOOST_COMPONENT} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
        set_target_properties(Boost::${_RAKAU_BOOST_COMPONENT} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${Boost_${_RAKAU_BOOST_UPPER_COMPONENT}_LIBRARY}")
    endif()
endforeach()
