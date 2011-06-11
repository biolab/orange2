project(orangegraph)

cmake_minimum_required(VERSION 2.6)
find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
find_package(PythonInterp REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/share/apps/cmake/modules)
find_package(PyQt4 REQUIRED)
find_package(SIP REQUIRED)
find_package(PythonLibrary REQUIRED)
include(SIPMacros)

include_directories(${QT_INCLUDES} ${SIP_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

#### orangegraph library ####
set(orangegraph_SRCS curve.cpp owgraph.cpp)
qt4_automoc(${orangegraph_SRCS})
add_library(orangegraph SHARED ${orangegraph_SRCS})
set_target_properties(orangegraph PROPERTIES PREFIX "")
target_link_libraries(orangegraph ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
install(TARGETS orangegraph DESTINATION lib)

#### Python bindings ####
set(SIP_INCLUDES ${PYQT4_SIP_DIR})
set(SIP_TAGS Qt_4_7_2 WS_X11)
add_sip_python_module(OrangeGraph OrangeGraph.sip orangegraph)


#### copying both libraries after building ####
get_target_property(CPP_LIB_LOCATION orangegraph LOCATION)
add_custom_command(
   TARGET orangegraph
   POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E copy ${CPP_LIB_LOCATION} ${PROJECT_SOURCE_DIR}/..
 )

get_target_property(PYTHON_LIB_LOCATION python_module_OrangeGraph LOCATION)
add_custom_command(
   TARGET python_module_OrangeGraph
   POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_LIB_LOCATION} ${PROJECT_SOURCE_DIR}/..
 )

