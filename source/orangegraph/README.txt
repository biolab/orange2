== How to build the OrangeGraph library and its python bindings: ==

= Prerequisites =

OrangeGraph uses CMake and SIP. They are both free software and are included in most Linux distributions. 
They can also be downloaded from their sites: 

CMake: http://www.cmake.org/cmake/resources/software.html
SIP: http://www.riverbankcomputing.co.uk/software/sip/download

OrangeGraph also needs the Qt libraries and headers (the qt-dev or qt-devel packages on Linux) and PyQt. 

= Compilation =

Follow the standard CMake instructions to compile OrangeGraph:
 - Use the CMake GUI, and load the orangegraph directory from there
 - From the command line: run "mkdir build", "cd build", "cmake .." and "make"

There is no need to call "make install", because "make" will copy both the C++ library and its Python bindings into the parent directory, with other Orange libraries. 

If any step reports an error, it is mostly likely some of the dependencies listed above are not installed or not found. 
