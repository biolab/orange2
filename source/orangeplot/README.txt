== How to build the orangeplot library and its python bindings: ==

= Prerequisites =

orangeplot uses CMake and SIP. They are both free software and are included in most Linux distributions. 
They can also be downloaded from their sites: 

CMake: http://www.cmake.org/cmake/resources/software.html
SIP: http://www.riverbankcomputing.co.uk/software/sip/download

orangeplot also needs the Qt libraries and headers (the qt-dev or qt-devel packages on Linux) and PyQt. 

= Compilation =

Follow the standard CMake instructions to compile orangeplot:
 1.a) Use the CMake GUI, and load the orangeplot directory from there. 
	The build directory must be a new or empty directory, and is usually called build and placed inside the source directory.
	This method is preferred as it asks you for your compiler settings.
 1.b) From the command line: run 
		mkdir build
		cd build
		cmake -G "MinGW Makefiles" ..
	If you use a compiler other than MinGW, replace "MinGW Makefiles" with the appropriate value. 
 2) Compile the program by calling 'make' in the build directory. 
 There is no need to call "make install", because "make" will copy both the C++ library and its Python bindings into the parent directory, with other Orange libraries. 
 
 Again, if you're using a different compiler or make, you may have to use 'nmake' instead of make, or compile the solution from within VisualStudio. 

If any step reports an error, it is mostly likely some of the dependencies listed above are not installed or not found. 
