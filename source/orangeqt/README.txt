== How to build the orangeqt library: ==

= Prerequisites =

orangeqt uses CMake and SIP. They are both free software and are included in most Linux distributions. 
They can also be downloaded from their sites: 

CMake: http://www.cmake.org/cmake/resources/software.html
SIP: http://www.riverbankcomputing.co.uk/software/sip/download

orangeqt also needs the Qt libraries and headers (the qt-dev or qt-devel packages on Linux) and PyQt. 

The Qt libraries can be downloaded from Nokia: https://qt.nokia.com/downloads/downloads#qt-lib. 

Download and install the latest version of the libraries that matches your operating system and compiler. Different version of compilers are compatible, but different compilers are not. 


= Compilation =

If you use GNU Make, it's enough to just call 'make' from the orangeqt directory. 

On windows, follow the standard CMake instructions to compile orangeqt:
 1.a) Use the CMake GUI, and load the orangeqt directory from there. 
	The build directory must be a new or empty directory, and is usually called build and placed inside the source directory.
	This method is preferred as it asks you for your compiler settings.

 1.b) From the command line: run 
		mkdir build
		cd build
		cmake -G "NMake Makefiles" ..
	If you use a compiler other than NMake, replace "NMake Makefiles" with the appropriate value. 
        The list of possible choices includes "Visual Studio 10", "Unix Makefiles", "NMake Makefiles" and many others. 
        It is exparing with every new compiler supported by CMake, the complete list is available in the GUI version of CMake. 

 2) This step depends on your selected compiler. If you chose any type of makefiles, run "make" or "nmake" in the build directory. If you used Visual Studio, open the Solution file located in the build directory. 

 There is no need to install the library, because CMake will copy both the library into the parent directory, with other Orange libraries. However, make sure that the generated library (orangeqt.so or orangeqt.pyd) is in Python's path. You can either adjust the path, or copy the library somewhere where Python will find it. 
 
 If any step reports an error, it is mostly likely some of the dependencies listed above are not installed or not found. 
