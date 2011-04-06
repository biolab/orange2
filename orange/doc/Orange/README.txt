Building
========

The documentation can be built with Sphinx 1.0 or newer. Download it at
http://sphinx.pocoo.org/. Also, Orange needs to be installed to build
the documentation. To build the documentation, run

    make html

which will create a directory "html" containing the documentation. If 
make is not installed on your machine, run

    sphinx-build -b html rst html

in orange/doc/Orange. The last two parameters are the input and output 
directory.

For the links to code to work, copy the rst/code directory into
html directory.

Structure
=========

The actual documentation is intermixed in orange/doc/Orange/rst 
and scripts in orange/Orange. 

Regression testing and scripts are in orange/Orange/rst/code. Additional
files, such as images, go into orange/Orange/rst/files.
