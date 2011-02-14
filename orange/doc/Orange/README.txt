Building
========

The documentation can be built with Sphinx 1.0 or newer. Download
it at http://sphinx.pocoo.org/. To build the documentation, run

    make html

which will create a directory "html" containing the documentation. If 
make is not installed on your machine, run

    sphinx-build -b html rst html

in orange/doc/Orange. The last two parameters are the input and output 
directory.
