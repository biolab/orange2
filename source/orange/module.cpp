#include "converts.hpp"

#include "root.hpp"
#include "c2py.hpp"

/* Technically, exceptions and warnings belong to entire
   module, not just cls_orange.cpp. On the other hand, they
   need to be in orange_core.

   As orange_core does not (yet) have a general .cpp file,
   I've put this here. */

PyObject *PyExc_OrangeKernel;
PyObject *PyExc_OrangeWarning;
PyObject *PyExc_OrangeKernelWarning;
PyObject *PyExc_OrangeAttributeWarning;

void raiseWarning(const char *s)
{ if (PyErr_Warn(PyExc_OrangeKernelWarning, const_cast<char *>(s)))
    throw mlexception(s);
}

