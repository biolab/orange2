#include "converts.hpp"

#include "root.hpp"
#include "c2py.hpp"

/* Technically, exceptions and warnings belong to entire
   module, not just cls_orange.cpp. On the other hand, they
   need to be in orange_core.

   As orange_core does not (yet) have a general .cpp file,
   I've put this here. */

PyObject *PyExc_OrangeKernel;
PyObject *PyExc_OrangeKernelWarning;
PyObject *PyExc_OrangeAttributeWarning;


PyObject *PyExc_OrangeWarning;
PyObject *PyExc_OrangeCompatibilityWarning;

void raiseWarning(bool exhaustive, const char *s)
{ if (   (!exhaustive || exhaustiveWarnings)
      && (PyErr_Warn(exhaustive ? PyExc_OrangeCompatibilityWarning : PyExc_OrangeKernelWarning, const_cast<char *>(s))))
    throw mlexception(s);
}


extern char excbuf[256]; // defined in errors.cpp

bool raiseWarning(PyObject *warnType, const char *s, ...)
{ 
  va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, s);
  #else
    va_start(vargs);
  #endif

  vsnprintf(excbuf, 512, s, vargs);

  return PyErr_Warn(warnType, const_cast<char *>(excbuf)) >= 0;
}


bool raiseCompatibilityWarning(const char *s, ...)
{ 
  va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, s);
  #else
    va_start(vargs);
  #endif

  vsnprintf(excbuf, 512, s, vargs);

  return PyErr_Warn(PyExc_OrangeCompatibilityWarning, const_cast<char *>(excbuf)) >= 0;
}