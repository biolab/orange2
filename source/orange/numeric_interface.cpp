#ifndef NO_NUMERIC

#include "numeric_interface.hpp"

bool importarray_called = false;
PyTypeObject *PyNumericArrayType = NULL;

PyTypeObject *doGetNumericArrayType()
{
  PyObject *numericModule = PyImport_ImportModule("Numeric");
  if (!numericModule)
    return NULL;

  PyNumericArrayType = (PyTypeObject *)PyDict_GetItemString(PyModule_GetDict(numericModule), "ArrayType");
  return PyNumericArrayType;
}

#endif