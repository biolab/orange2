#ifndef NO_NUMERIC

#include "numeric_interface.hpp"
#include "errors.hpp"

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


void numericToDouble(PyObject *args, double *&matrix, int &columns, int &rows)
{
  prepareNumeric();

  if (PyArray_Check(args))
    raiseErrorWho("numericToDouble", "invalid type (got '%s', expected 'ArrayType')", args->ob_type->tp_name);

  PyArrayObject *array = (PyArrayObject *)(args);
  if (array->nd != 2)
    raiseErrorWho("numericToDouble", "two-dimensional array expected");

  const int arrayType = array->descr->type_num;
  if ((arrayType == PyArray_CFLOAT) || (arrayType == PyArray_CDOUBLE) || (arrayType == PyArray_OBJECT))
    raiseErrorWho("numericToDouble", "ExampleTable cannot use arrays of complex numbers or Python objects", NULL);

  columns = array->dimensions[1];
  rows = array->dimensions[0];
  matrix = new double[columns * rows];
 
  const int &strideRow = array->strides[0];
  const int &strideCol = array->strides[1];
  
  double *matrixi = matrix;
  char *coli, *cole;

  for(char *rowi = array->data, *rowe = array->data + rows*strideRow; rowi != rowe; rowi += strideRow) {
    #define READLINE(TYPE) \
      for(coli = rowi, cole = rowi + columns*strideCol; coli != cole; *matrixi++ = double(*(TYPE *)coli), coli += strideCol); \
      break;

    switch (arrayType) {
      case PyArray_CHAR: READLINE(char)
      case PyArray_UBYTE: READLINE(unsigned char)
      case PyArray_SBYTE: READLINE(signed char)
      case PyArray_SHORT: READLINE(short)
      case PyArray_INT: READLINE(int)
      case PyArray_LONG: READLINE(long)
      case PyArray_FLOAT: READLINE(float)
      case PyArray_DOUBLE: READLINE(double)
    }

    #undef READLINE
  }
}


void numericToDouble(PyObject *args, double *&matrix, int &rows)
{
  prepareNumeric();

  if (PyArray_Check(args))
    raiseErrorWho("numericToDouble", "invalid type (got '%s', expected 'ArrayType')", args->ob_type->tp_name);

  PyArrayObject *array = (PyArrayObject *)(args);
  if (array->nd != 1)
    raiseErrorWho("numericToDouble", "one-dimensional array expected");

  const int arrayType = array->descr->type_num;
  if ((arrayType == PyArray_CFLOAT) || (arrayType == PyArray_CDOUBLE) || (arrayType == PyArray_OBJECT))
    raiseError("numericToDouble", "ExampleTable cannot use arrays of complex numbers or Python objects", NULL);

  rows = array->dimensions[0];
  matrix = new double[rows];
 
  const int &strideRow = array->strides[0];
  
  double *matrixi = matrix;
  char *rowi, *rowe;

  #define READLINE(TYPE) \
    for(rowi = array->data, rowe = array->data + rows*strideRow; rowi != rowe; *matrixi++ = double(*(TYPE *)rowi), rowi += strideRow); \
    break;

  switch (arrayType) {
    case PyArray_CHAR: READLINE(char)
    case PyArray_UBYTE: READLINE(unsigned char)
    case PyArray_SBYTE: READLINE(signed char)
    case PyArray_SHORT: READLINE(short)
    case PyArray_INT: READLINE(int)
    case PyArray_LONG: READLINE(long)
    case PyArray_FLOAT: READLINE(float)
    case PyArray_DOUBLE: READLINE(double)
  }

  #undef READLINE
}
#endif