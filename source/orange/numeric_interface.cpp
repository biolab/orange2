/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "numeric_interface.hpp"
#include "errors.hpp"

bool importarray_called = false;

PyObject *moduleNumeric = NULL, *moduleNumarray = NULL, *moduleNumpy = NULL;
PyObject *numericMaskedArray = NULL, *numarrayMaskedArray = NULL, *numpyMaskedArray = NULL;
PyTypeObject *PyNumericArrayType = NULL, *PyNumarrayArrayType = NULL, *PyNumpyArrayType = NULL;

void initializeNumTypes()
{
  PyObject *ma;
  
  moduleNumeric = PyImport_ImportModule("Numeric");
  if (moduleNumeric) {
    PyNumericArrayType = (PyTypeObject *)PyDict_GetItemString(PyModule_GetDict(moduleNumeric), "ArrayType");
    
    ma = PyImport_ImportModule("MA");
    if (ma)
      numericMaskedArray = PyDict_GetItemString(PyModule_GetDict(ma), "MaskedArray");
  }
  else
    PyErr_Clear();
  
  moduleNumarray = PyImport_ImportModule("numarray");
  if (moduleNumarray) {
    PyNumarrayArrayType = (PyTypeObject *)PyDict_GetItemString(PyModule_GetDict(moduleNumarray), "ArrayType");

    ma = PyImport_ImportModule("numarray.ma");
    if (ma)
      numarrayMaskedArray = PyDict_GetItemString(PyModule_GetDict(ma), "MaskedArray");
  }
  else
    PyErr_Clear();

  moduleNumpy = PyImport_ImportModule("numpy");
  if (moduleNumpy) {
    PyObject *mdict = PyModule_GetDict(moduleNumpy);
    PyNumpyArrayType = (PyTypeObject *)PyDict_GetItemString(mdict, "ndarray");
    
    ma = PyDict_GetItemString(mdict, "ma");
    if (ma)
      numpyMaskedArray = PyDict_GetItemString(PyModule_GetDict(ma), "MaskedArray");
  }
  else
    PyErr_Clear();
    
  importarray_called = true;
//  import_array();
}


// avoids unnecessarily importing the numeric modules
bool isSomeNumeric_wPrecheck(PyObject *args) {
  static char *numericNames[] = {"array", "numpy.ndarray", "ndarray", "numarray.numarraycore.NumArray", "NumArray", 0};
  for(char **nni = numericNames; *nni; nni++)
    if (!strcmp(args->ob_type->tp_name, *nni))
      return isSomeNumeric(args);
  return false;
}


bool isSomeNumeric(PyObject *obj)
{
  if (!importarray_called)
    initializeNumTypes();
    
  return     PyNumericArrayType && PyType_IsSubtype(obj->ob_type, PyNumericArrayType)
          || PyNumarrayArrayType && PyType_IsSubtype(obj->ob_type, PyNumarrayArrayType)
          || PyNumpyArrayType && PyType_IsSubtype(obj->ob_type, PyNumpyArrayType);
}
  
bool isSomeMaskedNumeric_wPrecheck(PyObject *args) {
  static char *numericNames[] = {"numpy.core.ma.MaskedArray", "numarray.ma.MA.MaskedArray", 0};
  for(char **nni = numericNames; *nni; nni++)
    if (!strcmp(args->ob_type->tp_name, *nni))
      return isSomeMaskedNumeric(args);
  return false;
}


bool isSomeMaskedNumeric(PyObject *obj)
{
  if (!importarray_called)
    initializeNumTypes();
    
  return     numarrayMaskedArray && PyType_IsSubtype(obj->ob_type, (PyTypeObject *)numarrayMaskedArray)
          || numpyMaskedArray && PyType_IsSubtype(obj->ob_type, (PyTypeObject *)numpyMaskedArray);
}
  

char getArrayType(PyObject *args)
{
  PyObject *res = PyObject_CallMethod(args, "typecode", NULL);
  if (!res) {
    PyErr_Clear();
    PyObject *ress = PyObject_GetAttrString(args, "dtype");
    if (ress) {
      res = PyObject_GetAttrString(ress, "char");
      Py_DECREF(ress);
    }
  }
  
  if (!res) {
    PyErr_Clear();
    return -1;
  }
  
  char cres = PyString_AsString(res)[0];
  Py_DECREF(res);
  return cres;
}


void numericToDouble(PyObject *args, double *&matrix, int &columns, int &rows)
{
  prepareNumeric();

  if (!isSomeNumeric(args))
    raiseErrorWho("numericToDouble", "invalid type (got '%s', expected 'ArrayType')", args->ob_type->tp_name);

  PyArrayObject *array = (PyArrayObject *)(args);
  if (array->nd != 2)
    raiseErrorWho("numericToDouble", "two-dimensional array expected");

  const char arrayType = getArrayType(array);
  if (!strchr(supportedNumericTypes, arrayType))
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
      case 'c':
      case 'b': READLINE(char)
      case 'B': READLINE(unsigned char)
      case 'h': READLINE(short)
      case 'H': READLINE(unsigned short)
      case 'i': READLINE(int)
      case 'I': READLINE(unsigned int)
      case 'l': READLINE(long)
      case 'L': READLINE(unsigned long)
      case 'f': READLINE(float)
      case 'd': READLINE(double)
    }

    #undef READLINE
  }
}


void numericToDouble(PyObject *args, double *&matrix, int &rows)
{
  prepareNumeric();

  if (!isSomeNumeric(args))
    raiseErrorWho("numericToDouble", "invalid type (got '%s', expected 'ArrayType')", args->ob_type->tp_name);

  PyArrayObject *array = (PyArrayObject *)(args);
  if (array->nd != 1)
    raiseErrorWho("numericToDouble", "one-dimensional array expected");

  const char arrayType = getArrayType(array);
  if (!strchr(supportedNumericTypes, arrayType))
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
      case 'c':
      case 'b': READLINE(char)
      case 'B': READLINE(unsigned char)
      case 'h': READLINE(short)
      case 'H': READLINE(unsigned short)
      case 'i': READLINE(int)
      case 'I': READLINE(unsigned int)
      case 'l': READLINE(long)
      case 'L': READLINE(unsigned long)
      case 'f': READLINE(float)
      case 'd': READLINE(double)
  }

  #undef READLINE
}
