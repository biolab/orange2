/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifndef __NUMERIC_INTERFACE
#define __NUMERIC_INTERFACE

#include "Python.h"

#ifdef _MSC_VER
  /* easier to do some ifdefing here than needing to define a special
     include in every project that includes this header */
  #include "../lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#else
  #include <numpy/arrayobject.h>
#endif

extern PyObject *moduleNumeric, *moduleNumarray, *moduleNumpy;
extern PyTypeObject *PyNumericArrayType, *PyNumarrayArrayType, *PyNumpyArrayType;
extern PyObject *numericMaskedArray, *numarrayMaskedArray, *numpyMaskedArray;


extern bool importarray_called;

bool isSomeNumeric(PyObject *);
bool isSomeMaskedNumeric(PyObject *);

// avoids unnecessarily importing the numeric modules
bool isSomeNumeric_wPrecheck(PyObject *args);
bool isSomeMaskedNumeric_wPrecheck(PyObject *);


void initializeNumTypes();
  
char getArrayType(PyObject *);
  
inline char getArrayType(PyArrayObject *args)
{ return getArrayType((PyObject *)(args)); }
  
inline void prepareNumeric()
{ if (!importarray_called)
    initializeNumTypes();
}

void numericToDouble(PyObject *num, double *&table, int &columns, int &rows);
void numericToDouble(PyObject *num, double *&table, int &rows);

static char supportedNumericTypes[] = "bBhHiIlLfdc";

#endif // __NUMERIC_INTERFACE
