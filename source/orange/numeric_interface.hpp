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

#ifndef NO_NUMERIC

  #include "Python.h"

extern PyObject *moduleNumeric, *moduleNumarray, *moduleNumpy;
extern PyTypeObject *PyNumericArrayType, *PyNumarrayArrayType, *PyNumpyArrayType;
extern PyObject *numericMaskedArray, *numarrayMaskedArray, *numpyMaskedArray;

#ifdef NUMPY
  #include "../lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#else
  #include "Numeric/arrayobject.h"
#endif

  extern bool importarray_called;

  bool isSomeNumeric(PyObject *);

  // avoids unnecessarily importing the numeric modules
  bool isSomeNumeric_wPrecheck(PyObject *args);


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

#else

  inline void prepareNumeric()
  { raiseErrorWho("import_array()", "this build does not support Numeric"); }

  bool isSomeNumeric(PyObject *);
  { raiseErrorWho("import_array()", "this build does not support Numeric"); }

  bool isSomeNumeric_wPrecheck(PyObject *);
  { raiseErrorWho("import_array()", "this build does not support Numeric"); }

  inline void numericToDouble(PyObject *num, double *&table, int &columns, int &rows)
  { raiseErrorWho("numericToDouble()", "this build does not support Numeric"); }

#endif // NO_NUMERIC
#endif // __NUMERIC_INTERFACE
