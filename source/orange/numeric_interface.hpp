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
  #include "Numeric/arrayobject.h"

  extern PyTypeObject *PyNumericArrayType;
  extern bool importarray_called;

  PyTypeObject *doGetNumericArrayType();

  inline void prepareNumeric()
  { if (!importarray_called) {
      import_array();
      importarray_called = true;
    }
  }

  inline PyTypeObject *getNumericArrayType()
  { return PyNumericArrayType ? PyNumericArrayType : doGetNumericArrayType(); }

#else

  inline void prepareNumeric()
  { raiseErrorWho("import_array()", "this build does not support Numeric"); }

  inline PyTypeObject *getNumericArrayType()
  { raiseErrorWho("import_array()", "this build does not support Numeric"); }

#endif // NO_NUMERIC
#endif // __NUMERIC_INTERFACE
