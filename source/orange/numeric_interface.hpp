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
