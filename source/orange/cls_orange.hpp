#ifndef __CLS_ORANGE_HPP
#define __CLS_ORANGE_HPP

#include <typeinfo>

#include "root.hpp"
#include "orange.hpp"

ORANGE_API PyObject *Orange_getattr(TPyOrange *self, PyObject *name);
ORANGE_API PyObject *Orange_getattr1(TPyOrange *self, const char *name);
ORANGE_API PyObject *Orange_getattr1(TPyOrange *self, PyObject *pyname);

ORANGE_API int Orange_setattrLow(TPyOrange *self, PyObject *pyname, PyObject *args, bool warn);
ORANGE_API int Orange_setattr1(TPyOrange *self, char *name, PyObject *args);
ORANGE_API int Orange_setattr1(TPyOrange *self, PyObject *pyname, PyObject *args);

int Orange_setattrDictionary(TPyOrange *self, const char *name, PyObject *args, bool warn);
int Orange_setattrDictionary(TPyOrange *self, PyObject *pyname, PyObject *args, bool warn);

ORANGE_API PyObject *packOrangeDictionary(PyObject *self);
ORANGE_API int unpackOrangeDictionary(PyObject *self, PyObject *dict);

ORANGE_API PyObject *Orange__reduce__(PyObject *self, PyObject *, PyObject *);

ORANGE_API PyObject *objectOnTheFly(PyObject *args, PyTypeObject *objectType);

ORANGE_API PyObject *callbackOutput(PyObject *self, PyObject *args, PyObject *kwds,
                         char *formatname1, char *formatname2 = NULL,
                         PyTypeObject *toBase = (PyTypeObject *)&PyOrOrange_Type);



ORANGE_API PyObject *PyOrange_DictProxy_New(TPyOrange *);
ORANGE_API extern PyTypeObject PyOrange_DictProxy_Type;

class ORANGE_API TPyOrange_DictProxy : public PyDictObject { public: TPyOrange *backlink; };

#endif
