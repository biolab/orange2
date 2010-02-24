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
