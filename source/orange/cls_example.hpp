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

#ifndef __CLS_EXAMPLE_HPP
#define __CLS_EXAMPLE_HPP

#include "orange.hpp"  
#include "examples.hpp"

// Example can be either a wrapped pointer or a wrapped reference.
// If it is a wrapped reference then a container's wrapper should be passed to lock.
class ORANGE_API TPyExample {
public:
  PyObject_HEAD
  PExample example;
  POrange lock;
};


PyObject *Example_FromExample(PyTypeObject *type, PExample example, POrange lock=POrange());

#define Example_FromWrappedExample(example)   Example_FromExample((PyTypeObject *)&PyOrExample_Type, example)
#define Example_FromExampleRef(example, lock) Example_FromExample((PyTypeObject *)&PyOrExample_Type, PExample(example), lock)
#define Example_FromExampleCopyRef(example)   Example_FromExample((PyTypeObject *)&PyOrExample_Type, PExample(mlnew TExample(example)))
#define Example_FromDomain(domain)            Example_FromExample((PyTypeObject *)&PyOrExample_Type, mlnew TExample(domain))

#define PyExample_AS_Example(op) (((TPyExample *)(op))->example)
#define PyExample_AS_ExampleReference(op) (((TPyExample *)(op))->example.getReference())


PyObject *convertToPythonNative(const TExample &, int natvt=1, bool tuples=false, PyObject *forDK = NULL, PyObject *forDC = NULL, PyObject *forSpecial = NULL);
bool convertFromPython(PyObject *args, TExample &, PDomain domain);
bool convertFromPythonExisting(PyObject *lst, TExample &example);

ORANGE_API int cc_Example(PyObject *obj, void *ptr);
ORANGE_API int ccn_Example(PyObject *obj, void *ptr);
ORANGE_API int ptr_Example(PyObject *obj, void *ptr);
ORANGE_API int ptrn_Example(PyObject *obj, void *ptr);

class TCharBuffer;

void Example_pack(const TExample &example, TCharBuffer &buf, PyObject *&otherValues);
void Example_unpack(TExample &example, TCharBuffer &buf, PyObject *&otherValues, int &otherValuesIndex);

#endif
