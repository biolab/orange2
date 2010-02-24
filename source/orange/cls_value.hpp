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


#ifndef __CLS_VALUE_HPP
#define __CLS_VALUE_HPP

#include "c2py.hpp"
#include "orange.hpp"

#include "values.hpp"
#include "vars.hpp"

extern ORANGE_API TOrangeType PyOrValue_Type;

class ORANGE_API TPyValue {
public:
  PyObject_HEAD
  TValue value;
  PVariable variable;
};

PyObject *Value_FromVariableValueType(PyTypeObject *type, PVariable var, const TValue &val);
#define Value_FromVariableValue(variable, value)  Value_FromVariableValueType((PyTypeObject *)&PyOrValue_Type, variable, value)
#define Value_FromValueType(type, value)          Value_FromVariableValueType(type, PVariable(), value)
#define Value_FromValue(value)                    Value_FromVariableValue(PVariable(), value)

inline PyObject *Value_FromVariable(PVariable variable)
{ return Value_FromVariableValue(variable, variable->DK()); }

inline PyObject *Value_FromVariableType(PyTypeObject *type, PVariable variable)
{ return Value_FromVariableValueType(type, variable, variable->DK()); }


PyObject *Value_FromArguments(PyTypeObject *type, PyObject *args);


#define PyValue_AS_Value(op) (((TPyValue *)(op))->value)
#define PyValue_AS_Variable(op) (((TPyValue *)(op))->variable)


PyObject *convertToPythonNative(const TValue &val);
PyObject *convertToPythonNative(const TValue &val, PVariable var);
PyObject *convertToPythonNative(const TPyValue *value);

bool convertFromPython(PyObject *args, TValue &value, PVariable var);
bool convertFromPython(PyObject *args, TPyValue *&value);

inline bool convertFromPython(PyObject *args, TValue &value)
{ return convertFromPython(args, value, PVariable()); }

class TCharBuffer;

bool Value_pack(const TValue &value, TCharBuffer &buf, PyObject *&otherValues);
bool Value_unpack(TValue &value, TCharBuffer &buf, PyObject *otherValues, int &otherValuesIndex);


#endif
