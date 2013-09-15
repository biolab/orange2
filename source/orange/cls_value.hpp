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
