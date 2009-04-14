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


#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "vars.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "learn.hpp"
#include "estimateprob.hpp"
#include "preprocessors.hpp"
#include "callback.hpp"

#include "cls_value.hpp"
#include "cls_example.hpp"
#include "lib_kernel.hpp"
#include "vectortemplates.hpp"


#include "externs.px"

#include "converts.hpp"
#include "cls_orange.hpp"
#include "slist.hpp"

WRAPPER(ExampleTable);

PStringList PStringList_FromArguments(PyObject *arg);

/* This was moved from lib_vectors.cpp:
    - nobody used it
    - lib_vectors.cpp is automatically generated and I'd hate to add this as an exception

int pt_FloatList(PyObject *args, void *floatlist)
{
  *(PFloatList *)(floatlist) = PFloatList_FromArguments(args);
  return PyErr_Occurred() ? -1 : 0;
}
*/


int pt_StringList(PyObject *args, void *stringList)
{ 
  PStringList &rsl = *(PStringList *)stringList;
  
  if (PyOrStringList_Check(args))
    rsl = PyOrange_AsStringList(args);
  else
    rsl = PStringList_FromArguments(args);
    
  return rsl ? 1 : 0;
}

int ptn_StringList(PyObject *args, void *stringList)
{
  if (args == Py_None) {
    *(PStringList *)stringList = PStringList();
    return 1;
  }
  
  return pt_StringList(args, stringList);
}


/* ************ PROGRESS CALLBACK ************ */

#include "progress.hpp"

PyObject *ProgressCallback_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrProgressCallback_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TProgressCallback_Python(), type), args);
  else
    return WrapNewOrange(mlnew TProgressCallback_Python(), type);
}


PyObject *ProgressCallback__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrProgressCallback_Type);
}


PyObject *ProgressCallback_call(PyObject *self, PyObject *targs, PyObject *keywords) PYDOC("(float[, Orange]) -> bool")
{
  PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrProgressCallback_Type) {
      PyErr_Format(PyExc_SystemError, "ProgressCallback.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    float f;
    POrange o;
    if (!PyArg_ParseTuple(targs, "f|O&:ProgressCallback", &f, ccn_Orange, &o))
      return PYNULL;

    return PyInt_FromLong(SELF_AS(TProgressCallback)(f, o) ? 1 : 0);
  PyCATCH
}

/* ************ VARIABLE ************ */

PVarList PVarList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::P_FromArguments(arg); }
PyObject *VarList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_FromArguments(type, arg); }
PyObject *VarList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Variable>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_new(type, arg, kwds); }
PyObject *VarList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_getitem(self, index); }
int       VarList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_setitem(self, index, item); }
PyObject *VarList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_getslice(self, start, stop); }
int       VarList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_setslice(self, start, stop, item); }
int       VarList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_len(self); }
PyObject *VarList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_richcmp(self, object, op); }
PyObject *VarList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_concat(self, obj); }
PyObject *VarList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_repeat(self, times); }
PyObject *VarList_str(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_str(self); }
PyObject *VarList_repr(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_str(self); }
int       VarList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_contains(self, obj); }
PyObject *VarList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Variable) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_append(self, item); }
PyObject *VarList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_extend(self, obj); }
PyObject *VarList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> int") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_count(self, obj); }
PyObject *VarList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> VarList") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_filter(self, args); }
PyObject *VarList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> int") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_index(self, obj); }
PyObject *VarList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_insert(self, args); }
PyObject *VarList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_native(self); }
PyObject *VarList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Variable") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_pop(self, args); }
PyObject *VarList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_remove(self, obj); }
PyObject *VarList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_reverse(self); }
PyObject *VarList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_sort(self, args); }
PyObject *VarList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, &PyOrVariable_Type>::_reduce(self); }


PVarListList PVarListList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::P_FromArguments(arg); }
PyObject *VarListList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_FromArguments(type, arg); }
PyObject *VarListList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of VarList>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_new(type, arg, kwds); }
PyObject *VarListList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_getitem(self, index); }
int       VarListList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_setitem(self, index, item); }
PyObject *VarListList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_getslice(self, start, stop); }
int       VarListList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_setslice(self, start, stop, item); }
int       VarListList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_len(self); }
PyObject *VarListList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_richcmp(self, object, op); }
PyObject *VarListList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_concat(self, obj); }
PyObject *VarListList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_repeat(self, times); }
PyObject *VarListList_str(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_str(self); }
PyObject *VarListList_repr(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_str(self); }
int       VarListList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_contains(self, obj); }
PyObject *VarListList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(VarList) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_append(self, item); }
PyObject *VarListList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_extend(self, obj); }
PyObject *VarListList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> int") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_count(self, obj); }
PyObject *VarListList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> VarListList") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_filter(self, args); }
PyObject *VarListList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> int") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_index(self, obj); }
PyObject *VarListList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_insert(self, args); }
PyObject *VarListList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_native(self); }
PyObject *VarListList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> VarList") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_pop(self, args); }
PyObject *VarListList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_remove(self, obj); }
PyObject *VarListList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_reverse(self); }
PyObject *VarListList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_sort(self, args); }
PyObject *VarListList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, &PyOrVarList_Type>::_reduce(self); }


PVarList knownVars(PyObject *keywords)
{
  PVarList variables;
  PyObject *pyknownVars=keywords ? PyDict_GetItemString(keywords, "use") : PYNULL;
  if (!pyknownVars || (pyknownVars == Py_None))
    return PVarList();

  if (PyOrVarList_Check(pyknownVars))
    variables = ((GCPtr<TVarList>)(PyOrange_AS_Orange(pyknownVars)));

  else if (PyOrDomain_Check(pyknownVars)) {
    PDomain domain = PyOrange_AsDomain(pyknownVars);
    variables = mlnew TVarList(domain->variables.getReference());
    ITERATE(TMetaVector, mi, domain->metas)
      variables->push_back((*mi).variable);
  }

  else
    variables= PVarList_FromArguments(pyknownVars);

  if (!variables)
    raiseError("invalid value for 'use' argument"); // PYERROR won't do - NULL is a valid value to return...

  return variables;
}


PDomain knownDomain(PyObject *keywords)
{
  PVarList variables;
  PyObject *pyknownDomain = keywords ? PyDict_GetItemString(keywords, "domain") : PYNULL;
  if (!pyknownDomain || (pyknownDomain == Py_None))
    return PDomain();

  if (!PyOrDomain_Check(pyknownDomain))
    raiseError("invalid value for 'domain' argument"); // PYERROR won't do - NULL is a valid value to return...

  return PyOrange_AsDomain(pyknownDomain);
}


TMetaVector *knownMetas(PyObject *keywords)
{ 
  if (!keywords)
    return NULL;

  PyObject *pyknownDomain = PyDict_GetItemString(keywords, "domain");
  if (pyknownDomain && (pyknownDomain != Py_None)) {
    if (!PyOrDomain_Check(pyknownDomain))
      raiseError("invalid value for 'domain' argument"); // PYERROR won't do - NULL is a valid value to return...
    return &PyOrange_AsDomain(pyknownDomain)->metas;
  }

  pyknownDomain = PyDict_GetItemString(keywords, "use");
  if (pyknownDomain && PyOrDomain_Check(pyknownDomain))
    return &PyOrange_AsDomain(pyknownDomain)->metas;
  
  return NULL;
}

ABSTRACT(Variable, Orange)
C_NAMED(EnumVariable, Variable, "([name=, values=, autoValues=, distributed=, getValueFrom=])")
C_NAMED(FloatVariable, Variable, "([name=, startValue=, endValue=, stepValue=, distributed=, getValueFrom=])")

PyObject *PyVariable_MakeStatus_FromLong(long ok);

/* Left for compatibility (also put into the header, as for others */
PyObject *MakeStatus()
{ PyObject *mt=PyModule_New("MakeStatus");
  PyModule_AddObject(mt, "OK", PyVariable_MakeStatus_FromLong((long)TVariable::OK));
  PyModule_AddObject(mt, "MissingValues", PyVariable_MakeStatus_FromLong((long)TVariable::MissingValues));
  PyModule_AddObject(mt, "NoRecognizedValues", PyVariable_MakeStatus_FromLong((long)TVariable::NoRecognizedValues));
  PyModule_AddObject(mt, "Incompatible", PyVariable_MakeStatus_FromLong((long)TVariable::Incompatible));
  PyModule_AddObject(mt, "NotFound", PyVariable_MakeStatus_FromLong((long)TVariable::NotFound));
  return mt;
}

PYCLASSCONSTANT(Variable, MakeStatus, MakeStatus())


PyObject *Variable_getExisting(PyObject *, PyObject *args) PYARGS(METH_VARARGS | METH_STATIC, "(name, type[, fixedOrderValues[, otherValues, failOn]]) -> (Variable|None, status)")
{
  PyTRY
    char *varName;
    int varType;
    PStringList values;
    PStringList unorderedValues_asList;
    int failOn = TVariable::Incompatible;
    
    if (!PyArg_ParseTuple(args, "si|O&O&i:Variable.getExisting", &varName, &varType, ptn_StringList, &values, ptn_StringList, &unorderedValues_asList, &failOn))
      return NULL;
    
    set<string> unorderedValues;
    if (unorderedValues_asList)
      unorderedValues.insert(unorderedValues_asList->begin(), unorderedValues_asList->end());
      
    int status;
    PVariable var = TVariable::getExisting(varName, varType, values.getUnwrappedPtr(), &unorderedValues, failOn, &status);
    return Py_BuildValue("NN", WrapOrange(var), PyVariable_MakeStatus_FromLong(status));
  PyCATCH
}


PyObject *Variable_make(PyObject *, PyObject *args) PYARGS(METH_VARARGS | METH_STATIC, "(name, type[, fixedOrderValues[, otherValues, createNewOn]]) -> (Variable|None, status)")
{
  PyTRY
    char *varName;
    int varType;
    PStringList values;
    PStringList unorderedValues_asList;
    int createNewOn = TVariable::Incompatible;
    
    if (!PyArg_ParseTuple(args, "si|O&O&i:Variable.make", &varName, &varType, ptn_StringList, &values, ptn_StringList, &unorderedValues_asList, &createNewOn))
      return NULL;
    
    set<string> unorderedValues;
    if (unorderedValues_asList)
      unorderedValues.insert(unorderedValues_asList->begin(), unorderedValues_asList->end());
    
    int status;  
    PVariable var = TVariable::make(varName, varType, values.getUnwrappedPtr(), &unorderedValues, createNewOn, &status);
    return Py_BuildValue("NN", WrapOrange(var), PyVariable_MakeStatus_FromLong(status));
  PyCATCH
}


#include "stringvars.hpp"
C_NAMED(StringVariable, Variable, "([name=])")

#include "pythonvars.hpp"
C_NAMED(PythonVariable, Variable, "([name=])")

PyObject *PythonValue_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(SomeValue, "([object])")
{
  if (!PyTuple_Size(args))
    return WrapNewOrange(mlnew TPythonValue(), type);

  if (PyTuple_Size(args)==1)
    return WrapNewOrange(mlnew TPythonValue(PyTuple_GET_ITEM(args, 0)), type);

  else
    PYERROR(PyExc_TypeError, "PythonValue.__init__ expects up to one Python object", PYNULL);
}


PyObject *PythonValueSpecial_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Orange, "(int)") ALLOWS_EMPTY
{
  int vtype = 1;
  if (!PyArg_ParseTuple(args, "|i:PythonValueSpecial.__init__", &vtype))
    return PYNULL;

  return WrapNewOrange(mlnew TPythonValueSpecial(vtype), type);
}


int PythonValue_set_value(PyObject *self, PyObject *value)
{
  Py_INCREF(value);
  SELF_AS(TPythonValue).value = value;
  return 0;
}


PyObject *PythonValue_get_value(PyObject *self)
{
  PyObject *res = SELF_AS(TPythonValue).value;
  Py_INCREF(res);
  return res;
}


PyObject *PythonValue__reduce__(PyObject *self)
{
  return Py_BuildValue("O(O)", (PyObject *)(self->ob_type), SELF_AS(TPythonValue).value);
}


PyObject *Variable_getattr(TPyOrange *self, PyObject *name)
{
  if (PyString_Check(name) && !strcmp(PyString_AsString(name), "attributes")
      && (!self->orange_dict || !PyDict_Contains(self->orange_dict, name))) {
    PyObject *dict = PyDict_New();
    Orange_setattrDictionary(self, name, dict, false);
    Py_DECREF(dict);
  }

  return Orange_getattr(self, name);
}

PyObject *Variable_randomvalue(PyObject *self, PyObject *args) PYARGS(0, "() -> Value")
{ PyTRY
    CAST_TO(TVariable, var);
    if (args && !PyArg_ParseTuple(args, ""))
      PYERROR(PyExc_TypeError, "no parameters expected", PYNULL);

    return Value_FromVariableValue(PyOrange_AsVariable(self), var->randomValue());
  PyCATCH
}

PyObject *Variable_firstvalue(PyObject *self, PyObject *args) PYARGS(0, "() -> Value | None")
{ PyTRY
    CAST_TO(TVariable, var);
    if (args && !PyArg_ParseTuple(args, ""))
      PYERROR(PyExc_TypeError, "no parameters expected", PYNULL);

    TValue val;
    if (!var->firstValue(val)) RETURN_NONE;

    return Value_FromVariableValue(PyOrange_AsVariable(self), val);
  PyCATCH
}

PyObject *Variable_nextvalue(PyObject *self, PyObject *val) PYARGS(METH_O, "(value)  -> Value | None")
{ PyTRY
    CAST_TO(TVariable, var);
    if (   !PyOrValue_Check(val)
        || (PyValue_AS_Variable(val) ? (PyValue_AS_Variable(val) != var) : (PyValue_AS_Value(val).varType != var->varType)))
      PYERROR(PyExc_TypeError, "invalid value parameter", PYNULL);

    TValue sval = PyValue_AS_Value(val);

    if (!var->nextValue(sval))
      RETURN_NONE;

    return Value_FromVariableValue(PyOrange_AsVariable(self), sval);
  PyCATCH
}


PyObject *Variable_computeValue(PyObject *self, PyObject *args) PYARGS(METH_O, "(example) -> Value")
{ PyTRY
    CAST_TO(TVariable, var);
    if (!PyOrExample_Check(args))
      PYERROR(PyExc_TypeError, "Variable.computeValue: 'Example' expected", PYNULL);

    const TExample &ex = PyExample_AS_ExampleReference(args);

    int idx = ex.domain->getVarNum(var, false);
    if (idx != ILLEGAL_INT)
      return Value_FromVariableValue(var, ex[idx]);

    if (!var->getValueFrom)
      PYERROR(PyExc_SystemError, "Variable.computeValue: 'getValueFrom' not defined", PYNULL);

    return Value_FromVariableValue(var, var->computeValue(PyExample_AS_ExampleReference(args)));
  PyCATCH
}


PyObject *Variable_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(value) -> Value")
{ PyTRY
    NO_KEYWORDS

    PyObject *object;
    TValue value;

    if (   !PyArg_ParseTuple(args, "O:Variable.__call__", &object)
        || !convertFromPython(object, value, PyOrange_AsVariable(self)))
      return PYNULL;

    return Value_FromVariableValue(PyOrange_AsVariable(self), value);
  PyCATCH
}


PyObject *Variable_DC(PyObject *self) PYARGS(METH_NOARGS, "() -> DC")
{
  PyTRY
    PVariable var = PyOrange_AsVariable(self);
    return Value_FromVariableValue(var, var->DC());
  PyCATCH
}



PyObject *Variable_DK(PyObject *self) PYARGS(METH_NOARGS, "() -> DK")
{
  PyTRY
    PVariable var = PyOrange_AsVariable(self);
    return Value_FromVariableValue(var, var->DK());
  PyCATCH
}


PyObject *Variable_specialValue(PyObject *self, PyObject *arg) PYARGS(METH_O, "(int) -> special value")
{
  PyTRY
    int valType;
    if (!convertFromPython(arg, valType))
      return PYNULL;
    PVariable var = PyOrange_AsVariable(self);
    return Value_FromVariableValue(var, var->specialValue(valType));
  PyCATCH
}



PyObject *replaceVarWithEquivalent(PyObject *pyvar)
{
  PVariable newVar = PyOrange_AsVariable(pyvar);
  TEnumVariable *enewVar = newVar.AS(TEnumVariable);
  TVariable *oldVar = TVariable::getExisting(newVar->name, newVar->varType, enewVar ? enewVar->values.getUnwrappedPtr() : NULL, NULL, TVariable::Incompatible);
  if (oldVar && oldVar->isEquivalentTo(newVar.getReference())) {
    if (newVar->sourceVariable)
      oldVar->sourceVariable = newVar->sourceVariable;
    if (newVar->getValueFrom)
      oldVar->getValueFrom = newVar->getValueFrom;
    Py_DECREF(pyvar);
    return WrapOrange(PVariable(oldVar));
  }
  return pyvar;
}


PyObject *Variable__reduce__(PyObject *self)
{
	PyTRY
		return Py_BuildValue("O(ON)", getExportedFunction("__pickleLoaderVariable"), self->ob_type, packOrangeDictionary(self));
	PyCATCH
}

PyObject *__pickleLoaderVariable(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, dictionary)")
{
  PyTRY
    PyTypeObject *type;
    PyObject *dict;
	  if (!PyArg_ParseTuple(args, "OO:__pickleLoaderEnumVariable", &type, &dict))
		  return NULL;
		PyObject *emptyTuple = PyTuple_New(0);
		PyObject *pyVar = type->tp_new(type, emptyTuple, NULL);
		Py_DECREF(emptyTuple);
		if (unpackOrangeDictionary(pyVar, dict) == -1)
		  PYERROR(PyExc_AttributeError, "cannot construct the variable from the pickle", PYNULL)
		return replaceVarWithEquivalent(pyVar);
	PyCATCH
}

PyObject *EnumVariable__reduce__(PyObject *self)
{
	PyTRY
		return Py_BuildValue("O(ON)", getExportedFunction("__pickleLoaderEnumVariable"), self->ob_type, packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderEnumVariable(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, dictionary)")
{
  PyTRY
    PyTypeObject *type;
    PyObject *dict;
	  if (!PyArg_ParseTuple(args, "OO:__pickleLoaderEnumVariable", &type, &dict))
		  return NULL;

    char *name = NULL;
    TStringList *values = NULL;
    
    PyObject *pyname = PyDict_GetItemString(dict, "name");
    if (pyname)
      name = PyString_AsString(pyname);

    PyObject *pyvalues = PyDict_GetItemString(dict, "values");
    if (pyvalues)
      values = PyOrange_AsStringList((TPyOrange *)pyvalues).getUnwrappedPtr();
      
    TVariable *var = TVariable::getExisting(name, TValue::INTVAR, values, NULL);
    PVariable pvar = var;
    if (!var) {
      TEnumVariable *evar = new TEnumVariable(name ? name : "");
      pvar = evar;
      if (values)
        const_PITERATE(TStringList, vi, values)
          evar->addValue(*vi);
    }
    
    PyObject *pyvar = WrapOrange(pvar);

    PyObject *d_key, *d_value;
    Py_ssize_t i = 0;
    while (PyDict_Next(dict, &i, &d_key, &d_value)) {
      if (   strcmp("values", PyString_AsString(d_key))
          && Orange_setattrLow((TPyOrange *)pyvar, d_key, d_value, false) < 0
         ) {
          Py_DECREF(pyvar);
          return NULL;
        }
    }
 
    return replaceVarWithEquivalent(pyvar);
	PyCATCH
}

PyObject *EnumVariable_getitem_sq(PyObject *self, int index)
{ PyTRY
    CAST_TO(TEnumVariable, var)
    if (!var->values || (index<0) || (index>=int(var->values->size())))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);
    return Value_FromVariableValue(PyOrange_AsVariable(self), TValue(index));
  PyCATCH
}


PyObject *FloatVariable_getitem_sq(PyObject *self, int index)
{ PyTRY
    CAST_TO(TFloatVariable, var);
    if ((var->stepValue<=0) || (var->startValue>var->endValue))
      PYERROR(PyExc_IndexError, "interval not specified", PYNULL);
    
    float maxInd = (var->endValue - var->startValue)/var->stepValue;

    if ((index<0) || (index>maxInd))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);
    return Value_FromVariableValue(PyOrange_AsVariable(self), TValue(var->startValue+var->stepValue*index));
  PyCATCH
}


bool convertFromPythonWithVariable(PyObject *obj, string &str)
{ return convertFromPythonWithML(obj, str, *FindOrangeType(typeid(TVariable))); }


bool varListFromDomain(PyObject *boundList, PDomain domain, TVarList &boundSet, bool allowSingle, bool checkForIncludance)
{ if (PyOrVarList_Check(boundList)) {
    PVarList variables = PyOrange_AsVarList(boundList);
    if (checkForIncludance)
      const_PITERATE(TVarList, vi, variables)
        if (!domain || (domain->getVarNum(*vi, false)==ILLEGAL_INT)) {
          PyErr_Format(PyExc_IndexError, "variable '%s' does not exist in the domain", (*vi)->name.c_str());
          return false;
        }
    boundSet=variables.getReference();
    return true;
  }
  
  if (PySequence_Check(boundList)) {
    PyObject *iterator = PyObject_GetIter(boundList);
    if (iterator) {
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
        PVariable variable=varFromArg_byDomain(item, domain, checkForIncludance);
        Py_DECREF(item);
        if (!variable) {
          Py_DECREF(iterator);
          return false;
        }
        boundSet.push_back(variable);
      }
        
      Py_DECREF(iterator);
      return true;
    }
  }
      
  else if (allowSingle) {
    PVariable variable=varFromArg_byDomain(boundList, domain, checkForIncludance);
    if (variable) {
      boundSet.push_back(variable);
      return true;
    }
  }
  PYERROR(PyExc_TypeError, "invalid argument (list of variables expected)", false);
}


// Given a parameter from Python and a domain, it returns a variable.
// Python's parameter can be a string name, an index or Variable
PVariable varFromArg_byDomain(PyObject *obj, PDomain domain, bool checkForIncludance)
{ PVariable var;
  if (domain) {
    PyTRY
      if (PyString_Check(obj)) {
        const char *attr = PyString_AS_STRING(obj);
        PVariable res = domain->getVar(attr, true, false);
        if (!res)
          PyErr_Format(PyExc_IndexError, "attribute '%s' not found", attr);
        return res;
      }
      if (PyInt_Check(obj)) {
        int idx = PyInt_AsLong(obj);

        if (idx<0) {
          PVariable res = domain->getMetaVar(idx, false);
          if (!res)
            PyErr_Format(PyExc_IndexError, "meta attribute %i not found", idx);
          return res;
        }

        if (idx>=int(domain->variables->size()))
          PYERROR(PyExc_IndexError, "index out of range", PVariable());

        return domain->getVar(idx);
      }
    PyCATCH_r(PVariable())
  }

  if (PyOrVariable_Check(obj)) {
    PVariable var(PyOrange_AsVariable(obj));
    if (checkForIncludance)
      if (!domain || (domain->getVarNum(var, false)==ILLEGAL_INT))
        PYERROR(PyExc_IndexError, "variable does not exist in the domain", PVariable());
    return var;
  }

  PYERROR(PyExc_TypeError, "invalid type for variable", PVariable());
}


bool varListFromVarList(PyObject *boundList, PVarList varlist, TVarList &boundSet, bool allowSingle, bool checkForIncludance)
{ if (PyOrVarList_Check(boundList)) {
    PVarList variables = PyOrange_AsVarList(boundList);
    if (checkForIncludance)
      const_PITERATE(TVarList, vi, variables) {
        TVarList::const_iterator fi(varlist->begin()), fe(varlist->end());
        for(; (fi!=fe) && (*fi != *vi); fi++);
        if (fi==fe) {
          PyErr_Format(PyExc_IndexError, "variable '%s' does not exist in the domain", (*vi)->name.c_str());
          return false;
        }
      }
    boundSet = variables.getReference();
    return true;
  }
  
  if (PyList_Check(boundList)) {
    for(int pos=0, max=PyList_Size(boundList); pos<max; pos++) {
      PyObject *li=PyList_GetItem(boundList, pos);
      if (!li)
        PYERROR(PyExc_TypeError, "can't read the argument list", false);
      PVariable variable = varFromArg_byVarList(li, varlist, checkForIncludance);
      if (!variable)
        return false;
      boundSet.push_back(variable);
    }
    return true;
  }
  else if (allowSingle) {
    PVariable variable = varFromArg_byVarList(boundList, varlist, checkForIncludance);
    if (!variable)
      return false;
    boundSet.push_back(variable);
    return true;
  }

  PYERROR(PyExc_TypeError, "invalid attribute for list of variables", false);
}


// Given a parameter from Python and a list of variables, it returns a variable.
// Python's parameter can be a string name, an index or Variable
PVariable varFromArg_byVarList(PyObject *obj, PVarList varlist, bool checkForIncludance)
{ PVariable var;
  if (varlist) {
    PyTRY
      if (PyString_Check(obj)) {
        char *s = PyString_AS_STRING(obj);
        TVarList::const_iterator fi(varlist->begin()), fe(varlist->end());
        for(; (fi!=fe) && ((*fi)->name != s); fi++);
        if (fi==fe) {
          PyErr_Format(PyExc_IndexError, "variable '%s' does not exist in the domain", s);
          return PVariable();
        }
        else
          return *fi;
      }
    PyCATCH_r(PVariable())
  }

  if (PyOrVariable_Check(obj)) {
    PVariable var(PyOrange_AsVariable(obj));
    if (checkForIncludance) {
      TVarList::const_iterator fi(varlist->begin()), fe(varlist->end());
      for(; (fi!=fe) && (*fi != var); fi++);
      if (fi==fe)
        PYERROR(PyExc_IndexError, "variable does not exist in the domain", PVariable());
    }
    return var;
  }

  PYERROR(PyExc_TypeError, "invalid type for variable", PVariable());
}


bool varNumFromVarDom(PyObject *pyvar, PDomain domain, int &attrNo)
{ 
  PVariable var=varFromArg_byDomain(pyvar, domain);
  if (!var)
    return false; // varFromArg_byDomain has already set the error message

  PITERATE(TVarList, vi, domain->attributes)
    if (*vi==var) {
      attrNo = vi-domain->attributes->begin();
      return true;
    }

  attrNo = domain->getMetaNum(var, false);
  return attrNo != ILLEGAL_INT;
}


bool weightFromArg_byDomain(PyObject *pyweight, PDomain domain, int &weightID)
{
  if (!pyweight || (pyweight == Py_None))
    weightID = 0;

  else if (PyInt_Check(pyweight))
    weightID =  PyInt_AsLong(pyweight);

  else {
    PVariable var = varFromArg_byDomain(pyweight, domain);
    if (!var)
      PYERROR(PyExc_TypeError, "invalid or unknown weight attribute", false);
  
    weightID = domain->getVarNum(var);
  }

  return true;
}


static PExampleGenerator *ptw_examplegenerator;

int ptw_weightByDomainCB(PyObject *args, void *weight)
{ 
  PDomain dom = ptw_examplegenerator ? (*ptw_examplegenerator)->domain : PDomain();
  ptw_examplegenerator = NULL;
  return weightFromArg_byDomain(args, dom, *(int *)weight) ? 1 : 0;
}

converter pt_weightByGen(PExampleGenerator &peg)
{ 
  ptw_examplegenerator = &peg;
  return ptw_weightByDomainCB;
}


PyObject *StringValue_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SomeValue, "(string)")
{ char *s;
  if (!PyArg_ParseTuple(args, "s:StringValue", &s))
    return PYNULL;

  return WrapNewOrange(mlnew TStringValue(s), type);
}


PyObject *StringValue__reduce__(PyObject *self)
{
  return Py_BuildValue("O(s)", (PyObject *)(self->ob_type), SELF_AS(TStringValue).value.c_str());
}


/* ************ ATTRIBUTED FLOAT LIST ************ */

PyObject *AttributedFloatList_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(FloatList, "(attributes, list)") ALLOWS_EMPTY
{
  PyObject *ob1 = NULL, *ob2 = NULL;
  if (!PyArg_UnpackTuple(args, "AttributedFloatList.new", 0, 2, &ob1, &ob2))
    return PYNULL;

  PyObject *wabl = ListOfUnwrappedMethods<PAttributedFloatList, TAttributedFloatList, float>::_new(type, ob2 ? ob2 : ob1, keywds);

  if (ob2) {
    PVarList attributes = PVarList_FromArguments(ob1);
    if (!attributes)
      return PYNULL;

    PyOrange_AsAttributedFloatList(wabl)->attributes = attributes;
  }

  return wabl;
}


PyObject * /*no pyxtract!*/ FloatList_getitem_sq(TPyOrange *self, int index);
int        /*no pyxtract!*/ FloatList_setitem_sq(TPyOrange *self, int index, PyObject *item);

int AttributedList_getIndex(const int &listsize, PVarList attributes, PyObject *index)
{
  int res;

  if (!listsize)
    PYERROR(PyExc_IndexError, "the list is empty", ILLEGAL_INT);

  if (PyInt_Check(index)) {
    res = (int)PyInt_AsLong(index);
    if (res < 0)
      res += listsize;
  }

  else {
    if (!attributes)
      PYERROR(PyExc_AttributeError, "variable list not defined, need integer indices", ILLEGAL_INT);

    if (PyOrVariable_Check(index)) {
      PVariable var = PyOrange_AsVariable(index);
      TVarList::const_iterator vi(attributes->begin()), ve(attributes->end());
      int ind = 0;
      for(; vi!=ve; vi++, ind++)
        if (*vi == var) {
          res = ind;
          break;
        }

      if (vi == ve) {
        PyErr_Format(PyExc_AttributeError, "attribute '%s' not found in the list", var->name.c_str());
        return ILLEGAL_INT;
      }
    }
    
    else if (PyString_Check(index)) {
      const char *name = PyString_AsString(index);
      TVarList::const_iterator vi(attributes->begin()), ve(attributes->end());
      int ind = 0;
      for(; vi!=ve; vi++, ind++)
        if ((*vi)->name == name) {
          res = ind;
          break;
        }

      if (vi == ve) {
        PyErr_Format(PyExc_AttributeError, "attribute '%s' not found in the list", name);
        return ILLEGAL_INT;
      }
    }

    else {
      PyErr_Format(PyExc_TypeError, "cannot index the list by '%s'", index->ob_type->tp_name);
      return ILLEGAL_INT;
    }
  }

  if ((res >= listsize) || (res < 0)) {
    PyErr_Format(PyExc_IndexError, "index %i out of range 0-%i", res, listsize-1);
    return ILLEGAL_INT;
  }

  return res;
}


PyObject *AttributedFloatList_getitem(TPyOrange *self, PyObject *index)
{
  PyTRY 
    CAST_TO(TAttributedFloatList, aflist)

    const int ind = AttributedList_getIndex(aflist->size(), aflist->attributes, index);
    if (ind == ILLEGAL_INT)
      return PYNULL;

    return FloatList_getitem_sq(self, ind);
  PyCATCH
}


int AttributedFloatList_setitem(TPyOrange *self, PyObject *index, PyObject *value)
{
  PyTRY 
    CAST_TO_err(TAttributedFloatList, aflist, -1)

    const int ind = AttributedList_getIndex(aflist->size(), aflist->attributes, index);
    if (ind == ILLEGAL_INT)
      return -1;

    return FloatList_setitem_sq(self, ind, value);
  PyCATCH_1
}



PyObject *AttributedBoolList_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(BoolList, "(attributes, list)") ALLOWS_EMPTY
{
  PyObject *ob1 = NULL, *ob2 = NULL;
  if (!PyArg_UnpackTuple(args, "AttributedBoolList.new", 0, 2, &ob1, &ob2))
    return PYNULL;

  PyObject *wabl = ListOfUnwrappedMethods<PAttributedBoolList, TAttributedBoolList, bool>::_new(type, ob2 ? ob2 : ob1, keywds);

  if (ob2) {
    PVarList attributes = PVarList_FromArguments(ob1);
    if (!attributes)
      return PYNULL;

    PyOrange_AsAttributedBoolList(wabl)->attributes = attributes;
  }

  return wabl;
}


PyObject * /*no pyxtract!*/ BoolList_getitem_sq(TPyOrange *self, int index);
int        /*no pyxtract!*/ BoolList_setitem_sq(TPyOrange *self, int index, PyObject *item);


PyObject *AttributedBoolList_getitem(TPyOrange *self, PyObject *index)
{
  PyTRY 
    CAST_TO(TAttributedBoolList, aflist)

    const int ind = AttributedList_getIndex(aflist->size(), aflist->attributes, index);
    if (ind == ILLEGAL_INT)
      return PYNULL;

    return BoolList_getitem_sq(self, ind);
  PyCATCH
}


int AttributedBoolList_setitem(TPyOrange *self, PyObject *index, PyObject *value)
{
  PyTRY 
    CAST_TO_err(TAttributedBoolList, aflist, -1)

    const int ind = AttributedList_getIndex(aflist->size(), aflist->attributes, index);
    if (ind == ILLEGAL_INT)
      return -1;

    return BoolList_setitem_sq(self, ind, value);
  PyCATCH_1
}


/* ************ DOMAIN ************ */

#include "domain.hpp"

const TMetaDescriptor *metaDescriptorFromArg(TDomain &domain, PyObject *rar)
{
  TMetaDescriptor *desc = NULL;

  if (PyString_Check(rar))
    desc = domain.metas[string(PyString_AsString(rar))];

  else if (PyOrVariable_Check(rar))
    desc = domain.metas[PyOrange_AsVariable(rar)->name];

  else if (PyInt_Check(rar))
    desc = domain.metas[PyInt_AsLong(rar)];

  else
    PYERROR(PyExc_TypeError, "invalid meta descriptor", NULL);

  if (!desc)
    PYERROR(PyExc_AttributeError, "meta attribute does not exist", NULL);

  return desc;
}


PyObject *Domain_metaid(TPyOrange *self, PyObject *rar) PYARGS(METH_O, "(name | descriptor) -> int")
{ PyTRY
    const TMetaDescriptor *desc = metaDescriptorFromArg(SELF_AS(TDomain), rar);
    return desc ? PyInt_FromLong(desc->id) : NULL;
  PyCATCH
}


PyObject *Domain_isOptionalMeta(TPyOrange *self, PyObject *rar) PYARGS(METH_O, "(name | int | descriptor) -> bool")
{
  PyTRY
    const TMetaDescriptor *desc = metaDescriptorFromArg(SELF_AS(TDomain), rar);
    return desc ? PyBool_FromLong(desc->optional ? 1 : 0) : NULL;

  PyCATCH
}


PyObject *Domain_hasmeta(TPyOrange *self, PyObject *rar) PYARGS(METH_O, "(name | int | descriptor) -> bool")
{
  PyTRY
    CAST_TO(TDomain, domain)

    TMetaDescriptor *desc = NULL;

    if (PyString_Check(rar))
      desc = domain->metas[string(PyString_AsString(rar))];

    else if (PyOrVariable_Check(rar))
      desc = domain->metas[PyOrange_AsVariable(rar)->name];

    else if (PyInt_Check(rar))
      desc = domain->metas[PyInt_AsLong(rar)];

    else
      PYERROR(PyExc_TypeError, "invalid meta descriptor", NULL);

    return PyBool_FromLong(desc ? 1 : 0);
  PyCATCH
}


PyObject *Domain_getmeta(TPyOrange *self, PyObject *rar) PYARGS(METH_O, "(name | int) -> Variable")
{ PyTRY
    const TMetaDescriptor *desc = metaDescriptorFromArg(SELF_AS(TDomain), rar);
    return desc ? WrapOrange(desc->variable) : NULL;
  PyCATCH
}


PyObject *Domain_getmetasLow(const TDomain &domain)
{
  PyObject *dict = PyDict_New();
  const_ITERATE(TMetaVector, mi, domain.metas)
    PyDict_SetItem(dict, PyInt_FromLong((*mi).id), WrapOrange((*mi).variable));
  return dict;
}


PyObject *Domain_getmetasLow(const TDomain &domain, const int optional)
{
  PyObject *dict = PyDict_New();
  const_ITERATE(TMetaVector, mi, domain.metas)
    if (optional == (*mi).optional)
      PyDict_SetItem(dict, PyInt_FromLong((*mi).id), WrapOrange((*mi).variable));
  return dict;
}


PyObject *Domain_getmetas(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([optional]) -> {int: Variable}")
{ PyTRY
    if (PyTuple_Size(args) && (PyTuple_GET_ITEM(args, 0) != Py_None)) {
      int opt;
      if (!PyArg_ParseTuple(args, "i:Domain.getmetas", &opt))
        return NULL;

      return Domain_getmetasLow(SELF_AS(TDomain), opt);
    }

    return Domain_getmetasLow(SELF_AS(TDomain));
  PyCATCH
}


PyObject *Domain_addmeta(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(id, descriptor[, optional]) -> None")
{ PyTRY
    CAST_TO(TDomain, domain);

    int id;
    PVariable var;
    int opt = 0;
    if (!PyArg_ParseTuple(args, "iO&|i", &id, cc_Variable, &var, &opt))
      return PYNULL;

    domain->metas.push_back(TMetaDescriptor(id, var, opt));
    domain->domainHasChanged();
    RETURN_NONE;
  PyCATCH
}


bool convertMetasFromPython(PyObject *dict, TMetaVector &metas)
{
  Py_ssize_t pos = 0;
  PyObject *pykey, *pyvalue;
  while (PyDict_Next(dict, &pos, &pykey, &pyvalue)) {
    if (!PyOrVariable_Check(pyvalue)) {
      PyErr_Format(PyExc_TypeError, "parsing meta attributes: dictionary value at position '%i' should be 'Variable', not '%s'", pos-1, pyvalue->ob_type->tp_name);
      return false;
    }
    if (!PyInt_Check(pykey) || (PyInt_AsLong(pykey)>=0))
      PYERROR(PyExc_TypeError, "parsing meta attributes: dictionary keys should be meta-ids (negative integers)", false);

    metas.push_back(TMetaDescriptor((int)PyInt_AsLong(pykey), PyOrange_AsVariable(pyvalue)));
  }

  return true;
}


PyObject *Domain_addmetasLow(TDomain &domain, PyObject *dict, const int opt = 0)
{
  TMetaVector metas;
  if (!convertMetasFromPython(dict, metas))
    return PYNULL;

  ITERATE(TMetaVector, mi, metas) {
    (*mi).optional = opt;
    domain.metas.push_back(*mi);
  }

  domain.domainHasChanged();

  RETURN_NONE;
}


PyObject *Domain_addmetas(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "({id: descriptor, id: descriptor, ...}[, optional]) -> None")
{ PyTRY
    PyObject *pymetadict;
    int opt = 0;
    if (!PyArg_ParseTuple(args, "O|i", &pymetadict, &opt))
      PYERROR(PyExc_AttributeError, "Domain.addmetas expects a dictionary with id's and descriptors, optionally follow by an int flag 'optional'", PYNULL);

    return Domain_addmetasLow(SELF_AS(TDomain), pymetadict, opt);
  PyCATCH
}


bool removeMeta(PyObject *rar, TMetaVector &metas)
{ TMetaVector::iterator mvi(metas.begin()), mve(metas.end());

  if (PyInt_Check(rar)) {
    int id = PyInt_AsLong(rar);
    while((mvi!=mve) && ((*mvi).id!=id))
      mvi++;
  }
  else if (PyOrVariable_Check(rar))
    while((mvi!=mve) && ((*mvi).variable!=PyOrange_AsVariable(rar)))
      mvi++;
  else if (PyString_Check(rar)) {
    char *metaname = PyString_AsString(rar);
    while((mvi!=mve) && ((*mvi).variable->name!=metaname))
      mvi++;
  }
  else
    mvi=mve;

  if (mvi==mve)
    PYERROR(PyExc_AttributeError, "meta value not found", false);

  metas.erase(mvi);
  return true;
}


PyObject *Domain_removemeta(TPyOrange *self, PyObject *rar) PYARGS(METH_O, "({id0:desc0, id1:desc1, ...}) | ([id0|desc0, id1|desc1, ...]) -> None")
{ PyTRY
    CAST_TO(TDomain, domain);

    if (PyDict_Check(rar)) {
      Py_ssize_t pos=0;
      PyObject *key, *value;
      TMetaVector newMetas=domain->metas;
      TMetaVector::iterator mve=domain->metas.end();

      while (PyDict_Next(rar, &pos, &key, &value)) {
        if (!PyInt_Check(key) || !PyOrVariable_Check(value))
          PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

        long idx=PyInt_AsLong(key);
        TMetaVector::iterator mvi(newMetas.begin());
        for(; (mvi!=mve) && ( ((*mvi).id!=idx) || (*mvi).variable!=PyOrange_AsVariable(value)); mvi++);
        if (mvi==mve)
          PYERROR(PyExc_AttributeError, "meta not found", PYNULL);

        newMetas.erase(mvi);
      }
      domain->metas=newMetas;
      domain->domainHasChanged();
    }

    else if (PyList_Check(rar)) {
      TMetaVector newMetas=domain->metas;
      for(int pos=0, noel=PyList_Size(rar); pos!=noel; pos++)
        if (!removeMeta(PyList_GetItem(rar, pos), newMetas))
          return PYNULL;
      domain->metas=newMetas;
      domain->domainHasChanged();
    }
  
    else if (!removeMeta(rar, domain->metas))
      return PYNULL;

    RETURN_NONE;
  PyCATCH
}


PyObject *Domain_hasDiscreteAttributes(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(includeClass=0) -> int")
{
  PyTRY
      PyObject *includeClass = PYNULL;
      if (!PyArg_ParseTuple(args, "|O:Domain.hasDiscreteAttributes", &includeClass))
        return PYNULL;

      return PyInt_FromLong(SELF_AS(TDomain).hasDiscreteAttributes(!includeClass || PyObject_IsTrue(includeClass)!=0) ? 1 : 0);
  PyCATCH
}


PyObject *Domain_hasContinuousAttributes(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(includeClass=0) -> int")
{
  PyTRY
      PyObject *includeClass = PYNULL;
      if (!PyArg_ParseTuple(args, "|O:Domain.hasContinuousAttributes", &includeClass))
        return PYNULL;

      return PyInt_FromLong(SELF_AS(TDomain).hasContinuousAttributes(!includeClass || PyObject_IsTrue(includeClass)!=0) ? 1 : 0);
  PyCATCH
}


PyObject *Domain_hasOtherAttributes(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(includeClass=0) -> int")
{
  PyTRY
      PyObject *includeClass = PYNULL;
      if (!PyArg_ParseTuple(args, "|O:Domain.hasOtherAttributes", &includeClass))
        return PYNULL;

      return PyInt_FromLong(SELF_AS(TDomain).hasOtherAttributes(!includeClass || PyObject_IsTrue(includeClass)!=0) ? 1 : 0);
  PyCATCH
}


int Domain_len(TPyOrange *self)
{ PyTRY
    CAST_TO_err(TDomain, domain, -1);
    return domain->variables->size();
  PyCATCH_1
}


PyObject *Domain_index(PyObject *self, PyObject *arg) PYARGS(METH_O, "(variable) -> int")
{
  PyTRY
    CAST_TO(TDomain, domain);

    PVariable variable = varFromArg_byDomain(arg, domain, true);
    return variable ? PyInt_FromLong(domain->getVarNum(variable)) : PYNULL;
  PyCATCH
}


int Domain_contains(PyObject *self, PyObject *arg)
{
  PyTRY
    CAST_TO_err(TDomain, domain, -1);

    PVariable variable = varFromArg_byDomain(arg, domain, true);
    PyErr_Clear();
    return variable ? 1 : 0;
  PyCATCH_1
}

CONSTRUCTOR_KEYWORDS(Domain, "source")

PyObject *Domain_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "(list-of-attrs | domain [, hasClass | classVar | None] [,domain | list-of-attrs | source=domain])")
{ PyTRY
    PyObject *list;
    PyObject *arg1 = PYNULL;
    PyObject *arg2 = PYNULL;

    if (PyArg_ParseTuple(args, "O|OO", &list, &arg1, &arg2)) {

      if (keywds) {
        if (PyDict_Size(keywds)>1)
          PYERROR(PyExc_TypeError, "Domain() accepts only one keyword argument ('source')", PYNULL);
        if (PyDict_Size(keywds)==1) {
          PyObject *arg3 = PyDict_GetItemString(keywds, "source");
          if (!arg3)
            PYERROR(PyExc_TypeError, "Domain: invalid keywords argument ('source' expected)", PYNULL);
          if (arg1 && arg2) {
            PYERROR(PyExc_TypeError, "Domain: too many arguments", PYNULL);
          }
          else
            if (!arg1)
              arg1 = arg3;
            else
              arg2 = arg3;
        }
      }


      if (PyOrDomain_Check(list)) {

        PDomain dom = PyOrange_AsDomain(list);

        if (arg1)
          if (PyString_Check(arg1) || PyOrVariable_Check(arg1)) {
            PVariable classVar = varFromArg_byDomain(arg1, dom, false);
            if (!classVar)
              return PYNULL;
            TVarList attributes = dom->variables.getReference();
            int vnumint = dom->getVarNum(classVar, false);
            if (vnumint>=0)
              attributes.erase(attributes.begin()+vnumint);
            return WrapNewOrange(mlnew TDomain(classVar, attributes), type);
          }
          else if (PyInt_Check(arg1) || (arg1==Py_None)) {
            TVarList attributes = dom->variables.getReference();
            if (PyObject_IsTrue(arg1))
              return WrapNewOrange(CLONE(TDomain, dom), type);
            else
              return WrapNewOrange(mlnew TDomain(PVariable(), dom->variables.getReference()), type);
          }
          else 
            PYERROR(PyExc_TypeError, "Domain: invalid arguments for constructor (I'm unable to guess what you meant)", PYNULL);

        return WrapNewOrange(CLONE(TDomain, dom), type);
      }

      /* Now, arg1 can be either 
           - NULL
           - source (i.e. Domain or list of variables) 
           - boolean that tells whether we have a class
           - class variable
         If arg1 is present but is not source, arg2 can be source
      */

      PVarList source;
      PVariable classVar;
      bool hasClass = true;

      if (arg1) {
        if (PyOrDomain_Check(arg1))
          source = PyOrange_AsDomain(arg1)->variables;
        else if (PyOrVarList_Check(arg1))
          source = PyOrange_AsVarList(arg1);
        else if (PyList_Check(arg1))
          source = PVarList_FromArguments(arg1);
        else if (PyOrVariable_Check(arg1))
          classVar = PyOrange_AsVariable(arg1);
        else
          hasClass = (PyObject_IsTrue(arg1) != 0);
      }
      if (arg2) {
        if (source) {
          PYERROR(PyExc_TypeError, "Domain: invalid argument 3", PYNULL);
        }
        else
          if (PyOrDomain_Check(arg2)) {
            PDomain sourceDomain = PyOrange_AsDomain(arg2);
            source = mlnew TVarList(sourceDomain->variables.getReference());
            ITERATE(TMetaVector, mi, sourceDomain->metas)
              source->push_back((*mi).variable);
          }

          else if (PyOrVarList_Check(arg2))
            source = PyOrange_AsVarList(arg2);
          else if (PyList_Check(arg2))
            source = PVarList_FromArguments(arg2);
      }

      TVarList variables;
      if (!varListFromVarList(list, source, variables, true, false))
        return PYNULL;

      if (hasClass && !classVar && variables.size()) {
        classVar = variables.back();
        variables.erase(variables.end()-1);
      }
      
      return WrapNewOrange(mlnew TDomain(classVar, variables), type);
    }

    PYERROR(PyExc_TypeError, "invalid parameters (list of 'Variable' expected)", PYNULL);

  PyCATCH
}


PyObject *Domain__reduce__(PyObject *self)
{
  CAST_TO(TDomain, domain)

  
  return Py_BuildValue("O(ONNNN)N", getExportedFunction("__pickleLoaderDomain"),
                                   self->ob_type,
                                   WrapOrange(domain->attributes),
                                   WrapOrange(domain->classVar),
                                   Domain_getmetasLow(SELF_AS(TDomain), false),
                                   Domain_getmetasLow(SELF_AS(TDomain), true),
                                   packOrangeDictionary(self));
}

PyObject *__pickleLoaderDomain(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, attributes, classVar, metas)")
{
  PyTRY {
    if (!args || !PyTuple_Check(args) || (PyTuple_Size(args) != 5))
      PYERROR(PyExc_TypeError, "invalid arguments for the domain unpickler", NULL);

    PyTypeObject *type = (PyTypeObject *)PyTuple_GET_ITEM(args, 0);
    PyObject *attributes = PyTuple_GET_ITEM(args, 1);
    PyObject *classVar = PyTuple_GET_ITEM(args, 2);
    PyObject *req_metas = PyTuple_GET_ITEM(args, 3);
    PyObject *opt_metas = PyTuple_GET_ITEM(args, 4);

    if (!PyOrVarList_Check(attributes) || !PyDict_Check(req_metas) || !PyDict_Check(opt_metas))
      PYERROR(PyExc_TypeError, "invalid arguments for the domain unpickler", NULL);

  
    TDomain *domain = NULL;
    if (classVar == Py_None)
      domain = new TDomain(PVariable(), PyOrange_AsVarList(attributes).getReference());
    else if (PyOrVariable_Check(classVar))
      domain = new TDomain(PyOrange_AsVariable(classVar), PyOrange_AsVarList(attributes).getReference());
    else
      PYERROR(PyExc_TypeError, "invalid arguments for the domain unpickler", NULL);
      

    PyObject *pydomain = WrapNewOrange(domain, type);

    PyObject *res;
    res = Domain_addmetasLow(*domain, req_metas, false);
    if (!res) {
      Py_DECREF(pydomain);
      return NULL;
    }
    Py_DECREF(res);

    res = Domain_addmetasLow(*domain, opt_metas, true);
    if (!res) {
      Py_DECREF(pydomain);
      return NULL;
    }
    Py_DECREF(res);

    return pydomain;
  }
  PyCATCH
}


PyObject *Domain_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example) -> Example")
{ PyTRY
    NO_KEYWORDS

    TExample *ex;
    if (!PyArg_ParseTuple(args, "O&", ptr_Example, &ex))
      PYERROR(PyExc_TypeError, "invalid parameters (Example expected)", PYNULL);

    return Example_FromWrappedExample(PExample(mlnew TExample(PyOrange_AsDomain(self), *ex)));
  PyCATCH
}


PyObject *Domain_getitem(TPyOrange *self, PyObject *index)
{ PyTRY
    PVariable var = varFromArg_byDomain(index, PyOrange_AsDomain(self), true);
    return var ? WrapOrange(var) : PYNULL;
  PyCATCH
}


PyObject *Domain_getitem_sq(TPyOrange *self, int index)
{ PyTRY
    CAST_TO(TDomain, domain)
    if ((index<0) || (index>=int(domain->variables->size())))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);
    return WrapOrange(domain->getVar(index));
  PyCATCH
}


PyObject *Domain_getslice(TPyOrange *self, int start, int stop)
{ 
  PyTRY
    CAST_TO(TDomain, domain);

    int ds=domain->variables->size();
    if (start>ds) 
      start=ds;
    else if (start<0)
      start=0;

    if (stop>ds) stop=ds;
    else if (stop<0) stop=0;
    stop-=start;
    PyObject *list=PyList_New(stop);
    if (!list)
      return NULL;
    TVarList::iterator vi(domain->variables->begin()+start);
    for(int i=0; i<stop; i++)
      PyList_SetItem(list, i, WrapOrange(*(vi++)));
    return list;
  PyCATCH
}


const string &nonamevar = string("<noname>");

inline const string &namefrom(const string &name)
{ return name.length() ? name : nonamevar; }

string TDomain2string(TPyOrange *self)
{ CAST_TO_err(TDomain, domain, "<invalid domain>")

  string res;
  
  int added=0;
  PITERATE(TVarList, vi, domain->variables)
    res+=(added++ ? ", " : "[") + namefrom((*vi)->name);

  if (added) {
    res+="]";
    if (domain->metas.size())
      res+=", {";
  }
  else 
    if (domain->metas.size())
      res+="{";

  added=0;
  ITERATE(TMetaVector, mi, domain->metas) {
    char pls[256];
    sprintf(pls, "%s%i:%s", (added++) ? ", " : "", int((*mi).id), namefrom((*mi).variable->name).c_str());
    res+=pls;
  }
  if (added)
    res+="}";

  return res;
}



PyObject *Domain_repr(TPyOrange *pex)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)pex, NULL, NULL, "repr", "str");
    if (result)
      return result;

    return PyString_FromString(TDomain2string(pex).c_str());
  PyCATCH
}


PyObject *Domain_str(TPyOrange *pex)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)pex, NULL, NULL, "str", "repr");
    if (result)
      return result;

    return PyString_FromString(TDomain2string(pex).c_str());
  PyCATCH
}


int Domain_set_classVar(PyObject *self, PyObject *arg) PYDOC("Domain's class attribute")
{
  PyTRY
    CAST_TO_err(TDomain, domain, -1);
  
    if (arg==Py_None)
      domain->removeClass();
    else
      if (PyOrVariable_Check(arg))
        domain->changeClass(PyOrange_AsVariable(arg));
      else PYERROR(PyExc_AttributeError, "invalid type for class", -1)

    return 0;
  PyCATCH_1
}

PyObject *Domain_checksum(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "() -> crc")
{ return PyInt_FromLong(SELF_AS(TDomain).sumValues()); }


/* ************ RANDOM GENERATORS ************** */

#include "random.hpp"

C_UNNAMED(RandomGenerator, Orange, "() -> 32-bit random int")

PyObject *RandomGenerator_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "([int])")
{ PyTRY
      int i = 0;
      if (!PyArg_ParseTuple(args, "|i:RandomGenerator.__new__", &i))
        return PYNULL;

      return WrapNewOrange(mlnew TRandomGenerator(i), type);
  PyCATCH
}


PyObject *RandomGenerator__reduce__(PyObject *self)
{
  cMersenneTwister &mt = SELF_AS(TRandomGenerator).mt;

  return Py_BuildValue("O(Os#ii)N", getExportedFunction("__pickleLoaderRandomGenerator"),
                                    self->ob_type,
                                    (char *)(mt.state), (mt.next-mt.state + mt.left + 1) * sizeof(long),
                                    mt.next - mt.state,
                                    mt.left,
                                    packOrangeDictionary(self));
}


PyObject *__pickleLoaderRandomGenerator(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, state, next_offset, left)")
{
  PyTypeObject *type;
  int offs;
  int left;
  char *buff;
  int bufsize;
  if (!PyArg_ParseTuple(args, "Os#ii", &type, &buff, &bufsize, &offs, &left))
    return PYNULL;

  TRandomGenerator *rg = new TRandomGenerator;

  cMersenneTwister &mt = rg->mt;
  memcpy(mt.state, buff, bufsize);
  mt.next = mt.state + offs;
  mt.left = left;

  return WrapNewOrange(rg, type);
}


PyObject *RandomGenerator_reset(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([new_seed]) -> None")
{ PyTRY
    int seed = numeric_limits<int>::min();
    if (!PyArg_ParseTuple(args, "|i:RandomGenerator.reset", &seed))
      return PYNULL;

    if (seed != numeric_limits<int>::min())
      SELF_AS(TRandomGenerator).initseed = seed;

    SELF_AS(TRandomGenerator).reset();
    RETURN_NONE; 
  PyCATCH
}

PyObject *RandomGenerator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("() -> 32-bit random int")
{ PyTRY
    NO_KEYWORDS

    if (args) {
      if (PyTuple_Size(args) == 1) {
        return PyInt_FromLong((long)SELF_AS(TRandomGenerator).randlong(PyInt_AsLong(PyTuple_GET_ITEM(args, 0))));
      }
      PYERROR(PyExc_TypeError, "zero or one argument expected", PYNULL);
    }
    
    return PyInt_FromLong((long)SELF_AS(TRandomGenerator)());
  PyCATCH
}
  

PyObject *stdRandomGenerator()
{ return WrapOrange(globalRandom); }

PYCONSTANTFUNC(globalRandom, stdRandomGenerator)


/* ************ EXAMPLE GENERATOR ************ */

TFiletypeDefinition::TFiletypeDefinition(const char *an, PyObject *al, PyObject *as)
: name(an),
  loader(al),
  saver(as)
{
  if (loader == Py_None)
    loader = PYNULL;
  else
    Py_INCREF(loader);

  if (saver == Py_None)
    saver = PYNULL;
  else
    Py_INCREF(saver);
}


TFiletypeDefinition::TFiletypeDefinition(const TFiletypeDefinition &other)
: name(other.name),
  extensions(other.extensions),
  loader(other.loader),
  saver(other.saver)
{
  Py_XINCREF(loader);
  Py_XINCREF(saver);
}


TFiletypeDefinition::~TFiletypeDefinition()
{
  Py_XDECREF(loader);
  Py_XDECREF(saver);
}


vector<TFiletypeDefinition> filetypeDefinitions;

/* lower case to avoid any ambiguity problems (don't know how various compilers can react when
   registerFiletype is cast by the code produced by pyxtract */
ORANGE_API void registerFiletype(const char *name, const vector<string> &extensions, PyObject *loader, PyObject *saver)
{
  TFiletypeDefinition ftd(name, loader, saver);
  ftd.extensions = extensions;
  filetypeDefinitions.push_back(ftd);
}

bool fileExists(const string &s);
const char *getExtension(const char *name);


vector<TFiletypeDefinition>::iterator findFiletypeByExtension(const char *name, bool needLoader, bool needSaver, bool exhaustive)
{
  const char *extension = getExtension(name);

  if (extension) {
    ITERATE(vector<TFiletypeDefinition>, fi, filetypeDefinitions)
      if ((!needLoader || (*fi).loader) && (!needSaver || (*fi).saver))
        ITERATE(TStringList, ei, (*fi).extensions)
          if (*ei == extension)
            return fi;
  }

  else if (exhaustive) {
    ITERATE(vector<TFiletypeDefinition>, fi, filetypeDefinitions)
      if ((!needLoader || (*fi).loader) && (!needSaver || (*fi).saver))
        ITERATE(TStringList, ei, (*fi).extensions)
          if (fileExists(name + *ei))
            return fi;
  }

  return filetypeDefinitions.end();
}


PyObject *registerFileType(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(name, extensions, loader, saver) -> None")
{
  char *name;
  PyObject *pyextensions, *loader, *saver;
  if (!PyArg_ParseTuple(args, "sOOO:registerFiletype", &name, &loader, &saver, &pyextensions))
    return PYNULL;

  TFiletypeDefinition ftd(name, loader, saver);

  if (PyString_Check(pyextensions))
    ftd.extensions.push_back(PyString_AsString(pyextensions));
  else {
    PStringList extensions = PStringList_FromArguments(pyextensions);
    if (!extensions)
      return PYNULL;
    ftd.extensions = extensions.getReference();
  }
 
  vector<TFiletypeDefinition>::iterator fi(filetypeDefinitions.begin()), fe(filetypeDefinitions.begin());
  for(; (fi != fe) && ((*fi).name != name); fi++);

  if (fi==fe)
    filetypeDefinitions.push_back(ftd);
  else
    *fi = ftd;

  RETURN_NONE;
}


extern char *fileTypes[][2];

PyObject *getRegisteredFileTypes(PyObject *, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> ((extension, description, loader, saver), ...)")
{
  char *(*t)[2] = fileTypes;
  while(**t)
    t++;

  int builtIns = t-fileTypes;
  int i = 0;
  PyObject *types = PyTuple_New(builtIns + filetypeDefinitions.size());
  for(t = fileTypes; **t; t++)
    PyTuple_SetItem(types, i++, Py_BuildValue("ss", (*t)[0], (*t)[1]));

  ITERATE(vector<TFiletypeDefinition>, fi, filetypeDefinitions) {
    string exts;
    ITERATE(TStringList, ei, (*fi).extensions)
      exts += (exts.size() ? " *" : "*") + *ei;

    PyObject *ploader = (*fi).loader, *psaver = (*fi).saver;
    if (!ploader) {
      ploader = Py_None;
      Py_INCREF(Py_None);
    }
    if (!psaver) {
      psaver = Py_None;
      Py_INCREF(Py_None);
    }
    PyTuple_SetItem(types, i++, Py_BuildValue("ssOO", (*fi).name.c_str(), exts.c_str(), ploader, psaver));
  }

  return types;
}

#include "examplegen.hpp"
#include "table.hpp"
#include "filter.hpp"


PyObject *loadDataByPython(PyTypeObject *type, char *filename, PyObject *argstuple, PyObject *keywords, bool exhaustiveFilesearch, bool &fileFound)
{
  vector<TFiletypeDefinition>::iterator fi = findFiletypeByExtension(filename, true, false, exhaustiveFilesearch);
  fileFound = fi!=filetypeDefinitions.end();

  if (!fileFound) 
    return PYNULL;

  PyObject *res = PyObject_Call((*fi).loader, argstuple, keywords);
  if (!res)
    throw pyexception();
  if (res == Py_None)
    return res;

  bool gotTuple = PyTuple_Check(res);
  PyObject *res1 = gotTuple ? PyTuple_GET_ITEM(res, 0) : res;
    
  if (PyOrExampleTable_Check(res1))
    return res;

  PExampleGenerator gen;
  if (!exampleGenFromParsedArgs(res1, gen)) {
    Py_DECREF(res);
    return PYNULL;
  }

  TExampleTable *table = gen.AS(TExampleTable);
  if (!table) {
    Py_DECREF(res);
    return PYNULL;
  }

  if (gotTuple) {
    PyObject *nres = PyTuple_New(PyTuple_Size(res));
    PyTuple_SetItem(nres, 0, WrapNewOrange(table, type));
    for(int i = 1; i < PyTuple_Size(res); i++)
      PyTuple_SetItem(nres, i, PyTuple_GET_ITEM(res, i));
      
    Py_DECREF(res);
    return nres;
  }
  else {
    Py_DECREF(res);
    return WrapNewOrange(table, type);
  }
}

bool readUndefinedSpecs(PyObject *keyws, char *&DK, char *&DC);


bool readBoolFlag(PyObject *keywords, char *flag)
{
  PyObject *pyflag = keywords ? PyDict_GetItemString(keywords, flag) : PYNULL;
  return pyflag && PyObject_IsTrue(pyflag);
}

bool hasFlag(PyObject *keywords, char *flag)
{
  return keywords && (PyDict_GetItemString(keywords, flag) != PYNULL);
}


TExampleTable         *readTable(char *filename, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, const char *DK, const char *DC, bool noExcOnUnknown = false, bool noCodedDiscrete = false, bool noClass = false);
TExampleGenerator *readGenerator(char *filename, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, const char *DK, const char *DC, bool noExcOnUnknown = false, bool noCodedDiscrete = false, bool noClass = false);

PyObject *encodeStatus(const vector<int> &status);
PyObject *encodeStatus(const vector<pair<int, int> > &metaStatus);

char *obsoleteFlags[] = {"dontCheckStored", "dontStore", "use", "useMetas", "domain", 0 };


PyObject *loadDataFromFileNoSearch(PyTypeObject *type, char *filename, PyObject *argstuple, PyObject *keywords, bool generatorOnly = false)
{
  PyObject *res;
  
  bool pythonFileFound;
  res = loadDataByPython(type, filename, argstuple, keywords, false, pythonFileFound);
  if (res) {
    if (res != Py_None) {
      if (!PyTuple_Check(res))
        return res;

      PyObject *pygen = PyTuple_GetItem(res, 0);
      Py_INCREF(pygen);
      
      if (PyTuple_Size(res) >= 2)
        Orange_setattrDictionary((TPyOrange *)pygen, "attributeLoadStatus", PyTuple_GET_ITEM(res, 1), false);
      if (PyTuple_Size(res) >= 3)
        Orange_setattrDictionary((TPyOrange *)pygen, "metaAttributeLoadStatus", PyTuple_GET_ITEM(res, 2), false);
      return pygen;
    }
      
    else
      Py_DECREF(Py_None);
  }

  PyErr_Clear();

  for(char * const *of = obsoleteFlags; *of; of++)
    if (hasFlag(keywords, *of))
      raiseWarning("flag '%s' is not supported any longer", *of);

  int createNewOn = TVariable::Incompatible;
  if (hasFlag(keywords, "createNewOn"))
    convertFromPython(PyDict_GetItemString(keywords, "createNewOn"), createNewOn);

  char *DK = NULL, *DC = NULL;
  if (!readUndefinedSpecs(keywords, DK, DC))
    return PYNULL;

  char *errs = NULL;
  vector<int> status;
  vector<pair<int, int> > metaStatus;
  try {
    TExampleGenerator *generator = 
      generatorOnly ? readGenerator(filename, createNewOn, status, metaStatus, DK, DC, false, readBoolFlag(keywords, "noCodedDiscrete"), readBoolFlag(keywords, "noClass"))
                    : readTable(filename, createNewOn, status, metaStatus, DK, DC, false, readBoolFlag(keywords, "noCodedDiscrete"), readBoolFlag(keywords, "noClass"));
    if (generator) {
      PyObject *pygen = WrapNewOrange(generator, type);
      PyObject *pystatus = encodeStatus(status);
      PyObject *pymetastatus = encodeStatus(metaStatus);
      Orange_setattrDictionary((TPyOrange *)pygen, "attributeLoadStatus", pystatus, false);
      Orange_setattrDictionary((TPyOrange *)pygen, "metaAttributeLoadStatus", pymetastatus, false);
      Py_DECREF(pystatus);
      Py_DECREF(pymetastatus);
      return pygen;
    }
  }
  catch (mlexception err) { 
    errs = strdup(err.what());
  }

  res = loadDataByPython(type, filename, argstuple, keywords, true, pythonFileFound);
  if (res)
    return res;

  if (pythonFileFound) {
    PYERROR(PyExc_SystemError, "cannot load the file", PYNULL);
  }
  else {
    PyErr_SetString(PyExc_SystemError, errs);
    free(errs);
    return PYNULL;
  }
}

PyObject *loadDataFromFilePath(PyTypeObject *type, char *filename, PyObject *argstuple, PyObject *keywords, bool generatorOnly, const char *path)
{
  if (!path) {
    return NULL;
  }
  
  #if defined _WIN32
  const char sep = ';';
  const char pathsep = '\\';
  #else
  const char sep = ':';
  const char pathsep = '/';
  #endif
  const int flen = strlen(filename);

  for(const char *pi = path, *pe=pi; *pi; pi = pe+1) {
    for(pe = pi; *pe && *pe != sep; pe++);
    const int plen = pe-pi;
     char *npath = strncpy(new char[plen+flen+2], pi, pe-pi);
    if (!plen || (pe[plen] != pathsep)) {
      npath[plen] = pathsep;
      strcpy(npath+plen+1, filename);
    }
    else {
      strcpy(npath+plen, filename);
    }
    PyObject *res = loadDataFromFileNoSearch(type, npath, argstuple, keywords, generatorOnly);
    PyErr_Clear();
    if (res) {
      return res;
    }
    if (!*pe)
      break;
  }
  
  return NULL;
}

PyObject *loadDataFromFile(PyTypeObject *type, char *filename, PyObject *argstuple, PyObject *keywords, bool generatorOnly = false)
{
  PyObject *ptype, *pvalue, *ptraceback;
  PyObject *res;
  
  res = loadDataFromFileNoSearch(type, filename, argstuple, keywords, generatorOnly);
  if (res) {
    return res;
  }

  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  
  PyObject *configurationModule = PyImport_ImportModule("orngConfiguration");
  if (configurationModule) {
    PyObject *datasetsPath = PyDict_GetItemString(PyModule_GetDict(configurationModule), "datasetsPath");
    if (datasetsPath)
      res = loadDataFromFilePath(type, filename, argstuple, keywords, generatorOnly, PyString_AsString(datasetsPath));
      Py_DECREF(configurationModule);
  }
  else {
    PyErr_Clear();
  }

  if (!res) {
    res = loadDataFromFilePath(type, filename, argstuple, keywords, generatorOnly, getenv("ORANGE_DATA_PATH"));
  }
  
  if (res) {
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    return res;
  }
  
  PyErr_Restore(ptype, pvalue, ptraceback);
  return PYNULL; 
}
  

int pt_ExampleGenerator(PyObject *args, void *egen)
{ 
  *(PExampleGenerator *)(egen) = PyOrExampleGenerator_Check(args) ? PyOrange_AsExampleGenerator(args)
                                                                  : PExampleGenerator(readListOfExamples(args));

  if (!*(PExampleGenerator *)(egen))
    PYERROR(PyExc_TypeError, "invalid example generator", 0)
  else
    return 1;
}


static PDomain ptd_domain;

int ptdf_ExampleGenerator(PyObject *args, void *egen)
{ 
  egen = NULL;

  try {
    if (PyOrExampleGenerator_Check(args)) {
      PExampleGenerator gen = PyOrange_AsExampleGenerator(args);
      if (gen->domain == ptd_domain)
        *(PExampleGenerator *)(egen) = gen;
      else
        *(PExampleGenerator *)(egen) = mlnew TExampleTable(ptd_domain, gen);
    }
    else
      *(PExampleGenerator *)(egen) = PExampleGenerator(readListOfExamples(args, ptd_domain));

    ptd_domain = PDomain();

    if (!*(PExampleGenerator *)(egen))
      PYERROR(PyExc_TypeError, "invalid example generator", 0)
    else
      return 1;
  }

  catch (...) {
    ptd_domain = PDomain();
    throw;
  }
}


converter ptd_ExampleGenerator(PDomain domain)
{ 
  ptd_domain = domain;
  return ptdf_ExampleGenerator;
}


CONSTRUCTOR_KEYWORDS(ExampleGenerator, "domain use useMetas dontCheckStored dontStore filterMetas DC DK NA noClass noCodedDiscrete")

// Class ExampleGenerator is abstract in C++; this constructor returns the derived classes

NO_PICKLE(ExampleGenerator)

PyObject *ExampleGenerator_new(PyTypeObject *type, PyObject *argstuple, PyObject *keywords) BASED_ON(Orange, "(filename)")
{  
  PyTRY
    char *filename = NULL;
    if (PyArg_ParseTuple(argstuple, "s", &filename))
      return loadDataFromFile(type, filename, argstuple, keywords, true);
    else
      return PYNULL;
  PyCATCH;
}


PExampleGenerator exampleGenFromParsedArgs(PyObject *args)
{
 if (PyOrOrange_Check(args)) {
   if (PyOrExampleGenerator_Check(args))
      return PyOrange_AsExampleGenerator(args);
    else 
      PYERROR(PyExc_TypeError, "example generator expected", NULL);
  }
  return PExampleGenerator(readListOfExamples(args));
}


PExampleGenerator exampleGenFromArgs(PyObject *args, int &weightID)
{ 
  PyObject *examples, *pyweight = NULL;
  if (!PyArg_UnpackTuple(args, "exampleGenFromArgs", 1, 2, &examples, &pyweight))
    return PExampleGenerator();

  PExampleGenerator egen = exampleGenFromParsedArgs(examples);
  if (!egen || !weightFromArg_byDomain(pyweight, egen->domain, weightID))
    return PExampleGenerator();

  return egen;
}


PExampleGenerator exampleGenFromArgs(PyObject *args)
{ 
  if (PyTuple_GET_SIZE(args) != 1)
    PYERROR(PyExc_TypeError, "exampleGenFromArgs: examples expected", PExampleGenerator())

  return exampleGenFromParsedArgs(PyTuple_GET_ITEM(args, 0));
}


PyObject *ExampleGenerator_native(PyObject *self, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "([nativity, tuple=]) -> examples")
{ PyTRY
    bool tuples = false;
    PyObject *forDC = NULL;
    PyObject *forDK = NULL;
    PyObject *forSpecial = NULL;
    if (keyws) {
      PyObject *pytuples = PyDict_GetItemString(keyws, "tuple");
      tuples = pytuples && (PyObject_IsTrue(pytuples) != 0);

      forDC = PyDict_GetItemString(keyws, "substituteDC");
      forDK = PyDict_GetItemString(keyws, "substituteDK");
      forSpecial = PyDict_GetItemString(keyws, "substituteOther");
    }

    int natvt=2;
    if (args && !PyArg_ParseTuple(args, "|i", &natvt) || ((natvt>=2)))
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);
    CAST_TO(TExampleGenerator, eg);

    PyObject *list=PyList_New(0);
    EITERATE(ei, *eg)
      if (natvt<=1) {
        PyObject *obj=convertToPythonNative(*ei, natvt, tuples, forDK, forDC, forSpecial);
        PyList_Append(list, obj);
        Py_DECREF(obj);
      }
        // What happens with: convertToPythonNative((*ei, natvt, tuples))? Funny.
      else {
        PyObject *example=Example_FromExampleCopyRef(*ei);
        if (!example) {
          PyMem_DEL(list);
          PYERROR(PyExc_SystemError, "out of memory", PYNULL);
        }      
        PyList_Append(list, example);
        Py_DECREF(example);
      }

    return list;
  PyCATCH
}


PVariableFilterMap PVariableFilterMap_FromArguments(PyObject *arg);

int VariableFilterMap_setitemlow(TVariableFilterMap *aMap, PVariable var, PyObject *pyvalue);

inline PVariableFilterMap sameValuesMap(PyObject *dict, PDomain dom)
{ TVariableFilterMap *vfm = mlnew TVariableFilterMap;
  PVariableFilterMap wvfm = vfm;

  Py_ssize_t pos=0;
  PyObject *pykey, *pyvalue;
  while (PyDict_Next(dict, &pos, &pykey, &pyvalue)) {
    PVariable var = varFromArg_byDomain(pykey, dom, true);
    if (!var || (VariableFilterMap_setitemlow(vfm, var, pyvalue) < 0))
      return PVariableFilterMap();
  }

  return wvfm;
}

inline PPreprocessor pp_sameValues(PyObject *dict, PDomain dom)
{ PVariableFilterMap vfm = sameValuesMap(dict, dom);
  return vfm ? mlnew TPreprocessor_take(vfm) : PPreprocessor();
}

inline PFilter filter_sameValues(PyObject *dict, PDomain domain, PyObject *kwds = PYNULL)
{ PVariableFilterMap svm = sameValuesMap(dict, domain);
  if (!svm)
    return PFilter();

  PyObject *pyneg = kwds ? PyDict_GetItemString(kwds, "negate") : NULL;
  return TPreprocessor_take::constructFilter(svm, domain, true, pyneg && PyObject_IsTrue(pyneg)); 
}

PyObject *applyPreprocessor(PPreprocessor preprocessor, PExampleGenerator gen, bool weightGiven, int weightID)
{ if (!preprocessor)
    return PYNULL;

  int newWeight;
  PExampleGenerator newGen = preprocessor->call(gen, weightID, newWeight);
  return weightGiven ? Py_BuildValue("Ni", WrapOrange(newGen), newWeight) : WrapOrange(newGen);
}



PyObject *applyFilter(PFilter filter, PExampleGenerator gen, bool weightGiven, int weightID);

PyObject *ExampleGenerator_select(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
/* This function is a spaghetti for compatibility reasons. Most of its functionality
   has been moved to specialized functions (changeDomain, filter). The only two
   functions that 'select' should be used for is selection of examples by vector
   of bools or indices (LongList) */
{ 
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    /* ***** SELECTION BY VALUES OF ATTRIBUTES GIVEN AS KEYWORDS ***** */
    /* Deprecated: use method 'filter' instead */
    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyPreprocessor(pp_sameValues(keywords, eg->domain), weg, false, 0);
    }


    PyObject *mplier;
    PyObject *pyweight = NULL;
    if (PyArg_ParseTuple(args, "O|O", &mplier, &pyweight)) {
      PyObject *pyneg = keywords ? PyDict_GetItemString(keywords, "negate") : NULL;
      bool negate = pyneg && PyObject_IsTrue(pyneg);
      bool secondArgGiven = (pyweight != NULL);

      /* ***** SELECTION BY VECTOR OF BOOLS ****** */
      if (PyList_Check(mplier) && PyList_Size(mplier) && PyInt_Check(PyList_GetItem(mplier, 0))) {
        int nole = PyList_Size(mplier);
        
        TExampleTable *newTable = mlnew TExampleTable(eg->domain);
        PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error
        int i = 0;

        if (secondArgGiven) {
          if (PyInt_Check(pyweight)) {
            int compVal = (int)PyInt_AsLong(pyweight);
            TExampleIterator ei = eg->begin();
            for(; ei && (i<nole); ++ei) {
              PyObject *lel = PyList_GetItem(mplier, i++);
              if (!PyInt_Check(lel))
                break;

              if (negate != (PyInt_AsLong(lel)==compVal))
                newTable->addExample(*ei);
            }

            if ((i==nole) && !ei)
              return WrapOrange(newGen);
          }
        }
        else {
          TExampleIterator ei = eg->begin();
          for(; ei && (i<nole); ++ei) {
            PyObject *lel = PyList_GetItem(mplier, i++);
            if (negate != (PyObject_IsTrue(lel) != 0))
              newTable->addExample(*ei);
          }

          if ((i==nole) && !ei)
            return WrapOrange(newGen);
        }
      }

      PyErr_Clear();


      /* ***** SELECTION BY LONGLIST ****** */
      if (PyOrLongList_Check(mplier)) {
        PLongList llist = PyOrange_AsLongList(mplier);
        TLongList::iterator lli(llist->begin()), lle(llist->end());
        
        TExampleTable *newTable = mlnew TExampleTable(eg->domain);
        PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

        TExampleIterator ei = eg->begin();

        if (secondArgGiven) {
          if (!PyInt_Check(pyweight))
            PYERROR(PyExc_AttributeError, "example selector must be an integer", PYNULL);

          int compVal = (int)PyInt_AsLong(pyweight);
          for(; ei && (lli!=lle); ++ei, lli++)
            if (negate != (*lli==compVal))
              newTable->addExample(*ei);
        }
        else {
          for(; ei && (lli != lle); ++ei, lli++)
            if (negate != (*lli != 0))
              newTable->addExample(*ei);
        }

        if ((lli==lle) && !ei)
          return WrapOrange(newGen);

        PYERROR(PyExc_IndexError, "ExampleGenerator.select: invalid list size", PYNULL)
      }

      PyErr_Clear();

      /* ***** CHANGING DOMAIN ***** */
      /* Deprecated: use method 'translate' instead. */
      if (PyOrDomain_Check(mplier)) {
        PyObject *wrappedGen = WrapOrange(PExampleTable(mlnew TExampleTable(PyOrange_AsDomain(mplier), weg)));
        return secondArgGiven ? Py_BuildValue("NO", wrappedGen, pyweight) : wrappedGen;
      }


      /* ***** SELECTION BY VECTOR OF NAMES, INDICES AND VARIABLES ****** */
      /* Deprecated: use method 'translate' instead. */
      TVarList attributes;
      if (varListFromDomain(mplier, eg->domain, attributes, true, false)) {
        PDomain newDomain;
        TVarList::iterator vi, ve;
        for(vi = attributes.begin(), ve = attributes.end(); (vi!=ve) && (*vi!=eg->domain->classVar); vi++);
        if (vi==ve)
          newDomain = mlnew TDomain(PVariable(), attributes);
        else {
          attributes.erase(vi);
          newDomain = mlnew TDomain(eg->domain->classVar, attributes);
        }

        PyObject *wrappedGen = WrapOrange(PExampleTable(mlnew TExampleTable(newDomain, weg)));
        return secondArgGiven ? Py_BuildValue("NO", wrappedGen, pyweight) : wrappedGen;
      }

      PyErr_Clear();


      /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS DICTIONARY ***** */
      /* Deprecated: use method 'filter' instead. */
      if (PyDict_Check(mplier)) {
        int weightID;
        if (weightFromArg_byDomain(pyweight, eg->domain, weightID))
          return applyFilter(filter_sameValues(mplier, eg->domain), weg, secondArgGiven, weightID);
      }


      /* ***** PREPROCESSING ***** */
      /* Deprecated: call preprocessor instead. */
      if (PyOrPreprocessor_Check(mplier)) {
        int weightID;
        if (weightFromArg_byDomain(pyweight, eg->domain, weightID)) {

          PExampleGenerator res;
          int newWeight;
          PyTRY
            NAME_CAST_TO(TPreprocessor, mplier, pp);
            if (!pp)
              PYERROR(PyExc_TypeError, "invalid object type (preprocessor announced, but not passed)", PYNULL)
            res = (*pp)(weg, weightID, newWeight);
          PyCATCH

          return secondArgGiven ? Py_BuildValue("Ni", WrapOrange(res), newWeight) : WrapOrange(res);
        }
      }

      /* ***** APPLY FILTER ***** */
      /* Deprecated: use method 'filter' instead. */
      if (PyOrFilter_Check(mplier)) {
        int weightID;
        if (weightFromArg_byDomain(pyweight, eg->domain, weightID))
          return applyFilter(PyOrange_AsFilter(mplier), weg, secondArgGiven, weightID);
      }
    }
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);
  PyCATCH
}


PyObject *ExampleGenerator_filter(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(list-of-attribute-conditions | filter)")
{
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyFilter(filter_sameValues(keywords, eg->domain, keywords), weg, false, 0);
    }

    if (PyTuple_Size(args)==1) {
      PyObject *arg = PyTuple_GET_ITEM(args, 0);

      if (PyDict_Check(arg))
        return applyFilter(filter_sameValues(arg, eg->domain, keywords), weg, false, 0);

      if (PyOrFilter_Check(arg))
          return applyFilter(PyOrange_AsFilter(arg), weg, false, 0);
    }

    PYERROR(PyExc_AttributeError, "ExampleGenerator.filter expects a list of conditions or orange.Filter", PYNULL)
  PyCATCH
}


PyObject *ExampleGenerator_translate(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "translate(domain | list-of-attributes) -> ExampleTable")
{
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    PDomain domain;
    if (PyArg_ParseTuple(args, "O&", cc_Domain, &domain))
      return WrapOrange(PExampleTable(mlnew TExampleTable(domain, weg)));

    PyObject *pargs, *guard = NULL;
    if (args && (PyTuple_Size(args)==1))
      pargs = guard = PyTuple_GET_ITEM(args, 0);
    else
      pargs = args;

    /* ***** SELECTION BY VECTOR OF NAMES, INDICES AND VARIABLES ****** */
    TVarList attributes;
    if (varListFromDomain(pargs, eg->domain, attributes, true, false)) {
      PDomain newDomain;
      TVarList::iterator vi, ve;
      for(vi = attributes.begin(), ve = attributes.end(); (vi!=ve) && (*vi!=eg->domain->classVar); vi++);
      if (vi==ve)
        newDomain = mlnew TDomain(PVariable(), attributes);
      else {
        attributes.erase(vi);
        newDomain = mlnew TDomain(eg->domain->classVar, attributes);
      }

      Py_XDECREF(guard);
      return WrapOrange(PExampleTable(mlnew TExampleTable(newDomain, weg)));
    }

    PYERROR(PyExc_AttributeError, "ExampleGenerator.translate expects a list of attributes or orange.Domain", PYNULL)
 PyCATCH
}


PyObject *multipleSelectLow(TPyOrange *self, PyObject *pylist, bool reference)
{ PyTRY
    if (!PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
    
    vector<int> indices;
    int i, sze = PyList_Size(pylist);
    for(i = 0; i<sze; i++) {
      PyObject *lel = PyList_GetItem(pylist, i);
      if (!PyInt_Check(lel))
        PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
      indices.push_back(int(PyInt_AsLong(lel)));
    }
    sort(indices.begin(), indices.end());

    CAST_TO(TExampleGenerator, eg);
    TExampleTable *newTable = reference ? mlnew TExampleTable(eg, (int)0)
                                        : mlnew TExampleTable(eg->domain);
    PExampleGenerator newGen(newTable);

    TExampleGenerator::iterator ei(eg->begin());
    vector<int>::iterator ii(indices.begin()), iie(indices.end());
    i = 0;
    while(ei && (ii!=iie)) {
      if (*ii == i) {
        newTable->addExample(*ei);
        ii++;
      }
      else {
        i++;
        ++ei;
      }
    }

    if (ii!=iie)
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);

    return WrapOrange(newGen);
  PyCATCH
}


PyObject *ExampleGenerator_getitems(TPyOrange *self, PyObject *pylist)  PYARGS(METH_O, "(indices) -> ExampleTable")
{ return multipleSelectLow(self, pylist, false); }


PyObject *ExampleGenerator_checksum(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "() -> crc")
{ PyTRY
    return PyInt_FromLong(SELF_AS(TExampleGenerator).checkSum()); 
  PyCATCH
}


const char *getExtension(const char *name);

PyObject *saveTabDelimited(PyObject *, PyObject *args, PyObject *keyws);
PyObject *saveC45(PyObject *, PyObject *args);
PyObject *saveTxt(PyObject *, PyObject *args, PyObject *keyws);
PyObject *saveCsv(PyObject *, PyObject *args, PyObject *keyws);
PyObject *saveBasket(PyObject *, PyObject *args);

PyObject *ExampleGenerator_save(PyObject *self, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "(filename) -> None")
{
  char *filename;
  if (!PyArg_ParseTuple(args, "s:ExampleGenerator.save", &filename))
    return PYNULL;

  const char *extension = getExtension(filename);
  if (!extension)
    PYERROR(PyExc_TypeError, "file name must have an extension", PYNULL);


  PyObject *newargs = PyTuple_New(PyTuple_Size(args) + 1);
  PyObject *el;

  el = PyTuple_GET_ITEM(args, 0);
  Py_INCREF(el);
  PyTuple_SetItem(newargs, 0, el);

  Py_INCREF(self);
  PyTuple_SetItem(newargs, 1, self);

  for(int i = 1, e = PyTuple_Size(args); i < e; i++) {
    el = PyTuple_GET_ITEM(args, i);
    Py_INCREF(el);
    PyTuple_SetItem(newargs, i+1, el);
  }

  PyObject *res = PYNULL;

  vector<TFiletypeDefinition>::iterator fi = findFiletypeByExtension(filename, false, true, false);
  if (fi != filetypeDefinitions.end())
    res = PyObject_Call((*fi).saver, newargs, keyws);
  else if (!strcmp(extension, ".tab"))
    res = saveTabDelimited(NULL, newargs, keyws);
  else if (!strcmp(extension, ".txt"))
    res = saveTxt(NULL, newargs, keyws);
  else if (!strcmp(extension, ".csv"))
    res = saveCsv(NULL, newargs, keyws);
  else if (!strcmp(extension, ".names") || !strcmp(extension, ".data") || !strcmp(extension, ".test"))
    res = saveC45(NULL, newargs);
  else if (!strcmp(extension, ".basket"))
    res = saveBasket(NULL, newargs);
  else
    PyErr_Format(PyExc_AttributeError, "unknown file format (%s)", extension);

  Py_DECREF(newargs);
  return res;
}


PyObject *ExampleGenerator_weight(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(weightID)")
{
  PyObject *pyw = PYNULL;
  if (!PyArg_ParseTuple(args, "|O:ExampleGenerator.weight", &pyw))
    return PYNULL;

  CAST_TO(TExampleGenerator, egen)
  if (!pyw)
    return PyInt_FromLong(egen->numberOfExamples());

  int weightID;
  if (!varNumFromVarDom(pyw, egen->domain, weightID))
    return PYNULL;

  float weight = 0.0;
  PEITERATE(ei, egen)
    weight += WEIGHT(*ei);

  return PyFloat_FromDouble(weight);
}


PExampleGeneratorList PExampleGeneratorList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::P_FromArguments(arg); }
PyObject *ExampleGeneratorList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_FromArguments(type, arg); }
PyObject *ExampleGeneratorList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ExampleGenerator>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_new(type, arg, kwds); }
PyObject *ExampleGeneratorList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_getitem(self, index); }
int       ExampleGeneratorList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_setitem(self, index, item); }
PyObject *ExampleGeneratorList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_getslice(self, start, stop); }
int       ExampleGeneratorList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_setslice(self, start, stop, item); }
int       ExampleGeneratorList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_len(self); }
PyObject *ExampleGeneratorList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_richcmp(self, object, op); }
PyObject *ExampleGeneratorList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_concat(self, obj); }
PyObject *ExampleGeneratorList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_repeat(self, times); }
PyObject *ExampleGeneratorList_str(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_str(self); }
PyObject *ExampleGeneratorList_repr(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_str(self); }
int       ExampleGeneratorList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_contains(self, obj); }
PyObject *ExampleGeneratorList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ExampleGenerator) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_append(self, item); }
PyObject *ExampleGeneratorList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_extend(self, obj); }
PyObject *ExampleGeneratorList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> int") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_count(self, obj); }
PyObject *ExampleGeneratorList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ExampleGeneratorList") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_filter(self, args); }
PyObject *ExampleGeneratorList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> int") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_index(self, obj); }
PyObject *ExampleGeneratorList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_insert(self, args); }
PyObject *ExampleGeneratorList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_native(self); }
PyObject *ExampleGeneratorList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ExampleGenerator") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_pop(self, args); }
PyObject *ExampleGeneratorList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_remove(self, obj); }
PyObject *ExampleGeneratorList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_reverse(self); }
PyObject *ExampleGeneratorList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_sort(self, args); }
PyObject *ExampleGeneratorList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, &PyOrExampleGenerator_Type>::_reduce(self); }
  

/* ************ EXAMPLE TABLE ************ */

#include "table.hpp"
#include "numeric_interface.hpp"

TExampleTable *readListOfExamples(PyObject *args)
{ 
  if (isSomeNumeric_wPrecheck(args))
    return readListOfExamples(args, PDomain(), false);

  if (PySequence_Check(args)) {
    int size=PySequence_Size(args);
    if (!size)
      PYERROR(PyExc_TypeError, "can't construct a table from an empty sequence", (TExampleTable *)NULL);

    TExampleTable *table=NULL;
    PyObject *pex = NULL;
 
    try {
      for(int i=0; i<size; i++) {
        PyObject *pex = PySequence_GetItem(args, i);
        if (!pex || !PyOrExample_Check(pex)) {
          Py_XDECREF(pex);
          mldelete table;
          PyErr_Format(PyExc_TypeError, "invalid sequence element at %i", i);
          return NULL;
        }
        if (!i)
          table = mlnew TExampleTable(PyExample_AS_Example(pex)->domain);
        table->addExample(PyExample_AS_ExampleReference(pex));
        Py_DECREF(pex);
        pex = NULL;
      }
    }
    catch (...) {
      delete table;
      Py_XDECREF(pex);
      throw;
    }

    return table;
  }

  PYERROR(PyExc_TypeError, "a list of examples expected", NULL);
}


TExampleTable *readListOfExamples(PyObject *args, PDomain domain, bool filterMetas)
{ 
  PyArrayObject *array = NULL, *mask = NULL;
  
  if (isSomeNumeric_wPrecheck(args)) {
    array = (PyArrayObject *)(args);
  }
  else if (isSomeMaskedNumeric_wPrecheck(args)) {
    array = (PyArrayObject *)(args);
    mask = (PyArrayObject *)PyObject_GetAttrString(args, "mask");
    if (PyBool_Check((PyObject *)mask)) {
      Py_DECREF((PyObject *)mask);
      mask = NULL;
    }
  }
    
  if (array) {  
      if (array->nd != 2)
        PYERROR(PyExc_AttributeError, "two-dimensional array expected for an ExampleTable", NULL);

     PVarList variables;
     TVarList::const_iterator vi, ve;

      if (!domain) {
        TVarList lvariables;
        char vbuf[20];
        for(int i = 0, e = array->dimensions[1]; i < e; i++) {
          sprintf(vbuf, "a%i", i+1);
          lvariables.push_back(mlnew TFloatVariable(vbuf));
        }
        domain = mlnew TDomain(PVariable(), lvariables);
        variables = domain->variables;
        ve = variables->end();
      }

      else {
        if (array->dimensions[1] != domain->variables->size())
          PYERROR(PyExc_AttributeError, "the number of columns in the array doesn't match the number of attributes", NULL);

       variables = domain->variables;
       ve = variables->end();
       for(vi = variables->begin(); vi!=ve; vi++)
				  if (((*vi)->varType != TValue::INTVAR) && ((*vi)->varType != TValue::FLOATVAR))
					  PYERROR(PyExc_TypeError, "cannot read the value of attribute '%s' from an array (unsupported attribute type)", NULL);
			}

      const char arrayType = getArrayType(array);
      if (!strchr(supportedNumericTypes, arrayType)) {
        PyErr_Format(PyExc_AttributeError, "Converting arrays of type '%c' is not supported (use one of '%s')", arrayType, supportedNumericTypes);
        return NULL;
      }
    
      TExampleTable *table = mlnew TExampleTable(domain);
      TExample *nex = NULL;
      table->reserve(array->dimensions[0]);

      const int &strideRow = array->strides[0];
      const int &strideCol = array->strides[1];
      
      // If there's no mask, the mask pointers will equal the data pointer to avoid too many if's
      const int &strideMaskRow = mask ? mask->strides[0] : strideRow;
      const int &strideMaskCol = mask ? mask->strides[1] : strideCol;

      try {
        TExample::iterator ei;
        char *rowPtr = array->data;
        char *maskRowPtr = mask ? mask->data : array->data;

        for(int row = 0, rowe = array->dimensions[0]; row < rowe; row++, rowPtr += strideRow, maskRowPtr += strideMaskRow) {
          char *elPtr = rowPtr;
          char *maskPtr = maskRowPtr;
          TExample *nex = mlnew TExample(domain);

          #define ARRAYTYPE(TYPE) \
            for(ei = nex->begin(), vi = variables->begin(); vi!=ve; vi++, ei++, elPtr += strideCol, maskPtr += strideMaskCol) \
              if ((*vi)->varType == TValue::INTVAR) \
                intValInit(*ei, *(TYPE *)elPtr, mask && !*maskPtr ? valueDK : valueRegular); \
              else \
                floatValInit(*ei, *(TYPE *)elPtr, mask && !*maskPtr ? valueDK : valueRegular); \
            break;

          switch (arrayType) {
            case 'c':
            case 'b': ARRAYTYPE(char)
            case 'B': ARRAYTYPE(unsigned char)
            case 'h': ARRAYTYPE(short)
            case 'H': ARRAYTYPE(unsigned short)
            case 'i': ARRAYTYPE(int)
            case 'I': ARRAYTYPE(unsigned int)
            case 'l': ARRAYTYPE(long)
            case 'L': ARRAYTYPE(unsigned long)

            case 'f':
              for(ei = nex->begin(), vi = variables->begin(); vi!=ve; vi++, ei++, elPtr += strideCol, maskPtr += strideMaskCol)
                if ((*vi)->varType == TValue::INTVAR)
                  intValInit(*ei, int(floor(0.5 + *(float *)elPtr)), mask && !*maskPtr ? valueDK : valueRegular);
                else
                  floatValInit(*ei, *(float *)elPtr, mask && !*maskPtr ? valueDK : valueRegular);
              break;

            case 'd':
              for(ei = nex->begin(), vi = variables->begin(); vi!=ve; vi++, ei++, elPtr += strideCol, maskPtr += strideMaskCol)
                if ((*vi)->varType == TValue::INTVAR)
                  intValInit(*ei, int(floor(0.5 + *(double *)elPtr)), mask && !*maskPtr ? valueDK : valueRegular);
                else
                  floatValInit(*ei, *(double *)elPtr, mask && !*maskPtr ? valueDK : valueRegular);
              break;

          }

          #undef ARRAYTYPE

          table->addExample(nex);
          nex = NULL;
        }
      }
      catch (...) {
        mldelete table;
        mldelete nex;
        throw;
      }

      return table;
  }

  if (PyList_Check(args)) {
    int size=PyList_Size(args);
    if (!size)
      PYERROR(PyExc_TypeError, "can't construct a table from an empty list", (TExampleTable *)NULL);

    TExampleTable *table = mlnew TExampleTable(domain);;
 
    try {
      for(int i=0; i<size; i++) {
        PyObject *pex = PyList_GetItem(args, i);
        if (PyOrExample_Check(pex))
          table->addExample(PyExample_AS_ExampleReference(pex), filterMetas);
        else {
          TExample example(domain);
          if (!convertFromPythonExisting(pex, example)) {
            mldelete table;
            PyObject *type, *value, *tracebk;
            PyErr_Fetch(&type, &value, &tracebk);
            if (type) {
              //PyErr_Restore(type, value, tracebk);
              const char *oldes = PyString_AsString(value);
              PyErr_Format(type, "%s (at example %i)", oldes, i);
              Py_DECREF(type);
              Py_XDECREF(value);
              Py_XDECREF(tracebk);
              return NULL;
            }
          }
          table->addExample(example);
        }
      }

      return table;
    }
    catch (...) {
      mldelete table;
      throw;
    }
  }

  PYERROR(PyExc_TypeError, "invalid arguments", NULL);
}


CONSTRUCTOR_KEYWORDS(ExampleTable, "domain use useMetas dontCheckStored dontStore filterMetas DC DK NA noClass noCodedDiscrete createNewOn")

PyObject *ExampleTable_new(PyTypeObject *type, PyObject *argstuple, PyObject *keywords) BASED_ON(ExampleGenerator, "(filename | domain[, examples] | examples)")
{  
  PyTRY

    char *filename = NULL;
    if (PyArg_ParseTuple(argstuple, "s", &filename))
      return loadDataFromFile(type, filename, argstuple, keywords, false);

    PyErr_Clear();

    PExampleGenerator egen;
    PyObject *args = PYNULL;
    if (PyArg_ParseTuple(argstuple, "O&|O", cc_ExampleTable, &egen, &args))
      return WrapNewOrange(mlnew TExampleTable(egen, !args || (PyObject_IsTrue(args) == 0)), type);

    PyErr_Clear();

    if (PyArg_ParseTuple(argstuple, "O", &args)) {
      if (PyOrDomain_Check(args))
        return WrapNewOrange(mlnew TExampleTable(PyOrange_AsDomain(args)), type);

      TExampleTable *res = readListOfExamples(args);
      if (res)
        return WrapNewOrange(res, type);
      PyErr_Clear();

      // check if it's a list of generators
      if (PyList_Check(args)) {
        TExampleGeneratorList eglist;
        PyObject *iterator = PyObject_GetIter(args);
        PyObject *item = PyIter_Next(iterator);
        for(; item; item = PyIter_Next(iterator)) {
          if (!PyOrExampleGenerator_Check(item)) {
            Py_DECREF(item);
            break;
          }
          eglist.push_back(PyOrange_AsExampleGenerator(item));
          Py_DECREF(item);
        }
        Py_DECREF(iterator);
        if (!item)
          return WrapNewOrange(mlnew TExampleTable(PExampleGeneratorList(eglist)), type);
      }

      PYERROR(PyExc_TypeError, "invalid arguments for constructor (domain or examples or both expected)", PYNULL);
    }

    PyErr_Clear();

    PDomain domain;
    if (PyArg_ParseTuple(argstuple, "O&O", cc_Domain, &domain, &args)) {
      bool filterMetas = readBoolFlag(keywords, "filterMetas");

      if (PyOrExampleGenerator_Check(args))
        return WrapNewOrange(mlnew TExampleTable(domain, PyOrange_AsExampleGenerator(args), filterMetas), type);
      else {
        TExampleTable *res = readListOfExamples(args, domain, filterMetas);
        return res ? WrapNewOrange(res, type) : PYNULL;
      }
    }

    PYERROR(PyExc_TypeError, "invalid arguments for ExampleTable.__init__", PYNULL);

  PyCATCH
}


PyObject *ExampleTable__reduce__(PyObject *self)
{
  CAST_TO(TExampleTable, table)

  if (!table->ownsExamples || table->lock) {
    PExampleTable lock = table->lock;
    TCharBuffer buf(1024);
    const int lockSize = lock->size();
    buf.writeInt(table->size());
    PEITERATE(ei, table) {
      int index = 0;
      PEITERATE(li, lock) {
        if (&*li == &*ei)
          break;
        index++;
      }
      if (index == lockSize) {
        PYERROR(PyExc_SystemError, "invalid example reference discovered in the table", PYNULL);
      }
        
      buf.writeInt(index);
    }
    
    return Py_BuildValue("O(ONs#)O", getExportedFunction("__pickleLoaderExampleReferenceTable"),
                                      self->ob_type,
                                      WrapOrange(table->lock),
                                      buf.buf, buf.length(),
                                      packOrangeDictionary(self));
  }
  
  else {
    TCharBuffer buf(1024);
    PyObject *otherValues = NULL;

    buf.writeInt(table->size());
    PEITERATE(ei, table)
      Example_pack(*ei, buf, otherValues);

    if (!otherValues) {
      otherValues = Py_None;
      Py_INCREF(otherValues);
    }
    
    return Py_BuildValue("O(ONs#N)O", getExportedFunction("__pickleLoaderExampleTable"),
                                      self->ob_type,
                                      WrapOrange(table->domain),
                                      buf.buf, buf.length(),
                                      otherValues,
                                      packOrangeDictionary(self));
  }
}


PyObject *__pickleLoaderExampleTable(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, domain, packed_values, other_values)")
{
  PyTRY
    PyTypeObject *type;
    PDomain domain;
    char *buf;
    int bufSize;
    PyObject *otherValues;

    if (!PyArg_ParseTuple(args, "OO&s#O:__pickleLoaderExampleTable", &type, cc_Domain, &domain, &buf, &bufSize, &otherValues))
      return NULL;

    TCharBuffer cbuf(buf);
    int otherValuesIndex = 0;

    int noOfEx = cbuf.readInt();
    TExampleTable *newTable = new TExampleTable(domain);
    try {
      newTable->reserve(noOfEx);
      for(int i = noOfEx; i--;)
        Example_unpack(newTable->new_example(), cbuf, otherValues, otherValuesIndex);

      return WrapNewOrange(newTable, type);
    }
    catch (...) {
      delete newTable;
      throw;
    }
  PyCATCH
}


PyObject *__pickleLoaderExampleReferenceTable(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, lockedtable, indices)")
{
  PyTRY
    PyTypeObject *type;
    PExampleTable table;
    char *buf;
    int bufSize;

    if (!PyArg_ParseTuple(args, "OO&s#:__pickleLoaderExampleReferenceTable", &type, cc_ExampleTable, &table, &buf, &bufSize))
      return NULL;

    
    TCharBuffer cbuf(buf);
    int noOfEx = cbuf.readInt();
   
    TExampleTable *newTable = new TExampleTable(table, 1);
    try {
      newTable->reserve(noOfEx);
      for(int i = noOfEx; i--;)
        newTable->addExample(table->at(cbuf.readInt()));
      return WrapNewOrange(newTable, type);
    }
    catch (...) {
      delete newTable;
      throw;
    }
  PyCATCH
}


#define EXAMPLE_LOCK(tab) (((tab)->ownsExamples || !(tab)->lock) ? PExampleGenerator(tab) : (tab)->lock)

PyObject *ExampleTable_native(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "([nativity, tuple=]) -> examples")
{ PyTRY
    int natvt=2;
    if (args && !PyArg_ParseTuple(args, "|i", &natvt) || ((natvt>=3)))
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

    if (natvt<2)
      return ExampleGenerator_native(self, args, keywords);

    CAST_TO(TExampleTable, table);

    PyObject *list=PyList_New(table->numberOfExamples());
    int i=0;
    PExampleGenerator lock = EXAMPLE_LOCK(PyOrange_AsExampleTable(self));
    EITERATE(ei, *table) {
      // here we wrap a reference to example, so we must pass a self's wrapper
      PyObject *example = Example_FromExampleRef(*ei, lock);
      if (!example) {
        PyMem_DEL(list);
        PYERROR(PyExc_SystemError, "out of memory", PYNULL);
      }      
      PyList_SetItem(list, i++, example);
    }

    return list;
  PyCATCH
}

/*
PyTypeObject *gsl_matrixType = NULL;

bool load_gsl()
{
  if (gsl_matrixType)
    return true;

  PyObject *matrixModule = PyImport_ImportModule("pygsl.matrix");
  if (!matrixModule)
    return false;

  gsl_matrixType = (PyTypeObject *)PyDict_GetItemString(PyModule_GetDict(matrixModule), "matrix");
  return gsl_matrixType != NULL;
}
*/


/* Not in .hpp (to be parsed by pyprops) since these only occur in arguments to numpy conversion function */

PYCLASSCONSTANT_INT(ExampleTable, Multinomial_Ignore, 0)
PYCLASSCONSTANT_INT(ExampleTable, Multinomial_AsOrdinal, 1)
PYCLASSCONSTANT_INT(ExampleTable, Multinomial_Error, 2)



PyObject *packMatrixTuple(PyObject *X, PyObject *y, PyObject *w, char *contents)
{
  int left = (*contents && *contents != '/') ? 1 : 0;

  char *cp = strchr(contents, '/');
  if (cp)
    cp++;

  int right = cp ? strlen(cp) : 0;

  PyObject *res = PyTuple_New(left + right);
  if (left) {
    Py_INCREF(X);
    PyTuple_SetItem(res, 0, X);
  }

  if (cp)
    for(; *cp; cp++)
      if ((*cp == 'c') || (*cp == 'C')) {
        Py_INCREF(y);
        PyTuple_SetItem(res, left++, y);
      }
      else {
        Py_INCREF(w);
        PyTuple_SetItem(res, left++, w);
      }

  Py_DECREF(X);
  Py_DECREF(y);
  Py_DECREF(w);
  return res;
}


void parseMatrixContents(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                         bool &hasClass, bool &classVector, bool &weightVector, bool &classIsDiscrete, int &columns,
                         vector<bool> &include);


inline bool storeNumPyValue(double *&p, const TValue &val, signed char *&m, const PVariable attr, const int &row)
{
  if (val.isSpecial()) {
    if (m) {
      *p++ = 0;
      *m++ = 1;
    }
    else {
      PyErr_Format(PyExc_TypeError, "value of attribute '%s' in example '%i' is undefined", attr->name.c_str(), row);
      return false;
    }
  }
    
  else if (val.varType == TValue::FLOATVAR) {
    *p++ = val.floatV;
    if (m)
      *m++ = 0;
  }
  
  else if (val.varType == TValue::INTVAR) {
    *p++ = float(val.intV);
    if (m)
      *m++ = 0;
  }

  else {
    *p++ = ILLEGAL_FLOAT;
    if (m)
      *m++ = 1;
  }
  
  return true;
}


PyObject *ExampleTable_toNumericOrMA(PyObject *self, PyObject *args, PyObject *keywords, PyObject **module, PyObject **maskedArray = NULL)
{
  PyTRY
    prepareNumeric();
    if (!*module || maskedArray && !*maskedArray)
      PYERROR(PyExc_ImportError, "cannot import the necessary numeric module for conversion", PYNULL);

    // These references are all borrowed
    PyObject *moduleDict = PyModule_GetDict(*module);
	  PyObject *mzeros = PyDict_GetItemString(moduleDict, "zeros");
	  if (!mzeros)
	    PYERROR(PyExc_AttributeError, "numeric module has no function 'zeros'", PYNULL);

    char *contents = NULL;
    int weightID = 0;
    int multinomialTreatment = 1;
    if (!PyArg_ParseTuple(args, "|sii:ExampleTable.toNumeric", &contents, &weightID, &multinomialTreatment))
      return PYNULL;

    if (!contents)
      contents = "a/cw";

    PExampleGenerator egen = PyOrange_AsExampleGenerator(self);

    bool hasClass, classVector, weightVector, classIsDiscrete;
    vector<bool> include;
    int columns;
    parseMatrixContents(egen, weightID, contents, multinomialTreatment,
                            hasClass, classVector, weightVector, classIsDiscrete, columns, include);

    int rows = egen->numberOfExamples();
    PVariable classVar = egen->domain->classVar;

    PyObject *X, *y, *w, *mask = NULL, *masky = NULL;
    double *Xp, *yp, *wp;
    signed char *mp = NULL, *mpy = NULL;
    if (columns) {
      X = PyObject_CallFunction(mzeros, "(ii)s", rows, columns, "d");
      if (!X)
        return PYNULL;
        
      Xp = (double *)((PyArrayObject *)X)->data;

      if (maskedArray) {
        mask = PyObject_CallFunction(mzeros, "(ii)s", rows, columns, "b");
        mp = (signed char *)((PyArrayObject *)mask)->data;
      }
    }
    else {
      X = Py_None;
      Py_INCREF(X);
      Xp = NULL;
    }

    if (classVector) {
      y = PyObject_CallFunction(mzeros, "(i)s", rows, "d");
      if (!y)
        return PYNULL;
      yp = (double *)((PyArrayObject *)y)->data;

      if (maskedArray) {
        masky = PyObject_CallFunction(mzeros, "(i)s", rows, "b");
        mpy = (signed char *)((PyArrayObject *)masky)->data;
      }
    }
    else {
      y = Py_None;
      Py_INCREF(y);
      yp = NULL;
    }


    if (weightVector) {
      w = PyObject_CallFunction(mzeros, "(i)s", rows, "d");
      if (!w)
        return PYNULL;
      wp = (double *)((PyArrayObject *)w)->data;
    }
    else {
      w = Py_None;
      Py_INCREF(w);
      wp = NULL;
    }
  
    try {
      int row = 0;
      TExampleGenerator::iterator ei(egen->begin());
      for(; ei; ++ei, row++) {
        int col = 0;
      
        /* This is all optimized assuming that each symbol (A, C, W) only appears once. 
           If it would be common for them to appear more than once, we could cache the
           values, but since this is unlikely, caching would only slow down the conversion */
        for(const char *cp = contents; *cp && (*cp!='/'); cp++) {
          switch (*cp) {
            case 'A':
            case 'a': {
              const TVarList &attributes = egen->domain->attributes.getReference();
              TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
              TExample::iterator eei((*ei).begin());
              vector<bool>::const_iterator bi(include.begin());
              for(; vi != ve; eei++, vi++, bi++)
                if (*bi && !storeNumPyValue(Xp, *eei, mp, *vi, row))
                  return PYNULL;
              break;
            }

            case 'C':
            case 'c': 
              if (hasClass && !storeNumPyValue(Xp, (*ei).getClass(), mp, classVar, row)) 
                return PYNULL;
              break;

            case 'W':
            case 'w': 
              if (weightID)
                *Xp++ = WEIGHT(*ei);
                if (maskedArray)
                  *mp++ = 0;
              break;

            case '0':
              *Xp++ = 0.0;
              if (maskedArray)
                *mp++ = 0;
              break;

            case '1':
              *Xp++ = 1.0;
              if (maskedArray)
                *mp++ = 0;
              break;
          }
        }

        if (yp && !storeNumPyValue(yp, (*ei).getClass(), mpy, classVar, row))
          return PYNULL;

        if (wp)
          *wp++ = WEIGHT(*ei);
      }

      if (maskedArray) {
        PyObject *args, *maskedX = NULL, *maskedy = NULL;

        bool err = false;
        
        if (mask) {
          args = Py_BuildValue("OOiOO", X, Py_None, 1, Py_None, mask);
          maskedX = PyObject_CallObject(*maskedArray, args);
          Py_DECREF(args);
          if (!maskedX) {
            PyErr_Clear();
            args = Py_BuildValue("OOOi", X, mask, Py_None, 1);
            maskedX = PyObject_CallObject(*maskedArray, args);
            Py_DECREF(args);
          }
          err = !maskedX;
        }

        if (!err && masky) {
          args = Py_BuildValue("OOiOO", y, Py_None, 1, Py_None, masky);
          maskedy = PyObject_CallObject(*maskedArray, args);
          Py_DECREF(args);
          if (!maskedy) {
            PyErr_Clear();
            args = Py_BuildValue("OOOi", y, masky, Py_None, 1);
            maskedy = PyObject_CallObject(*maskedArray, args);
            Py_DECREF(args);
          }
          err = !maskedy;
        }

        if (err) {
          Py_DECREF(X);
          Py_DECREF(y);
          Py_DECREF(w);
          Py_XDECREF(maskedX);
          Py_XDECREF(mask);
          Py_XDECREF(masky);
          return PYNULL;
        }

        if (mask) {
          Py_DECREF(X);
          Py_DECREF(mask);
          X = maskedX;
        }

        if (masky) {
          Py_DECREF(y);
          Py_DECREF(masky);
          y = maskedy;
        }
      }

      return packMatrixTuple(X, y, w, contents);
    }
    catch (...) {
      Py_DECREF(X);
      Py_DECREF(y);
      Py_DECREF(w);
      Py_XDECREF(mask);
      Py_XDECREF(masky);
      throw;
    }
  PyCATCH
}


PyObject *ExampleTable_toNumeric(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumeric);
}


PyObject *ExampleTable_toNumericMA(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumeric, &numericMaskedArray);
}

// this is for compatibility
PyObject *ExampleTable_toMA(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericMA(self, args, keywords);
}

PyObject *ExampleTable_toNumarray(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumarray);
}


PyObject *ExampleTable_toNumarrayMA(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumarray, &numarrayMaskedArray);
}

PyObject *ExampleTable_toNumpy(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumpy);
}


PyObject *ExampleTable_toNumpyMA(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "([contents='a/cw'[, weightID=0[, multinomialTreatment=1]]) -> matrix(-ces)")
{
  return ExampleTable_toNumericOrMA(self, args, keywords, &moduleNumpy, &numpyMaskedArray);
}



int ExampleTable_nonzero(PyObject *self)
{ PyTRY
    return SELF_AS(TExampleGenerator).numberOfExamples() ? 1 : 0;
  PyCATCH_1
}

int ExampleTable_len_sq(PyObject *self) 
{ PyTRY
    return SELF_AS(TExampleGenerator).numberOfExamples();
  PyCATCH_1
}


PyObject *ExampleTable_append(PyObject *self, PyObject *args) PYARGS(METH_O, "(example) -> None")
{ PyTRY
    CAST_TO(TExampleTable, table)

    if (table->ownsExamples) {
      if (!convertFromPythonExisting(args, table->new_example())) {
        table->delete_last();
        return PYNULL;
      }
    }
    else {
      if (!PyOrExample_Check(args) || (((TPyExample *)(args))->lock != table->lock))
        PYERROR(PyExc_TypeError, "tables containing references to examples can only append examples from the same table", PYNULL);

      table->addExample(PyExample_AS_ExampleReference(args));
    }
    RETURN_NONE;
  PyCATCH
}

PyObject *ExampleTable_extend(PyObject *self, PyObject *args) PYARGS(METH_O, "(examples) -> None")
{ PyTRY 
    CAST_TO(TExampleTable, table)

    if (PyOrExampleGenerator_Check(args)) {
      PExampleGenerator gen = PyOrange_AsExampleGenerator(args);
      if (args==self) {
        TExampleTable temp(gen, false);
        table->addExamples(PExampleGenerator(temp));
      }
      else {
        if (!table->ownsExamples
              && (table->lock != gen)
              && (!gen.is_derived_from(TExampleTable) || (table->lock != gen.AS(TExampleTable)->lock)))
            PYERROR(PyExc_TypeError, "tables containing references to examples can only extend by examples from the same table", PYNULL);
        table->addExamples(gen);
      }
      RETURN_NONE;
    }

    TExample example(table->domain);
    if (PyList_Check(args)) {
      int i, size = PyList_Size(args);

      // We won't append until we know we can append all
      // (don't want to leave the work half finished)
      if (!table->ownsExamples) {
        for (i = 0; i<size; i++) {
          PyObject *pyex = PyList_GET_ITEM(args, i);
          if (!PyOrExample_Check(pyex) || (((TPyExample *)(pyex))->lock != table->lock))
            PYERROR(PyExc_TypeError, "tables containing references to examples can only extend by examples from the same table", PYNULL);
        }    
      }

      for(i = 0; i<size; i++) {
        PyObject *pex = PyList_GET_ITEM(args, i);
        if (!convertFromPythonExisting(pex, example))
          return PYNULL;

        table->addExample(example);
      }

      RETURN_NONE;
    }


    PYERROR(PyExc_TypeError, "invalid argument for ExampleTable.extend", PYNULL);
  PyCATCH
}


PyObject *ExampleTable_getitem_sq(TPyOrange *self, int idx)
{
  PyTRY
    CAST_TO(TExampleTable, table);

    if (idx<0)
      idx += table->numberOfExamples();

    if ((idx<0) || (idx>=table->numberOfExamples()))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);

    // here we wrap a reference to example, so we must pass self's wrapper
    return Example_FromExampleRef((*table)[idx], EXAMPLE_LOCK(PyOrange_AsExampleTable(self)));
  PyCATCH
}


int ExampleTable_setitem_sq(TPyOrange *self, int idx, PyObject *pex)
{ 
  PyTRY
    CAST_TO_err(TExampleTable, table, -1);

    if (idx>table->numberOfExamples())
      PYERROR(PyExc_IndexError, "index out of range", -1);

    if (!pex) {
      table->erase(idx);
      return 0;
    }

    if (!table->ownsExamples) {
      if (!PyOrExample_Check(pex) || (((TPyExample *)(pex))->lock != table->lock))
        PYERROR(PyExc_TypeError, "tables containing references to examples can contain examples from the same table", -1);

      (*table)[idx] = TExample(table->domain, PyExample_AS_ExampleReference(pex));
      return 0;
    }

    if (PyOrExample_Check(pex)) {
      (*table)[idx] = TExample(table->domain, PyExample_AS_ExampleReference(pex));
      return 0;
    }

    TExample example(table->domain);
    if (convertFromPythonExisting(pex, example)) {
      (*table)[idx] = example;
      return 0;
    }

    PYERROR(PyExc_TypeError, "invalid parameter type (Example expected)", -1)
  PyCATCH_1
}


PyObject *ExampleTable_getslice(TPyOrange *self, int start, int stop)
{ 
  PyTRY
    CAST_TO(TExampleTable, table);

    if (stop>table->numberOfExamples())
      stop=table->numberOfExamples();

    if (start>stop)
      start=stop;
    
    PyObject *list=PyList_New(stop-start);
    int i=0;
    PExampleGenerator lock = EXAMPLE_LOCK(PyOrange_AsExampleTable(self));
    while(start<stop) {
      // here we wrap a reference to example, so we must pass a self's wrapper
      PyObject *example=Example_FromExampleRef((*table)[start++], lock);
      if (!example) {
        PyMem_DEL(list);
        PYERROR(PyExc_SystemError, "out of memory", PYNULL);
      }      
      PyList_SetItem(list, i++, (PyObject *)example);
    }

    return list;
  PyCATCH
}  


int ExampleTable_setslice(TPyOrange *self, int start, int stop, PyObject *args)
{ 
  PyTRY
    CAST_TO_err(TExampleTable, table, -1);

    if (stop>int(table->size()))
      stop = table->size();

    if (start>stop)
      PYERROR(PyExc_IndexError, "index out of range", -1);

    int inspoint = stop;

    try {
      if (PyOrExampleGenerator_Check(args)) {
        PExampleGenerator gen = PyOrange_AsExampleGenerator(args);
        if (args==(PyObject *)self) {
          TExampleTable tab(gen, false);
          EITERATE(ei, tab)
            table->insert(inspoint++, *ei);
        }
        else
          if (!table->ownsExamples
                && (table->lock != gen)
                && (!gen.is_derived_from(TExampleTable) || (table->lock != gen.AS(TExampleTable)->lock)))
              PYERROR(PyExc_TypeError, "tables containing references to examples can only contain examples from the same table", -1);
          PEITERATE(ei, gen)
            table->insert(inspoint++, *ei);
      }

      else {
        TExample example(table->domain);
        if (PyList_Check(args)) {
          int size = PyList_Size(args);

          for(int i = 0; i<size; i++) {
            PyObject *pex = PyList_GetItem(args, i);

            if (table->ownsExamples) {
              if (!convertFromPythonExisting(pex, example)) {
                table->erase(stop, inspoint);
                return -1;
              }
              table->insert(inspoint++, example);
            }
            else {
              if (!PyOrExample_Check(pex) || (((TPyExample *)(pex))->lock != table->lock)) {
                table->erase(stop, inspoint);
                PYERROR(PyExc_TypeError, "tables containing references to examples can only extend by examples from the same table", -1);
              }
              table->insert(inspoint++, PyExample_AS_ExampleReference(pex));
            }
          }
        }
        else
          PYERROR(PyExc_TypeError, "invalid argument for ExampleTable.__setslice__", -1);
      }
    }
    catch (...) {
      table->erase(stop, inspoint);
      throw;
    }

    table->erase(start, stop);

    return 0;
  PyCATCH_1
}


PyObject *applyFilterL(PFilter filter, PExampleTable gen)
{ if (!filter)
    return PYNULL;

  PyObject *list=PyList_New(0);
  filter->reset();
  PExampleGenerator lock = EXAMPLE_LOCK(gen);
  PEITERATE(ei, gen)
    if (filter->operator()(*ei)) {
      PyObject *obj=Example_FromExampleRef(*ei, lock);
      PyList_Append(list, obj);
      Py_DECREF(obj);
    }

  return list;
}


PyObject *applyFilterP(PFilter filter, PExampleTable gen)
{ if (!filter)
    return PYNULL;
  
  TExampleTable *newTable = mlnew TExampleTable(PExampleGenerator(gen), 1);
  PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error
  filter->reset();
  PEITERATE(ei, gen)
    if (filter->operator()(*ei))
      newTable->addExample(*ei);

  return WrapOrange(newGen);
}


PyObject *filterSelectionVectorLow(TFilter &filter, PExampleGenerator egen);

PyObject *applyFilterB(PFilter filter, PExampleTable gen)
{ 
  return filter ? filterSelectionVectorLow(filter.getReference(), gen) : PYNULL;
}


PyObject *ExampleTable_getitemsref(TPyOrange *self, PyObject *pylist)   PYARGS(METH_O, "(indices) -> ExampleTable")
{ return multipleSelectLow(self, pylist, true); }


PyObject *ExampleTable_selectLow(TPyOrange *self, PyObject *args, PyObject *keywords, const int toList)
{ 
  PyTRY
    CAST_TO(TExampleTable, eg);
    PExampleGenerator weg = PExampleGenerator(PyOrange_AS_Orange(self));
    PExampleGenerator lock = EXAMPLE_LOCK(eg);

    /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS KEYWORDS ***** */
    /* Deprecated: use 'filter' instead */
    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      switch (toList) {
        case 2: return applyFilterB(filter_sameValues(keywords, eg->domain), weg);
        case 1: return applyFilterL(filter_sameValues(keywords, eg->domain), weg);
        default: return applyFilterP(filter_sameValues(keywords, eg->domain), weg);
      }
    }

    PyObject *mplier;
    int index;
    if (PyArg_ParseTuple(args, "O|i", &mplier, &index)) {
      PyObject *pyneg = keywords ? PyDict_GetItemString(keywords, "negate") : NULL;
      bool negate = pyneg && PyObject_IsTrue(pyneg);
      bool indexGiven = (PyTuple_Size(args)==2);

      /* ***** SELECTION BY PYLIST ****** */
      if (PyList_Check(mplier)) {
        if (PyList_Size(mplier) != eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "example selector of invalid length", PYNULL);

        int i = 0;
        switch (toList) {

          case 1: {
            PyObject *list = PyList_New(0);

            if (indexGiven)
              EITERATE(ei, *eg) {
                PyObject *lel = PyList_GetItem(mplier, i++);
                if (!PyInt_Check(lel))
                  PYERROR(PyExc_IndexError, "example selector must be an integer index", PYNULL)

                if (negate != (index==PyInt_AsLong(lel))) {
                  PyObject *pyex = Example_FromExampleRef(*ei, lock);
                  PyList_Append(list, pyex);
                  Py_DECREF(pyex);
                }
              }
            else
              EITERATE(ei, *eg)
                if (negate != (PyObject_IsTrue(PyList_GetItem(mplier, i++)) != 0)) {
                  PyObject *pyex = Example_FromExampleRef(*ei, lock);
                  PyList_Append(list, pyex);
                  Py_DECREF(pyex);
                }

            return list;
          }


          // this is a pervesion, but let's support it as a kind of syntactic sugar...
          case 2: {
            const int lsize = PyList_Size(mplier);
            TBoolList *selection = new TBoolList(lsize);
            PBoolList pselection = selection;
            TBoolList::iterator si(selection->begin());
            if (indexGiven)
              for(int i = 0; i < lsize; i++) {
                PyObject *lel = PyList_GetItem(mplier, i);
                if (!PyInt_Check(lel))
                  PYERROR(PyExc_IndexError, "example selector must be an integer index", PYNULL)

                *si++ = negate != (index == PyInt_AsLong(lel));
              }
            else
              for(int i = 0; i < lsize; *si++ = negate != (PyObject_IsTrue(PyList_GetItem(mplier, i++)) != 0));

            return WrapOrange(pselection);
          }


          default: {
            TExampleTable *newTable = mlnew TExampleTable(lock, 1); //locks to weg but does not copy
            PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

            if (indexGiven)
              EITERATE(ei, *eg) {
                PyObject *lel = PyList_GetItem(mplier, i++);
                if (!PyInt_Check(lel))
                  PYERROR(PyExc_IndexError, "example selector must be an integer index", PYNULL)

                if (negate != (index==PyInt_AsLong(lel)))
                  newTable->addExample(*ei);
              }
            else
              EITERATE(ei, *eg)
                if (negate != (PyObject_IsTrue(PyList_GetItem(mplier, i++)) != 0))
                  newTable->addExample(*ei);

            return WrapOrange(newGen);
          }
        }
      }

      /* ***** SELECTION BY LONGLIST ****** */
      else if (PyOrLongList_Check(mplier)) {
        PLongList llist = PyOrange_AsLongList(mplier);
        if (int(llist->size()) != eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "select: invalid list size", PYNULL)

        TLongList::iterator lli(llist->begin()), lle(llist->end());
        TExampleIterator ei = eg->begin();
        
        switch (toList) {
          case 1: {
            PyObject *list = PyList_New(0);
            for(; ei && (lli!=lle); ++ei, lli++)
              if (negate != (indexGiven ? (*lli==index) : (*lli!=0))) {
                PyObject *pyex = Example_FromExampleRef(*ei, lock);
                PyList_Append(list, pyex);
                Py_DECREF(pyex);
              }
            return list;
          }

          case 2: {
            TBoolList *selection = new TBoolList(llist->size());
            PBoolList pselection = selection;
            for(TBoolList::iterator si(selection->begin()); lli != lle; *si++ = negate != (indexGiven ? (*lli++ == index) : (*lli++ != 0)));

            return WrapOrange(pselection);
          }

          default: {
            TExampleTable *newTable = mlnew TExampleTable(lock, 1);
            PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

            for(;ei && (lli!=lle); ++ei, lli++)
              if (negate != (indexGiven ? (*lli==index) : (*lli!=0)))
                newTable->addExample(*ei);

            return WrapOrange(newGen);
          }
        }
      }

      PyErr_Clear();


      /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS DICTIONARY ***** */
      /* Deprecated: use method 'filter' instead. */
      if (PyDict_Check(mplier))
        switch (toList) {
          case 2: return applyFilterB(filter_sameValues(mplier, eg->domain), weg);
          case 1: return applyFilterL(filter_sameValues(mplier, eg->domain), weg);
          default: return applyFilterP(filter_sameValues(mplier, eg->domain), weg);
        }

      else if (PyOrFilter_Check(mplier))
        switch (toList) {
          case 2: return applyFilterB(PyOrange_AsFilter(mplier), weg);
          case 1: return applyFilterL(PyOrange_AsFilter(mplier), weg);
          default: return applyFilterP(PyOrange_AsFilter(mplier), weg);
        }
    }

  PYERROR(PyExc_TypeError, "invalid example selector type", PYNULL);
  PyCATCH
}


PyObject *ExampleTable_selectlist(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExampleTable_selectLow(self, args, keywords, 1); 
  PyCATCH
}


PyObject *ExampleTable_selectref(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExampleTable_selectLow(self, args, keywords, 0); 
  PyCATCH
}


PyObject *ExampleTable_selectbool(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExampleTable_selectLow(self, args, keywords, 2); 
  PyCATCH
}


PyObject *ExampleTable_filterlist(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(list-of-attribute-conditions | filter)")
{
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyFilterL(filter_sameValues(keywords, eg->domain, keywords), weg);
    }

    if (PyTuple_Size(args)==1) {
      PyObject *arg = PyTuple_GET_ITEM(args, 0);

      if (PyDict_Check(arg))
        return applyFilterL(filter_sameValues(arg, eg->domain, keywords), weg);

      if (PyOrFilter_Check(arg))
          return applyFilterL(PyOrange_AsFilter(arg), weg);
    }

    PYERROR(PyExc_AttributeError, "ExampleGenerator.filterlist expects a list of conditions or orange.Filter", PYNULL)
  PyCATCH
}


PyObject *ExampleTable_filterref(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(list-of-attribute-conditions | filter)")
{
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyFilterP(filter_sameValues(keywords, eg->domain, keywords), weg);
    }

    if (PyTuple_Size(args)==1) {
      PyObject *arg = PyTuple_GET_ITEM(args, 0);

      if (PyDict_Check(arg))
        return applyFilterP(filter_sameValues(arg, eg->domain, keywords), weg);

      if (PyOrFilter_Check(arg))
          return applyFilterP(PyOrange_AsFilter(arg), weg);
    }

    PYERROR(PyExc_AttributeError, "ExampleGenerator.filterlist expects a list of conditions or orange.Filter", PYNULL)
  PyCATCH
}


PyObject *ExampleTable_filterbool(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(list-of-attribute-conditions | filter)")
{
  PyTRY
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyFilterB(filter_sameValues(keywords, eg->domain, keywords), weg);
    }

    if (PyTuple_Size(args)==1) {
      PyObject *arg = PyTuple_GET_ITEM(args, 0);

      if (PyDict_Check(arg))
        return applyFilterB(filter_sameValues(arg, eg->domain, keywords), weg);

      if (PyOrFilter_Check(arg))
          return applyFilterB(PyOrange_AsFilter(arg), weg);
    }

    PYERROR(PyExc_AttributeError, "ExampleGenerator.filterlist expects a list of conditions or orange.Filter", PYNULL)
  PyCATCH
}


PyObject *ExampleTable_randomexample(TPyOrange *self) PYARGS(0, "() -> Example")
{ PyTRY
    CAST_TO(TExampleTable, table);
    TExample example(table->domain);
    table->randomExample(example);
    return Example_FromExampleCopyRef(example);
  PyCATCH
}


PyObject *ExampleTable_removeDuplicates(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([weightID=0]]) -> None")
{ PyTRY
    if (PyTuple_Size(args) > 1)
      PYERROR(PyExc_AttributeError, "at most one argument (weight) expected", PYNULL);

    CAST_TO(TExampleTable, table);

    int weightID = 0;
    if (PyTuple_Size(args) && !weightFromArg_byDomain(PyTuple_GET_ITEM(args, 0), table->domain, weightID))
      return PYNULL;

    table->removeDuplicates(weightID);
    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_shuffle(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None")
{
  PyTRY
    SELF_AS(TExampleTable).shuffle();
    RETURN_NONE;
  PyCATCH
}

PyObject *ExampleTable_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{ PyTRY
    CAST_TO(TExampleTable, table);

    if (!args || !PyTuple_Size(args)) {
      table->sort();
      RETURN_NONE;
    }

    PyObject *alist = PyTuple_GET_ITEM(args, 0);
    /* If the first argument is nor list nor tuple, the whole argument is taken as a list
       i.e., data.sort("age", "prescr") is interpreted the same as data.sort(["age", "prescr"])
       All references are borrowed. */
    if ((PyTuple_Size(args) > 1) || (!PyList_Check(alist) && !PyTuple_Check(alist)))
      alist = args;

    TVarList attributes;
    if (varListFromDomain(alist, table->domain, attributes, true, true)) {
      vector<int> order;
      for(TVarList::reverse_iterator vi(attributes.rbegin()), ve(attributes.rend()); vi!=ve; vi++)
        order.push_back(table->domain->getVarNum(*vi));
      table->sort(order);
      RETURN_NONE;
    }

    PYERROR(PyExc_TypeError, "invalid arguments (none, or a list of attributes expected)", PYNULL);

  PyCATCH
}


PyObject *ExampleTable_addMetaAttribute(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(id[, Value=1.0]) -> None")
{ PyTRY
    CAST_TO(TExampleTable, table);

    PyObject *pyid;
    PyObject *pyvalue=PYNULL;
    if (!PyArg_ParseTuple(args, "O|O", &pyid, &pyvalue))
      PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

    int id;
    PVariable metavariable;
    if (PyInt_Check(pyid)) {
      id = PyInt_AsLong(pyid);
      metavariable = table->domain->getMetaVar(id, false);
    }
    else if (PyString_Check(pyid)) {
      id = table->domain->getMetaNum(string(PyString_AsString(pyid)));
      metavariable = table->domain->getMetaVar(id, false);
    }
    else if (PyOrVariable_Check(pyid)) {
      metavariable = PyOrange_AsVariable(pyid);
      id = table->domain->getMetaNum(metavariable);
    }

    TValue value;
    if (!pyvalue)
      if (metavariable && metavariable->varType != TValue::FLOATVAR)
        value = metavariable->DK();
      else
        value = TValue(float(1.0));
    else if (!convertFromPython(pyvalue, value, metavariable))
      PYERROR(PyExc_AttributeError, "invalid value argument", PYNULL);

    table->addMetaAttribute(id, value);

    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_removeMetaAttribute(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(id) -> None")
{ PyTRY
    CAST_TO(TExampleTable, table);

    PyObject *pyid;
    PyObject *pyvalue=PYNULL;
    if (!PyArg_ParseTuple(args, "O|O", &pyid, &pyvalue))
      PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

    int id;
    if (PyInt_Check(pyid))
      id = PyInt_AsLong(pyid);
    else if (PyString_Check(pyid))
      id = table->domain->getMetaNum(string(PyString_AsString(pyid)));
    else if (PyOrVariable_Check(pyid))
      id = table->domain->getMetaNum(PyOrange_AsVariable(pyid));

    table->removeMetaAttribute(id);

    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_changeDomain(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(Domain) -> None")
{ PyTRY
    CAST_TO(TExampleTable, table);
    if (!table->ownsExamples)
      PYERROR(PyExc_TypeError, "tables containing references to examples cannot change domain", PYNULL);

    PDomain domain;
    if (!PyArg_ParseTuple(args, "O&", cc_Domain, &domain))
      PYERROR(PyExc_AttributeError, "domain argument expected", PYNULL);

    table->changeDomain(domain);
    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_hasMissingValues(TPyOrange *self) PYARGS(0, "() -> bool")
{
  PyTRY
    return PyInt_FromLong(SELF_AS(TExampleTable).hasMissing()); 
  PyCATCH
}


PyObject *ExampleTable_hasMissingClasses(TPyOrange *self) PYARGS(0, "() -> bool")
{
  PyTRY
    return PyInt_FromLong(SELF_AS(TExampleTable).hasMissingClass()); 
  PyCATCH
}
/* ************ TRANSFORMVALUE ************ */

#include "transval.hpp"
BASED_ON(TransformValue, Orange)

PyObject *TransformValue_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTransformValue_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TTransformValue_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTransformValue_Python(), type);
}


PyObject *TransformValue__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrTransformValue_Type);
}


PyObject *TransformValue_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(value) -> Value")
{ PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTransformValue_Type) {
      PyErr_Format(PyExc_SystemError, "TransformValue.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    CAST_TO(TTransformValue, tv)
  
    TPyValue *value;
    if (!convertFromPython(args, value))
      return PYNULL;

    tv->transform(value->value);
    value->variable=PVariable();
    return (PyObject *)value;
  PyCATCH
}


/* ************ DISTRIBUTION ************ */

#include "distvars.hpp"

PyObject *convertToPythonNative(const TDiscDistribution &disc)
{ int e = disc.size();
  PyObject *pylist = PyList_New(e);
  for (int i = 0; i<e; i++)
    PyList_SetItem(pylist, i, PyFloat_FromDouble((double)(disc[i])));
  return pylist;
}

PyObject *convertToPythonNative(const TContDistribution &cont)
{ PyObject *pydict = PyDict_New();
  const_ITERATE(TContDistribution, ci, cont) {
    PyObject *key = PyFloat_FromDouble((double)((*ci).first));
    PyObject *val = PyFloat_FromDouble((double)((*ci).second));
    PyDict_SetItem(pydict, key, val);
    Py_DECREF(key);
    Py_DECREF(val);
  }
  return pydict;
}


bool convertFromPython(PyObject *pylist, TDiscDistribution &disc)
{
  if (!PyList_Check(pylist))
    PYERROR(PyExc_TypeError, "list expected", false);
    
  disc.clear();
  float d;
  for(int i = 0, e = PyList_Size(pylist); i!=e; i++) {
    if (!PyNumber_ToFloat(PyList_GET_ITEM(pylist, i), d))
      PYERROR(PyExc_TypeError, "non-number in DiscDistribution as list", false);
    disc.set(TValue(i), d);
  }

  return true;
}


PyObject *convertToPythonNative(const TDistribution &dist, int)
{ const TDiscDistribution *disc = dynamic_cast<const TDiscDistribution *>(&dist);
  if (disc)
    return convertToPythonNative(*disc);

  const TContDistribution *cont = dynamic_cast<const TContDistribution *>(&dist);
  if (cont)
    return convertToPythonNative(*cont);

  PYERROR(PyExc_TypeError, "cannot convert to native python object", PYNULL);
}


/* Class Distribution has a constructor, but it constructs an instance of either DiscDistribution
   or ContDistribution. Class Distribution is thus essentially abstract for Python, although it has
   a constructor. */

NO_PICKLE(Distribution) 

PyObject *Distribution_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SomeValue, "(attribute[, examples[, weightID]])")
{
  PyTRY
    PExampleGenerator gen;
    PyObject *pyvar;
    int weightID = 0;
    if (!PyArg_ParseTuple(args, "O|O&O&:Distribution.new", &pyvar, &pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;

    TDistribution *dist;

    if (!gen) {
      if (PyOrVariable_Check(pyvar))
        dist = TDistribution::create(PyOrange_AsVariable(pyvar));
      else if (PyList_Check(pyvar)) {
        TDiscDistribution *ddist = mlnew TDiscDistribution();
        if (!convertFromPython(pyvar, *ddist)) {
          mldelete ddist;
          raiseError("invalid arguments");
        }
        else
          dist = ddist;
      }
      else
        raiseError("invalid arguments");
    }
    else {
      if (PyOrVariable_Check(pyvar))
        dist = TDistribution::fromGenerator(gen, PyOrange_AsVariable(pyvar), weightID);
      else {
        PVariable var = varFromArg_byDomain(pyvar, gen->domain, false);
        if (!var)
          return PYNULL;

        dist = TDistribution::fromGenerator(gen, var, weightID);
      }
    }

    /* We need to override the type (don't want to lie it's Distribution).
       The exception is if another type is prescribed. */
    return type==(PyTypeObject *)(&PyOrDistribution_Type) ? WrapOrange(PDistribution(dist)) : WrapNewOrange(dist, type);
  PyCATCH
}


PyObject *Distribution_native(PyObject *self, PyObject *) PYARGS(0, "() -> list | dictionary")
{ 
  PyTRY
    return convertToPythonNative(*PyOrange_AS_Orange(self).AS(TDistribution), 1);
  PyCATCH
}


TDiscDistribution *getDiscDistribution(PyObject *self)
{ TDiscDistribution *disc = PyOrange_AS_Orange(self).AS(TDiscDistribution);
  if (!disc)
    PyErr_Format(PyExc_TypeError, "invalid distribution type (expected DiscDistribution, got '%s')", TYPENAME(typeid(PyOrange_AS_Orange(self).getReference())));
  return disc;
}


TContDistribution *getContDistribution(PyObject *self)
{ TContDistribution *cont = PyOrange_AS_Orange(self).AS(TContDistribution);
  if (!cont)
    PyErr_Format(PyExc_TypeError, "invalid distribution type (expected ContDistribution, got '%s')", TYPENAME(typeid(PyOrange_AS_Orange(self).getReference())));
  return cont;
}



float *Distribution_getItemRef(PyObject *self, PyObject *index, float *float_idx=NULL)
{ 
  TDiscDistribution *disc = PyOrange_AS_Orange(self).AS(TDiscDistribution);
  if (disc) {
    int ind=-1;
    if (PyInt_Check(index))
      ind = (int)PyInt_AsLong(index);
    else {
      if (!disc->variable)
        PYERROR(PyExc_SystemError, "invalid distribution (no variable)", (float *)NULL);
      TValue val;
      if (convertFromPython(index, val, disc->variable) && !val.isSpecial()) 
        ind=int(val);
    }

    if (ind<0)
      PYERROR(PyExc_IndexError, "invalid index for distribution", (float *)NULL);

    if (ind<int(disc->size()))
      return &disc->at(ind);
    
    PyErr_Format(PyExc_IndexError, "index %i is out of range (0-%i)", ind, disc->size()-1);
    return (float *)NULL;
  }

  TContDistribution *cont = PyOrange_AS_Orange(self).AS(TContDistribution);
  if (cont) {
    float ind;
    if (PyNumber_ToFloat(index, ind)) {
      if (float_idx)
        *float_idx = ind;
    }
    else {
      TValue val;
      if (convertFromPython(index, val, cont->variable) && !val.isSpecial()) {
        ind = float(val);
        if (float_idx)
          *float_idx = ind;
      }
      else
        PYERROR(PyExc_IndexError, "invalid index type (float expected)", NULL);
    }

    TContDistribution::iterator mi=cont->find(ind);
    if (mi!=cont->end())
      return &(*mi).second;
  }

  PYERROR(PyExc_IndexError, "invalid index", (float *)NULL);
}


PyObject *Distribution_getitem(PyObject *self, PyObject *index)
{ PyTRY
    float *prob=Distribution_getItemRef(self, index);
    return prob ? PyFloat_FromDouble(*prob) : PYNULL;
  PyCATCH
}


int Distribution_setitem(PyObject *self, PyObject *index, PyObject *item)
{ PyTRY
    PyObject *flt = PyNumber_Float(item);
    if (!flt)
      PYERROR(PyExc_TypeError, "float expected", -1);
    
    float val=(float)PyFloat_AsDouble(flt);
    Py_DECREF(flt);

    if (PyOrValue_Check(index)) {
      SELF_AS(TDistribution).set(PyValue_AS_Value(index), val);
      return 0;
    }

    float *prob = Distribution_getItemRef(self, index);
    if (!prob)
      return -1;
      
    *prob = val;
    return 0;
  PyCATCH_1
}


string convertToString(const PDistribution &distribution)
{ 
  const TDiscDistribution *disc = distribution.AS(TDiscDistribution);
  if (disc) {
    string res = "<";
    char buf[128];
    const_PITERATE(TDiscDistribution, di, disc) {
      if (res.size()>1)
        res += ", ";
      sprintf(buf, "%.3f", *di);
      res += buf;
    }
    return res+">";
  }

  const TContDistribution *cont = distribution.AS(TContDistribution);
  if (cont) {
    string res = "<";
    char buf[128];
    const_PITERATE(TContDistribution, di, cont) {
      if (res.size()>1)
        res += ", ";
      sprintf(buf, "%.3f: %.3f", (*di).first, (*di).second);
      res += buf;
    }
    return res+">";
  }

  raiseErrorWho("convertToString(PDistribution)", "invalid distribution");
  return string();
}


PyObject *Distribution_str(PyObject *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;

    return PyString_FromString(convertToString(PyOrange_AsDistribution(self)).c_str());
  PyCATCH
}


PyObject *Distribution_repr(PyObject *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "repr", "str");
    if (result)
      return result;

    return PyString_FromString(convertToString(PyOrange_AsDistribution(self)).c_str());
  PyCATCH
}


PyObject *Distribution_normalize(PyObject *self) PYARGS(0, "() -> None")
{ PyTRY
    SELF_AS(TDistribution).normalize();
    RETURN_NONE;
  PyCATCH
} 


PyObject *Distribution_modus(PyObject *self) PYARGS(0, "() -> Value")
{ PyTRY
    CAST_TO(TDistribution, dist)
    return Value_FromVariableValue(dist->variable, dist->highestProbValue());
  PyCATCH
}


PyObject *Distribution_random(PyObject *self) PYARGS(0, "() -> Value")
{ PyTRY
    CAST_TO(TDistribution, dist)
    return Value_FromVariableValue(dist->variable, dist->randomValue());
  PyCATCH
}
   


PDiscDistribution list2discdistr(PyObject *args, PyTypeObject *type = NULL)
{
  TDiscDistribution *udist = mlnew TDiscDistribution();
  PDiscDistribution disc = type ? PDistribution(udist) : PDistribution(udist, type);
  for(int i = 0, e = PyList_Size(args); i<e; i++) {
    PyObject *flt = PyNumber_Float(PyList_GetItem(args, i));
    if (!flt) {
      PyErr_Format(PyExc_TypeError, "invalid element at index %i (float expected)", i);
      return PDiscDistribution();
    }
    udist->addint(i, (float)PyFloat_AsDouble(flt));
    Py_DECREF(flt);
  }

  return disc;
}


PyObject *DiscDistribution_new(PyTypeObject *type, PyObject *targs, PyObject *) BASED_ON(Distribution, "[list of floats] | DiscDistribution")
{ PyTRY {
    if (!PyTuple_Size(targs)) {
      return WrapNewOrange(mlnew TDiscDistribution(), type);
    }

    if (PyTuple_Size(targs)==1) {
      PyObject *args = PyTuple_GetItem(targs, 0);

      if (PyList_Check(args)) {
        PDiscDistribution disc = list2discdistr(args, type);
        if (disc)
          return WrapOrange(disc);
      }

      else if (PyOrDiscDistribution_Check(args)) {
        Py_INCREF(args);
        return args;
      }

      else if (PyOrEnumVariable_Check(args))
        return WrapNewOrange(mlnew TDiscDistribution(PyOrange_AsVariable(args)), type);
    }

    PYERROR(PyExc_TypeError, "invalid arguments for distribution constructor", PYNULL);
  }
  PyCATCH;
}
      

PyObject *DiscDistribution__reduce__(PyObject *self)
{
  PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    TCharBuffer buf(sizeof(float)*(disc->size()+2));
    buf.writeFloatVector(disc->distribution);

    return Py_BuildValue("O(Os#)N", getExportedFunction("__pickleLoaderDiscDistribution"),
                                    self->ob_type,
                                    buf.buf, buf.length(),
                                    packOrangeDictionary(self));
  PyCATCH
}


PyObject *__pickleLoaderDiscDistribution(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_distribution)")
{
  PyTRY
    PyTypeObject *type;
    char *buf;
    int bufSize;
    if (!PyArg_ParseTuple(args, "Os#:__pickleLoadDiscDistribution", &type, &buf, &bufSize))
      return PYNULL;

    const int &size = (int &)*buf;
    buf += sizeof(int);

    return WrapNewOrange(new TDiscDistribution((float *)buf, size), type);
  PyCATCH
}


int pt_DiscDistribution(PyObject *args, void *dist)
{ if (PyOrDiscDistribution_Check(args)) {
    *(PDiscDistribution *)(dist) = PyOrange_AsDiscDistribution(args);
    return 1;
  }
  else if (PyList_Check(args)) {
    *(PDiscDistribution *)(dist) = PyOrange_AsDiscDistribution(args);
    if (dist)
      return 1;
  }

  PYERROR(PyExc_TypeError, "invalid discrete distribution", 0)
}


PyObject *DiscDistribution_getitem_sq(PyObject *self, int ind)
{ 
  PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    if (!disc)
      return PYNULL;

    if ((ind<0) || (ind>=int(disc->size())))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);

    return PyFloat_FromDouble(double(disc->at(ind)));
  PyCATCH
}


int DiscDistribution_len(PyObject *self)
{ PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    return disc ? (int)disc->size() : -1;
  PyCATCH_1
}


PyObject *DiscDistribution_keys(PyObject *self) PYARGS(0, "() -> [string] | [float]")
{ PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    if (!disc)
      return PYNULL;

    if (!disc->variable)
      PYERROR(PyExc_TypeError, "invalid distribution (no variable)", PYNULL);

    PyObject *nl=PyList_New(disc->variable->noOfValues());
    int i=0;
    PStringList vals=disc->variable.AS(TEnumVariable)->values;
    PITERATE(TStringList, ii, vals)
      PyList_SetItem(nl, i++, PyString_FromString((*ii).c_str()));
    return nl;
  PyCATCH
}


PyObject *DiscDistribution_items(PyObject *self) PYARGS(0, "() -> [(string, float)] | [(float, float)]")
{ PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    if (!disc)
      return PYNULL;

    if (!disc->variable)
      PYERROR(PyExc_TypeError, "invalid distribution (no variable)", PYNULL);

    PyObject *nl=PyList_New(disc->variable->noOfValues());
    TDiscDistribution::const_iterator ci(disc->begin());
    int i=0;
    PStringList vals=disc->variable.AS(TEnumVariable)->values;
    PITERATE(TStringList, ii, vals)
      PyList_SetItem(nl, i++, Py_BuildValue("sf", (*ii).c_str(), *(ci++)));
    return nl;
  PyCATCH
}


PyObject *DiscDistribution_values(PyObject *self) PYARGS(0, "() -> list")
{ PyTRY
    TDiscDistribution *disc = getDiscDistribution(self);
    if (!disc)
      return PYNULL;

    PyObject *nl = PyList_New(disc->size());
    int i = 0;
    const_PITERATE(TDiscDistribution, ci, disc)
      PyList_SetItem(nl, i++, PyFloat_FromDouble(*ci));
    return nl;
  PyCATCH
}
    

PyObject *DiscDistribution_add(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(value, weight) -> Value")
{ PyTRY
    CAST_TO(TDiscDistribution, dist)

    PyObject *index;
    float weight = 1.0;
    if (!PyArg_ParseTuple(args, "O|f", &index, &weight))
      PYERROR(PyExc_TypeError, "DiscDistribution.add: invalid arguments", PYNULL);

    if (PyInt_Check(index)) {
      dist->addint(int(PyInt_AsLong(index)), weight);
      RETURN_NONE;
    }

    TValue val;
    if (!dist->variable || !convertFromPython(index, val, dist->variable))
      PYERROR(PyExc_TypeError, "DiscDistriubtion.add: cannot convert the arguments to a Value", PYNULL);

    dist->add(val, weight);
    RETURN_NONE;
  PyCATCH;
}


PyObject *ContDistribution_new(PyTypeObject *type, PyObject *targs, PyObject *) BASED_ON(Distribution, "[dist of float:float] | DiscDistribution")
{ PyTRY {

    if (!PyTuple_Size(targs))
      return WrapNewOrange(mlnew TContDistribution(), type);

    if (PyTuple_Size(targs) == 1) {
      PyObject *args = PyTuple_GetItem(targs, 0);

      if (PyDict_Check(args)) {
        TContDistribution *udist = mlnew TContDistribution();
        PContDistribution cont = PDistribution(udist);
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(args, &pos, &key, &value)) {
          PyObject *flt = PyNumber_Float(key);
          if (!flt) {
            PyErr_Format(PyExc_TypeError, "invalid key at index %i (float expected)", pos);
            return false;
          }
          float ind = (float) PyFloat_AsDouble(flt);
          Py_DECREF(flt);

          flt = PyNumber_Float(value);
          if (!flt) {
            PyErr_Format(PyExc_TypeError, "invalid value at index %i (float expected)", pos);
            return false;
          }

          udist->addfloat(ind, (float)PyFloat_AsDouble(flt));
          Py_DECREF(flt);
        }

        return WrapOrange(cont);
      }

      else if (PyOrDistribution_Check(args)) {
        Py_INCREF(args);
        return args;
      }

      else if (PyOrFloatVariable_Check(args))
        return WrapNewOrange(mlnew TContDistribution(PyOrange_AsVariable(args)), type);
    }

    PYERROR(PyExc_TypeError, "invalid arguments for distribution constructor", PYNULL);

  }
  PyCATCH;
}
      

PyObject *ContDistribution__reduce__(PyObject *self)
{
  PyTRY
    TContDistribution *cont = getContDistribution(self);
    TCharBuffer buf(sizeof(float) * 2 * (cont->size()  +  5));

    buf.writeInt(cont->size());
    PITERATE(TContDistribution, ci, cont) {
      buf.writeFloat((*ci).first);
      buf.writeFloat((*ci).second);
    }

    buf.writeFloat(cont->sum);
    buf.writeFloat(cont->sum2);

    return Py_BuildValue("O(Os#)N", getExportedFunction("__pickleLoaderContDistribution"),
                                    self->ob_type,
                                    buf.buf, buf.length(),
                                    packOrangeDictionary(self));
  PyCATCH
}


PyObject *__pickleLoaderContDistribution(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_distribution)")
{
  PyTRY
    PyTypeObject *type;
    char *pbuf;
    int bufSize;
    if (!PyArg_ParseTuple(args, "Os#:__pickleLoadDiscDistribution", &type, &pbuf, &bufSize))
      return PYNULL;

    TContDistribution *cdi = new TContDistribution();

    TCharBuffer buf(pbuf);
    for(int size = buf.readInt(); size--; ) {
      // cannot call buf.readFloat() in the make_pair call since we're not sure about the
      // order in which the arguments are evaluated
      const float p1 = buf.readFloat();
      const float p2 = buf.readFloat();
      cdi->insert(cdi->end(), make_pair(p1, p2));
    }

    cdi->sum = buf.readFloat();
    cdi->sum2 = buf.readFloat();

    return WrapNewOrange(cdi, type);
  PyCATCH
}


int ContDistribution_len(PyObject *self)
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    return cont ? (int)cont->size() : -1;
  PyCATCH_1
}


PyObject *ContDistribution_keys(PyObject *self) PYARGS(0, "() -> [string] | [float]")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;

    PyObject *nl=PyList_New(cont->size());
    int i=0;
    PITERATE(TContDistribution, ci, cont)
      PyList_SetItem(nl, i++, PyFloat_FromDouble((double)(*ci).first));
    return nl;
  PyCATCH
}


PyObject *ContDistribution_items(PyObject *self) PYARGS(0, "() -> [(string, float)] | [(float, float)]")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;

    PyObject *nl=PyList_New(cont->size());
    int i=0;
    PITERATE(TContDistribution, ci, cont)
      PyList_SetItem(nl, i++, Py_BuildValue("ff", (*ci).first, (*ci).second));
    return nl;
  PyCATCH
}


PyObject *ContDistribution_values(PyObject *self) PYARGS(0, "() -> list")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;

    PyObject *nl = PyList_New(cont->size());
    int i = 0;
    const_PITERATE(TContDistribution, ci, cont)
      PyList_SetItem(nl, i++, PyFloat_FromDouble((*ci).second));
    return nl;
  PyCATCH
}
    

PyObject *ContDistribution_percentile(PyObject *self, PyObject *arg) PYARGS(METH_VARARGS, "(int) -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;

    float perc;
    if (!PyArg_ParseTuple(arg, "f:ContDistribution.percentile", &perc))
      return PYNULL;

    return PyFloat_FromDouble(cont->percentile(perc));
  PyCATCH
}


PyObject *ContDistribution_add(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(value, weight) -> Value")
{ PyTRY
    CAST_TO(TContDistribution, dist)

    PyObject *index;
    float weight = 1.0;
    if (!PyArg_ParseTuple(args, "O|f", &index, &weight))
      PYERROR(PyExc_TypeError, "DiscDistribution.add: invalid arguments", PYNULL);

    float f;
    if (PyNumber_ToFloat(index, f)) {
      dist->addfloat(f);
      RETURN_NONE;
    }

    TValue val;
    if (!convertFromPython(index, val, dist->variable))
      PYERROR(PyExc_TypeError, "ContDistriubtion.add: invalid arguments", PYNULL);

    dist->add(val, weight);
    RETURN_NONE;
  PyCATCH;
}


PyObject *ContDistribution_error(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;
    
    return PyFloat_FromDouble(cont->error());
  PyCATCH
}


PyObject *ContDistribution_average(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;
    
    return PyFloat_FromDouble(cont->average());
  PyCATCH
}


PyObject *ContDistribution_dev(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;
    
    return PyFloat_FromDouble(cont->dev());
  PyCATCH
}


PyObject *ContDistribution_var(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    if (!cont)
      return PYNULL;
    
    return PyFloat_FromDouble(cont->var());
  PyCATCH
}


PyObject *ContDistribution_density(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x) -> float")
{ PyTRY
    TContDistribution *cont = getContDistribution(self);
    float x;
    if (!cont || !PyArg_ParseTuple(args, "f:ContDistribution.density", &x))
      return PYNULL;
    
    return PyFloat_FromDouble(cont->p(x));
  PyCATCH
}


PyObject *GaussianDistribution_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Distribution, "(mean, sigma) | (distribution) | () -> distribution") ALLOWS_EMPTY
{ PyTRY
    float mean = 0.0, sigma = 1.0;

    if (PyArg_ParseTuple(args, "|ff", &mean, &sigma))
      return WrapNewOrange(mlnew TGaussianDistribution(mean, sigma), type);

    PyErr_Clear();

    PDistribution dist;
    if (PyArg_ParseTuple(args, "O&", &cc_Distribution, &dist))
      return WrapNewOrange(mlnew TGaussianDistribution(dist), type);

    PYERROR(PyExc_TypeError, "GaussianDistribution expects mean and sigma, or distribution or nothing", PYNULL)

  PyCATCH
}


PyObject *GaussianDistribution_average(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    return PyFloat_FromDouble(SELF_AS(TGaussianDistribution).average());
  PyCATCH
}


PyObject *GaussianDistribution_error(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    return PyFloat_FromDouble(SELF_AS(TGaussianDistribution).error());
  PyCATCH
}


PyObject *GaussianDistribution_dev(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    return PyFloat_FromDouble(SELF_AS(TGaussianDistribution).dev());
  PyCATCH
}


PyObject *GaussianDistribution_var(PyObject *self) PYARGS(0, "() -> float")
{ PyTRY
    return PyFloat_FromDouble(SELF_AS(TGaussianDistribution).var());
  PyCATCH
}


PyObject *GaussianDistribution_density(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x) -> float")
{ PyTRY
    float x;
    if (!PyArg_ParseTuple(args, "f:GaussianDistribution.density", &x))
      return PYNULL;
    
    return PyFloat_FromDouble(SELF_AS(TGaussianDistribution).p(x));
  PyCATCH
}


/* We redefine new (removed from below!) and add mapping methods
*/

PyObject *getClassDistribution(PyObject *type, PyObject *args) PYARGS(METH_VARARGS, "(examples[, weightID]) -> Distribution")
{ PyTRY
    int weightID;
    PExampleGenerator gen = exampleGenFromArgs(args, weightID);
    if (!gen)
      return PYNULL;
    return WrapOrange(getClassDistribution(gen, weightID));
  PyCATCH
}


/* modified new (defined below), modified getitem, setitem */

PDomainDistributions PDomainDistributions_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::P_FromArguments(arg); }
PyObject *DomainDistributions_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_FromArguments(type, arg); }
PyObject *DomainDistributions_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_getslice(self, start, stop); }
int       DomainDistributions_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_setslice(self, start, stop, item); }
PyObject *DomainDistributions_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_getitem(self, index); }
int       DomainDistributions_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_setitem(self, index, item); }
int       DomainDistributions_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_len(self); }
PyObject *DomainDistributions_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_richcmp(self, object, op); }
PyObject *DomainDistributions_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_concat(self, obj); }
PyObject *DomainDistributions_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_repeat(self, times); }
PyObject *DomainDistributions_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_str(self); }
PyObject *DomainDistributions_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_str(self); }
int       DomainDistributions_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_contains(self, obj); }
PyObject *DomainDistributions_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_append(self, item); }
PyObject *DomainDistributions_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_extend(self, obj); }
PyObject *DomainDistributions_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_count(self, obj); }
PyObject *DomainDistributions_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainDistributions") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_filter(self, args); }
PyObject *DomainDistributions_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_index(self, obj); }
PyObject *DomainDistributions_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_insert(self, args); }
PyObject *DomainDistributions_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_native(self); }
PyObject *DomainDistributions_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Distribution") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_pop(self, args); }
PyObject *DomainDistributions_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_remove(self, obj); }
PyObject *DomainDistributions_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_reverse(self); }
PyObject *DomainDistributions_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_sort(self, args); }
PyObject *DomainDistributions__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_reduce(self); }


/* Note that this is not like callable-constructors. They return different type when given
   parameters, while this one returns the same type, disregarding whether it was given examples or not.
*/
PyObject *DomainDistributions_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "(examples[, weightID] | <list of Distribution>) -> DomainDistributions") ALLOWS_EMPTY
{ PyTRY
    if (!args || !PyTuple_Size(args))
      return WrapNewOrange(mlnew TDomainDistributions(), type);
      
    int weightID;
    PExampleGenerator gen = exampleGenFromArgs(args, weightID);
    if (gen)
      return WrapNewOrange(mlnew TDomainDistributions(gen, weightID), type);

    PyErr_Clear();

    PyObject *obj = ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, &PyOrDistribution_Type>::_new(type, args, keywds);
    if (obj)
      if (obj!=Py_None)
        return obj;
      else
        Py_DECREF(obj);

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "DomainDistributions.__init__ expect examples or a list of Distributions", PYNULL);

  PyCATCH
}


/* We keep the sequence methods and add mapping interface */

int DomainDistributions_getItemIndex(PyObject *self, PyObject *args)
{ CAST_TO_err(TDomainDistributions, bas, -1);
  
  if (PyInt_Check(args)) {
    int i=(int)PyInt_AsLong(args);
    if ((i>=0) && (i<int(bas->size())))
      return i;
    else
      PYERROR(PyExc_IndexError, "index out of range", -1);
  }

  if (PyString_Check(args)) {
    char *s=PyString_AsString(args);
    PITERATE(TDomainDistributions, ci, bas)
      if ((*ci)->variable && ((*ci)->variable->name==s))
        return ci - bas->begin();

    PyErr_Format(PyExc_IndexError, "attribute '%s' not found in domain", s);
    return -1;
  }

  if (PyOrVariable_Check(args)) {
    PVariable var = PyOrange_AsVariable(args);
    PITERATE(TDomainDistributions, ci, bas)
      if ((*ci)->variable && ((*ci)->variable==var))
        return ci - bas->begin();

    PyErr_Format(PyExc_IndexError, "attribute '%s' not found in domain", var->name.length() ? var->name.c_str() : "<no name>");
    return -1;
  }

  PYERROR(PyExc_IndexError, "invalid index type", -1);
}


PyObject *DomainDistributions_getitem(PyObject *self, PyObject *args)
{ PyTRY
    int index=DomainDistributions_getItemIndex(self, args);
    if (index<0)
      return PYNULL;
    return WrapOrange(POrange(SELF_AS(TDomainDistributions).at(index)));
  PyCATCH
}


int DomainDistributions_setitem(PyObject *self, PyObject *args, PyObject *obj)
{ PyTRY
    PDistribution bas;

    if (!PyOrBasicAttrStat_Check(obj))
      PYERROR(PyExc_TypeError, "invalid Distribution object", -1);

    int index=DomainDistributions_getItemIndex(self, args);
    if (index==-1)
      return -1;

    SELF_AS(TDomainDistributions)[index] = PyOrange_AsDistribution(obj);
    return 0;
  PyCATCH_1
}


PDistributionList PDistributionList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::P_FromArguments(arg); }
PyObject *DistributionList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_FromArguments(type, arg); }
PyObject *DistributionList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Distribution>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_new(type, arg, kwds); }
PyObject *DistributionList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_getitem(self, index); }
int       DistributionList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_setitem(self, index, item); }
PyObject *DistributionList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_getslice(self, start, stop); }
int       DistributionList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_setslice(self, start, stop, item); }
int       DistributionList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_len(self); }
PyObject *DistributionList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_richcmp(self, object, op); }
PyObject *DistributionList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_concat(self, obj); }
PyObject *DistributionList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_repeat(self, times); }
PyObject *DistributionList_str(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_str(self); }
PyObject *DistributionList_repr(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_str(self); }
int       DistributionList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_contains(self, obj); }
PyObject *DistributionList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_append(self, item); }
PyObject *DistributionList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_extend(self, obj); }
PyObject *DistributionList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_count(self, obj); }
PyObject *DistributionList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DistributionList") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_filter(self, args); }
PyObject *DistributionList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_index(self, obj); }
PyObject *DistributionList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_insert(self, args); }
PyObject *DistributionList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_native(self); }
PyObject *DistributionList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Distribution") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_pop(self, args); }
PyObject *DistributionList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_remove(self, obj); }
PyObject *DistributionList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_reverse(self); }
PyObject *DistributionList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_sort(self, args); }
PyObject *DistributionList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, &PyOrDistribution_Type>::_reduce(self); }



/* ************ LEARNER ************ */

#include "classify.hpp"
#include "learn.hpp"

BASED_ON(EFMDataDescription, Orange)

PyObject *EFMDataDescription__reduce__(PyObject *self)
{
  CAST_TO(TEFMDataDescription, edd);

  TCharBuffer buf(0);
  buf.writeFloatVector(edd->averages);
  buf.writeFloatVector(edd->matchProbabilities);
  buf.writeInt(edd->originalWeight);
  buf.writeInt(edd->missingWeight);

  return Py_BuildValue("O(OOs#)N", getExportedFunction("__pickleLoaderEFMDataDescription"),
                                  WrapOrange(edd->domain),
                                  WrapOrange(edd->domainDistributions),
                                  buf.buf, buf.length(),
                                  packOrangeDictionary(self));
}


PyObject *__pickleLoaderEFMDataDescription(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(domain, domainDistributions, packed_data)")
{
  PDomain domain;
  PDomainDistributions domainDistributions; 
  char *pbuf;
  int bufSize;
  
  if (!PyArg_ParseTuple(args, "O&O&s#", ccn_Domain, &domain, ccn_DomainDistributions, &domainDistributions, &pbuf, &bufSize))
    return PYNULL;

  TEFMDataDescription *edd = new TEFMDataDescription(domain, domainDistributions);
  PEFMDataDescription wedd = edd;

  TCharBuffer buf(pbuf);
  buf.readFloatVector(edd->averages);
  buf.readFloatVector(edd->matchProbabilities);
  edd->originalWeight = buf.readInt();
  edd->missingWeight = buf.readInt();

  return WrapOrange(wedd);
}


ABSTRACT(LearnerFD, Learner)

PyObject *Learner_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrLearner_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TLearner_Python(), type), args);
  else
    return WrapNewOrange(mlnew TLearner_Python(), type);
}


PyObject *Learner__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrLearner_Type);
}


PyObject *Learner_call(PyObject *self, PyObject *targs, PyObject *keywords) PYDOC("(examples) -> Classifier")
{
  PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrLearner_Type) {
      PyErr_Format(PyExc_SystemError, "Learner.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    PExampleGenerator egen;
    int weight = 0;
    if (!PyArg_ParseTuple(targs, "O&|O&", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weight))
      PYERROR(PyExc_AttributeError, "Learner.__call__: examples and, optionally, weight attribute expected", PYNULL);

    // Here for compatibility with obsolete scripts
/*    if (PyTuple_Size(targs)==1) {
      if (((TPyOrange *)self)->orange_dict) {
        PyObject *pyweight = PyDict_GetItemString(((TPyOrange *)self)->orange_dict, "weight");
        if (pyweight && PyInt_Check(pyweight))
          weight = (int)PyInt_AsLong(pyweight);
      }
    }
*/
    PClassifier classfr = SELF_AS(TLearner)(egen, weight);
    if (!classfr)
      PYERROR(PyExc_SystemError, "learning failed", PYNULL);

    return WrapOrange(classfr);
  PyCATCH
}




/* ************ CLASSIFIERS ************ */

#include "classify.hpp"
#include "majority.hpp"

ABSTRACT(ClassifierFD, Classifier)

PyObject *DefaultClassifier_new(PyTypeObject *tpe, PyObject *args, PyObject *kw) BASED_ON(Classifier, "([defaultVal])") ALLOWS_EMPTY
{
  PyObject *arg1 = NULL, *arg2 = NULL;
  if (!PyArg_UnpackTuple(args, "DefaultClassifier.__new__", 0, 2, &arg1, &arg2))
    return PYNULL;

  if (!arg1)
    return WrapNewOrange(mlnew TDefaultClassifier(), tpe);

  if (!arg2) {
    if (PyOrVariable_Check(arg1))
      return WrapNewOrange(mlnew TDefaultClassifier(PyOrange_AsVariable(arg1)), tpe);
    TValue val;
    if (convertFromPython(arg1, val)) {
      PVariable var = PyOrValue_Check(arg1) ? PyValue_AS_Variable(arg1) : PVariable();
      return WrapNewOrange(mlnew TDefaultClassifier(var, val, PDistribution()), tpe);
    }
  }

  else
    if (PyOrVariable_Check(arg1)) {
      PVariable classVar = PyOrange_AsVariable(arg1);
      TValue val;
      if (convertFromPython(arg2, val, classVar))
        return WrapNewOrange(mlnew TDefaultClassifier(classVar, val, PDistribution()), tpe);
    }

  PYERROR(PyExc_TypeError, "DefaultClassifier's constructor expects a Variable, a Value or both", PYNULL);
}

C_NAMED(RandomLearner, Learner, "([probabilities=])")
C_NAMED(RandomClassifier, Classifier, "([probabilities=])")

PClassifierList PClassifierList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::P_FromArguments(arg); }
PyObject *ClassifierList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_FromArguments(type, arg); }
PyObject *ClassifierList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Classifier>)")  ALLOWS_EMPTY { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_new(type, arg, kwds); }
PyObject *ClassifierList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_getitem(self, index); }
int       ClassifierList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_setitem(self, index, item); }
PyObject *ClassifierList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_getslice(self, start, stop); }
int       ClassifierList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_setslice(self, start, stop, item); }
int       ClassifierList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_len(self); }
PyObject *ClassifierList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_richcmp(self, object, op); }
PyObject *ClassifierList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_concat(self, obj); }
PyObject *ClassifierList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_repeat(self, times); }
PyObject *ClassifierList_str(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_str(self); }
PyObject *ClassifierList_repr(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_str(self); }
int       ClassifierList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_contains(self, obj); }
PyObject *ClassifierList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Classifier) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_append(self, item); }
PyObject *ClassifierList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_extend(self, obj); }
PyObject *ClassifierList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> int") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_count(self, obj); }
PyObject *ClassifierList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ClassifierList") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_filter(self, args); }
PyObject *ClassifierList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> int") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_index(self, obj); }
PyObject *ClassifierList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_insert(self, args); }
PyObject *ClassifierList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_native(self); }
PyObject *ClassifierList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Classifier") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_pop(self, args); }
PyObject *ClassifierList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_remove(self, obj); }
PyObject *ClassifierList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_reverse(self); }
PyObject *ClassifierList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_sort(self, args); }
PyObject *ClassifierList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, &PyOrClassifier_Type>::_reduce(self); }


/* Not in .hpp (to be parsed by pyprops) since these only occur in arguments and only in Python */
/* Duplicated for compatibility (and also simplicity) */

PYCONSTANT_INT(GetValue, 0)
PYCONSTANT_INT(GetProbabilities, 1)
PYCONSTANT_INT(GetBoth, 2)

PYCLASSCONSTANT_INT(Classifier, GetValue, 0)
PYCLASSCONSTANT_INT(Classifier, GetProbabilities, 1)
PYCLASSCONSTANT_INT(Classifier, GetBoth, 2)


PyObject *Classifier_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrClassifier_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TClassifier_Python(), type), args);
  else
    return WrapNewOrange(mlnew TClassifier_Python(), type);
}


PyObject *Classifier__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrClassifier_Type);
}


PyObject *Classifier_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example[, format]) -> Value | distribution | (Value, distribution)")
{ PyTRY
    NO_KEYWORDS

    CAST_TO(TClassifier, classifier);

    if ((PyOrange_OrangeBaseClass(self->ob_type) == &PyOrClassifier_Type) && !dynamic_cast<TClassifier_Python *>(classifier)) {
      PyErr_Format(PyExc_SystemError, "Classifier.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }


    if (!classifier)
      PYERROR(PyExc_SystemError, "attribute error", PYNULL);

    TExample *example;
    int dist=0;
    if (!PyArg_ParseTuple(args, "O&|i", ptr_Example, &example, &dist))
      PYERROR(PyExc_TypeError, "attribute error; example (and, optionally, return type) expected", PYNULL);

    switch (dist) {
      case 0:
        return Value_FromVariableValue(classifier->classVar, (*classifier)(*example));

      case 1:
        return WrapOrange(classifier->classDistribution(*example));

      case 2:
        TValue val;
        PDistribution dist;
        classifier->predictionAndDistribution(*example, val, dist);
        return Py_BuildValue("NN", Value_FromVariableValue(classifier->classVar, val), WrapOrange(dist));
    }

    PYERROR(PyExc_AttributeError, "invalid parameter for classifier call", PYNULL);

  PyCATCH
}



// We override its [gs]etattr to add the classVar
int DefaultClassifier_set_defaultValue(PyObject *self, PyObject *args)
{ PyTRY
    return convertFromPython(args, SELF_AS(TDefaultClassifier).defaultVal, SELF_AS(TDefaultClassifier).classVar) ? 0 : -1;
  PyCATCH_1
}


PyObject *DefaultClassifier_get_defaultValue(PyObject *self)
{ PyTRY
    return Value_FromVariableValue(SELF_AS(TDefaultClassifier).classVar, SELF_AS(TDefaultClassifier).defaultVal);
  PyCATCH
}


/* ************ CLASSIFIERS FROM VAR ************ */

#include "classfromvar.hpp"
C_NAMED(ClassifierFromVar, Classifier, "([whichVar=, transformer=])")
C_NAMED(ClassifierFromVarFD, ClassifierFD, "([position=, transformer=])")

#include "cartesian.hpp"
C_NAMED(CartesianClassifier, ClassifierFD, "()")


/* ************ LOOKUP ************ */

#include "lookup.hpp"

C_CALL(LookupLearner, Learner, "([examples] [, weight=]) -/-> Classifier")
C_NAMED(ClassifierByExampleTable, ClassifierFD, "([examples=])")


PyObject *LookupLearner_call(PyObject *self, PyObject *targs, PyObject *keywords) PYDOC("(examples) -> Classifier | (classVar, attributes, examples) -> Classifier")
{ PyTRY

    NO_KEYWORDS

    PyObject *pyclassVar;
    PyObject *pyvarList;
    PExampleGenerator egen;
    int weight = 0;
    if (PyArg_ParseTuple(targs, "OOO&|O&", &pyclassVar, &pyvarList, pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weight)) {

      PVariable classVar = varFromArg_byDomain(pyclassVar, egen->domain);
      TVarList varList;
      if (!varListFromDomain(pyvarList, egen->domain, varList, true))
        return PYNULL;
      PDomain newdomain = mlnew TDomain(egen->domain->classVar, varList);
      PExampleTable etable = mlnew TExampleTable(newdomain, egen);
      PClassifier cbet = TLookupLearner()(etable, weight);
      cbet->classVar = classVar;
      return WrapOrange(cbet);
    }

    PyErr_Clear();
    return Learner_call(self, targs, keywords);
      
  PyCATCH
}

PyObject *ClassifierByExampleTable_boundset(PyObject *self) PYARGS(0, "() -> variables")
{ PyTRY
    TVarList &attributes=SELF_AS(TClassifierByExampleTable).domain->attributes.getReference();
    PyObject *list=PyList_New(attributes.size());
    for(int i=0, asize = attributes.size(); i<asize; i++)
      PyList_SetItem(list, i, WrapOrange(attributes[i]));
    return list;
  PyCATCH
}


PyObject *ClassifierByExampleTable_get_variables(PyObject *self)
{ return ClassifierByExampleTable_boundset(self); }
  

PyObject *ClassifierByLookupTable_boundset(PyObject *self) PYARGS(0, "() -> (variables)")
{ PyTRY
    TVarList vlist;
    SELF_AS(TClassifierByLookupTable).giveBoundSet(vlist);
    PyObject *res = PyTuple_New(vlist.size());
    int i = 0;
    ITERATE(TVarList, vi, vlist)
      PyTuple_SetItem(res, i++, WrapOrange(*vi));
    return res;
  PyCATCH
}


PyObject *ClassifierByLookupTable_get_variables(PyObject *self)
{ return ClassifierByLookupTable_boundset(self); }


PyObject *ClassifierByLookupTable_getindex(PyObject *self, PyObject *pyexample) PYARGS(METH_O, "(example) -> int")
{ PyTRY
    if (!PyOrExample_Check(pyexample))
      PYERROR(PyExc_TypeError, "invalid arguments; an example expected", PYNULL);

    return PyInt_FromLong(long(SELF_AS(TClassifierByLookupTable).getIndex(PyExample_AS_ExampleReference(pyexample))));
  PyCATCH
}


PValueList PValueList_FromArguments(PyObject *arg, PVariable var = PVariable());

/* Finishes up the initialization. If anything goes wrong, it deallocates the classifier */
bool initializeTables(PyObject *pyvlist, PyObject *pydlist, TClassifierByLookupTable *cblt)
{
  try {
    PValueList tvlist;
    PDistributionList tdlist; 

    if (pyvlist && (pyvlist != Py_None)) {
      tvlist = PValueList_FromArguments(pyvlist, cblt->classVar);
      if (!tvlist) {
        mldelete cblt;
        return false;
      }
      if (tvlist->size() != cblt->lookupTable->size()) {
        mldelete cblt;
        PYERROR(PyExc_AttributeError, "invalid size for 'lookup' list", false);
      }
      cblt->lookupTable = tvlist;
    }

    if (pydlist && (pydlist != Py_None)) {
      tdlist = PDistributionList_FromArguments(pydlist);
      if (!tdlist) {
        mldelete cblt;
        return false;
      }
      if (tdlist->size() != cblt->distributions->size()) {
        mldelete cblt;
        PYERROR(PyExc_AttributeError, "invalid size for 'distributions' list", false);
      }
      cblt->distributions = tdlist;
    }
  }

  catch (...) {
    mldelete cblt;
    throw;
  }

  return true;
}


PyObject *ClassifierByLookupTable1_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ClassifierByLookupTable, "(class-descriptor, descriptor)")
{ PyTRY
    PVariable vcl, vvl;
    PyObject *pyvlist = PYNULL;
    PyObject *pydlist = PYNULL;
    if (!PyArg_ParseTuple(args, "O&O&|OO", cc_Variable, &vcl, cc_Variable, &vvl, &pyvlist, &pydlist))
      PYERROR(PyExc_TypeError, "invalid parameter; two variables and, optionally, ValueList and DistributionList expected", PYNULL);

    TClassifierByLookupTable1 *cblt = mlnew TClassifierByLookupTable1(vcl, vvl);
    return initializeTables(pyvlist, pydlist, cblt) ? WrapNewOrange(cblt, type) : PYNULL;
  PyCATCH
}


PyObject *ClassifierByLookupTable1__reduce__(PyObject *self)
{ 
  CAST_TO(TClassifierByLookupTable1, cblt);
  return Py_BuildValue("O(OOOO)N", (PyObject *)(self->ob_type), 
                                   WrapOrange(cblt->classVar),
                                   WrapOrange(cblt->variable1),
                                   WrapOrange(cblt->lookupTable),
                                   WrapOrange(cblt->distributions),
                                   packOrangeDictionary(self));
}


PyObject *ClassifierByLookupTable2_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ClassifierByLookupTable, "(class-descriptor, desc0, desc1)")
{ PyTRY
    PVariable vcl, vvl1, vvl2;
    PyObject *pyvlist = PYNULL;
    PyObject *pydlist = PYNULL;
    if (!PyArg_ParseTuple(args, "O&O&O&|OO", cc_Variable, &vcl, cc_Variable, &vvl1, cc_Variable, &vvl2, &pyvlist, &pydlist))
      PYERROR(PyExc_TypeError, "invalid parameter; three variables expected", PYNULL);

    TClassifierByLookupTable2 *cblt = mlnew TClassifierByLookupTable2(vcl, vvl1, vvl2);
    return initializeTables(pyvlist, pydlist, cblt) ? WrapNewOrange(cblt, type) : PYNULL;
  PyCATCH
}


PyObject *ClassifierByLookupTable2__reduce__(PyObject *self)
{ 
  CAST_TO(TClassifierByLookupTable2, cblt);
  return Py_BuildValue("O(OOOOO)N", (PyObject *)(self->ob_type), 
                                   WrapOrange(cblt->classVar),
                                   WrapOrange(cblt->variable1),
                                   WrapOrange(cblt->variable2),
                                   WrapOrange(cblt->lookupTable),
                                   WrapOrange(cblt->distributions),
                                   packOrangeDictionary(self));
}


PyObject *ClassifierByLookupTable3_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ClassifierByLookupTable, "(class-descriptor, desc0, desc1, desc2)")
{ PyTRY
    PVariable vcl, vvl1, vvl2, vvl3;
    PyObject *pyvlist = PYNULL;
    PyObject *pydlist = PYNULL;
    if (!PyArg_ParseTuple(args, "O&O&O&O&|OO", cc_Variable, &vcl, cc_Variable, &vvl1, cc_Variable, &vvl2, cc_Variable, &vvl3, &pyvlist, &pydlist))
      PYERROR(PyExc_TypeError, "invalid parameter; four variables expected", PYNULL);

    TClassifierByLookupTable3 *cblt = mlnew TClassifierByLookupTable3(vcl, vvl1, vvl2, vvl3);
    return initializeTables(pyvlist, pydlist, cblt) ? WrapNewOrange(cblt, type) : PYNULL;
  PyCATCH
}

PyObject *ClassifierByLookupTable3__reduce__(PyObject *self)
{ 
  CAST_TO(TClassifierByLookupTable3, cblt);
  return Py_BuildValue("O(OOOOOO)N", (PyObject *)(self->ob_type), 
                                   WrapOrange(cblt->classVar),
                                   WrapOrange(cblt->variable1),
                                   WrapOrange(cblt->variable2),
                                   WrapOrange(cblt->variable3),
                                   WrapOrange(cblt->lookupTable),
                                   WrapOrange(cblt->distributions),
                                   packOrangeDictionary(self));
}



PyObject *ClassifierByLookupTable_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Classifier, "(class-descriptor, descriptor)")
{ 
  static newfunc constructors[] = {ClassifierByLookupTable1_new, ClassifierByLookupTable2_new, ClassifierByLookupTable3_new};
  static TOrangeType *types[] = {&PyOrClassifierByLookupTable1_Type, &PyOrClassifierByLookupTable2_Type, &PyOrClassifierByLookupTable3_Type};
  if (!args || (PyTuple_Size(args)<2))
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

/* arguments in form (list, classvar ...) */

  PyObject *g = PyTuple_GET_ITEM(args, 1);
  PVarList variables = PVarList_FromArguments(PyTuple_GET_ITEM(args, 1));

  if (variables) {
    int vsize = variables->size();
    int asize = PyTuple_Size(args);
    int i;

    if (!PyOrVariable_Check(PyTuple_GET_ITEM(args, 0)))
      PYERROR(PyExc_TypeError, "the second argument should be the class attribute", PYNULL);

    if (vsize <= 3) {
      PyObject *newargs = PyTuple_New(vsize + asize-1);
      PyObject *elm = NULL;
      int el = 0;

      elm = PyTuple_GET_ITEM(args, 0);
      Py_INCREF(elm);
      PyTuple_SetItem(newargs, el++, elm);

      const_PITERATE(TVarList, vi, variables)
        PyTuple_SetItem(newargs, el++, WrapOrange(*vi));

      for(i = 2; i != asize; i++) {
        elm = PyTuple_GET_ITEM(args, i);
        Py_INCREF(elm);
        PyTuple_SetItem(newargs, el++, elm);
      }

      try {
        PyObject *res = constructors[vsize-1](type == (PyTypeObject *)(&PyOrClassifierByLookupTable_Type) ? (PyTypeObject *)(types[vsize-1]) : type, newargs, kwds);
        Py_DECREF(newargs);
        return res;
      }
      catch (...) {
        Py_DECREF(newargs);
        throw;
      }
    }

    /* arguments in form (var1, var2, ..., classvar) */

    else {
      TClassifierByLookupTableN *cblt = mlnew TClassifierByLookupTableN(PyOrange_AsVariable(PyTuple_GET_ITEM(args, 0)), variables);

      PyObject *pyvl = asize>=3 ? PyTuple_GET_ITEM(args, 2) : PYNULL;
      PyObject *pydl = asize>=3 ? PyTuple_GET_ITEM(args, 3) : PYNULL;
      return initializeTables(pyvl, pydl, cblt) ? WrapNewOrange(cblt, type) : PYNULL;
    }
  }

  PyErr_Clear();

  int i = 0, e = PyTuple_Size(args);
  for(; (i<e) && PyOrVariable_Check(PyTuple_GET_ITEM(args, i)); i++);

  if ((i<2) || (i>4) || (e-i > 2))
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

  return constructors[i-2](type == (PyTypeObject *)(&PyOrClassifierByLookupTable_Type) ? (PyTypeObject *)(types[i-2]) : type, args, kwds);
}


PyObject *ClassifierByLookupTable__reduce__(PyObject *self)
{
  // Python class ClassifierByLookupTable represents C++'s ClassifierByLookupTableN
  CAST_TO(TClassifierByLookupTableN, cblt);

  return Py_BuildValue("O(OOOO)N", (PyObject *)(self->ob_type), 
                                   WrapOrange(cblt->classVar),
                                   WrapOrange(cblt->variables),
                                   WrapOrange(cblt->lookupTable),
                                   WrapOrange(cblt->distributions),
                                   packOrangeDictionary(self));
}

#include "lib_kernel.px"
