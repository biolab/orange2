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


/********************************

This file includes constructors and specialized methods for classes defined in project Kernel

*********************************/

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
#include "cls_orange.hpp"
#include "lib_kernel.hpp"
#include "vectortemplates.hpp"


#include "externs.px"

#include "converts.hpp"

WRAPPER(ExampleTable);



/* ************ VARIABLE ************ */

PVarList PVarList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::P_FromArguments(arg); }
PyObject *VarList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_FromArguments(type, arg); }
PyObject *VarList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Variable>)") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_new(type, arg, kwds); }
PyObject *VarList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_getitem(self, index); }
int       VarList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_setitem(self, index, item); }
PyObject *VarList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_getslice(self, start, stop); }
int       VarList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_setslice(self, start, stop, item); }
int       VarList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_len(self); }
PyObject *VarList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_concat(self, obj); }
PyObject *VarList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_repeat(self, times); }
PyObject *VarList_str(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_str(self); }
PyObject *VarList_repr(TPyOrange *self) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_str(self); }
int       VarList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_contains(self, obj); }
PyObject *VarList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Variable) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_append(self, item); }
PyObject *VarList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> int") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_count(self, obj); }
PyObject *VarList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> VarList") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_filter(self, args); }
PyObject *VarList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> int") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_index(self, obj); }
PyObject *VarList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_insert(self, args); }
PyObject *VarList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_native(self); }
PyObject *VarList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Variable") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_pop(self, args); }
PyObject *VarList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Variable) -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_remove(self, obj); }
PyObject *VarList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_reverse(self); }
PyObject *VarList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func] -> None") { return ListOfWrappedMethods<PVarList, TVarList, PVariable, (PyTypeObject *)&PyOrVariable_Type>::_sort(self, args); }


PVarListList PVarListList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::P_FromArguments(arg); }
PyObject *VarListList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_FromArguments(type, arg); }
PyObject *VarListList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of VarList>)") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_new(type, arg, kwds); }
PyObject *VarListList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_getitem(self, index); }
int       VarListList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_setitem(self, index, item); }
PyObject *VarListList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_getslice(self, start, stop); }
int       VarListList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_setslice(self, start, stop, item); }
int       VarListList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_len(self); }
PyObject *VarListList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_concat(self, obj); }
PyObject *VarListList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_repeat(self, times); }
PyObject *VarListList_str(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_str(self); }
PyObject *VarListList_repr(TPyOrange *self) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_str(self); }
int       VarListList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_contains(self, obj); }
PyObject *VarListList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(VarList) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_append(self, item); }
PyObject *VarListList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> int") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_count(self, obj); }
PyObject *VarListList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> VarListList") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_filter(self, args); }
PyObject *VarListList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> int") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_index(self, obj); }
PyObject *VarListList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_insert(self, args); }
PyObject *VarListList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_native(self); }
PyObject *VarListList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> VarList") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_pop(self, args); }
PyObject *VarListList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(VarList) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_remove(self, obj); }
PyObject *VarListList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_reverse(self); }
PyObject *VarListList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PVarListList, TVarListList, PVarList, (PyTypeObject *)&PyOrVarList_Type>::_sort(self, args); }

PVarList knownVars(PyObject *keywords)
{
  PVarList variables;
  PyObject *pyknownVars=keywords ? PyDict_GetItemString(keywords, "use") : PYNULL;
  if (!pyknownVars)
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

  PyDict_DelItemString(keywords, "use");
  return variables;
}


BASED_ON(Variable, Orange)
C_NAMED(IntVariable, Variable, "([name=, startValue=, endValue=, distributed=, getValueFrom=])")
C_NAMED(EnumVariable, Variable, "([name=, values=, autoValues=, distributed=, getValueFrom=])")
C_NAMED(FloatVariable, Variable, "([name=, startValue=, endValue=, stepValue=, distributed=, getValueFrom=])")

#include "stringvars.hpp"
C_NAMED(StringVariable, Variable, "([name=])")

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

PyObject *Variable_nextvalue(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(value)  -> Value | None")
{ PyTRY
    CAST_TO(TVariable, var);
    TPyValue *val;
    if (   !PyArg_ParseTuple(args, "O", &val)
        || !PyOrValue_Check(val)
        || (val->variable ? (val->variable!=var) : (val->value.varType!=var->varType)))
      PYERROR(PyExc_TypeError, "invalid value parameter", PYNULL);

    TValue sval=val->value;
    if (!var->nextValue(sval)) RETURN_NONE;

    return Value_FromVariableValue(PyOrange_AsVariable(self), sval);
  PyCATCH
}


PyObject *Variable_computeValue(PyObject *self, PyObject *args) PYARGS(METH_O, "(example) -> Value")
{
  CAST_TO(TVariable, var);
  if (!var->getValueFrom)
    PYERROR(PyExc_SystemError, "Variable.computeValue: 'getValueFrom' not defined", PYNULL);

  if (!PyOrExample_Check(args))
    PYERROR(PyExc_TypeError, "Variable.computeValue: 'Example' expected", PYNULL);

  return Value_FromVariableValue(var, var->computeValue(PyExample_AS_ExampleReference(args)));
}


PyObject *Variable_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(value) -> Value")
{ PyTRY
    SETATTRIBUTES

    PyObject *object;
    if (!PyArg_ParseTuple(args, "O", &object))
      PYERROR(PyExc_TypeError, "invalid parameter", PYNULL);

    TValue value;
    if (!convertFromPython(object, value, PyOrange_AsVariable(self)))
      return PYNULL;
    return Value_FromVariableValue(PyOrange_AsVariable(self), value);
  PyCATCH
}



bool convertFromPythonWithVariable(PyObject *obj, string &str)
{ return convertFromPythonWithML(obj, str, *FindOrangeType(typeid(TVariable))); }


bool varListFromDomain(PyObject *boundList, PDomain domain, TVarList &boundSet, bool allowSingle, bool checkForIncludance)
{ if (PyOrVarList_Check(boundList)) {
    PVarList variables = PyOrange_AsVarList(boundList);
    if (checkForIncludance)
      const_PITERATE(TVarList, vi, variables)
        if (!domain || (!exists(domain->variables.getReference(), *vi) && (domain->getMetaNum(*vi, false)==-1))) {
          PyErr_Format(PyExc_IndexError, "variable '%s' does not exist in the domain", (*vi)->name.c_str());
          return false;
        }
    boundSet=variables.getReference();
    return true;
  }
  
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
      if (PyString_Check(obj))
        return domain->getVar(PyString_AS_STRING(obj), true);
      if (PyInt_Check(obj)) {
        int idx=PyInt_AsLong(obj);
        if (idx<0) 
          return domain->getMetaVar(-idx);
        if (idx>=int(domain->variables->size()))
          PYERROR(PyExc_IndexError, "index out of range", PVariable());
        return domain->variables->at(idx);
      }
    PyCATCH_r(PVariable())
  }

  if (PyOrVariable_Check(obj)) {
    PVariable var(PyOrange_AsVariable(obj));
    if (checkForIncludance)
      if (!domain || (!exists(domain->variables.getReference(), var) && (domain->getMetaNum(var, false)==-1)))
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

  attrNo = -domain->getMetaNum(var, false);
  if (attrNo<0)
    return true;

  return false;
}


PyObject *StringValue_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SomeValue, "(string)")
{ char *s;
  if (!PyArg_ParseTuple(args, "s:StringValue", &s))
    return PYNULL;

  return WrapNewOrange(mlnew TStringValue(s), type);
}



/* ************ DOMAIN ************ */

#include "domain.hpp"

PyObject *Domain_metaid(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(name | descriptor) -> int")
{ PyTRY
    CAST_TO(TDomain, domain);

    PyObject *rar;
    if (!PyArg_ParseTuple(args, "O", &rar))
      PYERROR(PyExc_AttributeError, "string or variable expected", PYNULL);

    TMetaDescriptor *desc=(TMetaDescriptor *)NULL;
    if (PyString_Check(rar))
      desc=domain->metas[PyString_AsString(rar)];
    else if (PyOrVariable_Check(rar))
      desc=domain->metas[PyOrange_AS(TVariable, rar).name];

    if (!desc) PYERROR(PyExc_AttributeError, "meta variable does not exist", PYNULL);
    return PyInt_FromLong(desc->id);
  PyCATCH
}

PyObject *Domain_getmetas(TPyOrange *self, PyObject *args) PYARGS(0, "() -> {int: Variable}")
{ PyTRY
    CAST_TO(TDomain, domain);

    if (args && !PyArg_ParseTuple(args, ""))
      PYERROR(PyExc_AttributeError, "no arguments expected", PYNULL);

    PyObject *dict=PyDict_New();
    ITERATE(TMetaVector, mi, domain->metas)
      PyDict_SetItem(dict, PyInt_FromLong((*mi).id), WrapOrange((*mi).variable));

    return dict;
  PyCATCH
}
    
PyObject *Domain_addmeta(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "({id0:desc0, id1:desc1, ...}) | (id, descriptor) -> None")
{ PyTRY
    CAST_TO(TDomain, domain);

    int id;
    PyObject *pyvar;
    if (PyArg_ParseTuple(args, "iO", &id, &pyvar))
      if (!PyOrVariable_Check(pyvar))
        PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL)
      else 
        domain->metas.push_back(TMetaDescriptor(id, PyOrange_AsVariable(pyvar)));
    else if (PyArg_ParseTuple(args, "O", &pyvar))
      if (!PyDict_Check(pyvar))
        PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL)
      else {
        int pos=0;
        PyObject *key, *value;
        vector<TMetaDescriptor> tempMetas;
        while (PyDict_Next(pyvar, &pos, &key, &value))
          if (PyInt_Check(key) && PyOrVariable_Check(value))
            tempMetas.push_back(TMetaDescriptor(PyInt_AsLong(key), PyOrange_AsVariable(value)));
          else PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);
        ITERATE(vector<TMetaDescriptor>, mi, tempMetas)
          domain->metas.push_back(*mi);
      }
    else PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL)

    PyErr_Clear();

    RETURN_NONE;
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
  else
    mvi=mve;

  if (mvi==mve)
    PYERROR(PyExc_AttributeError, "meta value not found", false);

  metas.erase(mvi);
  return true;
}


PyObject *Domain_removemeta(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "({id0:desc0, id1:desc1, ...}) | ([id0|desc0, id1|desc1, ...]) -> None")
{ PyTRY
    CAST_TO(TDomain, domain);

    PyObject *rar;
    if (!PyArg_ParseTuple(args, "O", &rar))
      PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

    if (PyDict_Check(rar)) {
      int pos=0;
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
    }

    else if (PyList_Check(rar)) {
      TMetaVector newMetas=domain->metas;
      for(int pos=0, noel=PyList_Size(rar); pos!=noel; pos++)
        if (!removeMeta(PyList_GetItem(rar, pos), newMetas))
          return PYNULL;
      domain->metas=newMetas;
    }
  
    else if (!removeMeta(rar, domain->metas))
      return PYNULL;

    RETURN_NONE;
  PyCATCH
}




int Domain_len(TPyOrange *self)
{ PyTRY
    CAST_TO_err(TDomain, domain, -1);
    return domain->variables->size();
  PyCATCH_1
}


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
          if (PyOrDomain_Check(arg2))
            source = PyOrange_AsDomain(arg2)->variables;
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


PyObject *Domain_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example) -> Example")
{ PyTRY
    SETATTRIBUTES
    PyObject *pex;
    if (PyArg_ParseTuple(args, "O", &pex) && PyOrExample_Check(pex))
      return Example_FromWrappedExample(PExample(mlnew TExample(PyOrange_AsDomain(self), PyExample_AS_ExampleReference(pex))));
    PYERROR(PyExc_TypeError, "invalid parameters (Example expected)", PYNULL);
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
    return WrapOrange(domain->variables->at(index));
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

 

string TDomain2string(TPyOrange *self)
{ CAST_TO_err(TDomain, domain, "<invalid domain>")

  string res;
  
  int added=0;
  PITERATE(TVarList, vi, domain->variables)
    res+=(added++ ? ", " : "[") +(*vi)->name;

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
    sprintf(pls, "%s%i:%s", (added++) ? ", " : "", int((*mi).id), (*mi).variable->name.c_str());
    res+=pls;
  }
  if (added)
    res+="}";

  return res;
}



PyObject *Domain_repr(TPyOrange *pex)
{ PyTRY
    return PyString_FromString(TDomain2string(pex).c_str());
  PyCATCH
}


PyObject *Domain_str(TPyOrange *pex)
{ PyTRY
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



/* ************ RANDOM GENERATORS ************** */

#include "random.hpp"

C_UNNAMED(RandomGenerator, Orange, "() -> 32-bit random int")

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
    SETATTRIBUTES
    if (!PyArg_ParseTuple(args, ""))
      PYERROR(PyExc_TypeError, "no arguments expected", PYNULL);
    return PyInt_FromLong((long)SELF_AS(TRandomGenerator)());
  PyCATCH
}
  

PyObject *stdRandomGenerator()
{ return WrapOrange(globalRandom); }

PYCONSTANTFUNC(globalRandom, stdRandomGenerator)


/* ************ EXAMPLE GENERATOR ************ */

#include "examplegen.hpp"
BASED_ON(ExampleGenerator, Orange)

#include "table.hpp"
#include "filter.hpp"


int pt_ExampleGenerator(PyObject *args, void *egen)
{ *(PExampleGenerator *)(egen) = PyOrExampleGenerator_Check(args) ? PyOrange_AsExampleGenerator(args)
                                                                  : PExampleGenerator(readListOfExamples(args));

  if (!*(PExampleGenerator *)(egen))
    PYERROR(PyExc_TypeError, "invalid example generator", 0)
  else
    return 1;
}


PExampleGenerator exampleGenFromParsedArgs(PyObject *args)
{
 return PyOrOrange_Check(args) ? PyOrange_AsExampleGenerator(args)
                               : PExampleGenerator(readListOfExamples(args));
}

PExampleGenerator exampleGenFromArgs(PyObject *args, long *weightID)
{ PyObject *examples;
  int aweight = 0;
  if (PyArg_ParseTuple(args, (char *)(weightID ? "O|i" : "O"), &examples, &aweight)) {
    if (weightID) *weightID=aweight;

    PExampleGenerator egen=exampleGenFromParsedArgs(examples);
    if (egen)
      return egen;
  }
  
  PYERROR(PyExc_TypeError, "attribute error", PExampleGenerator());
}


PyObject *ExampleGenerator_native(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "([nativity, tuple=]) -> examples")
{ PyTRY
    bool tuples = false;
    if (keywords) {
      PyObject *pytuples=PyDict_GetItemString(keywords, "tuple");
      if (pytuples) {
        tuples = PyObject_IsTrue(pytuples)!=0;

        // We don't want to modify the original keywords dictionary
        // (God knows what it is - this might be called by apply(ExampleGenerator, args, dict)).
        keywords = PyDict_Copy(keywords);
        PyDict_DelItemString(keywords, "tuple");
        SETATTRIBUTES 
        Py_DECREF(keywords);
      }
      else
        SETATTRIBUTES
    }

    int natvt=2;
    if (args && !PyArg_ParseTuple(args, "|i", &natvt) || ((natvt>=2)))
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);
    CAST_TO(TExampleGenerator, eg);

    PyObject *list=PyList_New(0);
    EITERATE(ei, *eg)
      if (natvt<=1) {
        PyObject *obj=convertToPythonNative(*ei, natvt, tuples);
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

inline PPreprocessor pp_sameValues(PyObject *dict)
{ return mlnew TPreprocessor_take(PVariableFilterMap_FromArguments(dict)); }

inline PPreprocessor filter_sameValues(PyObject *dict, PDomain domain)
{ return TPreprocessor_take::constructFilter(PVariableFilterMap_FromArguments(dict), domain); }

PyObject *applyPreprocessor(PPreprocessor preprocessor, PExampleGenerator gen, bool weightGiven, int weightID)
{ if (!preprocessor)
    return PYNULL;

  int newWeight;
  PExampleGenerator newGen = preprocessor->call(gen, weightID, newWeight);
  return weightGiven ? Py_BuildValue("Ni", WrapOrange(newGen), newWeight) : WrapOrange(newGen);
}



PyObject *applyFilter(PFilter filter, PExampleGenerator gen, bool weightGiven, int weightID);

PyObject *ExampleGenerator_select(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ 
  PyTRY
    long weightID=-999;
    CAST_TO(TExampleGenerator, eg);
    PExampleGenerator weg = PyOrange_AsExampleGenerator(self);

    /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS KEYWORDS ***** */
    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return applyPreprocessor(pp_sameValues(keywords), weg, false, 0);
    }

    PyObject *mplier;
    if (PyArg_ParseTuple(args, "O|i", &mplier, &weightID)) {
      PyObject *pyneg= keywords ? PyDict_GetItemString(keywords, "negate") : NULL;
      bool negate = pyneg && PyObject_IsTrue(pyneg);
      bool weightGiven = (weightID!=-999);
      if (weightID==-1)
        weightID=0;

      /* ***** SELECTION BY VECTOR OF BOOLS ****** */
      if (PyList_Check(mplier) && PyList_Size(mplier) && PyInt_Check(PyList_GetItem(mplier, 0))) {
        int nole=PyList_Size(mplier);
        
        TExampleTable *newTable=mlnew TExampleTable(eg->domain);
        PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error
        int i=0;

        TExampleIterator ei=eg->begin();
        for(;ei; ++ei) {
          if (i==nole)
            break;
          PyObject *lel = PyList_GetItem(mplier, i++);
          if (!PyInt_Check(lel))
            break;

          else if (negate != (weightGiven ? (int(PyInt_AsLong(lel))==weightID) : (PyObject_IsTrue(lel)!=0)))
            newTable->addExample(*ei);
        }

        if ((i==nole) && !ei)
          return WrapOrange(newGen);
      }

      PyErr_Clear();


      /* ***** SELECTION BY LONGLIST ****** */
      if (PyOrLongList_Check(mplier)) {
        PLongList llist = PyOrange_AsLongList(mplier);
        TLongList::iterator lli(llist->begin()), lle(llist->end());
        
        TExampleTable *newTable=mlnew TExampleTable(eg->domain);
        PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

        TExampleIterator ei=eg->begin();
        for(;ei && (lli!=lle); ++ei, lli++)
          if (negate != (weightGiven ? (*lli==weightID) : !!*lli))
            newTable->addExample(*ei);

        if ((lli==lle) && !ei)
          return WrapOrange(newGen);

        PYERROR(PyExc_IndexError, "ExampleGenerator.select: invalid list size", PYNULL)
      }

      PyErr_Clear();

      /* ***** CHANGING DOMAIN ***** */
      if (PyOrDomain_Check(mplier)) {
        PyObject *wrappedGen=WrapOrange(PExampleTable(mlnew TExampleTable(PyOrange_AsDomain(mplier), weg)));
        return weightGiven ? Py_BuildValue("Ni", wrappedGen, weightID) : wrappedGen;
      }


      /* ***** SELECTION BY VECTOR OF NAMES, INDICES AND VARIABLES ****** */
      TVarList attributes;
      if (varListFromDomain(mplier, eg->domain, attributes, true, false)) {
        PDomain newDomain;
        TVarList::iterator vi, ve;
        for(vi=attributes.begin(), ve=attributes.end(); (vi!=ve) && (*vi!=eg->domain->classVar); vi++);
        if (vi==ve)
          newDomain=mlnew TDomain(PVariable(), attributes);
        else {
          attributes.erase(vi);
          newDomain=mlnew TDomain(eg->domain->classVar, attributes);
        }

        PyObject *wrappedGen=WrapOrange(PExampleTable(mlnew TExampleTable(newDomain, weg)));
        return weightGiven ? Py_BuildValue("Ni", wrappedGen, weightID) : wrappedGen;
      }

      PyErr_Clear();


      /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS DICTIONARY ***** */
      if (PyDict_Check(mplier))
        return applyFilter(pp_sameValues(mplier), weg, weightGiven, weightID);


      /* ***** PREPROCESSING ***** */
      if (PyOrPreprocessor_Check(mplier)) {
        PExampleGenerator res;
        int newWeight;
        PyTRY
          NAME_CAST_TO(TPreprocessor, mplier, pp);
          if (!pp)
            PYERROR(PyExc_TypeError, "invalid object type (preprocessor announced, but not passed)", PYNULL)
          res = (*pp)(weg, weightID, newWeight);
        PyCATCH

        return weightGiven ? Py_BuildValue("Ni", WrapOrange(res), newWeight) : WrapOrange(res);
      }

      /* ***** APPLY FILTER ***** */
      else if (PyOrFilter_Check(mplier))
        return applyFilter(PyOrange_AsFilter(mplier), weg, weightGiven, weightID);
    }
    PYERROR(PyExc_TypeError, "ExampleGenerator.select failed, tried:\n"
                             "- example selector (list of ints) of correct length\n"
                             "- list of attributes (names, indices, Variables)\n"
                             "- values of attributes (as keyword parameters)\n"
                             "- Preprocessor, Domain, Filter", PYNULL);
  PyCATCH
}


PExampleGeneratorList PExampleGeneratorList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::P_FromArguments(arg); }
PyObject *ExampleGeneratorList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_FromArguments(type, arg); }
inline PyObject *ExampleGeneratorList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ExampleGenerator>)") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_new(type, arg, kwds); }
inline PyObject *ExampleGeneratorList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_getitem(self, index); }
inline int       ExampleGeneratorList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_setitem(self, index, item); }
inline PyObject *ExampleGeneratorList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_getslice(self, start, stop); }
inline int       ExampleGeneratorList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_setslice(self, start, stop, item); }
inline int       ExampleGeneratorList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_len(self); }
inline PyObject *ExampleGeneratorList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_concat(self, obj); }
inline PyObject *ExampleGeneratorList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_repeat(self, times); }
inline PyObject *ExampleGeneratorList_str(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_str(self); }
inline PyObject *ExampleGeneratorList_repr(TPyOrange *self) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_str(self); }
inline int       ExampleGeneratorList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_contains(self, obj); }
inline PyObject *ExampleGeneratorList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ExampleGenerator) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_append(self, item); }
inline PyObject *ExampleGeneratorList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> int") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_count(self, obj); }
inline PyObject *ExampleGeneratorList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ExampleGeneratorList") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_filter(self, args); }
inline PyObject *ExampleGeneratorList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> int") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_index(self, obj); }
inline PyObject *ExampleGeneratorList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_insert(self, args); }
inline PyObject *ExampleGeneratorList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_native(self); }
inline PyObject *ExampleGeneratorList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ExampleGenerator") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_pop(self, args); }
inline PyObject *ExampleGeneratorList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ExampleGenerator) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_remove(self, obj); }
inline PyObject *ExampleGeneratorList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_reverse(self); }
inline PyObject *ExampleGeneratorList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PExampleGeneratorList, TExampleGeneratorList, PExampleGenerator, (PyTypeObject *)&PyOrExampleGenerator_Type>::_sort(self, args); }

  

/* ************ EXAMPLE TABLE ************ */

#include "table.hpp"
#include "readdata.hpp"


TExampleTable *readListOfExamples(PyObject *args)
{ if (PyList_Check(args)) {
    int size=PyList_Size(args);
    if (!size)
      PYERROR(PyExc_TypeError, "can't construct a table from an empty list", (TExampleTable *)NULL);

    TExampleTable *table=NULL;
 
    for(int i=0; i<size; i++) {
      PyObject *pex=PyList_GetItem(args, i);
      if (!PyOrExample_Check(pex)) {
        mldelete table;
        PYERROR(PyExc_TypeError, "example list is expected to consist of Example's", NULL);
      }
      if (!i)
        table=mlnew TExampleTable(PyExample_AS_Example(pex)->domain);
      table->addExample(PyExample_AS_ExampleReference(pex));
    }

    return table;
  }

  PYERROR(PyExc_TypeError, "invalid type", NULL);
}


PyObject *ExampleTable_new(PyTypeObject *type, PyObject *argstuple, PyObject *keywords) BASED_ON(ExampleGenerator, "(filename | domain | examples)")
{ PyTRY
    TExampleTable *res=NULL;

    PyObject *args;
    char *filename = NULL;

    if (PyArg_ParseTuple(argstuple, "|s", &filename))
      res = readData(filename, knownVars(keywords));

    else if (PyArg_ParseTuple(argstuple, "O", &args))
      if (PyOrOrange_Check(args)) {
        if (PyOrDomain_Check(args))
          res = mlnew TExampleTable(PyOrange_AsDomain(args));

        else if (PyOrExampleGenerator_Check(args))
          res = mlnew TExampleTable(PyOrange_AsExampleGenerator(args));
      }
      else
        res = readListOfExamples(args); // this will return an error if needed

    if (!res)
      PYERROR(PyExc_TypeError, "invalid type", PYNULL);

    PyErr_Clear();
    return WrapNewOrange(res, type);
  PyCATCH
}



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
    EITERATE(ei, *table) {
      // here we wrap a reference to example, so we must pass a self's wrapper
      PyObject *example = Example_FromExampleRef(*ei, PyOrange_AsExampleTable(self)); 
      if (!example) {
        PyMem_DEL(list);
        PYERROR(PyExc_SystemError, "out of memory", PYNULL);
      }      
      PyList_SetItem(list, i++, example);
    }

    return list;
  PyCATCH
}


int ExampleTable_len_sq(PyObject *self) 
{ PyTRY
    return SELF_AS(TExampleGenerator).numberOfExamples();
  PyCATCH_1
}


PyObject *ExampleTable_append(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples) ->None | (example) -> None")
{ PyTRY
    CAST_TO(TExampleTable, table)

    PyObject *parsed;
    if (PyArg_ParseTuple(args, "O", &parsed) && PyOrExample_Check(parsed)) {
      table->addExample(PyExample_AS_ExampleReference(parsed));
      RETURN_NONE;
    }

    PyErr_Clear();

    PExampleGenerator egen=exampleGenFromArgs(args);
    if (egen) {
      // to prevent cycling in t.append(t)
      TExampleTable tab(egen);
      EITERATE(ei, tab)
        table->addExample(*ei);
      RETURN_NONE;
    }

    if (PyList_Check(parsed)) {
      int size=PyList_Size(parsed);

      for(int i=0; i<size; i++) {
        PyObject *pex=PyList_GetItem(parsed, i);
        TExample example(table->domain);
        if (!convertFromPython(pex, example, table->domain))
          return PYNULL;

        table->addExample(example);
      }

      RETURN_NONE;
    }

    PYERROR(PyExc_TypeError, "invalid type", PYNULL);
  PyCATCH
}


PyObject *ExampleTable_getitem_sq(TPyOrange *self, int idx)
{
  PyTRY
    CAST_TO(TExampleTable, table);
    if (idx<0) idx+=table->numberOfExamples();
    if ((idx<0) || (idx>=table->numberOfExamples()))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);
    // here we wrap a reference to example, so we must pass a self's wrapper
    return Example_FromExampleRef((*table)[idx], PyOrange_AsExampleTable(self));
  PyCATCH
}


int ExampleTable_setitem_sq(TPyOrange *self, int idx, TPyExample *pex)
{ 
  PyTRY
    CAST_TO_err(TExampleTable, table, -1);

    if (idx>table->numberOfExamples())
      PYERROR(PyExc_IndexError, "index out of range", -1);

    if (!pex) {
      table->erase(idx);
      return 0;
    }

    if (!PyOrExample_Check(pex))
      PYERROR(PyExc_TypeError, "invalid parameter type (Example expected)", -1)

    (*table)[idx]=TExample(table->domain, PyExample_AS_ExampleReference(pex));
    return 0;
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
    while(start<stop) {
      // here we wrap a reference to example, so we must pass a self's wrapper
      PyObject *example=Example_FromExampleRef((*table)[start++], PyOrange_AsExampleTable(self));
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

    // this must be checked separately since exampleGenFromParsedArgs does not
    // (and cannot, due to unknown domain) read empty lists
    if (PyList_Check(args) && !PyList_Size(args)) {
      table->erase(start, stop);
      return 0;
    }

    PExampleGenerator egen = exampleGenFromParsedArgs(args);
    if (!egen)
      return -1;
      
    table->erase(start, stop);
    PEITERATE(ei, egen)
      table->insert(start++, *ei);

    return 0;
  PyCATCH_1
}


PyObject *applyFilterL(PFilter filter, PExampleGenerator gen, bool, int)
{ if (!filter)
    return PYNULL;

  PyObject *list=PyList_New(0);
  filter->reset();
  PEITERATE(ei, gen)
    if (filter->operator()(*ei)) {
      PyObject *obj=Example_FromExampleRef(*ei, gen);
      PyList_Append(list, obj);
      Py_DECREF(obj);
    }

  return list;
}


PyObject *applyFilterP(PFilter filter, PExampleGenerator gen, bool, int)
{ if (!filter) return PYNULL;

  TExamplePointerTable *newTable=mlnew TExamplePointerTable(gen->domain);
  PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error
  filter->reset();
  PEITERATE(ei, gen)
    if (filter->operator()(*ei))
      newTable->addExample(*ei);

  return WrapOrange(newGen);
}


PyObject *ExampleTable_multipleselect(TPyOrange *self, PyObject *args)  PYARGS(METH_VARARGS, "(indices) -> ExampleTable")
{ PyTRY
    CAST_TO(TExampleTable, eg);
    PyObject *pylist;
    if (   !PyArg_ParseTuple(args, "O", &pylist)
        || !PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
    
    TExampleTable *newTable=mlnew TExampleTable(eg->domain);
    PExampleGenerator newGen(newTable);

    for(int sze=PyList_Size(pylist), i=0; i<sze; i++) {
      PyObject *lel=PyList_GetItem(pylist, i);
      if (!PyInt_Check(lel))
        PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
      newTable->addExample(eg->at(int(PyInt_AsLong(lel))));
    }

    return WrapOrange(newGen);
  PyCATCH
}


PyObject *ExampleTable_multipleselectref(TPyOrange *self, PyObject *args)   PYARGS(METH_VARARGS, "(indices) -> ExamplePointerTable")
{ PyTRY
    CAST_TO(TExampleTable, eg);
    PyObject *pylist;
    if (   !PyArg_ParseTuple(args, "O", &pylist)
        || !PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
    
    TExamplePointerTable *newTable=mlnew TExamplePointerTable(eg->domain);
    PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

    for(int sze=PyList_Size(pylist), i=0; i<sze; i++) {
      PyObject *lel=PyList_GetItem(pylist, i);
      if (!PyInt_Check(lel))
        PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
      newTable->addExample(eg->at(int(PyInt_AsLong(lel))));
    }

    return WrapOrange(newGen);
  PyCATCH
}


PyObject *ExampleTable_selectLow(TPyOrange *self, PyObject *args, PyObject *keywords, bool toList)
{ 
  PyTRY
    int weightID=-999;
    CAST_TO(TExampleTable, eg);
    POrange lock = PyOrange_AS_Orange(self);
    PExampleGenerator weg = PExampleGenerator(lock);

    /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS KEYWORDS ***** */
    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return toList ? applyFilterL(filter_sameValues(keywords, eg->domain), weg, false, 0)
                    : applyFilterP(filter_sameValues(keywords, eg->domain), weg, false, 0);
    }

    /* ***** SELECTION BY PYLIST ****** */
    PyObject *mplier;
    if (PyArg_ParseTuple(args, "O|i", &mplier, &weightID)) {
      PyObject *pyneg= keywords ? PyDict_GetItemString(keywords, "negate") : NULL;
      bool negate=pyneg && PyObject_IsTrue(pyneg);
      bool weightGiven=(weightID!=-999);
      if (weightID==-1) weightID=0;

      if (PyList_Check(mplier)) {
        if (PyList_Size(mplier)!=eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "example selector of invalid length", PYNULL);

        int i=0;
        if (toList) {
          PyObject *list=PyList_New(eg->numberOfExamples());
          EITERATE(ei, *eg) {
            PyObject *lel=PyList_GetItem(mplier, i);
            if (!PyInt_Check(lel))
              PYERROR(PyExc_IndexError, "invalid objects in example selector", PYNULL)
            else if (negate != (weightGiven ? (int(PyInt_AsLong(lel))==weightID) : (PyObject_IsTrue(lel)!=0)))
              PyList_SetItem(list, i, Example_FromExampleRef(*ei, lock));
            i++;
          }

          return list;
        }
        
        else { // !toList

          TExamplePointerTable *newTable=mlnew TExamplePointerTable(eg->domain);
          PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

          EITERATE(ei, *eg) {
            PyObject *lel=PyList_GetItem(mplier, i++);
            if (!PyInt_Check(lel))
              PYERROR(PyExc_IndexError, "invalid objects in example selector", PYNULL)
            else if (negate != (weightGiven ? (int(PyInt_AsLong(lel))==weightID) : (PyObject_IsTrue(lel)!=0)))
              newTable->addExample(*ei);
          }

          return WrapOrange(newGen);
        }
      }

      /* ***** SELECTION BY LONGLIST ****** */
      else if (PyOrLongList_Check(mplier)) {
        PLongList llist = PyOrange_AsLongList(mplier);
        if (int(llist->size()) != eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "select: invalid list size", PYNULL)

        TLongList::iterator lli(llist->begin()), lle(llist->end());
        TExampleIterator ei=eg->begin();
        
        int i=0;
        if (toList) {
          PyObject *list=PyList_New(eg->numberOfExamples());
          for(;ei && (lli!=lle); ++ei, lli++)
            if (negate != (weightGiven ? (*lli==weightID) : !!*lli))
              PyList_SetItem(list, i++, Example_FromExampleRef(*ei, lock));
          return list;
        }

        else { // !toList
          TExamplePointerTable *newTable=mlnew TExamplePointerTable(eg->domain);
          PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

          for(;ei && (lli!=lle); ++ei, lli++)
            if (negate != (weightGiven ? (*lli==weightID) : !!*lli))
              newTable->addExample(*ei);

          return WrapOrange(newGen);
        }
      }

      PyErr_Clear();


      /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS DICTIONARY ***** */
      if (PyDict_Check(mplier))
        return toList ? applyFilterL(filter_sameValues(mplier, eg->domain), weg, weightGiven, weightID)
                      : applyFilterP(filter_sameValues(mplier, eg->domain), weg, weightGiven, weightID);

      else if (PyOrFilter_Check(mplier))
        return toList ? applyFilterL(PyOrange_AsFilter(mplier), weg, weightGiven, weightID)
                      : applyFilterP(PyOrange_AsFilter(mplier), weg, weightGiven, weightID);
    }

  PYERROR(PyExc_TypeError, "invalid example selector type", PYNULL);
  PyCATCH
}


PyObject *ExampleTable_selectlist(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExampleTable_selectLow(self, args, keywords, true); 
  PyCATCH
}


PyObject *ExampleTable_selectref(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExampleTable_selectLow(self, args, keywords, false); 
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
    int weightID=0;
    if (!PyArg_ParseTuple(args, "|i", &weightID))
      PYERROR(PyExc_AttributeError, "at most one int argument expected", PYNULL);

    CAST_TO(TExampleTable, table);
    table->removeDuplicates(weightID);
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
    int id;
    PyObject *pyvalue=PYNULL;
    if (!PyArg_ParseTuple(args, "i|O", &id, &pyvalue))
      PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

    TValue value;
    if (!pyvalue)
      value=TValue(float(1.0));
    else if (!convertFromPython(pyvalue, value))
      PYERROR(PyExc_AttributeError, "invalid value argument", PYNULL);

    SELF_AS(TExampleTable).addMetaAttribute(id, value);

    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_removeMetaAttribute(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(id) -> None")
{ PyTRY
    int id;
    if (!PyArg_ParseTuple(args, "i", &id))
      PYERROR(PyExc_AttributeError, "invalid arguments", PYNULL);

    SELF_AS(TExampleTable).removeMetaAttribute(id);

    RETURN_NONE;
  PyCATCH
}


PyObject *ExampleTable_changeDomain(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(Domain) -> None")
{ PyTRY
    PyObject *pydomain;
    if (   !PyArg_ParseTuple(args, "O", &pydomain)
        || !PyOrDomain_Check(pydomain))
      PYERROR(PyExc_AttributeError, "domain argument expected", PYNULL);

    CAST_TO(TExampleTable, table);
    table->changeDomain(PyOrange_AsDomain(pydomain));
    RETURN_NONE;
  PyCATCH
}


/* ************ EXAMPLEREFTABLE ************ */


PyObject *ExamplePointerTable_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(ExampleTable, "(example-table | domain)")
{ PyTRY
    PyObject *table;
    if (PyArg_ParseTuple(args, "O", &table))

      if (PyOrExampleTable_Check(table))
        return WrapNewOrange(mlnew TExamplePointerTable(PyOrange_AsExampleGenerator(table)), type);

      if (PyOrDomain_Check(args))
        return WrapNewOrange(mlnew TExamplePointerTable(PyOrange_AsDomain(table)), type);

    PYERROR(PyExc_TypeError, "invalid parameters (ExampleTable or Domain expected)", PYNULL);
  PyCATCH
}


PyObject *ExamplePointerTable_append(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples) -> None")
{ PyTRY
    CAST_TO(TExamplePointerTable, table)

    PyObject *parsed;
    if (   PyArg_ParseTuple(args, "O", &parsed)
        && (PyOrExampleTable_Check(parsed) || PyOrExamplePointerTable_Check(parsed))) {

      table->addExamples(PyOrange_AsExampleGenerator(parsed));
      RETURN_NONE;
    }

    PYERROR(PyExc_TypeError, "ExamplePointerTable.append expects ExampleTable or ExamplePointerTable", PYNULL);
  PyCATCH
}


int ExamplePointerTable_setitem_sq(TPyOrange *self, int idx, TPyExample *pex)
{ 
  PyTRY
    CAST_TO_err(TExampleTable, table, -1);

    if (idx>table->numberOfExamples())
      PYERROR(PyExc_IndexError, "index out of range", -1);

    if (!pex) {
      table->erase(idx);
      return 0;
    }

    PYERROR(PyExc_SystemError, "cannot set items", -1);
  PyCATCH_1
}



PyObject *ExamplePointerTable_selectLow(TPyOrange *self, PyObject *args, PyObject *keywords, bool toList)
{ 
  PyTRY
    int weightID=-999;
    CAST_TO(TExamplePointerTable, eg);
    POrange lock=PyOrange_AS_Orange(self);
    PExampleGenerator weg = PExampleGenerator(lock);

    /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS KEYWORDS ***** */
    if (!PyTuple_Size(args) && NOT_EMPTY(keywords)) {
      return toList ? applyFilterL(filter_sameValues(keywords, eg->domain), weg, false, 0)
                    : applyFilterP(filter_sameValues(keywords, eg->domain), weg, false, 0);
    }

    PyObject *mplier;
    if (PyArg_ParseTuple(args, "O|i", &mplier, &weightID)) {
      PyObject *pyneg= keywords ? PyDict_GetItemString(keywords, "negate") : NULL;
      bool negate=pyneg && PyObject_IsTrue(pyneg);
      bool weightGiven=(weightID!=-999);
      if (weightID==-1) weightID=0;

      if (PyList_Check(mplier)) {
        if (PyList_Size(mplier)!=eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "example selector of invalid length", PYNULL);

        int i=0;
        if (toList) {
          PyObject *list=PyList_New(eg->numberOfExamples());
          EITERATE(ei, *eg) {
            PyObject *lel=PyList_GetItem(mplier, i);
            if (!PyInt_Check(lel))
              PYERROR(PyExc_IndexError, "invalid objects in example selector", PYNULL)
            else if (negate != (weightGiven ? (int(PyInt_AsLong(lel))==weightID) : (PyObject_IsTrue(lel)!=0)))
              PyList_SetItem(list, i, Example_FromExampleRef(*ei, lock));
            i++;
          }

          return list;
        }
        
        else { // !toList

          TExamplePointerTable *newTable=mlnew TExamplePointerTable(eg->domain);
          PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

          EITERATE(ei, *eg) {
            PyObject *lel=PyList_GetItem(mplier, i++);
            if (!PyInt_Check(lel))
              PYERROR(PyExc_IndexError, "invalid objects in example selector", PYNULL)
            else if (negate != (weightGiven ? (int(PyInt_AsLong(lel))==weightID) : (PyObject_IsTrue(lel)!=0)))
              newTable->addExample(*ei);
          }

          return WrapOrange(newGen);
        }
      }

      /* ***** SELECTION BY LONGLIST ****** */
      else if (PyOrLongList_Check(mplier)) {
        PLongList llist = PyOrange_AsLongList(mplier);
        if (int(llist->size()) != eg->numberOfExamples())
          PYERROR(PyExc_IndexError, "select: invalid list size", PYNULL)

        TLongList::iterator lli(llist->begin()), lle(llist->end());
        TExampleIterator ei=eg->begin();
        
        int i=0;
        if (toList) {
          PyObject *list=PyList_New(eg->numberOfExamples());
          for(;ei && (lli!=lle); ++ei, lli++)
            if (negate != (weightGiven ? (*lli==weightID) : !!*lli))
              PyList_SetItem(list, i++, Example_FromExampleRef(*ei, lock));
          return list;
        }

        else { // !toList
          TExamplePointerTable *newTable=mlnew TExamplePointerTable(eg->domain);
          PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error

          for(;ei && (lli!=lle); ++ei, lli++)
            if (negate != (weightGiven ? (*lli==weightID) : !!*lli))
              newTable->addExample(*ei);

          return WrapOrange(newGen);
        }
      }

    PyErr_Clear();

    /* ***** SELECTING BY VALUES OF ATTRIBUTES GIVEN AS DICTIONARY ***** */
    if (PyDict_Check(mplier))
      return toList ? applyFilterL(filter_sameValues(mplier, eg->domain), weg, weightGiven, weightID)
                    : applyFilterP(filter_sameValues(mplier, eg->domain), weg, weightGiven, weightID);

    else if (PyOrFilter_Check(mplier))
      return toList ? applyFilterL(PyOrange_AsFilter(mplier), weg, weightGiven, weightID)
                    : applyFilterP(PyOrange_AsFilter(mplier), weg, weightGiven, weightID);
  }

  PYERROR(PyExc_TypeError, "invalid example selector type", PYNULL);
  PyCATCH
}

PyObject *ExamplePointerTable_selectlist(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExamplePointerTable_selectLow(self, args, keywords, true); 
  PyCATCH
}

PyObject *ExamplePointerTable_selectref(TPyOrange *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "see the manual for help")
{ PyTRY
    return ExamplePointerTable_selectLow(self, args, keywords, false); 
  PyCATCH
}


PyObject *ExamplePointerTable_multipleselect(TPyOrange *self, PyObject *args)  PYARGS(METH_VARARGS, "(indices) -> ExampleTable")
{ PyTRY
    CAST_TO(TExamplePointerTable, eg);
    PyObject *pylist;
    if (   !PyArg_ParseTuple(args, "O", &pylist)
        || !PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
    
    TExampleTable *newTable = mlnew TExampleTable(eg->domain);
    PExampleGenerator newGen(newTable);

    int noex = eg->numberOfExamples();
    for(int sze=PyList_Size(pylist), i=0; i<sze; i++) {
      PyObject *lel=PyList_GetItem(pylist, i);
      if (!PyInt_Check(lel))
        PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
      int r = int(PyInt_AsLong(lel));
      if ((r<0) || (r>=noex))
        PyErr_Format(PyExc_TypeError, "index %i out of range (0-%i)", r, noex-1);
      newTable->addExample(eg->operator[](int(PyInt_AsLong(lel))));
    }

    return WrapOrange(newGen);
  PyCATCH
}


PyObject *ExamplePointerTable_multipleselectref(TPyOrange *self, PyObject *args)  PYARGS(METH_VARARGS, "(indices) -> ExampleTable")
{ PyTRY
    CAST_TO(TExamplePointerTable, eg);
    PyObject *pylist;
    if (   !PyArg_ParseTuple(args, "O", &pylist)
        || !PyList_Check(pylist))
      PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
    
    TExamplePointerTable *newTable = mlnew TExamplePointerTable(eg->domain);
    PExampleGenerator newGen(newTable);

    int noex = eg->numberOfExamples();
    for(int sze=PyList_Size(pylist), i=0; i<sze; i++) {
      PyObject *lel=PyList_GetItem(pylist, i);
      if (!PyInt_Check(lel))
        PYERROR(PyExc_TypeError, "a list of example indices expected", PYNULL);
      int r = int(PyInt_AsLong(lel));
      if ((r<0) || (r>=noex))
        PyErr_Format(PyExc_TypeError, "index %i out of range (0-%i)", r, noex-1);
      newTable->addExample(eg->operator[](int(PyInt_AsLong(lel))));
    }

    return WrapOrange(newGen);
  PyCATCH
}



/* ************ TRANSFORMVALUE ************ */

#include "transval.hpp"
C_NAMED(TransformValue, Orange, "([subTransform=])")


PyObject *TransformValue_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(value) -> Value")
{ PyTRY
    SETATTRIBUTES

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

BASED_ON(Distribution, SomeValue)

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



PyObject *convertToPythonNative(const TDistribution &dist, int)
{ const TDiscDistribution *disc = dynamic_cast<const TDiscDistribution *>(&dist);
  if (disc)
    return convertToPythonNative(*disc);

  const TContDistribution *cont = dynamic_cast<const TContDistribution *>(&dist);
  if (cont)
    return convertToPythonNative(*cont);

  PYERROR(PyExc_TypeError, "cannot convert to native python object", PYNULL);
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
    float ind = numeric_limits<float>::quiet_NaN();
    if (PyNumber_Check(index)) {
      ind = PyNumber_AsFloat(index);
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
    }

    if (ind == numeric_limits<float>::quiet_NaN())
      PYERROR(PyExc_IndexError, "invalid index for distribution", (float *)NULL);

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

    TDiscDistribution *disc = PyOrange_AS_Orange(self).AS(TDiscDistribution);
    if (disc) {
      if (!PyInt_Check(index))
        PYERROR(PyExc_IndexError, "invalid index type", 0);
      disc->setint((int)PyInt_AsLong(index), val);
      return 0;
    }

    TContDistribution *cont = PyOrange_AS_Orange(self).AS(TContDistribution);
    if (cont) {
      PyObject *flt = PyNumber_Float(index);
      if (!flt)
        PYERROR(PyExc_TypeError, "float expected", -1);
      cont->setfloat((float)PyFloat_AsDouble(flt), val);
      Py_DECREF(flt);
      return 0;
    }
  PyCATCH_1

  PYERROR(PyExc_TypeError, "invalid distribution type for 'setitem'", -1);
}


string convertToString(const PDistribution &distribution)
{ 
  const TDiscDistribution *disc;
  distribution.dynamic_cast_to(disc);
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

  const TContDistribution *cont;
  distribution.dynamic_cast_to(cont);
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
    return PyString_FromString(convertToString(PyOrange_AsDistribution(self)).c_str());
  PyCATCH
}


PyObject *Distribution_repr(PyObject *self)
{ PyTRY
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


PyObject *ContDistribution_add(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(value, weight) -> Value")
{ PyTRY
    CAST_TO(TContDistribution, dist)

    PyObject *index;
    float weight = 1.0;
    if (!PyArg_ParseTuple(args, "O|f", &index, &weight))
      PYERROR(PyExc_TypeError, "DiscDistribution.add: invalid arguments", PYNULL);

    if (PyNumber_Check(index)) {
      dist->addfloat(PyNumber_AsFloat(index));
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
    PITERATE(TIdList, ii, vals)
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
    PITERATE(TIdList, ii, vals)
      PyList_SetItem(nl, i++, Py_BuildValue("sf", (*ii).c_str(), *(ci++)));
    return nl;
  PyCATCH
}




PyObject *ContDistribution_new(PyTypeObject *type, PyObject *targs, PyObject *) BASED_ON(Distribution, "[dist of float:float] | DiscDistribution")
{ PyTRY {

    if (!PyTuple_Size(targs))
      return WrapOrange(mlnew TContDistribution());

    if (PyTuple_Size(targs) == 1) {
      PyObject *args = PyTuple_GetItem(targs, 0);

      if (PyDict_Check(args)) {
        TContDistribution *udist = mlnew TContDistribution();
        PContDistribution cont = PDistribution(udist);
        PyObject *key, *value;
        int pos = 0;
        while (PyDict_Next(args, &pos, &key, &value)) {
          PyObject *flt = PyNumber_Float(key);
          if (!flt) {
            PyErr_Format(PyExc_TypeError, "invalid element at index %i (float expected)", pos);
            return false;
          }
          float ind = (float) PyFloat_AsDouble(flt);
          Py_DECREF(flt);

          flt = PyNumber_Float(value);
          if (!flt) {
            PyErr_Format(PyExc_TypeError, "invalid element at index %i (float expected)", pos);
            return false;
          }

          udist->addfloat(ind, (float)PyFloat_AsDouble(flt));
          Py_DECREF(flt);

          return WrapOrange(cont);
        }
      }

      else if (PyOrDistribution_Check(args)) {
        Py_INCREF(args);
        return args;
      }
    }

    PYERROR(PyExc_TypeError, "invalid arguments for distribution constructor", PYNULL);

  }
  PyCATCH;
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


PyObject *getClassDistribution(PyTypeObject *type, PyObject *args) PYARGS(METH_VARARGS, "(examples) -> Distribution")
{ PyTRY
    long weightID;
    PExampleGenerator gen=exampleGenFromArgs(args, &weightID);
    if (!gen)
      return PYNULL;
    return WrapOrange(getClassDistribution(gen));
  PyCATCH
}


PyObject *GaussianDistribution_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Distribution, "(mean, sigma)")
{ PyTRY
    float mean = 0.0, sigma = 1.0;
    if (args && PyTuple_Check(args) && PyTuple_Size(args))
      if (!PyArg_ParseTuple(args, "ff", &mean, &sigma))
        PYERROR(PyExc_TypeError, "GaussianDistribution expects mean and sigma, or nothing", PYNULL)

    return WrapNewOrange(mlnew TGaussianDistribution(mean, sigma), type);
  PyCATCH
}


/* We redefine new (removed from below!) and add mapping methods
*/

inline PDomainDistributions PDomainDistributions_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::P_FromArguments(arg); }
inline PyObject *DomainDistributions_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_FromArguments(type, arg); }
inline PyObject *DomainDistributions_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_getitem(self, index); }
inline int       DomainDistributions_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_setitem(self, index, item); }
inline PyObject *DomainDistributions_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_getslice(self, start, stop); }
inline int       DomainDistributions_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_setslice(self, start, stop, item); }
inline int       DomainDistributions_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_len(self); }
inline PyObject *DomainDistributions_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_concat(self, obj); }
inline PyObject *DomainDistributions_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_repeat(self, times); }
inline PyObject *DomainDistributions_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_str(self); }
inline PyObject *DomainDistributions_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_str(self); }
inline int       DomainDistributions_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_contains(self, obj); }
inline PyObject *DomainDistributions_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_append(self, item); }
inline PyObject *DomainDistributions_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_count(self, obj); }
inline PyObject *DomainDistributions_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainDistributions") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_filter(self, args); }
inline PyObject *DomainDistributions_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_index(self, obj); }
inline PyObject *DomainDistributions_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_insert(self, args); }
inline PyObject *DomainDistributions_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_native(self); }
inline PyObject *DomainDistributions_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Distribution") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_pop(self, args); }
inline PyObject *DomainDistributions_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_remove(self, obj); }
inline PyObject *DomainDistributions_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_reverse(self); }
inline PyObject *DomainDistributions_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_sort(self, args); }


/* Note that this is not like callable-constructors. They return different type when given
   parameters, while this one returns the same type, disregarding whether it was given examples or not.
*/
PyObject *DomainDistributions_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "(examples | <list of Distribution>) -> DomainDistributions")
{ PyTRY
    PyObject *obj = ListOfWrappedMethods<PDomainDistributions, TDomainDistributions, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_new(type, args, keywds);
    if (obj)
      if (obj!=Py_None)
        return obj;
      else {
        Py_DECREF(obj);
        PyErr_Clear();
      }

    long weightID;
    PExampleGenerator gen=exampleGenFromArgs(args, &weightID);
    if (gen)
      return WrapNewOrange(mlnew TDomainDistributions(gen, weightID), type);
      
    return PYNULL;
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




PDistributionList PDistributionList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::P_FromArguments(arg); }
PyObject *DistributionList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_FromArguments(type, arg); }
PyObject *DistributionList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Distribution>)") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_new(type, arg, kwds); }
PyObject *DistributionList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_getitem(self, index); }
int       DistributionList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_setitem(self, index, item); }
PyObject *DistributionList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_getslice(self, start, stop); }
int       DistributionList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_setslice(self, start, stop, item); }
int       DistributionList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_len(self); }
PyObject *DistributionList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_concat(self, obj); }
PyObject *DistributionList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_repeat(self, times); }
PyObject *DistributionList_str(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_str(self); }
PyObject *DistributionList_repr(TPyOrange *self) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_str(self); }
int       DistributionList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_contains(self, obj); }
PyObject *DistributionList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_append(self, item); }
PyObject *DistributionList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_count(self, obj); }
PyObject *DistributionList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DistributionList") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_filter(self, args); }
PyObject *DistributionList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> int") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_index(self, obj); }
PyObject *DistributionList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_insert(self, args); }
PyObject *DistributionList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_native(self); }
PyObject *DistributionList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Distribution") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_pop(self, args); }
PyObject *DistributionList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Distribution) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_remove(self, obj); }
PyObject *DistributionList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_reverse(self); }
PyObject *DistributionList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDistributionList, TDistributionList, PDistribution, (PyTypeObject *)&PyOrDistribution_Type>::_sort(self, args); }


/* ************ LEARNER ************ */

#include "classify.hpp"
#include "learn.hpp"
BASED_ON(LearnerFD, Learner)

PyObject *Learner_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrLearner_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TLearner_Python(), type), args);
  else
    return WrapNewOrange(mlnew TLearner_Python(), type);
}


PyObject *Learner_call(PyObject *self, PyObject *targs, PyObject *keywords) PYDOC("(examples) -> Classifier")
{
  PyTRY
    SETATTRIBUTES

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrLearner_Type) {
      PyErr_Format(PyExc_SystemError, "Learner.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    int weight = -1;
    PExampleGenerator egen;

    if (   !PyArg_ParseTuple(targs, "O&|i:Learner.__call__", pt_ExampleGenerator, &egen, &weight)
        && !PyArg_ParseTuple(targs, "(O&|i):Learner.__call__", pt_ExampleGenerator, &egen, &weight))
      return PYNULL;

    if (weight == -1) {
      weight = 0;

      PyObject **odict = _PyObject_GetDictPtr(self);
      if (*odict) {
        PyObject *pyweight = PyDict_GetItemString(*odict, "weight");
        if (pyweight && PyInt_Check(pyweight))
          weight = (int)PyInt_AsLong(pyweight);
      }
    }

    PClassifier classfr = SELF_AS(TLearner)(egen, weight);
    if (!classfr)
      PYERROR(PyExc_SystemError, "learning failed", PYNULL);

    return WrapOrange(classfr);
  PyCATCH
}




/* ************ CLASSIFIERS ************ */

#include "classify.hpp"

BASED_ON(ClassifierFD, Classifier)
C_NAMED(DefaultClassifier, Classifier, "([defaultVal=])")
C_NAMED(RandomClassifier, Classifier, "([probabilities=])")

PClassifierList PClassifierList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::P_FromArguments(arg); }
PyObject *ClassifierList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_FromArguments(type, arg); }
PyObject *ClassifierList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Classifier>)") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_new(type, arg, kwds); }
PyObject *ClassifierList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_getitem(self, index); }
int       ClassifierList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_setitem(self, index, item); }
PyObject *ClassifierList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_getslice(self, start, stop); }
int       ClassifierList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_setslice(self, start, stop, item); }
int       ClassifierList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_len(self); }
PyObject *ClassifierList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_concat(self, obj); }
PyObject *ClassifierList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_repeat(self, times); }
PyObject *ClassifierList_str(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_str(self); }
PyObject *ClassifierList_repr(TPyOrange *self) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_str(self); }
int       ClassifierList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_contains(self, obj); }
PyObject *ClassifierList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Classifier) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_append(self, item); }
PyObject *ClassifierList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> int") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_count(self, obj); }
PyObject *ClassifierList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ClassifierList") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_filter(self, args); }
PyObject *ClassifierList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> int") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_index(self, obj); }
PyObject *ClassifierList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_insert(self, args); }
PyObject *ClassifierList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_native(self); }
PyObject *ClassifierList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Classifier") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_pop(self, args); }
PyObject *ClassifierList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Classifier) -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_remove(self, obj); }
PyObject *ClassifierList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_reverse(self); }
PyObject *ClassifierList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func] -> None") { return ListOfWrappedMethods<PClassifierList, TClassifierList, PClassifier, (PyTypeObject *)&PyOrClassifier_Type>::_sort(self, args); }



PYCONSTANT(GetValue, PyInt_FromLong(0))
PYCONSTANT(GetProbabilities, PyInt_FromLong(1))
PYCONSTANT(GetBoth, PyInt_FromLong(2))

PYCLASSCONSTANT_INT(Classifier, GetValue, 0)
PYCLASSCONSTANT_INT(Classifier, GetProbabilities, 1)
PYCLASSCONSTANT_INT(Classifier, GetBoth, 2)


PyObject *Classifier_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrClassifier_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TClassifier_Python(), type), args);
  else
    return WrapNewOrange(mlnew TClassifier_Python(), type);
}


PyObject *Classifier_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example[, format]) -> Value | distribution | (Value, distribution)")
{ PyTRY
    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrClassifier_Type) {
      PyErr_Format(PyExc_SystemError, "Classifier.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES
    CAST_TO(TClassifier, classifier);

    if (!classifier)
      PYERROR(PyExc_SystemError, "attribute error", PYNULL);

    TPyExample *example;
    int dist=0;
    if (   !PyArg_ParseTuple(args, "O|i", &example, &dist)
        || !PyOrExample_Check(example))
      PYERROR(PyExc_TypeError, "attribute error (example expected)", PYNULL);

    switch (dist) {
      case 0:
        return Value_FromVariableValue(classifier->classVar, (*classifier)(PyExample_AS_ExampleReference(example)));

      case 1:
        return WrapOrange(classifier->classDistribution(PyExample_AS_ExampleReference(example)));

      case 2:
        return Py_BuildValue("NN", Value_FromVariableValue(classifier->classVar, (*classifier)(PyExample_AS_ExampleReference(example))),
                                   WrapOrange(classifier->classDistribution(PyExample_AS_ExampleReference(example))));
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
C_NAMED(ClassifierFromMeta, Classifier, "([whichID=, transformer=])")

#include "cartesian.hpp"
C_NAMED(CartesianClassifier, ClassifierFD, "()")


/* ************ LOOKUP ************ */

#include "lookup.hpp"

C_CALL(LookupLearner, Learner, "([examples] [, weight=]) -/-> Classifier")
C_NAMED(ClassifierByExampleTable, ClassifierFD, "([examples=])")


PyObject *ClassifierByExampleTable_boundset(PyObject *self) PYARGS(0, "() -> variables")
{ PyTRY
    TVarList &attributes=SELF_AS(TClassifierByExampleTable).domain->attributes.getReference();
    PyObject *list=PyList_New(attributes.size());
    for(int i=0, asize = attributes.size(); i<asize; i++)
      PyList_SetItem(list, i, WrapOrange(attributes[i]));
    return list;
  PyCATCH
}


PyObject *ClassifierByLookupTable_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Classifier, "(class-descriptor, descriptor)")
{ PyTRY
    PyObject *pycl, *pyvl;

    if (   !PyArg_ParseTuple(args, "OO", &pycl, &pyvl)
        || !PyOrVariable_Check(pycl)
        || !PyOrVariable_Check(pyvl))
      PYERROR(PyExc_TypeError, "invalid parameter; two variables expected", PYNULL);

    return WrapNewOrange(mlnew TClassifierByLookupTable(PyOrange_AsVariable(pycl), PyOrange_AsVariable(pyvl)), type);
  PyCATCH
}


PyObject *ClassifierByLookupTable_boundset(PyObject *self) PYARGS(0, "() -> variable")
{ PyTRY
    return Py_BuildValue("(N)", WrapOrange(SELF_AS(TClassifierByLookupTable).variable));
  PyCATCH
}


PyObject *ClassifierByLookupTable_getindex(PyObject *self, PyObject *pyexample) PYARGS(METH_O, "(example) -> int")
{ PyTRY
    if (!PyOrExample_Check(pyexample))
      PYERROR(PyExc_TypeError, "invalid arguments; an example expected", PYNULL);

    return PyInt_FromLong(long(SELF_AS(TClassifierByLookupTable).getIndex(PyExample_AS_ExampleReference(pyexample))));
  PyCATCH
}



PyObject *ClassifierByLookupTable2_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Classifier, "(class-descriptor, desc0, desc1)")
{ PyTRY
    PyObject *pycl, *pyvl1, *pyvl2;

    if (   !PyArg_ParseTuple(args, "OOO", &pycl, &pyvl1, &pyvl2)
        || !PyOrVariable_Check(pycl)
        || !PyOrVariable_Check(pyvl1)
        || !PyOrVariable_Check(pyvl2))
      PYERROR(PyExc_TypeError, "invalid parameter; three variables expected", PYNULL);

    return WrapNewOrange(mlnew TClassifierByLookupTable2(PyOrange_AsVariable(pycl), PyOrange_AsVariable(pyvl1), PyOrange_AsVariable(pyvl2)), type);
  PyCATCH
}


PyObject *ClassifierByLookupTable2_boundset(PyObject *self) PYARGS(0, "() -> [variable1, variable2]")
{ PyTRY
    return Py_BuildValue("(NN)", WrapOrange(SELF_AS(TClassifierByLookupTable2).variable1),
                                WrapOrange(SELF_AS(TClassifierByLookupTable2).variable2));
  PyCATCH
}


PyObject *ClassifierByLookupTable2_getindex(PyObject *self, PyObject *pyexample) PYARGS(METH_O, "(example) -> int")
{ PyTRY
    if (!PyOrExample_Check(pyexample))
      PYERROR(PyExc_TypeError, "invalid arguments; an example expected", PYNULL);

    return PyInt_FromLong(long(SELF_AS(TClassifierByLookupTable2).getIndex(PyExample_AS_ExampleReference(pyexample))));
  PyCATCH
}


PyObject *ClassifierByLookupTable3_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Classifier, "(class-descriptor, desc0, desc1, desc2)")
{ PyTRY
    PyObject *pycl, *pyvl1, *pyvl2, *pyvl3;

    if (   !PyArg_ParseTuple(args, "OOOO", &pycl, &pyvl1, &pyvl2, &pyvl3)
        || !PyOrVariable_Check(pycl)
        || !PyOrVariable_Check(pyvl1)
        || !PyOrVariable_Check(pyvl2)
        || !PyOrVariable_Check(pyvl3))
    PYERROR(PyExc_TypeError, "invalid parameter; four variables expected", PYNULL);

    return WrapNewOrange(mlnew TClassifierByLookupTable3(PyOrange_AsVariable(pycl), PyOrange_AsVariable(pyvl1), PyOrange_AsVariable(pyvl2), PyOrange_AsVariable(pyvl3)), type);
  PyCATCH
}

PyObject *ClassifierByLookupTable3_boundset(PyObject *self) PYARGS(0, "() -> [variable1, variable2, variable3]")
{ PyTRY
    return Py_BuildValue("(NNN)", WrapOrange(SELF_AS(TClassifierByLookupTable3).variable1),
                                  WrapOrange(SELF_AS(TClassifierByLookupTable3).variable2),
                                  WrapOrange(SELF_AS(TClassifierByLookupTable3).variable3));
  PyCATCH
}


PyObject *ClassifierByLookupTable3_getindex(PyObject *self, PyObject *pyexample) PYARGS(METH_O, "(example) -> int")
{ PyTRY
    if (!PyOrExample_Check(pyexample))
      PYERROR(PyExc_TypeError, "invalid arguments; an example expected", PYNULL);

    return PyInt_FromLong(long(SELF_AS(TClassifierByLookupTable3).getIndex(PyExample_AS_ExampleReference(pyexample))));
  PyCATCH
}

#include "lib_kernel.px"
