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

This file includes constructors and specialized methods for ML* object, defined in project Components

*********************************/

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "vars.hpp"
#include "distvars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "cls_value.hpp"
#include "cls_example.hpp"
#include "cls_orange.hpp"
#include "lib_kernel.hpp"
#include "callback.hpp"

#include "vectortemplates.hpp"


#include "converts.hpp"

#include "externs.px"


/* ************ COST ************ */

#include "cost.hpp"



PyObject *convertToPython(const PCostMatrix &matrix)
{ int dim=matrix->size();
  PyObject *pycost=PyList_New(dim);
  int i=0;
  const_PITERATE(TCostMatrix, ci, matrix) {
    PyObject *el=PyList_New(dim);
    matrix->at(i)->operator[](dim);
    int j=0;
    const_PITERATE(TDiscDistribution, di, matrix->at(i))
      PyList_SetItem(el, j++, PyFloat_FromDouble(*di));
    PyList_SetItem(pycost, i++, el);
  }

  return pycost;
}


bool readCostMatrix(PyObject *arg, TCostMatrix *&matrix)
{
  int dim;
  const int arglength = PyObject_Length(arg);
  if (matrix) {
    dim = matrix->size();
    if (dim != arglength) {
      PyErr_Format(PyExc_TypeError, "invalid sequence length (expected %i, got %i)", dim, arglength);
      return false;
    }
  }
  else {
    dim = arglength;
    matrix = mlnew TCostMatrix(dim);
  }

  PyObject *iter = PyObject_GetIter(arg);
  if (!iter)
    PYERROR(PyExc_TypeError, "sequence expected", false);

  int i, j;

  for(i = 0; i<dim; i++) {
    PyObject *item = PyIter_Next(iter);
    if (!item) {
      PyErr_Format(PyExc_TypeError, "matrix is too short (%i rows expected)", dim);
      break;
    }

    PyObject *subiter = PyObject_GetIter(item);
    Py_DECREF(item);

    if (!subiter) {
      PyErr_Format(PyExc_TypeError, "element %i is not a sequence", i);
      break;
    }

    for(j = 0; j<dim; j++) {
      PyObject *subitem = PyIter_Next(subiter);
      if (!subitem) {
        PyErr_Format(PyExc_TypeError, "element %i is too short (sequence with %i elements expected)", i, dim);
        break;
      }

      float f;
      bool ok = PyNumber_ToFloat(subitem, f);
      Py_DECREF(subitem);
      if (!ok) {
        PyErr_Format(PyExc_TypeError, "element at (%i, %i) is not a number", i, j);
        break;
      }
      
      // this cannot fail:
      matrix->setCost(i, j, f);
    }

    if (j<dim) {
      Py_DECREF(subiter);
      break;
    }

    PyObject *subitem = PyIter_Next(subiter);
    Py_DECREF(subiter);

    if (subitem) {
      PyErr_Format(PyExc_TypeError, "element %i is too long (sequence with %i elements expected)", i, dim);
      Py_DECREF(subitem);
      break;
    }
  }

  Py_DECREF(iter);

  if (i<dim) {
    mldelete matrix;
    return false;
  }

  return true;
}


PyObject *CostMatrix_new(PyTypeObject *type, PyObject *args) BASED_ON(Orange, "(list-of-list-of-prices) -> CostMatrix")
{
  PyTRY
    if (PyTuple_Size(args) == 1) {
      PyObject *arg = PyTuple_GET_ITEM(args, 0);
      
      if (PyInt_Check(arg))
        return WrapNewOrange(mlnew TCostMatrix(PyInt_AsLong(arg)), type);

      if (PyOrVariable_Check(arg))
        return WrapNewOrange(mlnew TCostMatrix(PyOrange_AsVariable(arg)), type);

      TCostMatrix *nm = NULL;
      return readCostMatrix(arg, nm) ? WrapNewOrange(nm, type) : PYNULL;
    }


    if (PyTuple_Size(args) == 2) {
      PyObject *arg1, *arg2;
      arg1 = PyTuple_GetItem(args, 0);
      arg2 = PyTuple_GetItem(args, 1);

      float inside;
      if (PyNumber_ToFloat(arg2, inside)) {
        if (PyInt_Check(arg1))
          return WrapNewOrange(mlnew TCostMatrix(PyInt_AsLong(arg1), inside), type);

        if (PyOrVariable_Check(arg1))
          return WrapNewOrange(mlnew TCostMatrix(PyOrange_AsVariable(arg1), inside), type);
      }

      if (PyOrVariable_Check(arg1)) {
        TCostMatrix *nm = mlnew TCostMatrix(PyOrange_AsVariable(arg1));
        return readCostMatrix(arg2, nm) ? WrapNewOrange(nm, type) : PYNULL;
      }
    }

    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);
  PyCATCH
}


PyObject *CostMatrix_native(PyObject *self) PYARGS(METH_O, "() -> list of lists of floats")
{ return convertToPython(PyOrange_AsCostMatrix(self)); }


int getCostIndex(PyObject *arg, TCostMatrix *matrix, char *error)
{
  if (PyInt_Check(arg)) {
    int pred = PyInt_AsLong(arg);
    if ((pred<0) || (pred >= matrix->size()))
      PYERROR(PyExc_IndexError, error, -1);
    return pred;
  }
  else {
    TValue val;
    return  convertFromPython(arg, val, matrix->classVar) ? int(val) : -1;
  }
}



PyObject *CostMatrix_getcost(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(predicted, correct) -> float")
{ 
  PyTRY
    CAST_TO(TCostMatrix, matrix);

    if (PyTuple_Size(args) != 2)
      PYERROR(PyExc_TypeError, "two arguments expected", PYNULL);

    PyObject *arg1 = PyTuple_GET_ITEM(args, 0);
    PyObject *arg2 = PyTuple_GET_ITEM(args, 1);
  
    int pred = getCostIndex(arg1, matrix, "predicted value out of range");
    int corr = getCostIndex(arg2, matrix, "correct value out of range");
    if ((pred==-1) || (corr==-1))
      return PYNULL;

    return PyFloat_FromDouble(matrix->getCost(pred, corr));
  PyCATCH
}


PyObject *CostMatrix_setcost(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(predicted, correct, cost) -> float")
{
  PyTRY
    CAST_TO(TCostMatrix, matrix);

    PyObject *arg1, *arg2;
    float cost;

    if (!PyArg_ParseTuple(args, "OOf:CostMatrix.setcost", &arg1, &arg2, &cost))
      return PYNULL;

    int pred = getCostIndex(arg1, matrix, "predicted value out of range");
    int corr = getCostIndex(arg2, matrix, "correct value out of range");
    if ((pred==-1) || (corr==-1))
      return PYNULL;

    matrix->setCost(pred, corr, cost);
    RETURN_NONE;
  PyCATCH
}


PyObject *CostMatrix_get_dimension(PyObject *self)
{
  return PyInt_FromLong(SELF_AS(TCostMatrix).size());
}

/* ************ BASSTAT ************ */

#include "basstat.hpp"

PyObject *BasicAttrStat_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "(variable, [min=, max=, avg=, dev=, n=]) -> BasicAttrStat")
{ PyTRY
    if (!PyOrVariable_Check(args))
      PYERROR(PyExc_TypeError, "invalid parameters for constructor", PYNULL);
    return WrapNewOrange(mlnew TBasicAttrStat(PyOrange_AsVariable(args)), type);
  PyCATCH
}

PyObject *BasicAttrStat_add(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(value[, weight]) -> None")
{ PyTRY
    float value, weight=1.0;
    if (!PyArg_ParseTuple(args, "f|f:BasicAttrStat.add", &value, &weight))
      return PYNULL;
    SELF_AS(TBasicAttrStat).add(value, weight);
    RETURN_NONE;
  PyCATCH
}


PyObject *BasicAttrStat_recompute(PyObject *self) PYARGS(METH_O, "() -> None")
{ PyTRY
    SELF_AS(TBasicAttrStat).recompute();
    RETURN_NONE;
  PyCATCH
}


/* We redefine new (removed from below!) and add mapping methods
*/

inline PDomainBasicAttrStat PDomainBasicAttrStat_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::P_FromArguments(arg); }
inline PyObject *DomainBasicAttrStat_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_FromArguments(type, arg); }
inline PyObject *DomainBasicAttrStat_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_getitem(self, index); }
inline int       DomainBasicAttrStat_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_setitem(self, index, item); }
inline PyObject *DomainBasicAttrStat_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_getslice(self, start, stop); }
inline int       DomainBasicAttrStat_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_setslice(self, start, stop, item); }
inline PyObject *DomainBasicAttrStat_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_richcmp(self, object, op); }
inline int       DomainBasicAttrStat_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_len(self); }
inline PyObject *DomainBasicAttrStat_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_concat(self, obj); }
inline PyObject *DomainBasicAttrStat_repeat(TPyOrange *self, int times) {     return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_repeat(self, times); }
inline PyObject *DomainBasicAttrStat_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_str(self); }
inline PyObject *DomainBasicAttrStat_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_str(self); }
inline int       DomainBasicAttrStat_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_contains(self, obj); }
inline PyObject *DomainBasicAttrStat_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(BasicAttrStat) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_append(self, item); }
inline PyObject *DomainBasicAttrStat_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> int") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_count(self, obj); }
inline PyObject *DomainBasicAttrStat_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainBasicAttrStat") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_filter(self, args); }
inline PyObject *DomainBasicAttrStat_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> int") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_index(self, obj); }
inline PyObject *DomainBasicAttrStat_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_insert(self, args); }
inline PyObject *DomainBasicAttrStat_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_native(self); }
inline PyObject *DomainBasicAttrStat_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> BasicAttrStat") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_pop(self, args); }
inline PyObject *DomainBasicAttrStat_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_remove(self, obj); }
inline PyObject *DomainBasicAttrStat_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_reverse(self); }
inline PyObject *DomainBasicAttrStat_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_sort(self, args); }


/* Note that this is not like callable-constructors. They return different type when given
   parameters, while this one returns the same type, disregarding whether it was given examples or not.
*/
PyObject *DomainBasicAttrStat_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "(examples | <list of BasicAttrStat>) -> DomainBasicAttrStat")
{ PyTRY
    PyObject *obj = ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, (PyTypeObject *)&PyOrBasicAttrStat_Type>::_new(type, args, keywds);
    if (obj)
      return obj;
    PyErr_Clear();

    int weightID;
    PExampleGenerator gen=exampleGenFromArgs(args, &weightID);
    if (gen)
      return WrapNewOrange(mlnew TDomainBasicAttrStat(gen, weightID), type);
      
    return PYNULL;
  PyCATCH
}


PyObject *DomainBasicAttrStat_purge(PyObject *self) PYARGS(METH_NOARGS, "None -> None")
{ PyTRY
    SELF_AS(TDomainBasicAttrStat).purge();
    RETURN_NONE
  PyCATCH
}

/* We keep the sequence methods and add mapping interface */

int DomainBasicAttrStat_getItemIndex(PyObject *self, PyObject *args)
{ CAST_TO_err(TDomainBasicAttrStat, bas, -1);
  
  if (PyInt_Check(args)) {
    int i=(int)PyInt_AsLong(args);
    if ((i>=0) && (i<int(bas->size())))
      return i;
    else
      PYERROR(PyExc_IndexError, "index out of range", -1);
  }

  if (PyString_Check(args)) {
    char *s=PyString_AsString(args);
    PITERATE(TDomainBasicAttrStat, ci, bas)
      if (*ci && (*ci)->variable && ((*ci)->variable->name==s))
        return ci - bas->begin();

    PyErr_Format(PyExc_IndexError, "attribute '%s' not found", s);
    return -1;
  }

  if (PyOrVariable_Check(args)) {
    PVariable var = PyOrange_AsVariable(args);
    PITERATE(TDomainBasicAttrStat, ci, bas)
      if (*ci && (*ci)->variable && ((*ci)->variable==var))
        return ci - bas->begin();

    PyErr_Format(PyExc_IndexError, "attribute '%s' not found", var->name.length() ? var->name.c_str() : "<no name>");
    return -1;
  }

  PYERROR(PyExc_IndexError, "invalid index type", -1);
}


PyObject *DomainBasicAttrStat_getitem(PyObject *self, PyObject *args)
{ PyTRY
    int index=DomainBasicAttrStat_getItemIndex(self, args);
    if (index<0)
      return PYNULL;
    return WrapOrange(POrange(SELF_AS(TDomainBasicAttrStat).at(index)));
  PyCATCH
}


int DomainBasicAttrStat_setitem(PyObject *self, PyObject *args, PyObject *obj)
{ PyTRY
    PBasicAttrStat bas;

    if (!PyOrBasicAttrStat_Check(obj))
      PYERROR(PyExc_TypeError, "invalid BasicAttrStat object", -1);

    int index=DomainBasicAttrStat_getItemIndex(self, args);
    if (index==-1)
      return -1;

    SELF_AS(TDomainBasicAttrStat)[index] = PyOrange_AsBasicAttrStat(obj);
    return 0;
  PyCATCH_1
}




/* ************ CONTINGENCY ************ */

#include "contingency.hpp"
#include "estimateprob.hpp"

BASED_ON(ContingencyClass, Contingency)

PDistribution *Contingency_getItemRef(PyObject *self, PyObject *index)
{ CAST_TO_err(TContingency, cont, (PDistribution *)NULL);
  if (!cont->outerVariable)
    PYERROR(PyExc_SystemError, "invalid contingency (no variable)", (PDistribution *)NULL);

  if (cont->outerVariable->varType==TValue::INTVAR) {
    int ind=-1;
    if (PyInt_Check(index))
      ind=(int)PyInt_AsLong(index);
    else {
      TValue val;
      if (convertFromPython(index, val, cont->outerVariable) && !val.isSpecial()) 
        ind=int(val);
    }

    if ((ind>=0) && (ind<int(cont->discrete->size())))
      return &cont->discrete->at(ind);
  }
  else if (cont->outerVariable->varType==TValue::FLOATVAR) {
    float ind = numeric_limits<float>::quiet_NaN();
    if (!PyNumber_ToFloat(index, ind)) {
      TValue val;
      if (convertFromPython(index, val, cont->outerVariable) && !val.isSpecial())
        ind = float(val);
    }

    TDistributionMap::iterator mi=cont->continuous->find(ind);
    if (mi!=cont->continuous->end())
      return &(*mi).second;

    PyErr_Format(PyExc_IndexError, "invalid index (%5.3f)", ind);
    return NULL;
  }

  PYERROR(PyExc_IndexError, "invalid index", (PDistribution *)NULL);
}


PyObject *Contingency_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "(outer_desc, inner_desc)")
{ PyTRY
    PVariable var1, var2;
    if (!PyArg_ParseTuple(args, "O&O&:Contingency.__new__", cc_Variable, &var1, cc_Variable, &var2))
      return PYNULL;

    return WrapNewOrange(mlnew TContingency(var1, var2), type);
  PyCATCH
}


PyObject *Contingency_add(PyObject *self, PyObject *args)  PYARGS(METH_VARARGS, "(outer_value, inner_value[, w=1]) -> None")
{
  PyTRY
    PyObject *pyouter, *pyinner;
    float w = 1.0;
    if (!PyArg_ParseTuple(args, "OO|f:Contingency.add", &pyouter, &pyinner, &w))
      return PYNULL;

   CAST_TO(TContingency, cont)

   TValue inval, outval;
    if (   !convertFromPython(pyinner, inval, cont->innerVariable)
        || !convertFromPython(pyouter, outval, cont->outerVariable))
      return PYNULL;

    cont->add(outval, inval, w);
    RETURN_NONE;
  PyCATCH
}


bool ContingencyClass_getValuePair(TContingencyClass *cont, PyObject *pyattr, PyObject *pyclass, TValue &attrval, TValue &classval)
{
  return    convertFromPython(pyattr, attrval, cont->getAttribute())
         && convertFromPython(pyclass, classval, cont->getClassVar());
}


bool ContingencyClass_getValuePair(TContingencyClass *cont, PyObject *args, char *s, TValue &attrval, TValue &classval)
{
  PyObject *pyattr, *pyclass;
  return    PyArg_ParseTuple(args, s, &pyattr, &pyclass)
         && ContingencyClass_getValuePair(cont, pyattr, pyclass, attrval, classval);
}


PyObject *ContingencyClass_add_attrclass(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(attribute_value, class_value[, w=1]) -> None")
{
  PyTRY
    CAST_TO(TContingencyClass, cont)

    PyObject *pyattr, *pyclass;
    TValue attrval, classval;
    float w = 1.0;
    if (   !PyArg_ParseTuple(args, "OO|f:ContingencyClass.add_attrclass", &pyattr, &pyclass, &w)
        || !ContingencyClass_getValuePair(cont, pyattr, pyclass, attrval, classval))
      return PYNULL;

    cont->add_attrclass(attrval, classval, w);
    RETURN_NONE;
  PyCATCH
}


PyObject *ContingencyClass_get_classVar(PyObject *self)
{
  return WrapOrange(SELF_AS(TContingencyClass).getClassVar());
}


PyObject *ContingencyClass_get_variable(PyObject *self)
{
  return WrapOrange(SELF_AS(TContingencyClass).getAttribute());
}


PyObject *ContingencyAttrClass_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(ContingencyClass, "(attribute, class attribute) | (attribute, examples[, weightID])")
{ PyTRY
    PVariable var1, var2;
    if (PyArg_ParseTuple(args, "O&O&:Contingency.__new__", cc_Variable, &var1, cc_Variable, &var2))
      return WrapNewOrange(mlnew TContingencyAttrClass(var1, var2), type);

    PyErr_Clear();

    PyObject *object1;
    PExampleGenerator gen;
    int weightID=0;
    if (PyArg_ParseTuple(args, "OO&|O&", &object1, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID)) {
      if (PyOrVariable_Check(object1))
        return WrapNewOrange(mlnew TContingencyAttrClass(gen, PyOrange_AsVariable(object1), weightID), type);

      int attrNo;
      if (varNumFromVarDom(object1, gen->domain, attrNo))
        return WrapNewOrange(mlnew TContingencyAttrClass(gen, attrNo, weightID), type);
    }
          
    PYERROR(PyExc_TypeError, "invalid type for ContingencyAttrClass constructor", PYNULL);   

  PyCATCH
}


PyObject *ContingencyAttrClass_p_class(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(attr_value[, class_value]) -> p | distribution of classes")
{
  PyTRY
    CAST_TO(TContingencyClass, cont);

    if (PyTuple_Size(args) == 1) {
      TValue attrval;
      if (!convertFromPython(PyTuple_GET_ITEM(args, 0), attrval, cont->outerVariable))
        return PYNULL;

      PDistribution dist = CLONE(TDistribution, cont->p_classes(attrval));
      if (!dist)
        PYERROR(PyExc_AttributeError, "no distribution", PYNULL);

      dist->normalize();
      return WrapOrange(dist);
    }

    else {
      TValue attrval, classval;
      if (!ContingencyClass_getValuePair(cont, args, "OO:ContingencyAttrClass.p_class", attrval, classval))
        return PYNULL;

      return PyFloat_FromDouble(cont->p_class(attrval, classval));
    }
  PyCATCH
}


PyObject *ContingencyClassAttr_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(ContingencyClass, "(attribute, class attribute) | (attribute, examples[, weightID])")
{ PyTRY
    PVariable var1, var2;
    if (PyArg_ParseTuple(args, "O&O&:Contingency.__new__", cc_Variable, &var1, cc_Variable, &var2))
      return WrapNewOrange(mlnew TContingencyClassAttr(var1, var2), type);

    PyErr_Clear();

    PyObject *object1;
    int weightID=0;
    PExampleGenerator gen;
    if (   PyArg_ParseTuple(args, "OO&|O&", &object1, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID)) {
      if (PyOrVariable_Check(object1))
        return WrapNewOrange(mlnew TContingencyClassAttr(gen, PyOrange_AsVariable(object1), weightID), type);
      else {
        int attrNo;
        if (varNumFromVarDom(object1, gen->domain, attrNo))
          return WrapNewOrange(mlnew TContingencyClassAttr(gen, attrNo, weightID), type);
      }
    }
          
  PyCATCH

  PYERROR(PyExc_TypeError, "invalid type for ContingencyClassAttr constructor", PYNULL);   
}


PyObject *ContingencyClassAttr_p_attr(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([attr_value, ]class_value) -> p | distribution of values")
{
  PyTRY
    CAST_TO(TContingencyClass, cont);

    if (PyTuple_Size(args) == 1) {
      TValue classval;
      if (!convertFromPython(PyTuple_GET_ITEM(args, 0), classval, cont->outerVariable))
        return PYNULL;

      PDistribution dist = CLONE(TDistribution, cont->p_attrs(classval));
      if (!dist)
        PYERROR(PyExc_AttributeError, "no distribution", PYNULL);

      dist->normalize();
      return WrapOrange(dist);
    }

    else {
      TValue attrval, classval;
      if (!ContingencyClass_getValuePair(cont, args, "OO:ContingencyClassAttr.p_attr", attrval, classval))
        return PYNULL;

      return PyFloat_FromDouble(cont->p_attr(attrval, classval));
    }
  PyCATCH
}


PyObject *ContingencyAttrAttr_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Contingency, "(outer_attr, inner_attr[, examples [, weight-id]])")
{ PyTRY
    PyObject *pyvar, *pyinvar;
    PExampleGenerator gen;
    int weightID=0;
    if (PyArg_ParseTuple(args, "OO|O&O&", &pyvar, &pyinvar, &pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      if (gen)
        return WrapNewOrange(mlnew TContingencyAttrAttr(
           varFromArg_byDomain(pyvar, gen->domain),
           varFromArg_byDomain(pyinvar, gen->domain),
           gen, weightID), type);

      else
        if (PyOrVariable_Check(pyvar) && PyOrVariable_Check(pyinvar))
          return WrapNewOrange(mlnew TContingencyAttrAttr(
            PyOrange_AsVariable(pyvar),
            PyOrange_AsVariable(pyinvar)),
            type);
  PyCATCH

  PYERROR(PyExc_TypeError, "ContingencyAttrAttr: two variables and (opt) examples and (opt) weight expected", PYNULL);
}



PyObject *ContingencyAttrAttr_p_attr(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(outer_value[, inner_value]) -> p | distribution of values")
{
  PyTRY
    CAST_TO(TContingencyAttrAttr, cont);

    PyObject *pyouter, *pyinner = PYNULL;
    TValue outerval, innerval;
    if (   !PyArg_ParseTuple(args, "O|O:ContingencyAttrAttr.p_attr", &pyouter, &pyinner)
        || !convertFromPython(pyouter, outerval, cont->outerVariable))
      return PYNULL;

    if (!pyinner) {
      PDistribution dist = CLONE(TDistribution, cont->p_attrs(outerval));
      if (!dist)
        PYERROR(PyExc_AttributeError, "no distribution", PYNULL);

      dist->normalize();
      return WrapOrange(dist);
    }

    else {
      if (!convertFromPython(pyinner, innerval, cont->innerVariable))
        return PYNULL;

      return PyFloat_FromDouble(cont->p_attr(outerval, innerval));
    }
  PyCATCH
}


PyObject *Contingency_normalize(PyObject *self, PyObject *) PYARGS(0,"() -> None")
{ PyTRY
    SELF_AS(TContingency).normalize();
    RETURN_NONE
  PyCATCH
}


PyObject *Contingency_getitem(PyObject *self, PyObject *index)
{ PyTRY
    PDistribution *dist=Contingency_getItemRef(self, index);
    if (!dist)
      return PYNULL;

    return WrapOrange(POrange(*dist));
  PyCATCH
}


PyObject *Contingency_getitem_sq(PyObject *self, int ind)
{ PyTRY
    CAST_TO(TContingency, cont)

    if (cont->outerVariable->varType!=TValue::INTVAR)
      PYERROR(PyExc_TypeError, "cannot iterate through contingency of continuous attribute", PYNULL);

    if ((ind<0) || (ind>=int(cont->discrete->size())))
      PYERROR(PyExc_IndexError, "index out of range", PYNULL);

    return WrapOrange(POrange(cont->discrete->at(ind)));
  PyCATCH
}


int Contingency_setitem(PyObject *self, PyObject *index, PyObject *item)
{ PyTRY
    if (!PyOrDistribution_Check(item))
      PYERROR(PyExc_TypeError, "Distribution expected", -1);

    PDistribution *dist=Contingency_getItemRef(self, index);
    if (!dist)
      return -1;

    *dist = CLONE(TDistribution, PyOrange_AsDistribution(item));
    return 0;
  PyCATCH_1
}

int Contingency_len(PyObject *self)
{ PyTRY
    CAST_TO_err(TContingency, cont, -1);
    if (cont->outerVariable)
      if (cont->outerVariable->varType==TValue::INTVAR)
        return cont->discrete->size();
      else if (cont->outerVariable->varType==TValue::FLOATVAR)
        return cont->continuous->size();

    return 0;
  PyCATCH_1
}


bool convertFromPython(PyObject *obj, PContingency &var, bool allowNull, PyTypeObject *type)
{ if (!type)
    type = (PyTypeObject *)&PyOrContingency_Type;
    
  if (allowNull && (!obj || (obj==Py_None))) {
    var=GCPtr<TContingency>();
    return true;
  }
  if (!type)
    type = (PyTypeObject *)FindOrangeType(typeid(TContingency));

  if (!obj || !PyObject_TypeCheck(obj, type)) {
    PyErr_Format(PyExc_TypeError, "expected '%s', got '%s'", type->tp_name, obj ? obj->ob_type->tp_name : "None");
    return false;
  }
  
  var=GCPtr<TContingency>(PyOrange_AS_Orange(obj));
  return true;
}


string convertToString(const PContingency &cont)
{ if (!cont->outerVariable)
    raiseError("invalid contingency ('outerVariable' not set)");

  if (cont->outerVariable->varType==TValue::INTVAR) {
    TValue val;
    cont->outerVariable->firstValue(val);

    string res="<";
    PITERATE(TDistributionVector, di, cont->discrete) {
      if (di!=cont->discrete->begin()) res+=", ";
      string vals;
      cont->outerVariable->val2str(val,vals);
      res+="'"+vals+"': "+convertToString(*di);
      cont->outerVariable->nextValue(val);
    }
    return res+">";
  }
  else if (cont->outerVariable->varType==TValue::FLOATVAR) {
    string res="<";
    char buf[128];

    PITERATE(TDistributionMap, di, cont->continuous) {
      if (di!=cont->continuous->begin()) res+=", ";
      sprintf(buf, "%.3f: ", (*di).first);
      res+=buf+convertToString((*di).second);
    }
    return res+">";
  }

  raiseError("invalid contingency");
  return string();
}

string convertToString(const PContingencyClass &cc)
{ return convertToString((const PContingency &)cc); }

PyObject *Contingency_str(PyObject *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;

    return PyString_FromString(convertToString(PyOrange_AsContingency(self)).c_str());
  PyCATCH
}


PyObject *Contingency_keys(PyObject *self) PYARGS(0, "() -> [string] | [float]")
{ PyTRY
    CAST_TO(TContingency, cont);
    if (cont->outerVariable)
      if (cont->outerVariable->varType==TValue::FLOATVAR) {
        PyObject *nl=PyList_New(cont->continuous->size());
        int i=0;
        PITERATE(TDistributionMap, ci, cont->continuous)
          PyList_SetItem(nl, i++, PyFloat_FromDouble((double)(*ci).first));
        return nl;
      }
      else if (cont->outerVariable->varType==TValue::INTVAR) {
        PyObject *nl=PyList_New(cont->outerVariable->noOfValues());
        int i=0;
        PStringList vals=cont->outerVariable.AS(TEnumVariable)->values;
        PITERATE(TIdList, ii, vals)
          PyList_SetItem(nl, i++, PyString_FromString((*ii).c_str()));
        return nl;
      }

    raiseError("Invalid contingency ('outerVariable' not set)");
    return PYNULL;
  PyCATCH
}

PyObject *Contingency_values(PyObject *self) PYARGS(0, "() -> [Distribution]")
{ PyTRY
    CAST_TO(TContingency, cont);
    if (cont->outerVariable)
      if (cont->outerVariable->varType==TValue::FLOATVAR) {
        PyObject *nl=PyList_New(cont->continuous->size());
        int i=0;
        PITERATE(TDistributionMap, ci, cont->continuous)
          PyList_SetItem(nl, i++, WrapOrange((*ci).second));
        return nl;
      }
      else if (cont->outerVariable->varType==TValue::INTVAR) {
        PyObject *nl=PyList_New(cont->discrete->size());
        int i=0;
        PITERATE(TDistributionVector, ci, cont->discrete)
          PyList_SetItem(nl, i++, WrapOrange(*ci));
        return nl;
      }

    PYERROR(PyExc_AttributeError, "Invalid contingency (no variable)", PYNULL);
  PyCATCH
}

PyObject *Contingency_items(PyObject *self) PYARGS(0, "() -> [(string, Distribution)] | [(float: Distribution)]")
{ PyTRY
    CAST_TO(TContingency, cont);
    if (cont->outerVariable)
      if (cont->outerVariable->varType==TValue::FLOATVAR) {
        PyObject *nl=PyList_New(cont->continuous->size());
        int i=0;
        PITERATE(TDistributionMap, ci, cont->continuous)
          PyList_SetItem(nl, i++, 
            Py_BuildValue("fN", (double)(*ci).first, WrapOrange((*ci).second)));
        return nl;
      }
      else if (cont->outerVariable->varType==TValue::INTVAR) {
        PyObject *nl=PyList_New(cont->outerVariable->noOfValues());
        int i=0;
        TIdList::const_iterator ii(cont->outerVariable.AS(TEnumVariable)->values->begin());
        PITERATE(TDistributionVector, ci, cont->discrete)
          PyList_SetItem(nl, i++, 
            Py_BuildValue("sN", (*(ii++)).c_str(), WrapOrange(*ci)));
        return nl;
      }

    PYERROR(PyExc_AttributeError, "Invalid contingency (no variable)", PYNULL);
  PyCATCH
}



/* We redefine new (removed from below!) and add mapping methods
*/

inline PDomainContingency PDomainContingency_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::P_FromArguments(arg); }
inline PyObject *DomainContingency_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_FromArguments(type, arg); }
inline PyObject *DomainContingency_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_getitem(self, index); }
inline int       DomainContingency_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_setitem(self, index, item); }
inline PyObject *DomainContingency_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_getslice(self, start, stop); }
inline int       DomainContingency_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_setslice(self, start, stop, item); }
inline PyObject *DomainContingency_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_richcmp(self, object, op); }
inline int       DomainContingency_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_len(self); }
inline PyObject *DomainContingency_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_concat(self, obj); }
inline PyObject *DomainContingency_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_repeat(self, times); }
inline PyObject *DomainContingency_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_str(self); }
inline PyObject *DomainContingency_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_str(self); }
inline int       DomainContingency_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_contains(self, obj); }
inline PyObject *DomainContingency_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Contingency) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_append(self, item); }
inline PyObject *DomainContingency_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> int") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_count(self, obj); }
inline PyObject *DomainContingency_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainContingency") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_filter(self, args); }
inline PyObject *DomainContingency_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> int") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_index(self, obj); }
inline PyObject *DomainContingency_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_insert(self, args); }
inline PyObject *DomainContingency_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_native(self); }
inline PyObject *DomainContingency_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Contingency") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_pop(self, args); }
inline PyObject *DomainContingency_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_remove(self, obj); }
inline PyObject *DomainContingency_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_reverse(self); }
inline PyObject *DomainContingency_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, (PyTypeObject *)&PyOrContingency_Type>::_sort(self, args); }



CONSTRUCTOR_KEYWORDS(DomainContingency, "classIsOuter")

PyObject *DomainContingency_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange, "(examples [, weightID] | <list of Contingency>) -> DomainContingency")
{ PyTRY
    PyObject *obj = ListOfWrappedMethods<PDomainContingency, TDomainContingency, PDomainContingency, (PyTypeObject *)&PyOrContingency_Type>::_new(type, args, keywds);
    if (obj)
      if (obj!=Py_None)
        return obj;
      else
        Py_DECREF(obj);

    PyErr_Clear();

    int weightID;
    PExampleGenerator gen=exampleGenFromArgs(args, &weightID);
    if (!gen)
      return PYNULL;

    bool classOuter = false;
    if (keywds) {
      PyObject *couter = PyDict_GetItemString(keywds, "classIsOuter");
      if (couter) {
        classOuter = (PyObject_IsTrue(couter) != 0);
        Py_DECREF(couter);
      }
    }

    return WrapNewOrange(mlnew TDomainContingency(gen, weightID, classOuter), type);
  PyCATCH
}


int pt_DomainContingency(PyObject *args, void *egen)
{ if (PyOrDomainContingency_Check(args)) {
    *(PDomainContingency *)(egen) = PyOrange_AsDomainContingency(args);
    return 1;
  }
  else {
    egen = NULL;
    PYERROR(PyExc_TypeError, "invalid domain contingency", 0);
  }
}





int DomainContingency_getItemIndex(PyObject *self, PyObject *args)
{ PyTRY
    CAST_TO_err(TDomainContingency, cont, -1);
  
    const bool &couter = cont->classIsOuter;

    if (PyInt_Check(args)) {
      int i=(int)PyInt_AsLong(args);
      if ((i>=0) && (i<int(cont->size())))
        return i;
      else
        PYERROR(PyExc_IndexError, "index out of range", -1);
    }

    if (PyString_Check(args)) {
      char *s=PyString_AsString(args);
      PITERATE(vector<PContingencyClass>, ci, cont)
        if (couter ? (*ci)->innerVariable && ((*ci)->innerVariable->name==s)
                   : (*ci)->outerVariable && ((*ci)->outerVariable->name==s))
          return ci - cont->begin();
      PYERROR(PyExc_IndexError, "invalid variable name", -1);
    }

    if (PyOrVariable_Check(args)) {
      PVariable var = PyOrange_AsVariable(args);
      PITERATE(vector<PContingencyClass>, ci, cont)
        if (couter ? (*ci)->innerVariable && ((*ci)->innerVariable==var)
                   : (*ci)->outerVariable && ((*ci)->outerVariable==var))
          return ci - cont->begin();
      PYERROR(PyExc_IndexError, "invalid variable", -1);
    }

    PYERROR(PyExc_TypeError, "invalid index type", -1);
  PyCATCH_1
}




PyObject *DomainContingency_getitem(PyObject *self, PyObject *args)
{ PyTRY
    int index=DomainContingency_getItemIndex(self, args);
    if (index<0)
      return PYNULL;

    return WrapOrange(POrange(SELF_AS(TDomainContingency)[index]));
  PyCATCH
}


int DomainContingency_setitem(PyObject *self, PyObject *args, PyObject *obj)
{ PyTRY
    PContingency cont;
    if (!convertFromPython(obj, cont))
      PYERROR(PyExc_TypeError, "invalid Contingency object", -1);

    int index = DomainContingency_getItemIndex(self, args);
    if (index==-1)
      return -1;

    SELF_AS(TDomainContingency)[index] = cont;
    return 0;
  PyCATCH_1
}


string convertToString(const PDomainContingency &cont)
{
  string res=string("{");
  const_PITERATE(TDomainContingency, di, cont) {
    if (di!=cont->begin()) res+=", ";
    res += (*di)->outerVariable->name+": "+convertToString(*di);
  }
  return res+"}";
}


PyObject *DomainContingency_normalize(PyObject *self, PyObject *) PYARGS(0, "() -> None")
{ PyTRY
    if (!PyOrange_AS_Orange(self))
      PYERROR(PyExc_SystemError, "NULL contingency matrix", PYNULL);

    SELF_AS(TDomainContingency).normalize();
    RETURN_NONE
  PyCATCH
}




/* ************ DISTANCE ************ */

#include "distance.hpp"
#include "distance_dtw.hpp"

BASED_ON(ExamplesDistance, Orange)
BASED_ON(ExamplesDistance_Normalized, ExamplesDistance)
C_NAMED(ExamplesDistance_Hamiltonian, ExamplesDistance, "()")
C_NAMED(ExamplesDistance_Maximal, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Manhattan, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Euclidean, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Relief, ExamplesDistance, "()")
C_NAMED(ExamplesDistance_DTW, ExamplesDistance_Normalized, "()")

BASED_ON(ExamplesDistanceConstructor, Orange)
C_CALL(ExamplesDistanceConstructor_Hamiltonian, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Hamiltonian")
C_CALL(ExamplesDistanceConstructor_Maximal, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Maximal")
C_CALL(ExamplesDistanceConstructor_Manhattan, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Manhattan")
C_CALL(ExamplesDistanceConstructor_Euclidean, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Euclidean")
C_CALL(ExamplesDistanceConstructor_Relief, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Relief")
C_CALL(ExamplesDistanceConstructor_DTW, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_DTW")



PyObject *ExamplesDistanceConstructor_call(PyObject *self, PyObject *uargs, PyObject *keywords) PYDOC("([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance")
{ PyTRY
    SETATTRIBUTES

    PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
    PExampleGenerator gen;
    int weightID = 0;
    PDomainDistributions dist;
    PDomainBasicAttrStat bstat;
    if (!PyArg_ParseTuple(uargs, "|OOOO:ExamplesDistanceConstructor.call", args+0, args+1, args+2, args+3))
      return PYNULL;

    PyObject **argp = args, **argc = args;
    for (int i=0; i<=3; i++, argp++)
      if (*argp)
        *argc++ = *argp;

    for(argp = args; argp!=argc; argp++) {
      if (PyOrDomainDistributions_Check(*argp))
        if (dist)
          PYERROR(PyExc_TypeError, "ExamplesDistanceConstructor.__call__: invalid arguments (DomainDistribution given twice)", PYNULL)
        else
          dist = PyOrange_AsDomainDistributions(*argp);
      else if (PyOrDomainBasicAttrStat_Check(*argp))
        if (bstat)
          PYERROR(PyExc_TypeError, "ExamplesDistanceConstructor.__call__: invalid arguments (DomainBasicAttrStat given twice)", PYNULL)
        else
          bstat = PyOrange_AsDomainBasicAttrStat(*argp);
      else {
        PExampleGenerator gen2 = exampleGenFromParsedArgs(*argp);
        if (!gen2)
          PYERROR(PyExc_TypeError, "ExamplesDistanceConstructor.__call__: invalid arguments", PYNULL)
        else if (gen)
          PYERROR(PyExc_TypeError, "ExamplesDistanceConstructor.__call__: invalid arguments (examples given twice)", PYNULL)
        else {
          gen = gen2;
          if (argp+1 != argc) {
            argp++;
            if (!weightFromArg_byDomain(*argp, gen->domain, weightID))
              return PYNULL;
          }
        }
      }
    }

    return WrapOrange(SELF_AS(TExamplesDistanceConstructor).call(gen, weightID, dist, bstat));
  PyCATCH
}


PyObject *ExamplesDistance_Normalized_attributeDistances(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example1, example2) -> [by-attribute distances as floats]")
{ PyTRY
    TExample *ex1, *ex2;
    if (!PyArg_ParseTuple(args, "O&O&:ExamplesDistance_Normalized.attributeDistances", ptr_Example, &ex1, ptr_Example, &ex2))
      PYERROR(PyExc_TypeError, "attribute error (two examples expected)", PYNULL);

    vector<float> difs;
    SELF_AS(TExamplesDistance_Normalized).getDifs(*ex1, *ex2, difs);
    
    PyObject *l = PyList_New(difs.size());
    for(int i = 0, e = difs.size(); i<e; e++)
      PyList_SetItem(l, i, PyFloat_FromDouble(difs[i]));
      
    return l;
  PyCATCH
}


PyObject *ExamplesDistance_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example1, example2) -> float")
{
  PyTRY
    SETATTRIBUTES
    TExample *ex1, *ex2;
    if (!PyArg_ParseTuple(args, "O&O&:ExamplesDistance_Normalized.__call__", ptr_Example, &ex1, ptr_Example, &ex2))
      PYERROR(PyExc_TypeError, "attribute error (two examples expected)", PYNULL);

    return PyFloat_FromDouble((double)(SELF_AS(TExamplesDistance)(*ex1, *ex2)));
  PyCATCH
}


PyObject *ExamplesDistance_DTW_alignment(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example1, example2) -> (distance, path)")
{
  PyTRY
    TExample *ex1, *ex2;
    if (!PyArg_ParseTuple(args, "O&O&:ExamplesDistance_DTW.attributeDistances", ptr_Example, &ex1, ptr_Example, &ex2))
      PYERROR(PyExc_TypeError, "attribute error (two examples expected)", PYNULL);

    PWarpPath warpPath;
    float distance = SELF_AS(TExamplesDistance_DTW)(*ex1, *ex2, warpPath);
    return Py_BuildValue("fO", distance, WrapOrange(warpPath));
  PyCATCH
}

/* ************ FINDNEAREST ************ */

#include "nearest.hpp"

BASED_ON(FindNearest, Orange)
C_NAMED(FindNearest_BruteForce, FindNearest, "([distance=, distanceID=, includeSame=])")

BASED_ON(FindNearestConstructor, Orange)
C_CALL(FindNearestConstructor_BruteForce, FindNearestConstructor, "([examples[, weightID[, distanceID]], distanceConstructor=, includeSame=]) -/-> FindNearest")


PyObject *FindNearestConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID[, distanceID]]) -> FindNearest")
{
  PyTRY
    SETATTRIBUTES
    PExampleGenerator egen;
    int weightID = 0;
    int distanceID = 0;
    PyObject *pydistanceID = PYNULL;

    if (   !PyArg_ParseTuple(args, "O&|O&O:FindNearestConstructor.__call__", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID, &pydistanceID)
        || !weightFromArg_byDomain(pydistanceID, egen->domain, distanceID))
      return PYNULL;

    return WrapOrange(SELF_AS(TFindNearestConstructor).call(egen, weightID, distanceID));
  PyCATCH
}



PyObject *FindNearest_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example, k) -> ExampleTable")
{
  PyTRY
    SETATTRIBUTES
    float k;
    TExample *example;
    // Both forms are allowed for compatibility with older versions
    if (!PyArg_ParseTuple(args, "fO&", &k, ptr_Example, &example)) {
      PyErr_Clear();
      if (!PyArg_ParseTuple(args, "O&f", ptr_Example, &example, &k))
        PYERROR(PyExc_TypeError, "attribute error (number and example expected)", PYNULL);
    }

    return WrapOrange(SELF_AS(TFindNearest).call(*example, k));
  PyCATCH
}




/* ************ FILTERS ************ */

#include "filter.hpp"

BASED_ON(ValueFilter, Orange)
C_NAMED(ValueFilter_discrete, ValueFilter, "([position=, acceptableValues=, acceptSpecial=])")
C_NAMED(ValueFilter_continuous, ValueFilter, "([position=, min=, max=, acceptSpecial=])")

BASED_ON(Filter, Orange)
C_CALL(Filter_random, Filter, "([examples], [negate=..., p=...]) -/-> ExampleTable")
C_CALL(Filter_hasSpecial, Filter, "([examples], [negate=..., domain=...]) -/-> ExampleTable")
C_CALL(Filter_hasClassValue, Filter, "([examples], [negate=..., domain=...]) -/-> ExampleTable")
C_CALL(Filter_sameValue, Filter, "([examples], [negate=..., domain=..., position=<int>, value=...]) -/-> ExampleTable")
C_CALL(Filter_values, Filter, "([examples], [negate=..., domain=..., values=<see the manual>) -/-> ExampleTable")


PValueFilterList PValueFilterList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::P_FromArguments(arg); }
PyObject *ValueFilterList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_FromArguments(type, arg); }
PyObject *ValueFilterList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ValueFilter>)") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_new(type, arg, kwds); }
PyObject *ValueFilterList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_getitem(self, index); }
int       ValueFilterList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_setitem(self, index, item); }
PyObject *ValueFilterList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_getslice(self, start, stop); }
int       ValueFilterList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_setslice(self, start, stop, item); }
int       ValueFilterList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_len(self); }
PyObject *ValueFilterList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_richcmp(self, object, op); }
PyObject *ValueFilterList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_concat(self, obj); }
PyObject *ValueFilterList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_repeat(self, times); }
PyObject *ValueFilterList_str(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_str(self); }
PyObject *ValueFilterList_repr(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_str(self); }
int       ValueFilterList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_contains(self, obj); }
PyObject *ValueFilterList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ValueFilter) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_append(self, item); }
PyObject *ValueFilterList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> int") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_count(self, obj); }
PyObject *ValueFilterList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ValueFilterList") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_filter(self, args); }
PyObject *ValueFilterList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> int") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_index(self, obj); }
PyObject *ValueFilterList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_insert(self, args); }
PyObject *ValueFilterList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_native(self); }
PyObject *ValueFilterList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ValueFilter") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_pop(self, args); }
PyObject *ValueFilterList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_remove(self, obj); }
PyObject *ValueFilterList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_reverse(self); }
PyObject *ValueFilterList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, (PyTypeObject *)&PyOrValueFilter_Type>::_sort(self, args); }


PyObject *applyFilter(PFilter filter, PExampleGenerator gen, bool weightGiven, int weightID)
{ if (!filter) return PYNULL;

  TExampleTable *newTable=mlnew TExampleTable(gen->domain);
  PExampleGenerator newGen(newTable); // ensure it gets deleted in case of error
  filter->reset();
  PEITERATE(ei, gen)
    if (filter->operator()(*ei))
      newTable->addExample(*ei);

  return weightGiven ? Py_BuildValue("Ni", WrapOrange(newGen), weightID) : WrapOrange(newGen);
}


PyObject *Filter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrFilter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TFilter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TFilter_Python(), type);
}


PyObject *Filter_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrFilter_Type) {
      PyErr_Format(PyExc_SystemError, "Filter.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES

    if (PyTuple_Size(args)==1) {
      PyObject *pyex = PyTuple_GET_ITEM(args, 0);
      if (PyOrExample_Check(pyex))
        return PyInt_FromLong(PyOrange_AsFilter(self)->call(PyExample_AS_ExampleReference(pyex)) ? 1 : 0);
    }

    PExampleGenerator egen = exampleGenFromArgs(args);
    if (!egen)
      PYERROR(PyExc_TypeError, "attribute error (example generator expected)", PYNULL);
    return applyFilter(PyOrange_AsFilter(self), egen, false, 0);
  PyCATCH
}



/* ************ RANDOM INDICES ******************** */
#include "trindex.hpp"

BASED_ON(MakeRandomIndices, Orange)
C_CALL3(MakeRandomIndices2, MakeRandomIndices2, MakeRandomIndices, "[n | gen [, p0]], [p0=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesMultiple, MakeRandomIndicesMultiple, MakeRandomIndices, "[n | gen [, p]], [p=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesN, MakeRandomIndicesN, MakeRandomIndices, "[n | gen [, p]], [p=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesCV, MakeRandomIndicesCV, MakeRandomIndices, "[n | gen [, folds]], [folds=, stratified=, randseed=] -/-> [int]")

PYCLASSCONSTANT_INT(MakeRandomIndices, StratifiedIfPossible, -1L)
PYCLASSCONSTANT_INT(MakeRandomIndices, NotStratified, 0L)
PYCLASSCONSTANT_INT(MakeRandomIndices, Stratified, 1L)

PYCONSTANT(StratifiedIfPossible, PyInt_FromLong(-1L))
PYCONSTANT(NotStratified, PyInt_FromLong(0L))
PYCONSTANT(Stratified, PyInt_FromLong(1L))

PyObject *MakeRandomIndices2_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    SETATTRIBUTES
    CAST_TO(TMakeRandomIndices2, mri2);

    int n;
    float f;
    PExampleGenerator egen;    
    PRandomIndices res;

    if (PyArg_ParseTuple(args, "i", &n)) {
      res = (*mri2)(n);
      goto out;
    }
    
    PyErr_Clear();
    if (PyArg_ParseTuple(args, "if", &n, &f)) {
      res = (*mri2)(n, f);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&", pt_ExampleGenerator, &egen)) {
      res = (*mri2)(egen);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&f", pt_ExampleGenerator, &egen, &f)) {
      res = (*mri2)(egen, f);
      goto out;
    }

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out: 
    if (!res)
      PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

    return WrapOrange(res);
  PyCATCH
}


PyObject *MakeRandomIndicesMultiple_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    SETATTRIBUTES
    CAST_TO(TMakeRandomIndicesMultiple, mrim)

    int n;
    float f;
    PExampleGenerator egen;
    PRandomIndices res;

    if (PyArg_ParseTuple(args, "i", &n)) {
      res = (*mrim)(n);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "if", &n, &f)) {
      res = (*mrim)(n, f);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&", pt_ExampleGenerator, &egen)) {
      res = (*mrim)(egen);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&f", pt_ExampleGenerator, &egen, &f)) {
      res = (*mrim)(egen, f);
      goto out;
    }

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
    if (!res)
      PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

    return WrapOrange(res);
  PyCATCH
}




PyObject *MakeRandomIndicesN_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    SETATTRIBUTES
    CAST_TO(TMakeRandomIndicesN, mriN)

    int n;
    PFloatList pyvector;
    PExampleGenerator egen;
    PRandomIndices res;

    if (PyArg_ParseTuple(args, "i", &n)) {
      res = (*mriN)(n);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "iO&", &n, cc_FloatList, &pyvector)) {
      res = (*mriN)(n, pyvector);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&", pt_ExampleGenerator, &egen)) {
      res = (*mriN)(egen);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&O&", pt_ExampleGenerator, &egen, cc_FloatList, &pyvector)) {
      res = (*mriN)(egen, pyvector);
      goto out;
    }

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
    if (!res)
      PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

    return WrapOrange(res);
  PyCATCH
}



PyObject *MakeRandomIndicesCV_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    SETATTRIBUTES
    CAST_TO(TMakeRandomIndicesCV, mriCV)

    int n, f;
    PExampleGenerator egen;
    PRandomIndices res;

    if (PyArg_ParseTuple(args, "i", &n)) {
      res = (*mriCV)(n);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "ii", &n, &f)) {
      res = (*mriCV)(n, f);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&", pt_ExampleGenerator, &egen)) {
      res = (*mriCV)(egen);
      goto out;
    }

    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O&i", pt_ExampleGenerator, &egen, &f)) {
      res = (*mriCV)(egen, f);
      goto out;
    }

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
    if (!res)
      PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

    return WrapOrange(res);
  PyCATCH
}


/* ************ PROBABILITY ESTIMATION ************ */

#include "estimateprob.hpp"

BASED_ON(ProbabilityEstimator, Orange)
BASED_ON(ProbabilityEstimatorConstructor, Orange)
C_NAMED(ProbabilityEstimator_FromDistribution, ProbabilityEstimator, "()")
C_CALL(ProbabilityEstimatorConstructor_relative, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_Laplace, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_m, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_kernel, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurve")
C_CALL(ProbabilityEstimatorConstructor_loess, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurve")

BASED_ON(ConditionalProbabilityEstimator, Orange)
BASED_ON(ConditionalProbabilityEstimatorConstructor, Orange)
C_NAMED(ConditionalProbabilityEstimator_FromDistribution, ConditionalProbabilityEstimator, "()")
C_NAMED(ConditionalProbabilityEstimator_ByRows, ConditionalProbabilityEstimator, "()")
C_CALL(ConditionalProbabilityEstimatorConstructor_ByRows, ConditionalProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ConditionalProbabilityEstimator_[FromDistribution|ByRows]")
C_CALL(ConditionalProbabilityEstimatorConstructor_loess, ConditionalProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurves")

PyObject *ProbabilityEstimatorConstructor_call(PyObject *self, PyObject *uargs, PyObject *keywords) PYDOC("([distribution[, apriori]] [example generator[, weight]]) -> ProbabilityEstimator")
{ PyTRY
    CAST_TO(TProbabilityEstimatorConstructor, cest);
    SETATTRIBUTES

    PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
    PDistribution dist, apriori;
    PExampleGenerator gen;
    int weightID = 0;
    if (!PyArg_ParseTuple(uargs, "|OOOO:ProbabilityEstimatorConstructor.call", args+0, args+1, args+2, args+3))
      return PYNULL;

    PyObject **argp = args, **argc = args;
    for (int i=0; i<=3; i++, argp++)
      if (*argp)
        *argc++ = *argp;

    argp = args;
    if ((argp != argc) && ((*argp==Py_None) || PyOrDistribution_Check(*argp))) {
      dist = (*argp==Py_None) ? PDistribution() : PyOrange_AsDistribution(*argp++);
      if ((argp != argc) && PyOrDistribution_Check(*argp))
        apriori = PyOrange_AsDistribution(*argp++);
    }
    if (argp != argc) {
      gen = exampleGenFromParsedArgs(*argp);
      if (gen) {
        argp++;
        if ((argp != argc) && !weightFromArg_byDomain(*(argp++), gen->domain, weightID))
          return PYNULL;
      }
    }

    if (argp != argc)
      PYERROR(PyExc_TypeError, "Invalid arguments for 'ProbabilityEstimatorConstructor.call'", PYNULL);

    return WrapOrange(cest->call(dist, apriori, gen, weightID));
  PyCATCH
}


PyObject *ProbabilityEstimator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(Value) -> float  |  () -> Distribution")
{ PyTRY
    CAST_TO(TProbabilityEstimator, cest);
    SETATTRIBUTES

    PyObject *pyobj = PYNULL;
    if (!PyArg_ParseTuple(args, "|O:ProbabilityEstimator.call", &pyobj)) 
      return PYNULL;
    
    if (pyobj) {
      TValue val;
      if (!convertFromPython(pyobj, val))
        PYERROR(PyExc_TypeError, "ProbabilityEstimator.call: cannot convert the arguments to a Value", PYNULL);
      return PyFloat_FromDouble((double)cest->call(val));
    }

    else
      return WrapOrange(cest->call());
  PyCATCH
}



PyObject *ConditionalProbabilityEstimatorConstructor_call(PyObject *self, PyObject *uargs, PyObject *keywords) PYDOC("([contingency[, apriori]] [example generator[, weight]]) -> ProbabilityEstimator")
{ PyTRY
    CAST_TO(TConditionalProbabilityEstimatorConstructor, cest);
    SETATTRIBUTES

    PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
    PContingency cont, apriori;
    PExampleGenerator gen;
    int weightID = 0;
    if (!PyArg_ParseTuple(uargs, "|OOOO:ProbabilityEstimatorConstructor.call", args, args+1, args+2, args+3))
      return PYNULL;

    PyObject **argp = args, **argc = args;
    for (int i=0; i<=3; i++, argp++)
      if (*argp)
        *argc++ = *argp;

    argp = args;
    if ((argp != argc) && ((*argp==Py_None) || PyOrContingency_Check(*argp))) {
      cont = (*argp==Py_None) ? PContingency() : PyOrange_AsContingency(*argp++);
      if ((argp != argc) && PyOrDistribution_Check(*argp))
        apriori = PyOrange_AsDistribution(*argp++);
    }
    if (argp != argc) {
      gen = exampleGenFromParsedArgs(*argp);
      if (gen) {
        argp++;
        if ((argp != argc) && !weightFromArg_byDomain(*(argp++), gen->domain, weightID))
          return PYNULL;
      }
    }

    if (argp != argc)
      PYERROR(PyExc_TypeError, "Invalid arguments for 'ProbabilityEstimatorConstructor.call'", PYNULL);

    return WrapOrange(cest->call(cont, apriori, gen, weightID));
  PyCATCH
}


PyObject *ConditionalProbabilityEstimator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(Value, Condition) -> float  |  (Condition) -> Distribution | () -> Contingency")
{ PyTRY
    CAST_TO(TConditionalProbabilityEstimator, cest);
    SETATTRIBUTES

    PyObject *pyobj1 = PYNULL, *pyobj2 = PYNULL;
    if (!PyArg_ParseTuple(args, "|OO:ProbabilityEstimator.call", &pyobj1, &pyobj2)) 
      return PYNULL;
    
    if (pyobj1 && pyobj2) {
      TValue val1, val2;
      if (!convertFromPython(pyobj1, val1) || !convertFromPython(pyobj2, val2))
        PYERROR(PyExc_TypeError, "ProbabilityEstimator.call: cannot convert the arguments to a Value", PYNULL);
      return PyFloat_FromDouble((double)cest->call(val1, val2));
    }

    else if (pyobj1) {
      TValue val;
      if (!convertFromPython(pyobj1, val))
        PYERROR(PyExc_TypeError, "ProbabilityEstimator.call: cannot convert the arguments to a Value", PYNULL);
      return WrapOrange(cest->call(val));
    }

    else
      return WrapOrange(cest->call());
  PyCATCH
}



/* ************ MEASURES ************ */

#include "measures.hpp"
#include "relief.hpp"

BASED_ON(MeasureAttribute, Orange)
BASED_ON(MeasureAttributeFromProbabilities, MeasureAttribute)

C_CALL(MeasureAttribute_info, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gini, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gainRatio, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gainRatioA, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_cost, MeasureAttributeFromProbabilities, "(cost=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")
C_CALL(MeasureAttribute_relevance, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")

C_CALL(MeasureAttribute_MSE, MeasureAttribute, "(estimate=, m=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")

C_CALL(MeasureAttribute_relief, MeasureAttribute, "(estimate=, m=, k=) | (attr, examples[, apriori] [,weightID]) -/-> float")

PyObject *MeasureNeeds()
{ PyObject *vartypes=PyModule_New("MeasureNeeds");
  PyModule_AddIntConstant(vartypes, "Generator", (int)TMeasureAttribute::Generator);
  PyModule_AddIntConstant(vartypes, "DomainContingency", (int)TMeasureAttribute::DomainContingency);
  PyModule_AddIntConstant(vartypes, "Contingency_Class", (int)TMeasureAttribute::Contingency_Class);
  return vartypes;
}

PYCLASSCONSTANT_INT(MeasureAttribute, NeedsGenerator, TMeasureAttribute::Generator)
PYCLASSCONSTANT_INT(MeasureAttribute, NeedsDomainContingency, TMeasureAttribute::DomainContingency)
PYCLASSCONSTANT_INT(MeasureAttribute, NeedsContingency_Class, TMeasureAttribute::Contingency_Class)

PYCLASSCONSTANT_INT(MeasureAttribute, IgnoreUnknowns, TMeasureAttribute::IgnoreUnknowns)
PYCLASSCONSTANT_INT(MeasureAttribute, ReduceByUnknowns, TMeasureAttribute::ReduceByUnknowns)
PYCLASSCONSTANT_INT(MeasureAttribute, UnknownsToCommon, TMeasureAttribute::UnknownsToCommon)
PYCLASSCONSTANT_INT(MeasureAttribute, UnknownsAsValue, TMeasureAttribute::UnknownsAsValue)

PYCLASSCONSTANT_FLOAT(MeasureAttribute, Rejected, ATTRIBUTE_REJECTED)

PyObject *MeasureAttribute_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrMeasureAttribute_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TMeasureAttribute_Python(), type), args);
  else
    return WrapNewOrange(mlnew TMeasureAttribute_Python(), type);
}


PyObject *MeasureAttribute_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(attr, xmpls[, apr, wght]) | (attr, domcont[, apr]) | (cont, clss-dist [,apr]) -> (float, meas-type)")
{ PyTRY
    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrMeasureAttribute_Type) {
      PyErr_Format(PyExc_SystemError, "MeasureAttribute.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES
    CAST_TO(TMeasureAttribute, meat)

    PyObject *arg1;
    PDistribution aprClDistr;

    // Try (contingency, class distribution, aprior class distribution)

    PContingency contingency;
    PDistribution clDistr;
    if (PyArg_ParseTuple(args, "O&O&|O&", cc_Contingency, &contingency, cc_Distribution, &clDistr, ccn_Distribution, &aprClDistr))
      return PyFloat_FromDouble((double)(meat->operator()(contingency, clDistr, aprClDistr)));

    PyErr_Clear();


    // Try (variable, domaincontingency, aprior class distribution)

    PDomainContingency dcont;
    if (PyArg_ParseTuple(args, "OO&|O&", &arg1, cc_DomainContingency, &dcont, ccn_Distribution, &aprClDistr)) {

        int attrNo;
      
        if (PyInt_Check(arg1)) {
          attrNo = int(PyInt_AsLong(arg1));
          if ((attrNo<0) || (attrNo>=dcont->size())) {
            PyErr_Format(PyExc_IndexError, "attribute index %i out of range for the given DomainContingency", attrNo);
            return PYNULL;
          }
        }

        else {
          TDomainContingency::const_iterator ci(dcont->begin()), ce(dcont->end());
          const bool &couter = dcont->classIsOuter;

          if (PyOrVariable_Check(arg1)) {
            PVariable var = PyOrange_AsVariable(arg1);
            for(attrNo = 0; (ci!=ce) && (couter ? ((*ci)->innerVariable != var) : ((*ci)->outerVariable != var)); ci++, attrNo++);

            if (ci==ce) {
              PyErr_Format(PyExc_IndexError, "attribute '%s' not in the given DomainContingency", var->name.c_str());
              return PYNULL;
            }
          }

          else if (PyString_Check(arg1)) {
            char *attrName = PyString_AsString(arg1);
            for(attrNo = 0; (ci!=ce) && (couter ? ((*ci)->innerVariable->name != attrName) : ((*ci)->outerVariable->name!=attrName)); ci++, attrNo++);

            if (ci==ce) {
              PyErr_Format(PyExc_IndexError, "attribute '%s' not in the given DomainContingency", attrName);
              return PYNULL;
            }
          }

          else {
            PyErr_Format(PyExc_IndexError, "cannot guess the attribute from the object of type '%s'", arg1->ob_type->tp_name);
            return PYNULL;
          }  
        }

        return PyFloat_FromDouble((double)(meat->operator()(attrNo, dcont, aprClDistr)));
    }


    PyErr_Clear();


    // Try (variable, examples, aprior class distribution, weight)

    PExampleGenerator egen;
    if ((PyTuple_Size(args) >= 2) && pt_ExampleGenerator(PyTuple_GET_ITEM(args, 1), &egen)) {

      // No need to INCREF (ParseArgs doesn't INCREF either)
      PyObject *arg3 = Py_None, *arg4 = Py_None;
      
      arg1 = PyTuple_GET_ITEM(args, 0);
    
      if (PyTuple_Size(args) == 4) {
        arg3 = PyTuple_GET_ITEM(args, 2);
        arg4 = PyTuple_GET_ITEM(args, 3);
      }

      else if (PyTuple_Size(args) == 3) {
        arg4 = PyTuple_GET_ITEM(args, 2);
        // This mess is for compatibility; previously, weightID could only be the 4th argument; 3rd had to be either Distribution or None if 4th was to be given
        if (PyOrDistribution_Check(arg4)) {
          arg3 = arg4;
          arg4 = Py_None;
        }
        else
          arg3 = Py_None;
      }

      int weightID=0;

      if (arg3 != Py_None)
        if (PyOrDistribution_Check(arg4))
          aprClDistr = PyOrange_AsDistribution(arg3);
        else
          PYERROR(PyExc_TypeError, "invalid argument 3 (Distribution or None expected)", PYNULL);

      if (arg4 != Py_None)
        if (!weightFromArg_byDomain(arg4, egen->domain, weightID))
          PYERROR(PyExc_TypeError, "invalid argument 4 (weightID or None expected)", PYNULL);

      if (PyOrVariable_Check(arg1))
        return PyFloat_FromDouble((double)(meat->operator()(PyOrange_AsVariable(arg1), egen, aprClDistr, weightID)));

      else if (PyInt_Check(arg1)) {
        int attrNo = PyInt_AsLong(arg1);
        if (attrNo >= (int)egen->domain->attributes->size()) {
          PyErr_Format(PyExc_IndexError, "attribute index %i out of range for the given DomainContingency", attrNo);
          return PYNULL;
        }
        return PyFloat_FromDouble((double)(meat->operator()(attrNo, egen, aprClDistr, weightID)));
      }

      else {
        int attrNo;
        if (!varNumFromVarDom(arg1, egen->domain, attrNo))
          PYERROR(PyExc_TypeError, "invalid argument 1 (attribute index, name or descriptor expected)", PYNULL);

        return PyFloat_FromDouble((double)(meat->operator()(attrNo, egen, aprClDistr, weightID)));
      }
    }

    PYERROR(PyExc_TypeError, "invalid set of parameters", PYNULL);
    return PYNULL;
  PyCATCH;
}


/* ************ EXAMPLE CLUSTERING ************ */

#include "exampleclustering.hpp"

BASED_ON(GeneralExampleClustering, Orange)
C_NAMED(ExampleCluster, Orange, "([left=, right=, distance=, centroid=])")
C_NAMED(ExampleClusters, GeneralExampleClustering, "([root=, quality=]")


PyObject *GeneralExampleClustering_exampleClusters(PyObject *self) PYARGS(METH_NOARGS, "() -> ExampleClusters")
{ return WrapOrange(SELF_AS(TGeneralExampleClustering).exampleClusters()); }


PyObject *GeneralExampleClustering_exampleSets(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> ExampleSets")
{ float cut = 0.0;
  if (!PyArg_ParseTuple(args, "|f", &cut))
    return PYNULL;

  return WrapOrange(SELF_AS(TGeneralExampleClustering).exampleSets(cut));
}


PyObject *GeneralExampleClustering_classifier(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> Classifier")
{ float cut = 0.0;
  if (!PyArg_ParseTuple(args, "|f", &cut))
    return PYNULL;

  return WrapOrange(SELF_AS(TGeneralExampleClustering).classifier(cut));
}


PyObject *GeneralExampleClustering_feature(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> Variable")
{ float cut = 0.0;
  if (!PyArg_ParseTuple(args, "|f", &cut))
    return PYNULL;

  return WrapOrange(SELF_AS(TGeneralExampleClustering).feature(cut));
}


#include "heatmap.hpp"

PyObject *HeatmapConstructor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Orange, "(ExampleTable[, baseHeatmap=None [, disregardClass=0]])")
{
  PExampleTable table;
  PHeatmapConstructor baseHeatmap;
  int disregardClass = 0;
  if (!PyArg_ParseTuple(args, "O&|O&i:HeatmapConstructor.__new__", cc_ExampleTable, &table, ccn_HeatmapConstructor, &baseHeatmap, &disregardClass))
    return NULL;
  return WrapNewOrange(mlnew THeatmapConstructor(table, baseHeatmap, (PyTuple_Size(args)==2) && !baseHeatmap, (disregardClass!=0)), type);
}


PyObject *HeatmapConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(squeeze) -> HeatmapList")
{
  float squeeze;
  if (!PyArg_ParseTuple(args, "f:HeatmapConstructor.__call__", &squeeze))
    return NULL;

  SETATTRIBUTES

  float absLow, absHigh;
  PHeatmapList hml = SELF_AS(THeatmapConstructor).call(squeeze, absLow, absHigh);
  return Py_BuildValue("Nff", WrapOrange(hml), absLow, absHigh);
}


PyObject *HeatmapConstructor_getLegend(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(width, height, gamma) -> bitmap")
{ int width, height;
  float gamma;
  if (!PyArg_ParseTuple(args, "iif:HeatmapConstructor.getLegend", &width, &height, &gamma))
    return NULL;

  int size;
  unsigned char *bitmap = SELF_AS(THeatmapConstructor).getLegend(width, height, gamma, size);
  PyObject *res = PyString_FromStringAndSize((const char *)bitmap, size);
  delete bitmap;
  return res;
}

BASED_ON(Heatmap, Orange)

PyObject *Heatmap_getBitmap(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(cell_width, cell_height, lowerBound, upperBound, gamma) -> bitmap")
{
  int cellWidth, cellHeight;
  float absLow, absHigh, gamma;
  if (!PyArg_ParseTuple(args, "iifff:Heatmap.getBitmap", &cellWidth, &cellHeight, &absLow, &absHigh, &gamma))
    return NULL;

  CAST_TO(THeatmap, hm)

  int size;
  unsigned char *bitmap = hm->heatmap2string(cellWidth, cellHeight, absLow, absHigh, gamma, size);
  PyObject *res = Py_BuildValue("Nii", PyString_FromStringAndSize((const char *)bitmap, size), cellWidth * hm->width, cellHeight * hm->height);
  delete bitmap;
  return res;
}


PyObject *Heatmap_getAverages(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(cell_width, cell_height, lowerBound, upperBound, gamma) -> bitmap")
{
  int width, height;
  float absLow, absHigh, gamma;
  if (!PyArg_ParseTuple(args, "iifff:Heatmap.getAverageBitmap", &width, &height, &absLow, &absHigh, &gamma))
    return NULL;

  CAST_TO(THeatmap, hm)

  int size;
  unsigned char *bitmap = hm->averages2string(width, height, absLow, absHigh, gamma, size);
  PyObject *res = Py_BuildValue("Nii", PyString_FromStringAndSize((const char *)bitmap, size), width, height * hm->height);
  delete bitmap;
  return res;
}


PyObject *Heatmap_getCellIntensity(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(row, column) -> float")
{
  int row, column;
  if (!PyArg_ParseTuple(args, "ii:Heatmap.getCellIntensity", &row, &column))
    return NULL;

  return PyFloat_FromDouble(SELF_AS(THeatmap).getCellIntensity(row, column));
}


PyObject *Heatmap_getRowIntensity(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(row) -> float")
{
  int row;
  if (!PyArg_ParseTuple(args, "i:Heatmap.getRowIntensity", &row))
    return NULL;

  return PyFloat_FromDouble(SELF_AS(THeatmap).getRowIntensity(row));
}

PHeatmapList PHeatmapList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::P_FromArguments(arg); }
PyObject *HeatmapList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_FromArguments(type, arg); }
PyObject *HeatmapList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Heatmap>)") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_new(type, arg, kwds); }
PyObject *HeatmapList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_getitem(self, index); }
int       HeatmapList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_setitem(self, index, item); }
PyObject *HeatmapList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_getslice(self, start, stop); }
int       HeatmapList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_setslice(self, start, stop, item); }
int       HeatmapList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_len(self); }
PyObject *HeatmapList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_richcmp(self, object, op); }
PyObject *HeatmapList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_concat(self, obj); }
PyObject *HeatmapList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_repeat(self, times); }
PyObject *HeatmapList_str(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_str(self); }
PyObject *HeatmapList_repr(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_str(self); }
int       HeatmapList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_contains(self, obj); }
PyObject *HeatmapList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_append(self, item); }
PyObject *HeatmapList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_count(self, obj); }
PyObject *HeatmapList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> HeatmapList") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_filter(self, args); }
PyObject *HeatmapList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_index(self, obj); }
PyObject *HeatmapList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_insert(self, args); }
PyObject *HeatmapList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_native(self); }
PyObject *HeatmapList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Heatmap") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_pop(self, args); }
PyObject *HeatmapList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_remove(self, obj); }
PyObject *HeatmapList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_reverse(self); }
PyObject *HeatmapList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, (PyTypeObject *)&PyOrHeatmap_Type>::_sort(self, args); }

#include "lib_components.px"
