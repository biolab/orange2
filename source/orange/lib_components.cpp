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


#ifdef _MSC_VER
#pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "vars.hpp"
#include "stringvars.hpp"
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
#include "slist.hpp"

#include "externs.px"


bool convertFromPython(PyObject *, PContingency &, bool allowNull=false, PyTypeObject *type=NULL);

/* ************ COST ************ */

#include "cost.hpp"



PyObject *convertToPython(const PCostMatrix &matrix)
{
	int dim = matrix->dimension;
	PyObject *pycost = PyList_New(dim);
	float *ci = matrix->costs;
	for(int i = 0; i < dim; i++) {
		PyObject *row = PyList_New(dim);
		for(int j = 0; j < dim; j++)
			PyList_SetItem(row, j, PyFloat_FromDouble(*ci++));
		PyList_SetItem(pycost, i, row);
	}
	return pycost;
}


bool readCostMatrix(PyObject *arg, TCostMatrix *&matrix)
{
	int dim;
	const int arglength = PyObject_Length(arg);
	if (matrix) {
		dim = matrix->dimension;
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
			matrix->cost(i, j) = f;
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


PyObject *CostMatrix_new(PyTypeObject *type, PyObject *args) BASED_ON(Orange - Orange.classification.CostMatrix, "(list-of-list-of-prices) -> CostMatrix")
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


PyObject *CostMatrix__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TCostMatrix, matrix);
	const int dim = matrix->dimension;
	return Py_BuildValue("O(Os#i)N", getExportedFunction("__pickleLoaderCostMatrix"),
		self->ob_type,
		matrix->costs, dim*dim*sizeof(float),
		dim,
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderCostMatrix(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_matrix, dimension)")
{
	PyTRY
		PyTypeObject *type;
	char *buf;
	int bufSize, dim;
	if (!PyArg_ParseTuple(args, "Os#i:__pickleLoaderCostMatrix", &type, &buf, &bufSize, &dim))
		return NULL;

	TCostMatrix *cm = new TCostMatrix(dim);
	memcpy(cm->costs, buf, bufSize);
	return WrapNewOrange(cm, type);
	PyCATCH
}


PyObject *CostMatrix_native(PyObject *self) PYARGS(METH_O, "() -> list of lists of floats")
{ return convertToPython(PyOrange_AsCostMatrix(self)); }


int getCostIndex(PyObject *arg, TCostMatrix *matrix, char *error)
{
	if (PyInt_Check(arg)) {
		int pred = PyInt_AsLong(arg);
		if ((pred<0) || (pred >= matrix->dimension))
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

	return PyFloat_FromDouble(matrix->cost(pred, corr));
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

	matrix->cost(pred, corr) = cost;
	RETURN_NONE;
	PyCATCH
}


/* ************ BASSTAT ************ */

#include "basstat.hpp"

PyObject *BasicAttrStat_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange - Orange.statistics.basic.Variable, "(variable, [examples, weightID, min=, max=, avg=, dev=, n=]) -> BasicAttrStat") ALLOWS_EMPTY
{
  PyTRY
    PyObject *pyvar = NULL;
    PExampleGenerator egen;
    int weightID = 0;
    if (!PyArg_ParseTuple(args, "|OO&i:BasicAttrStat.__new__", &pyvar, pt_ExampleGenerator, &egen, &weightID))
      return NULL;

    if (!pyvar)
      return WrapNewOrange(mlnew TBasicAttrStat(PVariable()), type);

    if (!egen) {
	    if (!PyOrVariable_Check(pyvar)) {
		    PyErr_Format(PyExc_TypeError, "BasicAttrStat expects a 'Variable', not a '%s'", pyvar->ob_type->tp_name);
		    return NULL;
	    }

	    return WrapNewOrange(mlnew TBasicAttrStat(PyOrange_AsVariable(pyvar)), type);
    }

    PVariable var = varFromArg_byDomain(pyvar, egen->domain, false);
    if (!var)
      return NULL;

    return WrapNewOrange(mlnew TBasicAttrStat(egen, var, weightID), type);
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


PyObject *BasicAttrStat_recompute(PyObject *self) PYARGS(METH_NOARGS, "() -> None")
{ PyTRY
SELF_AS(TBasicAttrStat).recompute();
RETURN_NONE;
PyCATCH
}


PyObject *BasicAttrStat_reset(PyObject *self) PYARGS(METH_NOARGS, "() -> None")
{ PyTRY
SELF_AS(TBasicAttrStat).reset();
RETURN_NONE;
PyCATCH
}


/* We redefine new (removed from below!) and add mapping methods
*/

PDomainBasicAttrStat PDomainBasicAttrStat_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::P_FromArguments(arg); }
PyObject *DomainBasicAttrStat_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_FromArguments(type, arg); }
PyObject *DomainBasicAttrStat_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_getitem(self, index); }
int       DomainBasicAttrStat_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_setitem(self, index, item); }
PyObject *DomainBasicAttrStat_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_getslice(self, start, stop); }
int       DomainBasicAttrStat_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       DomainBasicAttrStat_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_len(self); }
PyObject *DomainBasicAttrStat_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_richcmp(self, object, op); }
PyObject *DomainBasicAttrStat_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_concat(self, obj); }
PyObject *DomainBasicAttrStat_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_repeat(self, times); }
PyObject *DomainBasicAttrStat_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_str(self); }
PyObject *DomainBasicAttrStat_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_str(self); }
int       DomainBasicAttrStat_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_contains(self, obj); }
PyObject *DomainBasicAttrStat_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(BasicAttrStat) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_append(self, item); }
PyObject *DomainBasicAttrStat_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_extend(self, obj); }
PyObject *DomainBasicAttrStat_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> int") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_count(self, obj); }
PyObject *DomainBasicAttrStat_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainBasicAttrStat") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_filter(self, args); }
PyObject *DomainBasicAttrStat_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> int") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_index(self, obj); }
PyObject *DomainBasicAttrStat_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_insert(self, args); }
PyObject *DomainBasicAttrStat_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_native(self); }
PyObject *DomainBasicAttrStat_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> BasicAttrStat") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_pop(self, args); }
PyObject *DomainBasicAttrStat_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(BasicAttrStat) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_remove(self, obj); }
PyObject *DomainBasicAttrStat_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_reverse(self); }
PyObject *DomainBasicAttrStat_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_sort(self, args); }
PyObject *DomainBasicAttrStat__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_reduce(self); }


/* Note that this is not like callable-constructors. They return different type when given
parameters, while this one returns the same type, disregarding whether it was given examples or not.
*/
PyObject *DomainBasicAttrStat_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange - Orange.statistics.basic.Domain, "(examples | <list of BasicAttrStat>) -> DomainBasicAttrStat") ALLOWS_EMPTY
{ PyTRY
if (!args || !PyTuple_Size(args))
return WrapNewOrange(mlnew TDomainBasicAttrStat(), type);

int weightID;
PExampleGenerator gen = exampleGenFromArgs(args, weightID);
if (gen)
return WrapNewOrange(mlnew TDomainBasicAttrStat(gen, weightID), type);

PyErr_Clear();

PyObject *obj = ListOfWrappedMethods<PDomainBasicAttrStat, TDomainBasicAttrStat, PBasicAttrStat, &PyOrBasicAttrStat_Type>::_new(type, args, keywds);
if (obj)
return obj;

PyErr_Clear();
PYERROR(PyExc_TypeError, "DomainBasicAttrStat.__init__ expects examples or a list of BasicAttrStat", PYNULL);
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
		if (*ci && (*ci)->variable && ((*ci)->variable->get_name()==s))
			return ci - bas->begin();

	PyErr_Format(PyExc_IndexError, "attribute '%s' not found", s);
	return -1;
}

if (PyOrVariable_Check(args)) {
	PVariable var = PyOrange_AsVariable(args);
	PITERATE(TDomainBasicAttrStat, ci, bas)
		if (*ci && (*ci)->variable && ((*ci)->variable==var))
			return ci - bas->begin();

	PyErr_Format(PyExc_IndexError, "attribute '%s' not found", var->get_name().length() ? var->get_name().c_str() : "<no name>");
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



PyObject *PearsonCorrelation_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "(var1, var2, examples[, weightID]) -> PearsonCorrelation")
{
  PyTRY
    PyObject *pyvar1, *pyvar2;
    PExampleGenerator egen;
    int weightID = 0;
    if (!PyArg_ParseTuple(args, "OOO&|i:BasicAttrStat.__new__", &pyvar1, &pyvar2, pt_ExampleGenerator, &egen, &weightID))
      return NULL;

    PVariable var1 = varFromArg_byDomain(pyvar1, egen->domain, false);
    if (!var1)
      return NULL;

    PVariable var2 = varFromArg_byDomain(pyvar2, egen->domain, false);
    if (!var2)
      return NULL;

    return WrapNewOrange(mlnew TPearsonCorrelation(egen, var1, var2, weightID), type);
  PyCATCH
}



/* ************ CONTINGENCY ************ */

#include "contingency.hpp"
#include "estimateprob.hpp"

ABSTRACT(ContingencyClass - Orange.statistics.contingency.Class, Contingency)

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
	float ind;
	if (!PyNumber_ToFloat(index, ind)) {
		TValue val;
		if (convertFromPython(index, val, cont->outerVariable) && !val.isSpecial())
			ind = float(val);
		else
			PYERROR(PyExc_IndexError, "invalid index type (float expected)", NULL);
	}

	TDistributionMap::iterator mi=cont->continuous->find(ind);
	if (mi!=cont->continuous->end())
		return &(*mi).second;

	PyErr_Format(PyExc_IndexError, "invalid index (%5.3f)", ind);
	return NULL;
}

PYERROR(PyExc_IndexError, "invalid index", (PDistribution *)NULL);
}


PyObject *Contingency_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange - Orange.statistics.contingency.Table, "(outer_desc, inner_desc)")
{ PyTRY
PVariable var1, var2;
if (!PyArg_ParseTuple(args, "O&O&:Contingency.__new__", cc_Variable, &var1, cc_Variable, &var2))
return PYNULL;

return WrapNewOrange(mlnew TContingency(var1, var2), type);
PyCATCH
}


PyObject *ContingencyReduceCommon(PyObject *self, const char *loaderFunc)
{
	PyTRY
		CAST_TO(TContingency, cont);

	if (cont->varType == TValue::INTVAR) {
		PyObject *dvect = PyList_New(cont->discrete->size());
		int i = 0;
		PITERATE(TDistributionVector, di, cont->discrete)
			PyList_SetItem(dvect, i++, WrapOrange(*di));

		return Py_BuildValue("O(ON)N", getExportedFunction(loaderFunc),
			(PyObject *)(self->ob_type),
			dvect,
			packOrangeDictionary(self));
	}

	else if (cont->varType == TValue::FLOATVAR) {
		PyObject *dvect = PyList_New(cont->continuous->size());
		TCharBuffer buf(1024);
		int i = 0;
		PITERATE(TDistributionMap, di, cont->continuous) {
			buf.writeFloat((*di).first);
			PyList_SetItem(dvect, i++, WrapOrange((*di).second));
		}

		return Py_BuildValue("O(ONs#)N", getExportedFunction(loaderFunc),
			(PyObject *)(self->ob_type),
			dvect,
			buf.buf, buf.length(),
			packOrangeDictionary(self));
	}

	else
		PYERROR(PyExc_SystemError, "an instance of Contingency for this attribute type cannot be pickled", NULL);

	PyCATCH
}


PyObject *__pickleLoaderContingencyCommon(TContingency *cont, PyObject *args)
{
	PyTRY
		PyTypeObject *type;
	PyObject *dvect, *packedF = NULL;
	if (!PyArg_UnpackTuple(args, "__pickleLoaderContingency", 2, 3, &type, &dvect, &packedF)) {
		delete cont;
		return NULL;
	}

	if (packedF) {
		char *pbuf;
		Py_ssize_t bufSize;
		if (PyString_AsStringAndSize(packedF, &pbuf, &bufSize) == -1) {
			delete cont;
			return NULL;
		}
		TCharBuffer buf(pbuf);

		cont->continuous = new TDistributionMap();
		TDistributionMap &dmap = *cont->continuous;

		for(Py_ssize_t i = 0, e = PyList_Size(dvect); i < e; i++) {
			PyObject *dist = PyList_GetItem(dvect, i);
			if (!PyOrDistribution_Check(dist)) {
				delete cont;
				PYERROR(PyExc_TypeError, "a list of distributions expected", NULL);
			}

			dmap.insert(dmap.end(), pair<float, PDistribution>(buf.readFloat(), PyOrange_AsDistribution(dist)));
		}

		return WrapNewOrange(cont, type);
	}

	else {
		cont->discrete = new TDistributionVector();
		TDistributionVector &dvec = *cont->discrete;

		for(Py_ssize_t i = 0, e = PyList_Size(dvect); i < e; i++) {
			PyObject *dist = PyList_GetItem(dvect, i);
			if (!PyOrDistribution_Check(dist)) {
				delete cont;
				PYERROR(PyExc_TypeError, "a list of distributions expected", NULL);
			}

			dvec.push_back(PyOrange_AsDistribution(dist));
		}

		return WrapNewOrange(cont, type);
	}

	PyCATCH
}


PyObject *Contingency__reduce__(PyObject *self, const char *loaderFunc)
{
	return ContingencyReduceCommon(self, "__pickleLoaderContingency");
}

PyObject *__pickleLoaderContingency(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(list of PDistribution, [packed_floats])")
{
	return __pickleLoaderContingencyCommon(new TContingency(), args);
}


PyObject *ContingencyAttrClass__reduce__(PyObject *self, const char *loaderFunc)
{
	return ContingencyReduceCommon(self, "__pickleLoaderContingencyAttrClass");
}

PyObject *__pickleLoaderContingencyAttrClass(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(list of PDistribution, [packed_floats])")
{
	return __pickleLoaderContingencyCommon(new TContingencyAttrClass(), args);
}


PyObject *ContingencyClassAttr__reduce__(PyObject *self, const char *loaderFunc)
{
	return ContingencyReduceCommon(self, "__pickleLoaderContingencyClassAttr");
}

PyObject *__pickleLoaderContingencyClassAttr(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(list of PDistribution, [packed_floats])")
{
	return __pickleLoaderContingencyCommon(new TContingencyClassAttr(), args);
}


PyObject *ContingencyAttrAttr__reduce__(PyObject *self, const char *loaderFunc)
{
	return ContingencyReduceCommon(self, "__pickleLoaderContingencyAttrAttr");
}

PyObject *__pickleLoaderContingencyAttrAttr(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(list of PDistribution, [packed_floats])")
{
	return __pickleLoaderContingencyCommon(new TContingencyAttrAttr(), args);
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


PyObject *ContingencyClass_add_var_class(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(attribute_value, class_value[, w=1]) -> None")
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


PyObject *ContingencyAttrClass_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(ContingencyClass - Orange.statistics.contingency.VarClass, "(attribute, class attribute) | (attribute, examples[, weightID])")
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


PyObject *ContingencyClassAttr_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(ContingencyClass - Orange.statistics.contingency.ClassVar, "(attribute, class attribute) | (attribute, examples[, weightID])")
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


PyObject *ContingencyAttrAttr_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Contingency - Orange.statistics.contingency.VarVar, "(outer_attr, inner_attr[, examples [, weight-id]])")
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


PyObject *Contingency_getitem_sq(PyObject *self, Py_ssize_t ind)
{ PyTRY
CAST_TO(TContingency, cont)

if (cont->outerVariable->varType!=TValue::INTVAR)
PYERROR(PyExc_TypeError, "cannot iterate through contingency of continuous attribute", PYNULL);

if ((ind<0) || (ind>=Py_ssize_t(cont->discrete->size())))
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

Py_ssize_t Contingency_len(PyObject *self)
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


string convertToString(const PDistribution &);

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
	Py_ssize_t i=0;
	PITERATE(TDistributionMap, ci, cont->continuous)
		PyList_SetItem(nl, i++, PyFloat_FromDouble((double)(*ci).first));
	return nl;
}
else if (cont->outerVariable->varType==TValue::INTVAR) {
	PyObject *nl=PyList_New(cont->outerVariable->noOfValues());
	Py_ssize_t i=0;
	PStringList vals=cont->outerVariable.AS(TEnumVariable)->values;
	PITERATE(TStringList, ii, vals)
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
	Py_ssize_t i=0;
	PITERATE(TDistributionMap, ci, cont->continuous)
		PyList_SetItem(nl, i++, WrapOrange((*ci).second));
	return nl;
}
else if (cont->outerVariable->varType==TValue::INTVAR) {
	PyObject *nl=PyList_New(cont->discrete->size());
	Py_ssize_t i=0;
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
	Py_ssize_t i=0;
	PITERATE(TDistributionMap, ci, cont->continuous)
		PyList_SetItem(nl, i++,
		Py_BuildValue("fN", (double)(*ci).first, WrapOrange((*ci).second)));
	return nl;
}
else if (cont->outerVariable->varType==TValue::INTVAR) {
	PyObject *nl=PyList_New(cont->outerVariable->noOfValues());
	Py_ssize_t i=0;
	TStringList::const_iterator ii(cont->outerVariable.AS(TEnumVariable)->values->begin());
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

PDomainContingency PDomainContingency_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::P_FromArguments(arg); }
PyObject *DomainContingency_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_FromArguments(type, arg); }
PyObject *DomainContingency_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_getitem(self, index); }
int       DomainContingency_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_setitem(self, index, item); }
PyObject *DomainContingency_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_getslice(self, start, stop); }
int       DomainContingency_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       DomainContingency_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_len(self); }
PyObject *DomainContingency_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_richcmp(self, object, op); }
PyObject *DomainContingency_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_concat(self, obj); }
PyObject *DomainContingency_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_repeat(self, times); }
PyObject *DomainContingency_str(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_str(self); }
PyObject *DomainContingency_repr(TPyOrange *self) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_str(self); }
int       DomainContingency_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_contains(self, obj); }
PyObject *DomainContingency_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Contingency) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_append(self, item); }
PyObject *DomainContingency_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_extend(self, obj); }
PyObject *DomainContingency_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> int") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_count(self, obj); }
PyObject *DomainContingency_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> DomainContingency") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_filter(self, args); }
PyObject *DomainContingency_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> int") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_index(self, obj); }
PyObject *DomainContingency_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_insert(self, args); }
PyObject *DomainContingency_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_native(self); }
PyObject *DomainContingency_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Contingency") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_pop(self, args); }
PyObject *DomainContingency_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Contingency) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_remove(self, obj); }
PyObject *DomainContingency_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_reverse(self); }
PyObject *DomainContingency_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_sort(self, args); }
PyObject *DomainContingency__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PDomainContingency, TDomainContingency, PContingencyClass, &PyOrContingency_Type>::_reduce(self); }


CONSTRUCTOR_KEYWORDS(DomainContingency, "classIsOuter class_is_outer")

PyObject *DomainContingency_new(PyTypeObject *type, PyObject *args, PyObject *keywds) BASED_ON(Orange - Orange.statistics.contingency.Domain, "(examples [, weightID] | <list of Contingency>) -> DomainContingency") ALLOWS_EMPTY
{ PyTRY
if (!args || !PyTuple_Size(args))
return WrapNewOrange(mlnew TDomainContingency(), type);

int weightID;
PExampleGenerator gen = exampleGenFromArgs(args, weightID);
if (gen) {
	bool classOuter = false;
	if (keywds) {
		PyObject *couter = PyDict_GetItemString(keywds, "class_is_outer");
        if (!couter) {
            couter = PyDict_GetItemString(keywds, "classIsOuter");
        }
		if (couter) {
			classOuter = (PyObject_IsTrue(couter) != 0);
			Py_DECREF(couter);
		}
	}

	return WrapNewOrange(mlnew TDomainContingency(gen, weightID, classOuter), type);
}

PyObject *obj = ListOfWrappedMethods<PDomainContingency, TDomainContingency, PDomainContingency, &PyOrContingency_Type>::_new(type, args, keywds);
if (obj)
return obj;

PyErr_Clear();
PYERROR(PyExc_TypeError, "DomainContingency.__init__ expects examples or a list of Contingencies", PYNULL);
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


int ptn_DomainContingency(PyObject *args, void *egen)
{
	if (args == Py_None) {
		*(PDomainContingency *)(egen) = PDomainContingency();
		return 1;
	}
	else if (PyOrDomainContingency_Check(args)) {
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
	PITERATE(TDomainContingency, ci, cont)
		if (couter ? (*ci)->innerVariable && ((*ci)->innerVariable->get_name()==s)
			: (*ci)->outerVariable && ((*ci)->outerVariable->get_name()==s))
			return ci - cont->begin();
  PyErr_Format(PyExc_IndexError, "Domain contingency has no variable '%s'", s);
  return -1;
}

if (PyOrVariable_Check(args)) {
	PVariable var = PyOrange_AsVariable(args);
	PITERATE(TDomainContingency, ci, cont)
		if (couter ? (*ci)->innerVariable && ((*ci)->innerVariable==var)
			: (*ci)->outerVariable && ((*ci)->outerVariable==var))
			return ci - cont->begin();
  PyErr_Format(PyExc_IndexError, "Domain contingency has no variable '%s'", var->get_name().c_str());
  return -1;
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
		res += (*di)->outerVariable->get_name()+": "+convertToString(*di);
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


C_CALL3(ComputeDomainContingency, ComputeDomainContingency, Orange, "([examples, weightID]) -/-> DomainContingency")

PyObject *ComputeDomainContingency_call(PyObject *self, PyObject *args)
{
  int weightID;
  PExampleGenerator gen = exampleGenFromArgs(args, weightID);
  if (!gen)
    PYERROR(PyExc_AttributeError, "examples and, optionally, weight ID expected", PYNULL);

  return WrapOrange(SELF_AS(TComputeDomainContingency).call(gen, weightID));
}


/* ************ DOMAIN TRANSFORMER ************ */

#include "transdomain.hpp"

ABSTRACT(DomainTransformerConstructor, Orange)

PyObject *DomainTransformerConstructor_call(PyObject *self, PyObject *args)
{
  int weightID;
  PExampleGenerator gen = exampleGenFromArgs(args, weightID);
  if (!gen)
    PYERROR(PyExc_AttributeError, "examples and, optionally, weight ID expected", PYNULL);

  return WrapOrange(SELF_AS(TDomainTransformerConstructor).call(gen, weightID));
}



/* ************ DISTANCE ************ */

#include "distance.hpp"
#include "distance_dtw.hpp"

ABSTRACT(ExamplesDistance_Normalized - Orange.distances.ExamplesDistance_Normalized, ExamplesDistance)
C_NAMED(ExamplesDistance_Hamming - Orange.distances.Hamming, ExamplesDistance, "()")
C_NAMED(ExamplesDistance_Maximal - Orange.distances.Maximal, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Manhattan - Orange.distances.Manhattan, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Euclidean - Orange.distances.Euclidean, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Lp - Orange.distances.Lp, ExamplesDistance_Normalized, "()")
C_NAMED(ExamplesDistance_Relief - Orange.distances.Relief, ExamplesDistance, "()")
C_NAMED(ExamplesDistance_DTW - Orange.distances.DTW, ExamplesDistance_Normalized, "()")

C_CALL(ExamplesDistanceConstructor_Hamming - Orange.distances.HammingConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Hamming")
C_CALL(ExamplesDistanceConstructor_Maximal - Orange.distances.MaximalConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Maximal")
C_CALL(ExamplesDistanceConstructor_Manhattan - Orange.distances.ManhattanConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Manhattan")
C_CALL(ExamplesDistanceConstructor_Euclidean - Orange.distances.EuclideanConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Euclidean")
C_CALL(ExamplesDistanceConstructor_Lp - Orange.distances.LpConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Lp")
C_CALL(ExamplesDistanceConstructor_Relief - Orange.distances.ReliefConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_Relief")
C_CALL(ExamplesDistanceConstructor_DTW - Orange.distances.DTWConstructor, ExamplesDistanceConstructor, "([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance_DTW")


PyObject *ExamplesDistanceConstructor_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.distances.ExamplesDistanceConstructor, "<abstract>")
{
  if (type == (PyTypeObject *)&PyOrExamplesDistanceConstructor_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TExamplesDistanceConstructor_Python(), type), args);
  else
    return WrapNewOrange(mlnew TExamplesDistanceConstructor_Python(), type);
}


PyObject *ExamplesDistanceConstructor__reduce__(PyObject *self)
{
	return callbackReduce(self, PyOrExamplesDistanceConstructor_Type);
}


PyObject *ExamplesDistance_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.distances.ExamplesDistance, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrExamplesDistance_Type)
return setCallbackFunction(WrapNewOrange(mlnew TExamplesDistance_Python(), type), args);
else
return WrapNewOrange(mlnew TExamplesDistance_Python(), type);
}


PyObject *ExamplesDistance__reduce__(PyObject *self)
{
	return callbackReduce(self, PyOrExamplesDistance_Type);
}


PyObject *ExamplesDistanceConstructor_call(PyObject *self, PyObject *uargs, PyObject *keywords) PYDOC("([examples, weightID][, DomainDistributions][, DomainBasicAttrStat]) -/-> ExamplesDistance")
{ PyTRY
  if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrExamplesDistanceConstructor_Type) {
	  PyErr_Format(PyExc_SystemError, "ExamplesDistanceConstructor.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
	  return PYNULL;
  }

  NO_KEYWORDS

  PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
  PExampleGenerator gen;
  int weightID = 0;
  PDomainDistributions dist;
  PDomainBasicAttrStat bstat;
  if (!PyArg_UnpackTuple(uargs, "ExamplesDistanceConstructor.call", 0, 4, args+0, args+1, args+2, args+3))
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
for(int i = 0, e = difs.size(); i<e; i++)
PyList_SetItem(l, i, PyFloat_FromDouble(difs[i]));

return l;
PyCATCH
}


PyObject *ExamplesDistance_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example1, example2) -> float")
{
	PyTRY
		NO_KEYWORDS

		if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrExamplesDistance_Type) {
			PyErr_Format(PyExc_SystemError, "ExamplesDistance.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
			return PYNULL;
		}

		TExample *ex1, *ex2;
		if (!PyArg_ParseTuple(args, "O&O&:ExamplesDistance_Normalized.__call__", ptr_Example, &ex1, ptr_Example, &ex2))
			PYERROR(PyExc_TypeError, "attribute error (two examples expected)", PYNULL);

		return PyFloat_FromDouble((double)(SELF_AS(TExamplesDistance)(*ex1, *ex2)));
		PyCATCH
}



bool convertFromPython(PyObject *pyobj, TAlignment &align)
{
	return PyArg_ParseTuple(pyobj, "ii:convertFromPython(Alignment)", &align.i, &align.j) != 0;
}


PyObject *convertToPython(const TAlignment &align)
{
	return Py_BuildValue("ii", align.i, align.j);
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

ABSTRACT(FindNearest - Orange.core.FindNearest, Orange)
C_NAMED(FindNearest_BruteForce - Orange.classification.knn.FindNearest, FindNearest, "([distance=, distanceID=, includeSame=])")

ABSTRACT(FindNearestConstructor - Orange.core.FindNearestConstructor, Orange)
C_CALL(FindNearestConstructor_BruteForce - Orange.classification.knn.FindNearestConstructor, FindNearestConstructor, "([examples[, weightID[, distanceID]], distanceConstructor=, includeSame=]) -/-> FindNearest")


PyObject *FindNearestConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID[, distanceID]]) -> FindNearest")
{
	PyTRY
		NO_KEYWORDS

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
		NO_KEYWORDS

		float k;
	TExample *example;
	int needsClass = 0;
	// Both forms are allowed for compatibility with older versions
	if (!PyArg_ParseTuple(args, "fO&|i", &k, ptr_Example, &example, &needsClass)) {
		PyErr_Clear();
		if (!PyArg_ParseTuple(args, "O&f|i", ptr_Example, &example, &k, &needsClass))
			PYERROR(PyExc_TypeError, "attribute error (number and example, and an optional flag for class expected)", PYNULL);
	}

	return WrapOrange(SELF_AS(TFindNearest).call(*example, k, needsClass != 0));
	PyCATCH
}




/* ************ FILTERS ************ */

#include "filter.hpp"

ABSTRACT(ValueFilter, Orange)
C_NAMED(ValueFilter_discrete, ValueFilter, "([position=, oper=, values=, acceptSpecial=])")
C_NAMED(ValueFilter_continuous, ValueFilter, "([position=, oper=, min=, max=, acceptSpecial=])")
C_NAMED(ValueFilter_string, ValueFilter, "([position=, oper=, min=, max=])");
C_NAMED(ValueFilter_stringList, ValueFilter, "([position=, oper=, values=])");

C_CALL(Filter_random, Filter, "([examples], [negate=..., p=...]) -/-> ExampleTable")
C_CALL(Filter_hasSpecial, Filter, "([examples], [negate=..., domain=...]) -/-> ExampleTable")
C_CALL(Filter_isDefined, Filter, "([examples], [negate=..., domain=..., check=]) -/-> ExampleTable")
C_CALL(Filter_hasMeta, Filter, "([examples], [id=...]) -/-> ExampleTable")
C_CALL(Filter_hasClassValue, Filter, "([examples], [negate=..., domain=...]) -/-> ExampleTable")
C_CALL(Filter_sameValue, Filter, "([examples], [negate=..., domain=..., position=<int>, value=...]) -/-> ExampleTable")
C_CALL(Filter_values, Filter, "([examples], [negate=..., domain=..., values=<see the manual>) -/-> ExampleTable")


PValueFilterList PValueFilterList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::P_FromArguments(arg); }
PyObject *ValueFilterList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_FromArguments(type, arg); }
PyObject *ValueFilterList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ValueFilter>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_new(type, arg, kwds); }
PyObject *ValueFilterList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_getitem(self, index); }
int       ValueFilterList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_setitem(self, index, item); }
PyObject *ValueFilterList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_getslice(self, start, stop); }
int       ValueFilterList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       ValueFilterList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_len(self); }
PyObject *ValueFilterList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_richcmp(self, object, op); }
PyObject *ValueFilterList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_concat(self, obj); }
PyObject *ValueFilterList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_repeat(self, times); }
PyObject *ValueFilterList_str(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_str(self); }
PyObject *ValueFilterList_repr(TPyOrange *self) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_str(self); }
int       ValueFilterList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_contains(self, obj); }
PyObject *ValueFilterList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ValueFilter) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_append(self, item); }
PyObject *ValueFilterList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_extend(self, obj); }
PyObject *ValueFilterList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> int") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_count(self, obj); }
PyObject *ValueFilterList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ValueFilterList") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_filter(self, args); }
PyObject *ValueFilterList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> int") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_index(self, obj); }
PyObject *ValueFilterList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_insert(self, args); }
PyObject *ValueFilterList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_native(self); }
PyObject *ValueFilterList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ValueFilter") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_pop(self, args); }
PyObject *ValueFilterList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ValueFilter) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_remove(self, obj); }
PyObject *ValueFilterList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_reverse(self); }
PyObject *ValueFilterList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_sort(self, args); }
PyObject *ValueFilterList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PValueFilterList, TValueFilterList, PValueFilter, &PyOrValueFilter_Type>::_reduce(self); }


PyObject *applyFilterP(PFilter filter, PExampleTable gen);

PyObject *applyFilter(PFilter filter, PExampleGenerator gen, bool weightGiven, int weightID)
{ if (!filter) return PYNULL;

TExampleTable *newTable = mlnew TExampleTable(gen->domain);
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


PyObject *Filter__reduce__(PyObject *self)
{
	return callbackReduce(self, PyOrFilter_Type);
}


PyObject *Filter_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrFilter_Type) {
			PyErr_Format(PyExc_SystemError, "Filter.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
			return PYNULL;
		}

		CAST_TO(TFilter, filter);

		bool savedNegate = filter->negate;
		PyObject *res;

		try {
			if (!((TPyOrange *)self)->call_constructed && keywords) {
				const Py_ssize_t sze = PyDict_Size(keywords);
				PyObject *neg = sze >= 1 ? PyDict_GetItemString(keywords, "negate") : NULL;
				if ((sze > 1) || !neg)
					NO_KEYWORDS;
				filter->negate = (PyObject_IsTrue(neg) != 0);
			}

			if ((PyTuple_Size(args)==1) && PyOrExample_Check(PyTuple_GET_ITEM(args, 0))) {
				res = PyInt_FromLong(filter->call(PyExample_AS_ExampleReference(PyTuple_GET_ITEM(args, 0))) ? 1 : 0);
			}
			else {
				PExampleGenerator egen;
				int references = 0;
				if (!PyArg_ParseTuple(args, "O&|i:Filter.__call__", &pt_ExampleGenerator, &egen, &references)) {
					filter->negate = savedNegate;
					return PYNULL;
				}

				if (references) {
					if (!egen.is_derived_from(TExampleTable))
						PYERROR(PyExc_TypeError, "cannot return references to examples that are not in example table", PYNULL);
					res = applyFilterP(filter, egen);
				}
				else
					res = applyFilter(PyOrange_AsFilter(self), egen, false, 0);
			}

			filter->negate = savedNegate;
			return res;
		}
		catch(...) {
			filter->negate = savedNegate;
			throw;
		}
		PyCATCH
}

PyObject *Filter_deepCopy(PyObject *self) PYARGS(METH_NOARGS, "() -> filter")
{
	PyTRY
		CAST_TO(TFilter, filter);

    PFilter res = filter->deepCopy();
	  return WrapOrange(res);
	PyCATCH
}

PyObject *Filter_count(PyObject *self, PyObject *arg) PYARGS(METH_O, "(examples)")
{
	PyTRY
		PExampleGenerator egen = exampleGenFromParsedArgs(arg);
	if (!egen)
		PYERROR(PyExc_TypeError, "Filter.count: examples expected", PYNULL);

	CAST_TO(TFilter, filter);

	filter->reset();
	int count = 0;
	PEITERATE(ei, egen)
		if (filter->operator()(*ei))
			count++;

	return PyInt_FromLong(count);
	PyCATCH
}


PyObject *filterSelectionVectorLow(TFilter &filter, PExampleGenerator egen)
{
	TBoolList *selection = new TBoolList();
	PBoolList pselection = selection;
	const int nex = egen->numberOfExamples();
	if (nex > 0)
		selection->reserve(nex);

	filter.reset();
	PEITERATE(ei, egen)
		selection->push_back(filter(*ei));

	return WrapOrange(pselection);
}


PyObject *Filter_selectionVector(PyObject *self, PyObject *arg) PYARGS(METH_O, "(examples)")
{
	PyTRY
		PExampleGenerator egen = exampleGenFromParsedArgs(arg);
	if (!egen)
		PYERROR(PyExc_TypeError, "Filter.selectionVector: examples expected", PYNULL);

	CAST_TO(TFilter, filter);
	return filterSelectionVectorLow(SELF_AS(TFilter), egen);
	PyCATCH
}


PYXTRACT_IGNORE PyObject *AttributedBoolList_new(PyTypeObject *type, PyObject *args, PyObject *keywds);

int Filter_isDefined_set_check(PyObject *self, PyObject *arg)
{
	PyTRY
		CAST_TO_err(TFilter_isDefined, filter, -1)

		if (arg == Py_None) {
			filter->check = PAttributedBoolList();
			return 0;
		}


		PyObject *boollist = objectOnTheFly(arg, (PyTypeObject *)&PyOrAttributedBoolList_Type);

		//    PyObject *boollist = AttributedBoolList_new((PyTypeObject *)&PyOrAttributedBoolList_Type, arg, PYNULL);
		if (!boollist)
			return -1;

		PAttributedBoolList cli = PyOrange_AsAttributedBoolList(boollist);

		if (filter->domain) {
			if (cli->attributes) {
				TVarList::const_iterator di(filter->domain->variables->begin()), de(filter->domain->variables->end());
				TVarList::const_iterator fci(cli->attributes->begin()), fce(cli->attributes->end());
				for(; (di!=de) && (fci!=fce); di++, fci++)
					if (*di!=*fci) {
						PyErr_Format(PyExc_AttributeError, "attribute %s in the list does not match the filter's domain", (*fci)->get_name().c_str());
						return -1;
					}
					if (fci!=fce)
						PYERROR(PyExc_AttributeError, "the check list has more attributes than the filter's domain", -1);
			}
			else {
				/* we don't want to modify the list if this is a reference
				to a list from somewhere else */
				if (!PyOrAttributedBoolList_Check(arg))
					cli->attributes = filter->domain->variables;
			}
		}

		filter->check = cli;
		return 0;
		PyCATCH_1
}

#include "valuelisttemplate.hpp"


PStringList PStringList_FromArguments(PyObject *arg);

int Filter_values_setitem(PyObject *self, PyObject *pyvar, PyObject *args)
{
	PyTRY
		CAST_TO_err(TFilter_values, filter, -1);

	if (!filter->domain)
		PYERROR(PyExc_IndexError, "Filter_values.__getitem__ cannot work if 'domain' is not set", -1);

	PVariable var = varFromArg_byDomain(pyvar, filter->domain);
	if (!var)
		return -1;


	if (!args || (args == Py_None)) {
		filter->removeCondition(var);
		return 0;
	}


	if (var->varType == TValue::INTVAR) {
		if (PyList_Check(args)) {
			PValueList vlist = TValueListMethods::P_FromArguments(args, var);
			if (!vlist)
				return -1;
			filter->addCondition(var, vlist);
		}
		else if (PyTuple_Check(args)) {
			int oper;
			PyObject *obj;
			if (!PyArg_ParseTuple(args, "iO:Filter_values.__setitem__", &oper, &obj))
				return -1;
			if ((oper != TValueFilter::Equal) && (oper != TValueFilter::NotEqual))
				PYERROR(PyExc_AttributeError, "Filter_values.__setitem__: operations for discrete attributes can be only Equal or NotEqual", -1);
			PValueList vlist = TValueListMethods::P_FromArguments(obj, var);
			if (!vlist)
				return -1;
			filter->addCondition(var, vlist, oper == TValueFilter::NotEqual);
		}
		else {
			TValue val;
			if (!convertFromPython(args, val, var))
				return -1;

			filter->addCondition(var, val);
		}
	}

	else if (var->varType == TValue::FLOATVAR) {
		if (PyTuple_Check(args)) {
			int oper;
			float minv, maxv;
			if (!PyArg_ParseTuple(args, "if|f:Filter_values.__setitem__", &oper, &minv, &maxv))
				return -1;
			if ((PyTuple_Size(args) == 3) && (oper != TValueFilter::Between) && (oper != TValueFilter::Outside))
				PYERROR(PyExc_TypeError, "Filter_values.__setitem__: only one reference value expected for the given operator", -1);

			filter->addCondition(var, oper, minv, maxv);
		}
		else {
			float f;
			if (!PyNumber_ToFloat(args, f)) {
				PyErr_Format(PyExc_TypeError, "Filter_values.__setitem__: invalid condition for attribute '%s'", var->get_name().c_str());
				return -1;
			}
			filter->addCondition(var, TValueFilter::Equal, f, f);
		}
	}

	else if (var->varType == STRINGVAR) {
		if (PyString_Check(args))
			filter->addCondition(var, TValueFilter::Equal, PyString_AsString(args), string());

		else if (PyList_Check(args)) {
			PStringList slist = PStringList_FromArguments(args);
			if (!slist)
				return -1;
			filter->addCondition(var, slist);
		}

		else if (PyTuple_Check(args) && PyTuple_Size(args)) {
			char *mins, *maxs = NULL;
			int oper;
			if (!PyArg_ParseTuple(args, "is|s:Filter_values.__setitem__", &oper, &mins, &maxs))
				return -1;
			if ((PyTuple_Size(args) == 3) && (oper != TValueFilter::Between) && (oper != TValueFilter::Outside))
				PYERROR(PyExc_TypeError, "Filter_values.__setitem__: only one reference value expected for the given operator", -1);

			filter->addCondition(var, oper, mins, maxs ? maxs : string());
		}

		else {
			PyErr_Format(PyExc_TypeError, "Filter_values.__setitem__: invalid condition for attribute '%s'", var->get_name().c_str());
			return -1;
		}
	}

	else
		PYERROR(PyExc_TypeError, "Filter_values.__setitem__: unsupported attribute type", -1);

	return 0;
	PyCATCH_1
}


PyObject *Filter_values_getitem(PyObject *self, PyObject *args)
{
	PyTRY
		CAST_TO(TFilter_values, filter);

	PVariable var = varFromArg_byDomain(args, filter->domain);
	if (!var)
		return PYNULL;

	int position;
	TValueFilterList::iterator condi = filter->findCondition(var, 0, position);
	if (condi == filter->conditions->end()) {
		PyErr_Format(PyExc_IndexError, "no condition on '%s'", var->get_name().c_str());
		return PYNULL;
	}

	return WrapOrange(*condi);
	PyCATCH
}


PFilterList PFilterList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::P_FromArguments(arg); }
PyObject *FilterList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_FromArguments(type, arg); }
PyObject *FilterList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Filter>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_new(type, arg, kwds); }
PyObject *FilterList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_getitem(self, index); }
int       FilterList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_setitem(self, index, item); }
PyObject *FilterList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_getslice(self, start, stop); }
int       FilterList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       FilterList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_len(self); }
PyObject *FilterList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_richcmp(self, object, op); }
PyObject *FilterList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_concat(self, obj); }
PyObject *FilterList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_repeat(self, times); }
PyObject *FilterList_str(TPyOrange *self) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_str(self); }
PyObject *FilterList_repr(TPyOrange *self) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_str(self); }
int       FilterList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_contains(self, obj); }
PyObject *FilterList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Filter) -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_append(self, item); }
PyObject *FilterList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_extend(self, obj); }
PyObject *FilterList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Filter) -> int") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_count(self, obj); }
PyObject *FilterList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FilterList") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_filter(self, args); }
PyObject *FilterList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Filter) -> int") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_index(self, obj); }
PyObject *FilterList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_insert(self, args); }
PyObject *FilterList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_native(self); }
PyObject *FilterList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Filter") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_pop(self, args); }
PyObject *FilterList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Filter) -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_remove(self, obj); }
PyObject *FilterList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_reverse(self); }
PyObject *FilterList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_sort(self, args); }
PyObject *FilterList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PFilterList, TFilterList, PFilter, &PyOrFilter_Type>::_reduce(self); }


PyObject *Filter_conjunction_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Filter, "([filter-list])") ALLOWS_EMPTY
{
	if (!PyTuple_Size(args))
		return WrapNewOrange(mlnew TFilter_conjunction(), type);

	PFilterList flist = PFilterList_FromArguments(PyTuple_Size(args)>1 ? args : PyTuple_GET_ITEM(args, 0));
	if (!flist)
		return PYNULL;

	return WrapNewOrange(mlnew TFilter_conjunction(flist), type);
}


PyObject *Filter_disjunction_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Filter, "([filter-list])") ALLOWS_EMPTY
{
	if (!PyTuple_Size(args))
		return WrapNewOrange(mlnew TFilter_disjunction(), type);

	PFilterList flist = PFilterList_FromArguments(PyTuple_Size(args)>1 ? args : PyTuple_GET_ITEM(args, 0));
	if (!flist)
		return PYNULL;

	return WrapNewOrange(mlnew TFilter_disjunction(flist), type);
}

/* ************ IMPUTATION ******************** */

#include "imputation.hpp"

C_NAMED(TransformValue_IsDefined, TransformValue, "([value=])")

ABSTRACT(Imputer - Orange.feature.imputation.Imputer, Orange)
C_NAMED(Imputer_asValue - Orange.feature.imputation.Imputer_asValue, Imputer, "() -> Imputer_asValue")
C_NAMED(Imputer_model - Orange.feature.imputation.Imputer_model, Imputer, "() -> Imputer_model")
C_NAMED(Imputer_random - Orange.feature.imputation.Imputer_random, Imputer, "() -> Imputer_random")

PyObject *Imputer_defaults_new(PyTypeObject *tpe, PyObject *args) BASED_ON(Imputer - Orange.feature.imputation.Imputer_defaults, "(domain | example) -> Imputer_defaults")
{
	PyTRY
		if (PyTuple_Size(args) == 1) {
			PyObject *arg = PyTuple_GET_ITEM(args, 0);
			if (PyOrDomain_Check(arg))
				return WrapNewOrange(mlnew TImputer_defaults(PyOrange_AsDomain(arg)), tpe);
			if (PyOrExample_Check(arg))
				return WrapNewOrange(mlnew TImputer_defaults(PyExample_AS_Example(arg)), tpe);
		}

		PYERROR(PyExc_TypeError, "Imputer_defaults.__init__ expects an example or domain", PYNULL);
		PyCATCH
}

PyObject *Imputer_defaults__reduce__(PyObject *self)
{
	PyTRY
		return Py_BuildValue("O(N)N", self->ob_type,
		Example_FromWrappedExample(SELF_AS(TImputer_defaults).defaults),
		packOrangeDictionary(self));
	PyCATCH
}


ABSTRACT(ImputerConstructor - Orange.feature.imputation.ImputerConstructor, Orange)
C_CALL(ImputerConstructor_average - Orange.feature.imputation.ImputerConstructor_average, ImputerConstructor, "(examples[, weightID]) -> Imputer")
C_CALL(ImputerConstructor_minimal - Orange.feature.imputation.ImputerConstructor_minimal, ImputerConstructor, "(examples[, weightID]) -> Imputer")
C_CALL(ImputerConstructor_maximal - Orange.feature.imputation.ImputerConstructor_maximal, ImputerConstructor, "(examples[, weightID]) -> Imputer")
C_CALL(ImputerConstructor_model - Orange.feature.imputation.ImputerConstructor_model, ImputerConstructor, "(examples[, weightID]) -> Imputer")
C_CALL(ImputerConstructor_asValue - Orange.feature.imputation.ImputerConstructor_asValue, ImputerConstructor, "(examples[, weightID]) -> Imputer")
C_CALL(ImputerConstructor_random - Orange.feature.imputation.ImputerConstructor_random, ImputerConstructor, "(examples[, weightID]) -> Imputer")

PyObject *Imputer_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		NO_KEYWORDS

		if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrImputer_Type) {
			PyErr_Format(PyExc_SystemError, "Imputer.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
			return PYNULL;
		}

		if ((PyTuple_GET_SIZE(args) == 1) && PyOrExample_Check(PyTuple_GET_ITEM(args, 0))) {
			TExample example = PyExample_AS_ExampleReference(PyTuple_GET_ITEM(args, 0));
			return Example_FromWrappedExample(PExample(PyOrange_AsImputer(self)->call(example)));
		}

		int weightID = 0;
		PExampleGenerator gen = exampleGenFromArgs(args, weightID);
		if (gen)
			return WrapOrange(SELF_AS(TImputer)(gen, weightID));

		PYERROR(PyExc_TypeError, "example or examples expected", PYNULL);
		PyCATCH
}


PyObject *ImputerConstructor_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		NO_KEYWORDS

		int weightID = 0;
	PExampleGenerator gen = exampleGenFromArgs(args, weightID);
	if (!gen)
		return PYNULL;

	return WrapOrange(SELF_AS(TImputerConstructor)(gen, weightID));
	PyCATCH
}


/* ************ RANDOM INDICES ******************** */
#include "trindex.hpp"

ABSTRACT(MakeRandomIndices - Orange.data.sample.MakeRandomIndices, Orange)
C_CALL3(MakeRandomIndices2 - Orange.data.sample.MakeRandomIndices2, MakeRandomIndices2, MakeRandomIndices, "[n | gen [, p0]], [p0=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesMultiple - Orange.data.sample.MakeRandomIndicesMultiple, MakeRandomIndicesMultiple, MakeRandomIndices, "[n | gen [, p]], [p=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesN - Orange.data.sample.MakeRandomIndicesN, MakeRandomIndicesN, MakeRandomIndices, "[n | gen [, p]], [p=, stratified=, randseed=] -/-> [int]")
C_CALL3(MakeRandomIndicesCV - Orange.data.sample.MakeRandomIndicesCV, MakeRandomIndicesCV, MakeRandomIndices, "[n | gen [, folds]], [folds=, stratified=, randseed=] -/-> [int]")


PyObject *MakeRandomIndices2_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		CAST_TO(TMakeRandomIndices2, mri2);

	float savedP0 = mri2->p0;

	try {
		if (!((TPyOrange *)self)->call_constructed && keywords) {
			const Py_ssize_t sze = PyDict_Size(keywords);
			PyObject *neg = sze == 1 ? PyDict_GetItemString(keywords, "p0") : NULL;
			if ((sze > 1) || !neg)
				NO_KEYWORDS;
			if (Orange_setattr1((TPyOrange *)self, "p0", neg) == -1) {
				mri2->p0 = savedP0;
				return PYNULL;
			}
		}

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

		mri2->p0 = savedP0;
		PyErr_Clear();
		PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
		mri2->p0 = savedP0;
		if (!res)
			PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

		return WrapOrange(res);
	}
	catch (...) {
		mri2->p0 = savedP0;
		throw;
	}
	PyCATCH
}


PyObject *MakeRandomIndicesMultiple_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		CAST_TO(TMakeRandomIndicesMultiple, mrim)

		float savedP0 = mrim->p0;

	try {
		if (!((TPyOrange *)self)->call_constructed && keywords) {
			const Py_ssize_t sze = PyDict_Size(keywords);
			PyObject *neg = sze == 1 ? PyDict_GetItemString(keywords, "p0") : NULL;
			if ((sze > 1) || !neg)
				NO_KEYWORDS;
			if (Orange_setattr1((TPyOrange *)self, "p0", neg) == -1) {
				mrim->p0 = savedP0;
				return PYNULL;
			}
		}

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

		mrim->p0 = savedP0;
		PyErr_Clear();
		PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
		mrim->p0 = savedP0;

		if (!res)
			PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

		return WrapOrange(res);
	}

	catch(...) {
		mrim->p0 = savedP0;
		throw;
	}
	PyCATCH
}




PyObject *MakeRandomIndicesN_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		CAST_TO(TMakeRandomIndicesN, mriN)

		PFloatList savedP = mriN->p;

	try {
		if (!((TPyOrange *)self)->call_constructed && keywords) {
			const Py_ssize_t sze = PyDict_Size(keywords);
			PyObject *neg = sze == 1 ? PyDict_GetItemString(keywords, "p") : NULL;
			if ((sze > 1) || !neg)
				NO_KEYWORDS;
			if (Orange_setattr1((TPyOrange *)self, "p", neg) == -1) {
				mriN->p = savedP;
				return PYNULL;
			}
		}

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

		mriN->p = savedP;
		PyErr_Clear();
		PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
		mriN->p = savedP;

		if (!res)
			PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

		return WrapOrange(res);
	}

	catch(...) {
		mriN->p = savedP;
		throw;
	}
	PyCATCH
}



PyObject *MakeRandomIndicesCV_call(PyObject *self, PyObject *args, PyObject *keywords)
{
	PyTRY
		CAST_TO(TMakeRandomIndicesCV, mriCV)

		int savedFolds = mriCV->folds;

	try {
		if (!((TPyOrange *)self)->call_constructed && keywords) {
			const Py_ssize_t sze = PyDict_Size(keywords);
			PyObject *neg = sze == 1 ? PyDict_GetItemString(keywords, "folds") : NULL;
			if ((sze > 1) || !neg)
				NO_KEYWORDS;
			if (Orange_setattr1((TPyOrange *)self, "folds", neg) == -1) {
				mriCV->folds = savedFolds;
				return PYNULL;
			}
		}

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

		mriCV->folds = savedFolds;
		PyErr_Clear();
		PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

out:
		mriCV->folds = savedFolds;
		if (!res)
			PYERROR(PyExc_TypeError, "cannot construct RandomIndices", PYNULL);

		return WrapOrange(res);
	}
	catch(...) {
		mriCV->folds = savedFolds;
		throw;
	}
	PyCATCH
}


/* ************ PROBABILITY ESTIMATION ************ */

#include "estimateprob.hpp"

ABSTRACT(ProbabilityEstimator, Orange)
ABSTRACT(ProbabilityEstimatorConstructor, Orange)
C_NAMED(ProbabilityEstimator_FromDistribution, ProbabilityEstimator, "()")
C_CALL(ProbabilityEstimatorConstructor_relative, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_Laplace, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_m, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromDistribution")
C_CALL(ProbabilityEstimatorConstructor_kernel, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurve")
C_CALL(ProbabilityEstimatorConstructor_loess, ProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurve")

ABSTRACT(ConditionalProbabilityEstimator, Orange)
ABSTRACT(ConditionalProbabilityEstimatorConstructor, Orange)
C_NAMED(ConditionalProbabilityEstimator_FromDistribution, ConditionalProbabilityEstimator, "()")
C_NAMED(ConditionalProbabilityEstimator_ByRows, ConditionalProbabilityEstimator, "()")
C_CALL(ConditionalProbabilityEstimatorConstructor_ByRows, ConditionalProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ConditionalProbabilityEstimator_[FromDistribution|ByRows]")
C_CALL(ConditionalProbabilityEstimatorConstructor_loess, ConditionalProbabilityEstimatorConstructor, "([example generator, weight] | [distribution]) -/-> ProbabilityEstimator_FromCurves")


extern PyTypeObject PyOrProbabilityEstimator_Type_inh;

PProbabilityEstimatorList PProbabilityEstimatorList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::P_FromArguments(arg); }
PyObject *ProbabilityEstimatorList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_FromArguments(type, arg); }
PyObject *ProbabilityEstimatorList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ProbabilityEstimator>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_new(type, arg, kwds); }
PyObject *ProbabilityEstimatorList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_getitem(self, index); }
int       ProbabilityEstimatorList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_setitem(self, index, item); }
PyObject *ProbabilityEstimatorList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_getslice(self, start, stop); }
int       ProbabilityEstimatorList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       ProbabilityEstimatorList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_len(self); }
PyObject *ProbabilityEstimatorList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_richcmp(self, object, op); }
PyObject *ProbabilityEstimatorList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_concat(self, obj); }
PyObject *ProbabilityEstimatorList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_repeat(self, times); }
PyObject *ProbabilityEstimatorList_str(TPyOrange *self) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_str(self); }
PyObject *ProbabilityEstimatorList_repr(TPyOrange *self) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_str(self); }
int       ProbabilityEstimatorList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_contains(self, obj); }
PyObject *ProbabilityEstimatorList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ProbabilityEstimator) -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_append(self, item); }
PyObject *ProbabilityEstimatorList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_extend(self, obj); }
PyObject *ProbabilityEstimatorList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ProbabilityEstimator) -> int") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_count(self, obj); }
PyObject *ProbabilityEstimatorList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ProbabilityEstimatorList") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_filter(self, args); }
PyObject *ProbabilityEstimatorList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ProbabilityEstimator) -> int") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_index(self, obj); }
PyObject *ProbabilityEstimatorList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_insert(self, args); }
PyObject *ProbabilityEstimatorList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_native(self); }
PyObject *ProbabilityEstimatorList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ProbabilityEstimator") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_pop(self, args); }
PyObject *ProbabilityEstimatorList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ProbabilityEstimator) -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_remove(self, obj); }
PyObject *ProbabilityEstimatorList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_reverse(self); }
PyObject *ProbabilityEstimatorList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_sort(self, args); }
PyObject *ProbabilityEstimatorList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PProbabilityEstimatorList, TProbabilityEstimatorList, PProbabilityEstimator, &PyOrProbabilityEstimator_Type>::_reduce(self); }



PConditionalProbabilityEstimatorList PConditionalProbabilityEstimatorList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::P_FromArguments(arg); }
PyObject *ConditionalProbabilityEstimatorList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_FromArguments(type, arg); }
PyObject *ConditionalProbabilityEstimatorList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of ConditionalProbabilityEstimator>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_new(type, arg, kwds); }
PyObject *ConditionalProbabilityEstimatorList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_getitem(self, index); }
int       ConditionalProbabilityEstimatorList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_setitem(self, index, item); }
PyObject *ConditionalProbabilityEstimatorList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_getslice(self, start, stop); }
int       ConditionalProbabilityEstimatorList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       ConditionalProbabilityEstimatorList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_len(self); }
PyObject *ConditionalProbabilityEstimatorList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_richcmp(self, object, op); }
PyObject *ConditionalProbabilityEstimatorList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_concat(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_repeat(self, times); }
PyObject *ConditionalProbabilityEstimatorList_str(TPyOrange *self) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_str(self); }
PyObject *ConditionalProbabilityEstimatorList_repr(TPyOrange *self) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_str(self); }
int       ConditionalProbabilityEstimatorList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_contains(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(ConditionalProbabilityEstimator) -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_append(self, item); }
PyObject *ConditionalProbabilityEstimatorList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_extend(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ConditionalProbabilityEstimator) -> int") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_count(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ConditionalProbabilityEstimatorList") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_filter(self, args); }
PyObject *ConditionalProbabilityEstimatorList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ConditionalProbabilityEstimator) -> int") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_index(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_insert(self, args); }
PyObject *ConditionalProbabilityEstimatorList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_native(self); }
PyObject *ConditionalProbabilityEstimatorList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> ConditionalProbabilityEstimator") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_pop(self, args); }
PyObject *ConditionalProbabilityEstimatorList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(ConditionalProbabilityEstimator) -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_remove(self, obj); }
PyObject *ConditionalProbabilityEstimatorList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_reverse(self); }
PyObject *ConditionalProbabilityEstimatorList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_sort(self, args); }
PyObject *ConditionalProbabilityEstimatorList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PConditionalProbabilityEstimatorList, TConditionalProbabilityEstimatorList, PConditionalProbabilityEstimator, &PyOrConditionalProbabilityEstimator_Type>::_reduce(self); }


PyObject *ProbabilityEstimatorConstructor_call(PyObject *self, PyObject *uargs, PyObject *keywords) PYDOC("([distribution[, apriori]] [example generator[, weight]]) -> ProbabilityEstimator")
{
	PyTRY
		NO_KEYWORDS

		CAST_TO(TProbabilityEstimatorConstructor, cest);

	PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
	PDistribution dist, apriori;
	PExampleGenerator gen;
	int weightID = 0;
	if (!PyArg_UnpackTuple(uargs, "ProbabilityEstimatorConstructor.call", 0, 4, args+0, args+1, args+2, args+3))
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
NO_KEYWORDS

CAST_TO(TProbabilityEstimator, cest);

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
{
	PyTRY
		NO_KEYWORDS

		CAST_TO(TConditionalProbabilityEstimatorConstructor, cest);

	PyObject *args[4] = {PYNULL, PYNULL, PYNULL, PYNULL};
	PContingency cont, apriori;
	PExampleGenerator gen;
	int weightID = 0;
	if (!PyArg_UnpackTuple(uargs, "ProbabilityEstimatorConstructor.call", 0, 4, args, args+1, args+2, args+3))
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
{
	PyTRY
		NO_KEYWORDS

		CAST_TO(TConditionalProbabilityEstimator, cest);

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

#include "stat.hpp"

/* ************ MEASURES ************ */

#include "measures.hpp"
#include "relief.hpp"

BASED_ON(MeasureAttribute - Orange.feature.scoring.Measure, Orange)
ABSTRACT(MeasureAttributeFromProbabilities - Orange.core.MeasureAttributeFromProbabilities, MeasureAttribute)

C_CALL(MeasureAttribute_info - Orange.feature.scoring.InfoGain, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gini - Orange.feature.scoring.Gini, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gainRatio - Orange.feature.scoring.GainRatio, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_gainRatioA - Orange.core.MeasureAttribute_gainRatioA, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori) -/-> float")
C_CALL(MeasureAttribute_cost - Orange.feature.scoring.Cost, MeasureAttributeFromProbabilities, "(cost=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")
C_CALL(MeasureAttribute_relevance - Orange.feature.scoring.Relevance, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")
C_CALL(MeasureAttribute_logOddsRatio - Orange.core.MeasureAttribute_logOddsRatio, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")
C_CALL(MeasureAttribute_chiSquare - Orange.feature.scoring.MeasureAttribute_chiSquare, MeasureAttributeFromProbabilities, "(estimate=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")

C_CALL(MeasureAttribute_MSE - Orange.feature.scoring.MSE, MeasureAttribute, "(estimate=, m=) | (attr, examples[, apriori] [,weightID]) | (attrno, domain-cont[, apriori]) | (cont, class dist [,apriori]) -/-> float")

C_CALL(MeasureAttribute_relief - Orange.feature.scoring.Relief, MeasureAttribute, "(estimate=, m=, k=) | (attr, examples[, apriori] [,weightID]) -/-> float")

/* obsolete: */
PYCONSTANT(MeasureAttribute_splitGain, (PyObject *)&PyOrMeasureAttribute_gainRatio_Type)
PYCONSTANT(MeasureAttribute_retis, (PyObject *)&PyOrMeasureAttribute_MSE_Type)


PYCLASSCONSTANT_FLOAT(MeasureAttribute, Rejected, ATTRIBUTE_REJECTED)

PyObject *MeasureAttribute_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.feature.scoring.Measure, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrMeasureAttribute_Type)
return setCallbackFunction(WrapNewOrange(mlnew TMeasureAttribute_Python(), type), args);
else
return WrapNewOrange(mlnew TMeasureAttribute_Python(), type);
}


PyObject *MeasureAttribute__reduce__(PyObject *self)
{
	return callbackReduce(self, PyOrMeasureAttribute_Type);
}


PyObject *MeasureAttribute_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(attr, xmpls[, apr, wght]) | (attr, domcont[, apr]) | (cont, clss-dist [,apr]) -> (float, meas-type)")
{
	PyTRY
		NO_KEYWORDS

		if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrMeasureAttribute_Type) {
			PyErr_Format(PyExc_SystemError, "MeasureAttribute.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
			return PYNULL;
		}

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
						PyErr_Format(PyExc_IndexError, "attribute '%s' not in the given DomainContingency", var->get_name().c_str());
						return PYNULL;
					}
				}

				else if (PyString_Check(arg1)) {
					char *attrName = PyString_AsString(arg1);
					for(attrNo = 0; (ci!=ce) && (couter ? ((*ci)->innerVariable->get_name()!= attrName) : ((*ci)->outerVariable->get_name()!=attrName)); ci++, attrNo++);

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


PyObject *MeasureAttribute_thresholdFunction(PyObject *self, PyObject *args, PyObject *kwds) PYARGS(METH_VARARGS, "(attr, examples[, weightID]) | (contingency[, distribution]) -> list")
{
	PyTRY
		TFloatFloatList thresholds;

	PyObject *pyvar;
	PExampleGenerator gen;
	int weightID = 0;
	if (PyArg_ParseTuple(args, "OO&|i:MeasureAttribute_thresholdFunction", &pyvar, pt_ExampleGenerator, &gen, &weightID)) {
		PVariable var = varFromArg_byDomain(pyvar, gen->domain);
		if (!var)
			return NULL;

		SELF_AS(TMeasureAttribute).thresholdFunction(thresholds, var, gen, PDistribution(), weightID);
	}
	else {
		PyErr_Clear();

		PContingency cont;
		PDistribution cdist;
		if (PyArg_ParseTuple(args, "O&|O&", cc_Contingency, &cont, ccn_Distribution, &cdist)) {
			if (!cdist)
				cdist = cont->innerDistribution;

			SELF_AS(TMeasureAttribute).thresholdFunction(thresholds, cont, cdist);
		}
		else {
			PyErr_Clear();
			PYERROR(PyExc_TypeError, "MeasureAttribute.thresholdFunction expects a variable, generator[, weight], or contingency", PYNULL)
		}
	}

	PyObject *res = PyList_New(thresholds.size());
	Py_ssize_t li = 0;
	for(TFloatFloatList::const_iterator ti(thresholds.begin()), te(thresholds.end()); ti != te; ti++)
		PyList_SetItem(res, li++, Py_BuildValue("ff", ti->first, ti->second));
	return res;
	PyCATCH;
}



PyObject *MeasureAttribute_relief_pairGains(PyObject *self, PyObject *args, PyObject *kwds) PYARGS(METH_VARARGS, "(attr, examples) -> list")
{
	PyTRY
		PyObject *pyvar;
	PExampleGenerator gen;
	int weightID = 0;
	if (!PyArg_ParseTuple(args, "OO&|i:MeasureAttribute_pairGains", &pyvar, pt_ExampleGenerator, &gen, &weightID))
		return NULL;

	PVariable var = varFromArg_byDomain(pyvar, gen->domain);
	if (!var)
		return NULL;

	TPairGainAdder pairGains;
	SELF_AS(TMeasureAttribute_relief).pairGains(pairGains, var, gen, weightID);

	PyObject *res = PyList_New(pairGains.size());
	Py_ssize_t li = 0;
	for(TPairGainAdder::const_iterator ti(pairGains.begin()), te(pairGains.end()); ti != te; ti++)
		PyList_SetItem(res, li++, Py_BuildValue("(ff)f", ti->e1, ti->e2, ti->gain));
	return res;
	PyCATCH
}


PyObject *MeasureAttribute_relief_gainMatrix(PyObject *self, PyObject *args, PyObject *kwds) PYARGS(METH_VARARGS, "(attr, examples) -> SymMatrix")
{
	PyTRY
		PyObject *pyvar;
	PExampleGenerator gen;
	int weightID = 0;
	if (!PyArg_ParseTuple(args, "OO&|i:MeasureAttribute_gainMatrix", &pyvar, pt_ExampleGenerator, &gen, &weightID))
		return NULL;

	PVariable var = varFromArg_byDomain(pyvar, gen->domain);
	if (!var)
		return NULL;

	return WrapOrange(SELF_AS(TMeasureAttribute_relief).gainMatrix(var, gen, NULL, weightID, NULL, NULL));
	PyCATCH
}

PyObject *MeasureAttribute_bestThreshold(PyObject *self, PyObject *args, PyObject *kwds) PYARGS(METH_VARARGS, "(attr, examples) -> list")
{
	PyTRY
		PyObject *pyvar;
	PExampleGenerator gen;
	int weightID = 0;
	float minSubset = 0.0;
	if (!PyArg_ParseTuple(args, "OO&|if:MeasureAttribute_thresholdFunction", &pyvar, pt_ExampleGenerator, &gen, &weightID, &minSubset))
		return NULL;

	PVariable var = varFromArg_byDomain(pyvar, gen->domain);
	if (!var)
		return NULL;

	float threshold, score;
	PDistribution distribution;
	threshold = SELF_AS(TMeasureAttribute).bestThreshold(distribution, score, var, gen, PDistribution(), weightID);

	if (threshold == ILLEGAL_FLOAT)
		PYERROR(PyExc_SystemError, "cannot compute the threshold; check the number of instances etc.", PYNULL);

	return Py_BuildValue("ffO", threshold, score, WrapOrange(distribution));
	PyCATCH
}

/* ************ EXAMPLE CLUSTERING ************ */

#include "exampleclustering.hpp"

ABSTRACT(GeneralExampleClustering - Orange.core.GeneralExampleClustering, Orange)
C_NAMED(ExampleCluster - Orange.clustering.ExampleCluster, Orange, "([left=, right=, distance=, centroid=])")
C_NAMED(ExampleClusters - Orange.core.ExampleClusters, GeneralExampleClustering, "([root=, quality=]")


PyObject *GeneralExampleClustering_exampleClusters(PyObject *self) PYARGS(METH_NOARGS, "() -> ExampleClusters")
{
	PyTRY
		return WrapOrange(SELF_AS(TGeneralExampleClustering).exampleClusters());
	PyCATCH
}


PyObject *GeneralExampleClustering_exampleSets(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> ExampleSets")
{
	PyTRY
		float cut = 0.0;
	if (!PyArg_ParseTuple(args, "|f", &cut))
		return PYNULL;

	return WrapOrange(SELF_AS(TGeneralExampleClustering).exampleSets(cut));
	PyCATCH
}


PyObject *GeneralExampleClustering_classifier(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> Classifier")
{
	PyTRY
		float cut = 0.0;
	if (!PyArg_ParseTuple(args, "|f", &cut))
		return PYNULL;

	return WrapOrange(SELF_AS(TGeneralExampleClustering).classifier(cut));
	PyCATCH
}


PyObject *GeneralExampleClustering_feature(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([cut=0.0]) -> Variable")
{
	PyTRY
		float cut = 0.0;
	if (!PyArg_ParseTuple(args, "|f", &cut))
		return PYNULL;

	return WrapOrange(SELF_AS(TGeneralExampleClustering).feature(cut));
	PyCATCH
}


#include "calibrate.hpp"

C_CALL(ThresholdCA - Orange.wrappers.ThresholdCA, Orange, "([classifier, examples[, weightID, target value]]) -/-> (threshold, optimal CA, list of CAs))")

PyObject *ThresholdCA_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(classifier, examples[, weightID, target value]) -> (threshold, optimal CA, list of CAs)")
{
	PyTRY
		NO_KEYWORDS

		PClassifier classifier;
	PExampleGenerator egen;
	int weightID = 0;
	PyObject *pyvalue = NULL;
	int targetVal = -1;

	if (!PyArg_ParseTuple(args, "O&O&|O&O:ThresholdCA.__call__", cc_Classifier, &classifier, pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID, &pyvalue))
		return PYNULL;
	if (pyvalue) {
		TValue classVal;
		if (!convertFromPython(pyvalue, classVal, classifier->classVar))
			return PYNULL;
		if (classVal.isSpecial())
			PYERROR(PyExc_TypeError, "invalid target value", PYNULL);
		targetVal = classVal.intV;
	}

	TFloatFloatList *ffl = mlnew TFloatFloatList();
	PFloatFloatList wfl(ffl);
	float optThresh, optCA;
	optThresh = SELF_AS(TThresholdCA).call(classifier, egen, weightID, optCA, targetVal, ffl);

	PyObject *pyCAs = PyList_New(ffl->size());
	Py_ssize_t i = 0;
	PITERATE(TFloatFloatList, ffi, ffl)
		PyList_SetItem(pyCAs, i++, Py_BuildValue("ff", (*ffi).first, (*ffi).second));

	return Py_BuildValue("ffN", optThresh, optCA, pyCAs);
	PyCATCH
}


#include "symmatrix.hpp"

PyObject *SymMatrix_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "(dimension[, initialElement=0] | a list of lists)")
{
	PyTRY
		int dim;
	float init = 0;
	if (PyArg_ParseTuple(args, "i|f", &dim, &init)) {
		if (dim<1)
			PYERROR(PyExc_TypeError, "matrix dimension must be positive", PYNULL);

		return WrapNewOrange(mlnew TSymMatrix(dim, init), type);
	}

	PyErr_Clear();

	PyObject *arg;
	if (PyArg_ParseTuple(args, "O|f", &arg, &init)) {
		dim = PySequence_Size(arg);
		PyObject *iter = PyObject_GetIter(arg);
		if ((dim<0) || !iter)
			PYERROR(PyExc_TypeError, "SymMatrix.__init__ expects a list of lists or the dimension, and an optional default element", PYNULL);

#define UNKNOWN_F -1e30f

		TSymMatrix *symmatrix = mlnew TSymMatrix(dim, UNKNOWN_F);
		PyObject *subiter = NULL;

#define FREE_ALL  Py_DECREF(iter); delete symmatrix; Py_XDECREF(subiter);

		int i, j;

		for(i = 0; i<dim; i++) {
			PyObject *item = PyIter_Next(iter);
			if (!item) {
				FREE_ALL
					PYERROR(PyExc_SystemError, "matrix is shorter than promissed ('len' returned more elements than there actuall are)", PYNULL);
			}

			PyObject *subiter = PyObject_GetIter(item);
			Py_DECREF(item);

			if (!subiter) {
				FREE_ALL
					PyErr_Format(PyExc_TypeError, "row %i is not a sequence", i);
				return PYNULL;
			}

			for(j = 0;; j++) {
				PyObject *subitem = PyIter_Next(subiter);
				if (!subitem)
					break;

				float f;
				bool ok = PyNumber_ToFloat(subitem, f);
				Py_DECREF(subitem);
				if (!ok) {
					FREE_ALL
						PyErr_Format(PyExc_TypeError, "element at (%i, %i) is not a number", i, j);
					return PYNULL;
				}


				try {
					float &mae = symmatrix->getref(i, j);

					if ((mae != UNKNOWN_F) && (mae!=f)) {
						FREE_ALL
							PyErr_Format(PyExc_TypeError, "the element at (%i, %i) is asymmetric", i, j);
						return PYNULL;
					}

					mae = f;
				}
				catch (...) {
					FREE_ALL
						throw;
				}
			}

			Py_DECREF(subiter);
			subiter = NULL;
		}
		Py_DECREF(iter);

		float *e = symmatrix->elements;
		for(i = ((dim+1)*(dim+2)) >> 1; i--; e++)
			if (*e == UNKNOWN_F)
				*e = init;

		return WrapNewOrange(symmatrix, type);

#undef UNKNOWN_F
#undef FREE_ALL
	}

	PyErr_Clear();

	PYERROR(PyExc_TypeError, "SymMatrix.__init__ expects a list of lists or the dimension and the initial element", PYNULL);

	PyCATCH
}


PyObject *SymMatrix__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TSymMatrix, matrix);
	const int dim = matrix->dim;
	return Py_BuildValue("O(Os#i)N", getExportedFunction("__pickleLoaderSymMatrix"),
		self->ob_type,
		matrix->elements, sizeof(float) * (((dim+1) * (dim+2)) >> 1),
		dim,
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderSymMatrix(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_matrix, dimension)")
{
	PyTRY
		PyTypeObject *type;
	char *buf;
	int bufSize, dim;
	if (!PyArg_ParseTuple(args, "Os#i:__pickleLoaderCostMatrix", &type, &buf, &bufSize, &dim))
		return NULL;

	TSymMatrix *cm = new TSymMatrix(dim);
	memcpy(cm->elements, buf, bufSize);
	return WrapNewOrange(cm, type);
	PyCATCH
}


PyObject *SymMatrix_getValues(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "(None -> list of values)")
{
	PyTRY
	CAST_TO(TSymMatrix, matrix)

	PyObject* components_list = PyList_New(0);

	int i,j;
	for (i = 0; i < matrix->dim; i++) {
		for (j = i+1; j < matrix->dim; j++) {
			double value = 0;
			if (matrix->matrixType == 0)
				value = matrix->getitem(j,i);
			else
				value = matrix->getitem(i,j);

			PyObject *nel = Py_BuildValue("d", value);
			PyList_Append(components_list, nel);
			Py_DECREF(nel);
		}
	}

	return components_list;
	PyCATCH
}

PyObject *SymMatrix_getKNN(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "((i, K) -> list of values)")
{
	PyTRY
	CAST_TO(TSymMatrix, matrix)

	int kNN;
	int i;

	if (!PyArg_ParseTuple(args, "ii:SymMatrix.getKNN", &i, &kNN))
		return PYNULL;

	vector<int> closest;
	matrix->getknn(i, kNN, closest);

	PyObject* components_list = PyList_New(0);

	for (i = 0; i < closest.size(); i++) {
		PyObject *nel = Py_BuildValue("i", closest[i]);
		PyList_Append(components_list, nel);
		Py_DECREF(nel);
	}

	return components_list;
	PyCATCH
}

PyObject *SymMatrix_avgLinkage(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Clusters -> SymMatrix)")
{
	PyTRY
	CAST_TO(TSymMatrix, matrix);

	PyObject *clusters;

	if (!PyArg_ParseTuple(args, "O:SymMatrix.avgLinkage", &clusters))
		return PYNULL;

	int size = PyList_Size(clusters);

	TSymMatrix *symmatrix = new TSymMatrix(size);
	PSymMatrix wsymmatrix = symmatrix;

	symmatrix->matrixType = TSymMatrix::Symmetric;

	int i,j,k,l;
	for (i = 0; i < size; i++)
	{
		for (j	 = i; j < size; j++)
		{
			PyObject *cluster_i = PyList_GetItem(clusters, i);
			PyObject *cluster_j = PyList_GetItem(clusters, j);
			int size_i = PyList_Size(cluster_i);
			int size_j = PyList_Size(cluster_j);
			float sum = 0;
			for (k = 0; k < size_i; k++)
			{
				for (l = 0; l < size_j; l++)
				{
					int item_k = PyInt_AsLong(PyList_GetItem(cluster_i, k));
					int item_l = PyInt_AsLong(PyList_GetItem(cluster_j, l));

					if (item_k >= matrix->dim || item_l >= matrix->dim)
						raiseError("index out of range");

					sum += matrix->getitem(item_k, item_l);
				}
			}

			sum /= (size_i * size_j);
			symmatrix->getref(i, j) = sum;
		}
	}

	PyObject *pysymmatrix = WrapOrange(wsymmatrix);
	return pysymmatrix;

	PyCATCH
}

PyObject *SymMatrix_invert(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Invert type -> None)")
{
	/* ******************
	 * Types:
	 * 0 - [-X]
	 * 1 - [1 - X]
	 * 2 - [max - X]
	 * 3 - [1 / X]
	 ********************/
	PyTRY
	int type;

	if (!PyArg_ParseTuple(args, "i:SymMatrix.invert", &type))
		return NULL;

	if (type < 0 || type > 3)
		PYERROR(PyExc_AttributeError, "only types 0 to 3  are supported", PYNULL);

	CAST_TO(TSymMatrix, matrix);
	int i;
	float *e = matrix->elements;
	switch(type)
	{
		case 0:
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				*e = 0 - *e;
			break;

		case 1:
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				*e = 1 - *e;
			break;

		case 2:
			float maxval;
			maxval = 0;

			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				if (*e > maxval)
					maxval = *e;

			e = matrix->elements;
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				*e = maxval - *e;

			break;

		case 3:
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				if (*e == 0)
					raiseError("division by zero");
				*e = 1 / *e;
			break;
	}

	RETURN_NONE;
	PyCATCH
}

PyObject *SymMatrix_normalize(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Normalize type -> None)")
{
	/* ******************
	 * Types:
	 * 0 - [0, 1]
	 * 1 - Sigmoid
	 ********************/
	PyTRY
	int type;

	if (!PyArg_ParseTuple(args, "i:SymMatrix.normalize", &type))
		return NULL;

	if (type < 0 || type > 1)
		PYERROR(PyExc_AttributeError, "only types 0 and 1 are supported", PYNULL);

	CAST_TO(TSymMatrix, matrix);
	int i;
	float *e = matrix->elements;
	float maxval = *e;
	float minval = *e;
	switch(type)
	{
		case 0:
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++) {
				if (*e > maxval)
					maxval = *e;

				if (*e < minval)
					minval = *e;
			}
			//cout << "minval: " << minval << endl;
			//cout << "maxval: " << maxval << endl;

			e = matrix->elements;
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				*e = (*e - minval) / (maxval - minval);
			break;

		case 1:
			for(i = ((matrix->dim+1)*(matrix->dim+2)) >> 1; i--; e++)
				*e = 1 / (1 + exp(-(*e)));
			break;
	}

	RETURN_NONE;
	PyCATCH
}

PyObject *SymMatrix_get_items(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(List of items -> SymMatrix)")
{
	PyTRY
	CAST_TO(TSymMatrix, matrix);

	PyObject *items;

	if (!PyArg_ParseTuple(args, "O:SymMatrix.get_items", &items))
		return PYNULL;

	int size = PyList_Size(items);
	PyList_Sort(items);

	TSymMatrix *symmatrix = new TSymMatrix(size);
	PSymMatrix wsymmatrix = symmatrix;

	symmatrix->matrixType = matrix->matrixType;

	int i,j;
	for (i = 0; i < size; i++)
	{
		for (j = i; j < size; j++)
		{
			if (symmatrix->matrixType == TSymMatrix::Lower || symmatrix->matrixType == TSymMatrix::LowerFilled)
			{
				int item_i = PyInt_AsLong(PyList_GetItem(items, j));
				int item_j = PyInt_AsLong(PyList_GetItem(items, i));

				float value = matrix->getitem(item_i, item_j);
				symmatrix->getref(j, i) = value;
			}
			else
			{
				int item_i = PyInt_AsLong(PyList_GetItem(items, i));
				int item_j = PyInt_AsLong(PyList_GetItem(items, j));

				float value = matrix->getitem(item_i, item_j);
				symmatrix->getref(i, j) = value;
			}
		}
	}

	PyObject *pysymmatrix = WrapOrange(wsymmatrix);
	return pysymmatrix;
	PyCATCH
}

PyObject *SymMatrix_getitem_sq(PyObject *self, Py_ssize_t i)
{
	PyTRY
		CAST_TO(TSymMatrix, matrix)
		int dim = matrix->dim;

	if (i >= matrix->dim) {
		PyErr_Format(PyExc_IndexError, "index %i out of range 0-%i", i, matrix->dim-1);
		return PYNULL;
	}

	Py_ssize_t j;
	PyObject *row;
	switch (matrix->matrixType) {
			case TSymMatrix::Lower:
				row = PyTuple_New(i+1);
				for(j = 0; j<=i; j++)
					PyTuple_SetItem(row, j, PyFloat_FromDouble((double)matrix->getitem(i, j)));
				return row;

			case TSymMatrix::Upper:
				row = PyTuple_New(matrix->dim - i);
				for(j = i; j<dim; j++)
					PyTuple_SetItem(row, j-i, PyFloat_FromDouble((double)matrix->getitem(i, j)));
				return row;

			default:
				row = PyTuple_New(matrix->dim);
				for(j = 0; j<dim; j++)
					PyTuple_SetItem(row, j, PyFloat_FromDouble((double)matrix->getitem(i, j)));
				return row;
	}
	PyCATCH
}



PyObject *SymMatrix_getitem(PyObject *self, PyObject *args)
{
	PyTRY
		CAST_TO(TSymMatrix, matrix)

		if ((PyTuple_Check(args) && (PyTuple_Size(args) == 1)) || PyInt_Check(args)) {
			if (PyTuple_Check(args)) {
				args = PyTuple_GET_ITEM(args, 0);
				if (!PyInt_Check(args))
					PYERROR(PyExc_IndexError, "integer index expected", PYNULL);
			}

			return SymMatrix_getitem_sq(self, (int)PyInt_AsLong(args));
		}

		else if (PyTuple_Size(args) == 2) {
			if (!PyInt_Check(PyTuple_GET_ITEM(args, 0)) || !PyInt_Check(PyTuple_GET_ITEM(args, 1)))
				PYERROR(PyExc_IndexError, "integer indices expected", PYNULL);

			const int i = PyInt_AsLong(PyTuple_GET_ITEM(args, 0));
			const int j = PyInt_AsLong(PyTuple_GET_ITEM(args, 1));
			if ((j>i) && (matrix->matrixType == TSymMatrix::Lower))
				PYERROR(PyExc_IndexError, "index out of range for lower triangular matrix", PYNULL);

			if ((j<i) && (matrix->matrixType == TSymMatrix::Upper))
				PYERROR(PyExc_IndexError, "index out of range for upper triangular matrix", PYNULL);

			return PyFloat_FromDouble(matrix->getitem(i, j));
		}

		PYERROR(PyExc_IndexError, "one or two integer indices expected", PYNULL);
		PyCATCH
}


int SymMatrix_setitem(PyObject *self, PyObject *args, PyObject *obj)
{
	PyTRY
		if (PyTuple_Size(args) == 1)
			PYERROR(PyExc_AttributeError, "cannot set entire matrix row", -1);

	if (PyTuple_Size(args) != 2)
		PYERROR(PyExc_IndexError, "two integer indices expected", -1);

	PyObject *pyfl = PyNumber_Float(obj);
	if (!pyfl)
		PYERROR(PyExc_TypeError, "invalid matrix elements; a number expected", -1);
	float f = PyFloat_AsDouble(pyfl);
	Py_DECREF(pyfl);

	const int i = PyInt_AsLong(PyTuple_GET_ITEM(args, 0));
	const int j = PyInt_AsLong(PyTuple_GET_ITEM(args, 1));

	SELF_AS(TSymMatrix).getref(i, j) = f;
	return 0;
	PyCATCH_1
}


PyObject *SymMatrix_getslice(PyObject *self, Py_ssize_t start, Py_ssize_t stop)
{
	PyTRY
		CAST_TO(TSymMatrix, matrix)
		const int dim = matrix->dim;

	if (start>dim)
		start = dim;
	else if (start<0)
		start = 0;

	if (stop>dim)
		stop=dim;

	PyObject *res = PyTuple_New(stop - start);
	int i = 0;
	while(start<stop)
		PyTuple_SetItem(res, i++, SymMatrix_getitem_sq(self, start++));

	return res;
	PyCATCH
}


PyObject *SymMatrix_str(PyObject *self)
{ PyTRY
CAST_TO(TSymMatrix, matrix)
const int dim = matrix->dim;
const int mattype = matrix->matrixType;

float matmax = 0.0;
for(float *ei = matrix->elements, *ee = matrix->elements + ((dim*(dim+1))>>1); ei != ee; ei++) {
	const float tei = *ei<0 ? fabs(10.0 * *ei) : *ei;
	if (tei > matmax)
		matmax = tei;
}

const int plac = 4 + (fabs(matmax) < 1 ? 1 : int(ceil(log10((double)matmax))));
const int elements = (matrix->matrixType == TSymMatrix::Lower) ? (dim*(dim+1))>>1 : dim * dim;
char *smatr = new char[3 * dim + (plac+2) * elements];
char *sptr = smatr;
*(sptr++) = '(';
*(sptr++) = '(';

int i, j;
for(i = 0; i<dim; i++) {
	switch (mattype) {
				case TSymMatrix::Lower:
					for(j = 0; j<i; j++, sptr += (plac+2))
						sprintf(sptr, "%*.3f, ", plac, matrix->getitem(i, j));
					break;

				case TSymMatrix::Upper:
					for(j = i * (plac+2); j--; *(sptr++) = ' ');
					for(j = i; j < dim-1; j++, sptr += (plac+2))
						sprintf(sptr, "%*.3f, ", plac, matrix->getitem(i, j));
					break;

				default:
					for(j = 0; j<dim-1; j++, sptr += (plac+2))
						sprintf(sptr, "%*.3f, ", plac, matrix->getitem(i, j));
	}

	sprintf(sptr, "%*.3f)", plac, matrix->getitem(i, j));
	sptr += (plac+1);

	if (i!=dim-1) {
		sprintf(sptr, ",\n (");
		sptr += 4;
	}
}

sprintf(sptr, ")");

PyObject *res = PyString_FromString(smatr);
mldelete smatr;
return res;
PyCATCH
}


PyObject *SymMatrix_repr(PyObject *self)
{ return SymMatrix_str(self); }


#include "hclust.hpp"

C_NAMED(HierarchicalCluster - Orange.clustering.hierarchical.HierarchicalCluster, Orange, "()")
C_CALL3(HierarchicalClustering - Orange.clustering.hierarchical.HierarchicalClustering, HierarchicalClustering, Orange, "(linkage=)")

C_CALL3(HierarchicalClusterOrdering - Orange.clustering.hierarchical.HierarchicalClusterOrdering, HierarchicalClusterOrdering, Orange, "(progressCallback=)")

PyObject *HierarchicalClustering_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(distance matrix) -> HierarchicalCluster")
{
	PyTRY
		NO_KEYWORDS

		PSymMatrix symmatrix;

	if (!PyArg_ParseTuple(args, "O&:HierarchicalClustering", cc_SymMatrix, &symmatrix))
		return NULL;

	PHierarchicalCluster root = SELF_AS(THierarchicalClustering)(symmatrix);

	if (symmatrix->myWrapper->orange_dict) {
		PyObject *objects = PyDict_GetItemString(symmatrix->myWrapper->orange_dict, "objects");
		TPyOrange *pymapping = root->mapping->myWrapper;
		if (objects && (objects != Py_None)) {
			if (!pymapping->orange_dict)
				pymapping->orange_dict = PyOrange_DictProxy_New(pymapping);
			PyDict_SetItemString(pymapping->orange_dict, "objects", objects);
		}
	}
	return WrapOrange(root);
	PyCATCH
}

PyObject * HierarchicalClusterOrdering_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(hierarchical cluster, distance_matrix) -> None")
{
	PyTRY
		NO_KEYWORDS
		PHierarchicalCluster root;
		PSymMatrix matrix;
		if (!PyArg_ParseTuple(args, "O&O&:HierarchicalClustering", cc_HierarchicalCluster, &root, cc_SymMatrix, &matrix))
			return NULL;
		SELF_AS(THierarchicalClusterOrdering).operator ()(root, matrix);
		RETURN_NONE
	PyCATCH
}

Py_ssize_t HierarchicalCluster_len_sq(PyObject *self)
{
	CAST_TO_err(THierarchicalCluster, cluster, -1);
	return cluster->last - cluster->first;
}


PyObject *HierarchicalCluster_getitem_sq(PyObject *self, Py_ssize_t i)
{
	PyTRY
		CAST_TO(THierarchicalCluster, cluster);

	if (!cluster->mapping)
		PYERROR(PyExc_SystemError, "'HierarchicalCluster' misses 'mapping'", PYNULL);

	i += (i>=0) ? cluster->first : cluster->last;
	if ((i < cluster->first) || (i >= cluster->last)) {
		PyErr_Format(PyExc_IndexError, "index out of range 0-%i", cluster->last - cluster->first - 1);
		return PYNULL;
	}

	if (i >= cluster->mapping->size())
		PYERROR(PyExc_SystemError, "internal inconsistency in instance of 'HierarchicalCluster' ('mapping' too short)", PYNULL);

	const int elindex = cluster->mapping->at(i);

	if (cluster->mapping->myWrapper->orange_dict) {
		PyObject *objs = PyDict_GetItemString(cluster->mapping->myWrapper->orange_dict, "objects");
		if (objs && (objs != Py_None))
			return PySequence_GetItem(objs, elindex);
	}

	return PyInt_FromLong(elindex);
	PyCATCH
}


PyObject *HierarchicalCluster_get_left(PyObject *self)
{
	PyTRY
		CAST_TO(THierarchicalCluster, cluster);

	if (!cluster->branches)
		RETURN_NONE

		if (cluster->branches->size() > 2)
			PYERROR(PyExc_AttributeError, "'left' not defined (cluster has more than two subclusters)", PYNULL);

	return WrapOrange(cluster->branches->front());
	PyCATCH
}


PyObject *HierarchicalCluster_get_right(PyObject *self)
{
	PyTRY
		CAST_TO(THierarchicalCluster, cluster);

	if (!cluster->branches || (cluster->branches->size() < 2))
		RETURN_NONE;

	if (cluster->branches->size() > 2)
		PYERROR(PyExc_AttributeError, "'right' not defined (cluster has more than two subclusters", PYNULL);

	return WrapOrange(cluster->branches->back());
	PyCATCH
}


int HierarchicalClusterLowSet(PyObject *self, PyObject *arg, const int side)
{
	PyTRY
		static const char *sides[2] = {"left", "right"};

	if (!PyOrHierarchicalCluster_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "'HierarchicalCluster.%s' should be of type 'HierarchicalCluster' (got '%s')", sides[side], arg->ob_type->tp_name);
		return -1;
	}

	CAST_TO_err(THierarchicalCluster, cluster, -1);

	if (!cluster->branches)
		cluster->branches = mlnew THierarchicalClusterList(2);
	else
		if (cluster->branches->size() != 2)
			PYERROR(PyExc_AttributeError, "'left' not defined (cluster does not have (exactly) two subclusters)", -1);

	cluster->branches->at(side) = PyOrange_AsHierarchicalCluster(arg);
	return 0;
	PyCATCH_1
}


int HierarchicalCluster_set_left(PyObject *self, PyObject *arg)
{
	return HierarchicalClusterLowSet(self, arg, 0);
}

int HierarchicalCluster_set_right(PyObject *self, PyObject *arg)
{
	return HierarchicalClusterLowSet(self, arg, 1);
}


PyObject *HierarchicalCluster_swap(PyObject *self, PyObject *arg, PyObject *keyw) PYARGS(METH_NOARGS, "() -> None; swaps the sub clusters")
{
	PyTRY
		SELF_AS(THierarchicalCluster).swap();
	RETURN_NONE;
	PyCATCH
}

PyObject *HierarchicalCluster_permute(PyObject *self, PyObject *arg, PyObject *keys) PYARGS(METH_O, "(permutation) -> None")
{
	PyTRY
		PIntList ilist = ListOfUnwrappedMethods<PIntList, TIntList, int>::P_FromArguments(arg);
	if (!ilist)
		return PYNULL;

	SELF_AS(THierarchicalCluster).permute(ilist.getReference());
	RETURN_NONE;
	PyCATCH
}


PHierarchicalClusterList PHierarchicalClusterList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::P_FromArguments(arg); }
PyObject *HierarchicalClusterList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_FromArguments(type, arg); }
PyObject *HierarchicalClusterList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange - Orange.clustering.hierarchical.HierarchicalClusterList, "(<list of HierarchicalCluster>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_new(type, arg, kwds); }
PyObject *HierarchicalClusterList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_getitem(self, index); }
int       HierarchicalClusterList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_setitem(self, index, item); }
PyObject *HierarchicalClusterList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_getslice(self, start, stop); }
int       HierarchicalClusterList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       HierarchicalClusterList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_len(self); }
PyObject *HierarchicalClusterList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_richcmp(self, object, op); }
PyObject *HierarchicalClusterList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_concat(self, obj); }
PyObject *HierarchicalClusterList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_repeat(self, times); }
PyObject *HierarchicalClusterList_str(TPyOrange *self) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_str(self); }
PyObject *HierarchicalClusterList_repr(TPyOrange *self) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_str(self); }
int       HierarchicalClusterList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_contains(self, obj); }
PyObject *HierarchicalClusterList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(HierarchicalCluster) -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_append(self, item); }
PyObject *HierarchicalClusterList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_extend(self, obj); }
PyObject *HierarchicalClusterList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(HierarchicalCluster) -> int") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_count(self, obj); }
PyObject *HierarchicalClusterList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> HierarchicalClusterList") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_filter(self, args); }
PyObject *HierarchicalClusterList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(HierarchicalCluster) -> int") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_index(self, obj); }
PyObject *HierarchicalClusterList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_insert(self, args); }
PyObject *HierarchicalClusterList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_native(self); }
PyObject *HierarchicalClusterList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> HierarchicalCluster") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_pop(self, args); }
PyObject *HierarchicalClusterList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(HierarchicalCluster) -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_remove(self, obj); }
PyObject *HierarchicalClusterList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_reverse(self); }
PyObject *HierarchicalClusterList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_sort(self, args); }
PyObject *HierarchicalClusterList__reduce__(TPyOrange *self, PyObject *) PYARGS(METH_VARARGS, "()") { return ListOfWrappedMethods<PHierarchicalClusterList, THierarchicalClusterList, PHierarchicalCluster, &PyOrHierarchicalCluster_Type>::_reduce(self); }




#include "distancemap.hpp"

C_NAMED(DistanceMapConstructor - Orange.distances.DistanceMapConstructor, Orange, "(distanceMatrix=, order=)")



PyObject *DistanceMapConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(squeeze) -> DistanceMap")
{
	PyTRY
		NO_KEYWORDS

		float squeeze = 1.0;
	if (!PyArg_ParseTuple(args, "|f:DistanceMapConstructor.__call__", &squeeze))
		return NULL;

	float absLow, absHigh;
	PDistanceMap dm = SELF_AS(TDistanceMapConstructor).call(squeeze, absLow, absHigh);
	return Py_BuildValue("Nff", WrapOrange(dm), absLow, absHigh);
	PyCATCH
}


PyObject *DistanceMapConstructor_getLegend(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(width, height, gamma) -> bitmap")
{
	PyTRY
		int width, height;
	float gamma;
	if (!PyArg_ParseTuple(args, "iif:DistanceMapConstructor.getLegend", &width, &height, &gamma))
		return NULL;

	long size;
	unsigned char *bitmap = SELF_AS(TDistanceMapConstructor).getLegend(width, height, gamma, size);
	PyObject *res = PyString_FromStringAndSize((const char *)bitmap, (Py_ssize_t)size);
	delete bitmap;
	return res;
	PyCATCH
}


BASED_ON(DistanceMap - Orange.distances.DistanceMap, Orange)

PyObject *DistanceMap__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TDistanceMap, matrix);
	const int dim = matrix->dim;
	return Py_BuildValue("O(Os#iO)N", getExportedFunction("__pickleLoaderDistanceMap"),
		self->ob_type,
		matrix->cells, dim*dim*sizeof(float),
		dim,
		WrapOrange(matrix->elementIndices),
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderDistanceMap(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_matrix, dimension, elementIndices)")
{
	PyTRY
		PyTypeObject *type;
	char *buf;
	int bufSize, dim;
	PIntList elementIndices;
	if (!PyArg_ParseTuple(args, "Os#iO&:__pickleLoaderDistanceMap", &type, &buf, &bufSize, &dim, ccn_IntList, &elementIndices))
		return NULL;

	TDistanceMap *cm = new TDistanceMap(dim);
	memcpy(cm->cells, buf, bufSize);
	cm->elementIndices = elementIndices;
	return WrapNewOrange(cm, type);
	PyCATCH
}


PyObject *DistanceMap_getBitmap(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(cell_width, cell_height, lowerBound, upperBound, gamma) -> bitmap")
{
	PyTRY
		int cellWidth, cellHeight;
	float absLow, absHigh, gamma;
	int grid = 1;
	int matrixType = 2;
	if (!PyArg_ParseTuple(args, "iifff|ii:Heatmap.getBitmap", &cellWidth, &cellHeight, &absLow, &absHigh, &gamma, &grid, &matrixType))
		return NULL;

	CAST_TO(TDistanceMap, dm)

		long size;
	unsigned char *bitmap = dm->distanceMap2string(cellWidth, cellHeight, absLow, absHigh, gamma, grid!=0, matrixType, size);
	PyObject *res = Py_BuildValue("s#ii", (const char *)bitmap, size, cellWidth * dm->dim, cellHeight * dm->dim);
	delete bitmap;
	return res;
	PyCATCH
}


PyObject *DistanceMap_getCellIntensity(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(row, column) -> float")
{
	PyTRY
		int row, column;
	if (!PyArg_ParseTuple(args, "ii:DistanceMap.getCellIntensity", &row, &column))
		return NULL;

	const float ci = SELF_AS(TDistanceMap).getCellIntensity(row, column);
	if (ci == ILLEGAL_FLOAT)
		RETURN_NONE;

	return PyFloat_FromDouble(ci);
	PyCATCH
}


PyObject *DistanceMap_getPercentileInterval(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(lower_percentile, upper_percentile) -> (min, max)")
{
	PyTRY
		float lowperc, highperc;
	if (!PyArg_ParseTuple(args, "ff:DistanceMap_percentileInterval", &lowperc, &highperc))
		return PYNULL;

	float minv, maxv;
	SELF_AS(TDistanceMap).getPercentileInterval(lowperc, highperc, minv, maxv);
	return Py_BuildValue("ff", minv, maxv);
	PyCATCH
}



#include "graph.hpp"

extern PyTypeObject PyEdge_Type;

/* If objectsOnEdges==true, this is a proxy object; double's are actualy PyObject *,
but the references are owned by the graph. */
class TPyEdge {
public:
	PyObject_HEAD

		PGraph graph;
	int v1, v2;
	double *weights;
	bool objectsOnEdges;
	int weightsVersion;

	inline double *getWeights()
	{
		if (weightsVersion != (weights ? graph->lastAddition : graph->lastRemoval)) {
			weights = graph->getEdge(v1, v2);
			weightsVersion = graph->currentVersion;
		}
		return weights;
	}
};


#define DOUBLE_AS_PYOBJECT(x) (*(PyObject **)(void *)(&(x)))
PyObject *PyEdge_Getitem(TPyEdge *self, Py_ssize_t ind)
{
	PyTRY
		if ((ind >= self->graph->nEdgeTypes) || (ind < 0)) {
			PyErr_Format(PyExc_IndexError, "type %s out of range (0-%i)", ind, self->graph->nEdgeTypes);
			return PYNULL;
		}

		if (self->getWeights()) {
			const double w = self->weights[ind];

			if (!CONNECTED(w))
				RETURN_NONE;

			if (self->objectsOnEdges) {
				Py_INCREF(DOUBLE_AS_PYOBJECT(w));
				return DOUBLE_AS_PYOBJECT(w);
			}
			else
				return PyFloat_FromDouble(w);
		}

		else
			RETURN_NONE;
		PyCATCH
}


int PyEdge_Contains(TPyEdge *self, PyObject *pyind)
{
	PyTRY
		if (!PyInt_Check(pyind))
			PYERROR(PyExc_IndexError, "edge types must be integers", -1);

	int ind = int(PyInt_AsLong(pyind));
	if ((ind >= self->graph->nEdgeTypes) || (ind < 0)) {
		PyErr_Format(PyExc_IndexError, "edge type %i out of range (0-%i)", ind, self->graph->nEdgeTypes);
		return -1;
	}

	return self->getWeights() && CONNECTED(self->weights[ind]) ? 1 : 0;
	PyCATCH_1
}


int PyEdge_Setitem(TPyEdge *self, Py_ssize_t ind, PyObject *item)
{
	PyTRY
		if ((ind >= self->graph->nEdgeTypes) || (ind < 0)) {
			PyErr_Format(PyExc_IndexError, "type %s out of range (0-%i)", ind, self->graph->nEdgeTypes);
			return -1;
		}

		double w;
		bool noConnection = !item || (item == Py_None);
		if (noConnection)
			DISCONNECT(w);
		else {
			if (!self->objectsOnEdges && !PyNumber_ToDouble(item, w))
				PYERROR(PyExc_TypeError, "a number expected for edge weight", -1);
		}


		if (self->getWeights()) {
			if (self->objectsOnEdges) {
				// watch the order: first INCREF, then DECREF!
				if (!noConnection)
					Py_INCREF(item);
				if (CONNECTED(self->weights[ind]))
					Py_DECREF(DOUBLE_AS_PYOBJECT(self->weights[ind]));

				DOUBLE_AS_PYOBJECT(self->weights[ind]) = item;
			}

			else
				self->weights[ind] = w;

			if (noConnection) {
				double *w, *we;
				for(w = self->weights, we = self->weights + self->graph->nEdgeTypes; (w != we) && !CONNECTED(*w); w++);
				if (w == we) {
					self->graph->removeEdge(self->v1, self->v2);
					self->weights = NULL;
					self->weightsVersion = self->graph->currentVersion;
				}
			}
		}

		else {
			if (!noConnection) {
				double *weights = self->weights = self->graph->getOrCreateEdge(self->v1, self->v2);

				if (self->objectsOnEdges) {
					DOUBLE_AS_PYOBJECT(weights[ind]) = item;
					Py_INCREF(item);
				}
				else
					weights[ind] = w;

				self->weightsVersion = self->graph->currentVersion;
			}
		}

		return 0;
		PyCATCH_1
}


/*
// I've programmed this by mistake; but it's nice, so let it stay
// for the case we need it :)
PyObject *PyEdge_Str(TPyEdge *self)
{
PyTRY
int nEdgeTypes = self->graph->nEdgeTypes;

if (!self->getWeights()) {
if (nEdgeTypes == 1)
RETURN_NONE;
else {
PyObject *res = PyTuple_New(nEdgeTypes);
while(nEdgeTypes--) {
Py_INCREF(Py_None);
PyTuple_SET_ITEM(res, nEdgeTypes, Py_None);
}
return res;
}
}

if (nEdgeTypes == 1)
return PyFloat_FromDouble(*self->weights);
else {
PyObject *res = PyTuple_New(nEdgeTypes);
int i = 0;
for(double weights = self->weights; i != nEdgeTypes; weights++, i++)
PyTuple_SET_ITEM(res, i, PyFloat_FromDouble(*weights));
return res;
}
PyCATCH
}
*/

PyObject *PyEdge_Str(TPyEdge *self)
{
	PyTRY
		int nEdgeTypes = self->graph->nEdgeTypes;
	char *buf;

	if (!self->getWeights()) {
		if (nEdgeTypes == 1)
			return PyString_FromString("None");
		else {
			buf = new char[nEdgeTypes*6 + 2];
			char *b2 = buf;
			*b2++ = '(';
			while(nEdgeTypes--) {
				strcpy(b2, "None, ");
				b2 += 6;
			}
			b2[-1] = 0;
			b2[-2] = ')';
		}
	}

	else {
		if (self->objectsOnEdges) {
			if (nEdgeTypes == 1)
				return PyObject_Str(DOUBLE_AS_PYOBJECT(*self->weights));
			else {
				PyObject *dump = PyString_FromString("(");
				PyString_ConcatAndDel(&dump, PyObject_Str(DOUBLE_AS_PYOBJECT(*self->weights)));

				for(double *wi = self->weights+1, *we = self->weights + self->graph->nEdgeTypes; wi != we; wi++) {
					PyString_ConcatAndDel(&dump, PyString_FromString(", "));
					PyString_ConcatAndDel(&dump, PyObject_Str(DOUBLE_AS_PYOBJECT(*wi)));
				}

				PyString_ConcatAndDel(&dump, PyString_FromString(")"));
				return dump;
			}
		}
		else {
			if (nEdgeTypes == 1) {
				buf = new char[20];
				char *b2 = buf;
				sprintf(b2, "%-10g", *self->weights);
				for(; *b2 > 32; b2++);
				*b2 = 0;
			}
			else {
				buf = new char[nEdgeTypes*20];
				char *b2 = buf;
				*b2++ = '(';
				for(double *weights = self->weights, *wee = weights + nEdgeTypes; weights != wee; weights++) {
					if (CONNECTED(*weights)) {
						sprintf(b2, "%-10g", *weights);
						for(; *b2 > 32; b2++);
						*b2++ = ',';
						*b2++ = ' ';
					}
					else {
						strcpy(b2, "None, ");
						b2 += 6;
					}
				}
				b2[-1] = 0;
				b2[-2] = ')';
			}
		}
	}

	PyObject *res = PyString_FromString(buf);
	delete buf;
	return res;
	PyCATCH
}

Py_ssize_t PyEdge_Len(TPyEdge *self)
{ return self->graph->nEdgeTypes; }


int PyEdge_Nonzero(TPyEdge *self)
{ return self->getWeights() ? 1 : 0; }


int PyEdge_Traverse(TPyEdge *self, visitproc visit, void *arg)
{ PVISIT(self->graph)
return 0;
}


int PyEdge_Clear(TPyEdge *self)
{ self->graph = POrange();
return 0;
}


void PyEdge_Dealloc(TPyEdge *self)
{
	self->graph.~PGraph();
	self->ob_type->tp_free((PyObject *)self);
}


PyObject *PyEdge_Richcmp(TPyEdge *self, PyObject *j, int op)
{
	double ref;
	if (self->graph->nEdgeTypes != 1)
		PYERROR(PyExc_TypeError, "multiple-type edges cannot be compared", PYNULL);

	if (self->graph->nEdgeTypes != 1)
		PYERROR(PyExc_TypeError, "multiple-type edges cannot be compared to floats", PYNULL);

	if (!self->getWeights() || !CONNECTED(*self->weights))
		PYERROR(PyExc_TypeError, "edge does not exist", PYNULL);

	if (self->objectsOnEdges)
		return PyObject_RichCompare(DOUBLE_AS_PYOBJECT(*self->weights), j, op);

	if (!PyNumber_ToDouble(j, ref))
		PYERROR(PyExc_TypeError, "edge weights can only be compared to floats", PYNULL);

	const double &f = *self->weights;

	int cmp;
	switch (op) {
case Py_LT: cmp = (f<ref); break;
case Py_LE: cmp = (f<=ref); break;
case Py_EQ: cmp = (f==ref); break;
case Py_NE: cmp = (f!=ref); break;
case Py_GT: cmp = (f>ref); break;
case Py_GE: cmp = (f>=ref); break;
default:
	Py_INCREF(Py_NotImplemented);
	return Py_NotImplemented;
	}

	PyObject *res;
	if (cmp)
		res = Py_True;
	else
		res = Py_False;
	Py_INCREF(res);
	return res;
}


PyObject *PyEdge_Float(TPyEdge *self)
{
	if (self->graph->nEdgeTypes != 1)
		PYERROR(PyExc_TypeError, "multiple-type edges cannot be cast to floats", PYNULL);

	if (!self->getWeights() || !CONNECTED(*self->weights))
		PYERROR(PyExc_TypeError, "edge does not exist", PYNULL);

	return self->objectsOnEdges ? PyNumber_Float(DOUBLE_AS_PYOBJECT(*self->weights)) : PyFloat_FromDouble(*self->weights);
}

PyObject *PyEdge_Int(TPyEdge *self)
{
	if (self->graph->nEdgeTypes != 1)
		PYERROR(PyExc_TypeError, "multiple-type edges cannot be cast to numbers", PYNULL);

	if (!self->getWeights() || !CONNECTED(*self->weights))
		PYERROR(PyExc_TypeError, "edge does not exist", PYNULL);

	return self->objectsOnEdges ? PyNumber_Int(DOUBLE_AS_PYOBJECT(*self->weights)) : PyInt_FromLong(long(*self->weights));
}


PyNumberMethods PyEdge_as_number = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	(inquiry)PyEdge_Nonzero,                           /* nb_nonzero */
	0, 0, 0, 0, 0, 0, 0,
	(unaryfunc)PyEdge_Int,    /* nb_int */
	0,
	(unaryfunc)PyEdge_Float,  /* nb_float */
	0, 0,
};

static PySequenceMethods PyEdge_as_sequence = {
	(inquiry)PyEdge_Len,					/* sq_length */
	0,					/* sq_concat */
	0,					/* sq_repeat */
	(intargfunc)PyEdge_Getitem,					/* sq_item */
	0,					/* sq_slice */
	(intobjargproc)PyEdge_Setitem,					/* sq_ass_item */
	0,					/* sq_ass_slice */
	(objobjproc)PyEdge_Contains,		/* sq_contains */
	0,					/* sq_inplace_concat */
	0,					/* sq_inplace_repeat */
};

PyTypeObject PyEdge_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,					/* ob_size */
	"Graph.Edge",			/* tp_name */
	sizeof(TPyEdge),			/* tp_basicsize */
	0,					/* tp_itemsize */
	/* methods */
	(destructor)PyEdge_Dealloc, 		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	(reprfunc)PyEdge_Str,					/* tp_repr */
	&PyEdge_as_number,					/* tp_as_number */
	&PyEdge_as_sequence,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	(reprfunc)PyEdge_Str,					/* tp_str */
	0,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,			/* tp_flags */
	0,					/* tp_doc */
	(traverseproc)PyEdge_Traverse,					/* tp_traverse */
	(inquiry)PyEdge_Clear,					/* tp_clear */
	(richcmpfunc)PyEdge_Richcmp,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,			/* tp_iter */
	0,	/* tp_iternext */
	0,					/* tp_methods */
	0,					/* tp_members */
	0,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,                  /* tp_dictoffset */
	0,                             /* tp_init */
	PyType_GenericAlloc,                               /* tp_alloc */
	0,                               /* tp_new */
	_PyObject_GC_Del,                                  /* tp_free */
};


PYXTRACT_IGNORE int Orange_traverse(TPyOrange *self, visitproc visit, void *arg);
PYXTRACT_IGNORE int Orange_clear(TPyOrange *self);
PYXTRACT_IGNORE void Orange_dealloc(TPyOrange *self);


inline bool hasObjectsOnEdges(PyObject *graph)
{
	PyObject *dict = ((TPyOrange *)graph)->orange_dict;
    PyObject *ooe = NULL;
    if (dict) {
	    ooe = PyDict_GetItemString(dict, "objects_on_edges");
        if (!ooe) {
    	    ooe = PyDict_GetItemString(dict, "objectsOnEdges");
        }
    }
	return ooe && (PyObject_IsTrue(ooe) != 0);
}

inline bool hasObjectsOnEdges(PGraph graph)
{
	PyObject *dict = graph->myWrapper->orange_dict;
    PyObject *ooe = NULL;
    if (dict) {
	    ooe = PyDict_GetItemString(dict, "objects_on_edges");
        if (!ooe) {
    	    ooe = PyDict_GetItemString(dict, "objectsOnEdges");
        }
    }
	return ooe && (PyObject_IsTrue(ooe) != 0);
}

inline bool hasObjectsOnEdges(const TGraph *graph)
{
	PyObject *dict = graph->myWrapper->orange_dict;
    PyObject *ooe = NULL;
    if (dict) {
	    ooe = PyDict_GetItemString(dict, "objects_on_edges");
        if (!ooe) {
    	    ooe = PyDict_GetItemString(dict, "objectsOnEdges");
        }
    }
	return ooe && (PyObject_IsTrue(ooe) != 0);
}

void decrefEdge(double *weights, const int &nEdgeTypes)
{
	if (weights)
		for(double *we = weights, *wee = weights + nEdgeTypes; we != wee; we++)
			if (CONNECTED(*we))
				Py_DECREF(DOUBLE_AS_PYOBJECT(*we));
}

PyObject *PyEdge_New(PGraph graph, const int &v1, const int &v2, double *weights)
{
	TPyEdge *self = PyObject_GC_New(TPyEdge, &PyEdge_Type);
	if (self == NULL)
		return NULL;

	// The object constructor has never been called, so we must initialize it
	// before assigning to it
	self->graph.init();

	self->graph = graph;
	self->v1 = v1;
	self->v2 = v2;
	self->weights = weights;
	self->objectsOnEdges = hasObjectsOnEdges(graph);

	PyObject_GC_Track(self);
	return (PyObject *)self;
}


ABSTRACT(Graph - Orange.network.Graph, Orange)
RECOGNIZED_ATTRIBUTES(Graph, "objects forceMapping force_mapping returnIndices return_indices objectsOnEdges object_on_edges")

int Graph_getindex(TGraph *graph, PyObject *index)
{
	if (PyInt_Check(index)) {
		if (!graph->myWrapper->orange_dict)
			return PyInt_AsLong(index);
		PyObject *fmap = PyDict_GetItemString(graph->myWrapper->orange_dict, "force_mapping");
        if (!fmap) {
            fmap = PyDict_GetItemString(graph->myWrapper->orange_dict, "forceMapping");
        }
		if (!fmap || PyObject_IsTrue(fmap))
			return PyInt_AsLong(index);
	}

	if (graph->myWrapper->orange_dict) {
		PyObject *objs = PyDict_GetItemString(graph->myWrapper->orange_dict, "objects");
		if (objs && (objs != Py_None)) {

			if (PyDict_Check(objs)) {
				PyObject *pyidx = PyDict_GetItem(objs, index);
				if (!pyidx)
					return -1;
				if (!PyInt_Check(pyidx))
					PYERROR(PyExc_IndexError, "vertex index should be an integer", -1);
				return PyInt_AsLong(pyidx);
			}

			PyObject *iter = PyObject_GetIter(objs);
			if (!iter)
				PYERROR(PyExc_IndexError, "Graph.object should be iterable", -1);
			int i = 0;

			for(PyObject *item = PyIter_Next(iter); item; item = PyIter_Next(iter), i++) {
				int cmp = PyObject_Compare(item, index);
				Py_DECREF(item);
				if (PyErr_Occurred())
					return -1;
				if (!cmp) {
					Py_DECREF(iter);
					return i;
				}
			}

			Py_DECREF(iter);
			PYERROR(PyExc_IndexError, "index not found", -1);
		}
	}

	PYERROR(PyExc_IndexError, "invalid index type: should be integer (or 'objects' must be specified)", -1);
}


PyObject *Graph_nodesToObjects(TGraph *graph, const vector<int> &neighbours)
{
	if (graph->myWrapper->orange_dict) {
		PyObject *objs = PyDict_GetItemString(graph->myWrapper->orange_dict, "returnIndices");
		if (!objs || (PyObject_IsTrue(objs) == 0)) {
			objs = PyDict_GetItemString(graph->myWrapper->orange_dict, "objects");
			if (objs && (objs != Py_None)) {
				PyObject *res = PyList_New(neighbours.size());

				if (PyDict_Check(objs)) {
					// This is slow, but can't help...
					int el = 0;
					PyObject *key, *value;
					Py_ssize_t pos = 0;

					while (PyDict_Next(objs, &pos, &key, &value))
						if (!PyInt_Check(value)) {
							Py_DECREF(res);
							PYERROR(PyExc_IndexError, "values in Graph.objects dictionary should be integers", PYNULL);
						}

						for(vector<int>::const_iterator ni(neighbours.begin()), ne(neighbours.end()); ni!=ne; ni++, el++) {
							pos = 0;
							bool set = false;
							while (PyDict_Next(objs, &pos, &key, &value) && !set) {
								if (PyInt_AsLong(value) == *ni) {
									Py_INCREF(key);
									PyList_SetItem(res, el, key);
									set = true;
								}
							}

							if (!set) {
								Py_DECREF(res);
								PyErr_Format(PyExc_IndexError, "'objects' miss the key for vertex %i", *ni);
								return PYNULL;
							}
						}
				}
				else {
					Py_ssize_t el = 0;
					for(vector<int>::const_iterator ni(neighbours.begin()), ne(neighbours.end()); ni!=ne; ni++, el++) {
						PyObject *pyel = PySequence_GetItem(objs, *ni);
						if (!pyel) {
							Py_DECREF(res);
							return PYNULL;
						}
						else
							PyList_SetItem(res, el, pyel);
					}
				}
				return res;
			}
		}
	}

	return convertToPython(neighbours);
}


PyObject *Graph_getitem(PyObject *self, PyObject *args)
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *py1, *py2;
	int v1, v2, type = -1;

	if (   !PyArg_ParseTuple(args, "OO|i", &py1, &py2, &type)
		|| ((v1 = Graph_getindex(graph, py1)) < 0)
		|| ((v2 = Graph_getindex(graph, py2)) < 0))
		return PYNULL;

	if (PyTuple_Size(args) == 2) {
		PGraph graph = PyOrange_AS_Orange(self);
		return PyEdge_New(graph, v1, v2, graph->getEdge(v1, v2));
	}

	else {
		PGraph graph = PyOrange_AS_Orange(self);
		if ((type >= graph->nEdgeTypes) || (type < 0)) {
			PyErr_Format(PyExc_IndexError, "type %s out of range (0-%i)", type, graph->nEdgeTypes);
			return PYNULL;
		}
		double *weights = graph->getEdge(v1, v2);
		if (!weights || !CONNECTED(weights[type]))
			RETURN_NONE
		else {
			if (hasObjectsOnEdges(graph)) {
				PyObject *res = DOUBLE_AS_PYOBJECT(weights[type]);
				Py_INCREF(res);
				return res;
			}
			else
				return PyFloat_FromDouble(weights[type]);
		}
	}
	PyCATCH
}


PyObject *Graph_edgeExists(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(v1, v2[, type])")
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *py1, *py2;
	int v1, v2, type = -1;

	if (   !PyArg_ParseTuple(args, "OO|i", &py1, &py2, &type)
		|| ((v1 = Graph_getindex(graph, py1)) < 0)
		|| ((v2 = Graph_getindex(graph, py2)) < 0))
		return PYNULL;

	if (PyTuple_Size(args) == 2)
		return PyInt_FromLong(graph->getEdge(v1, v2) ? 1 : 0);

	else {
		PGraph graph = PyOrange_AS_Orange(self);
		if ((type >= graph->nEdgeTypes) || (type < 0)) {
			PyErr_Format(PyExc_IndexError, "type %s out of range (0-%i)", type, graph->nEdgeTypes);
			return PYNULL;
		}
		double *weights = graph->getEdge(v1, v2);
		return PyInt_FromLong(!weights || !CONNECTED(weights[type]) ? 0 : 1);
	}
	PyCATCH
}

int Graph_setitem(PyObject *self, PyObject *args, PyObject *item)
{
	PyTRY
		CAST_TO_err(TGraph, graph, -1);
	bool objectsOnEdges = hasObjectsOnEdges(graph);

	PyObject *py1, *py2;
	int v1, v2, type = -1;

	if (   !PyArg_ParseTuple(args, "OO|i", &py1, &py2, &type)
		|| ((v1 = Graph_getindex(graph, py1)) < 0)
		|| ((v2 = Graph_getindex(graph, py2)) < 0))
		return -1;

	if (PyTuple_Size(args) == 3) {
		if ((type >= graph->nEdgeTypes) || (type < 0)) {
			PyErr_Format(PyExc_IndexError, "type %i out of range (0-%i)", type, graph->nEdgeTypes);
			return -1;
		}

		double w;

		bool noConnection = !item || (item == Py_None);

		if (noConnection)
			DISCONNECT(w);
		else
			if (!objectsOnEdges && !PyNumber_ToDouble(item, w))
				PYERROR(PyExc_TypeError, "a number expected for edge weight", -1);

		// we call getOrCreateEdge only after we check all arguments, so we don't end up
		// with a half-created edge
		double *weights = graph->getOrCreateEdge(v1, v2);

		if (objectsOnEdges) {
			if (!noConnection)
				Py_INCREF(item);
			if (CONNECTED(weights[type]))
				Py_DECREF(DOUBLE_AS_PYOBJECT(weights[type]));

			DOUBLE_AS_PYOBJECT(weights[type]) = item;
		}
		else {
			weights[type] = w;
		}

		if (noConnection) {
			double *we, *wee;
			for(we = weights, wee = weights + graph->nEdgeTypes; (we != wee) && !CONNECTED(*we); we++);
			if (we == wee)
				graph->removeEdge(v1, v2);
		}

		return 0;
	}

	else {
		if (!item || (item == Py_None)) {
			if (objectsOnEdges)
				decrefEdge(graph->getEdge(v1, v2), graph->nEdgeTypes);
			graph->removeEdge(v1, v2);
			return 0;
		}

		if (graph->nEdgeTypes == 1) {
			double w;
			if (objectsOnEdges || PyNumber_ToDouble(item, w)) {
				double *weights = graph->getOrCreateEdge(v1, v2);
				if (objectsOnEdges) {
					DOUBLE_AS_PYOBJECT(*weights) = item;
					Py_INCREF(item);
				}
				else
					*weights = w;
				return 0;
			}
		}

		if (PySequence_Check(item)) {
			if (PySequence_Size(item) != graph->nEdgeTypes)
				PYERROR(PyExc_AttributeError, "invalid size of the list of edge weights", -1);

			double *ww = new double[graph->nEdgeTypes];
			double *wwi = ww;
			PyObject *iterator = PyObject_GetIter(item);
			if (iterator) {
				for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
					if (item == Py_None)
						DISCONNECT(*(wwi++));
					else
						if (objectsOnEdges) {
							DOUBLE_AS_PYOBJECT(*wwi++) = item;
						}
						else {
							if (!PyNumber_ToDouble(item, *(wwi++))) {
								Py_DECREF(item);
								Py_DECREF(iterator);
								PyErr_Format(PyExc_TypeError, "invalid number for edge type %i", wwi-ww-1);
								delete ww;
								return -1;
							}
							Py_DECREF(item); // no Py_DECREF if objectsOnEdges!
						}
				}
				Py_DECREF(iterator);
			}

			double *weights = graph->getOrCreateEdge(v1, v2);
			if (objectsOnEdges)
				decrefEdge(weights, graph->nEdgeTypes);
			memcpy(weights, ww, graph->nEdgeTypes * sizeof(double));
			return 0;
		}
	}

	PYERROR(PyExc_AttributeError, "arguments for __setitem__ are [v1, v2, type] = weight|None,  [v1, v2] = list | weight (if nEdgeType=1)", -1);
	PyCATCH_1
}


PyObject *Graph_getNeighbours(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertex[, edgeType])")
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *pyv;
	int vertex, edgeType = -1;
	if (   !PyArg_ParseTuple(args, "O|i:Graph.getNeighbours", &pyv, &edgeType)
		|| ((vertex = Graph_getindex(graph, pyv)) < 0))
		return PYNULL;

	vector<int> neighbours;
	if (PyTuple_Size(args) == 1)
		graph->getNeighbours(vertex, neighbours);
	else
		graph->getNeighbours(vertex, edgeType, neighbours);

	return Graph_nodesToObjects(graph, neighbours);
	PyCATCH
}


PyObject *Graph_getEdgesFrom(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertex[, edgeType])")
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *pyv;
	int vertex, edgeType = -1;
	if (   !PyArg_ParseTuple(args, "O|i:Graph.getNeighbours", &pyv, &edgeType)
		|| ((vertex = Graph_getindex(graph, pyv)) < 0))
		return PYNULL;

	vector<int> neighbours;
	if (PyTuple_Size(args) == 1)
		graph->getNeighboursFrom(vertex, neighbours);
	else
		graph->getNeighboursFrom(vertex, edgeType, neighbours);

	return Graph_nodesToObjects(graph, neighbours);
	PyCATCH
}

PyObject *Graph_addCluster(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertices) -> None")
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *pyv;

	if (!PyArg_ParseTuple(args, "O:Graph.addCluster", &pyv))
		return PYNULL;

	Py_ssize_t size = PyList_Size(pyv);
	int i,j;
	for (i = 0; i < size-1; i++)
	{
		for (j = i+1; j < size; j++)
		{
			int x = PyInt_AsLong(PyList_GetItem(pyv, i));
			int y = PyInt_AsLong(PyList_GetItem(pyv, j));
			//cout << x << " " << y;
			double* weight = graph->getOrCreateEdge(x, y);
			*weight = 1.0;
			//cout << "." << endl;
		}
	}

	RETURN_NONE;
	PyCATCH
}

bool lessLength (const set<int>& s1, const set<int>& s2)
{
	return s1.size() > s2.size();
}

bool moreLength (const vector<int>& s1, const vector<int>& s2)
{
	return s1.size() > s2.size();
}

PyObject *Graph_getConnectedComponents(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_NOARGS, "None -> list of [nodes]")
{
	PyTRY
	CAST_TO(TGraph, graph);
	//cout << "Graph_getConnectedComponents" << endl;
	int node = 0;
	vector<set<int> > components;
	set<int> all;

	while (node < graph->nVertices)
	{
		set<int> component = graph->getConnectedComponent(node);
		components.push_back(component);
		all.insert(component.begin(), component.end());

		while(node < graph->nVertices)
		{
			node++;
			if (all.find(node) == all.end())
				break;
		}
	}
	sort(components.begin(), components.end(), lessLength);

	PyObject* components_list = PyList_New(0);

	ITERATE(vector<set<int> >, si, components) {
		PyObject* component_list = PyList_New(0);

		ITERATE(set<int>, ni, *si) {
			PyObject *nel = Py_BuildValue("i", *ni);
			PyList_Append(component_list, nel);
			Py_DECREF(nel);
		}

		PyList_Append(components_list, component_list);
		Py_DECREF(component_list);
	}

	return components_list;
	PyCATCH
}

PyObject *Graph_getDegreeDistribution(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(distribution)")
{
	PyTRY
		CAST_TO(TGraph, graph);

		PyObject* degrees = PyDict_New();
		PyObject *nsize, *pydegree;

		int v;
		for (v = 0; v < graph->nVertices; v++)
		{
			vector<int> neighbours;
			graph->getNeighbours(v, neighbours);
			nsize = PyInt_FromLong(neighbours.size());

			pydegree = PyDict_GetItem(degrees, nsize); // returns borrowed reference!
			int newdegree = pydegree ? PyInt_AsLong(pydegree) + 1 : 1;

			pydegree = PyInt_FromLong(newdegree);
      PyDict_SetItem(degrees, nsize, pydegree);
      Py_DECREF(pydegree);
		}

		return degrees;
	PyCATCH
}

PyObject *Graph_getDegrees(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "degrees")
{
	PyTRY
		CAST_TO(TGraph, graph);

		PyObject* degrees = PyList_New(graph->nVertices);
		for(int v1 = 0; v1 < graph->nVertices; v1++)
		{
			PyList_SetItem(degrees, v1,  PyInt_FromLong(0));
		}

		vector<int> neighbours;
		for(int v1 = 0; v1 < graph->nVertices; v1++)
		{
			graph->getNeighboursFrom_Single(v1, neighbours);

			ITERATE(vector<int>, ni, neighbours)
			{
				int v1_degree = PyInt_AsLong(PyList_GetItem(degrees, v1));
				int v2_degree = PyInt_AsLong(PyList_GetItem(degrees, *ni));

				v2_degree++;
				v1_degree++;

				PyList_SetItem(degrees, v1,  PyInt_FromLong(v1_degree));
				PyList_SetItem(degrees, *ni, PyInt_FromLong(v2_degree));
			}
		}

		if (!graph->directed) {
			for(int v1 = 0; v1 < graph->nVertices; v1++) {
				int v1_degree = PyInt_AsLong(PyList_GetItem(degrees, v1));
				PyList_SetItem(degrees, v1,  PyInt_FromLong(v1_degree / 2));
			}
		}

		return degrees;
	PyCATCH
}


PyObject *multipleSelectLow(TPyOrange *self, PyObject *pylist, bool reference);

PyObject *Graph_getSubGraph(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(vertices) -> list of [v1, v2, ..., vn]")
{
	PyTRY
	CAST_TO(TGraph, graph);
	//cout << "Graph_getSubGraph" << endl;
	PyObject *vertices;

	if (!PyArg_ParseTuple(args, "O:Graph.getSubGraph", &vertices))
		return PYNULL;

	Py_ssize_t size = PyList_Size(vertices);
	PyList_Sort(vertices);

	TGraph *subgraph = new TGraphAsList(size, graph->nEdgeTypes, graph->directed);
	PGraph wsubgraph = subgraph;

	Py_ssize_t i;
	vector<int> neighbours;
	for (i = 0; i < size; i++) {
		int vertex = PyInt_AsLong(PyList_GetItem(vertices, i));

		graph->getNeighboursFrom_Single(vertex, neighbours);
		ITERATE(vector<int>, ni, neighbours) {
			if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1) {
				int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

				if (index != -1) {
					double* w = subgraph->getOrCreateEdge(i, index);
					double* oldw = graph->getOrCreateEdge(vertex, *ni);
					int j;
					for (j=0; j < subgraph->nEdgeTypes; j++) {
						w[j] = oldw[j];
					}
				}
			}
		}
	}

	PyObject *pysubgraph = WrapOrange(wsubgraph); //WrapNewOrange(subgraph, self->ob_type);

	// set graphs attribut items of type ExampleTable to subgraph
	PyObject *strItems = PyString_FromString("items");
	if (PyObject_HasAttr(self, strItems) == 1) {
		PyObject* items = PyObject_GetAttr(self, strItems);
		/*
		cout << PyObject_IsTrue(items) << endl;
		cout << PyObject_Size(items) << endl;
		cout << graph->nVertices << endl;
		*/
		if (PyObject_IsTrue(items) && PyObject_Size(items) == graph->nVertices) {
			PyObject* selection = multipleSelectLow((TPyOrange *)items, vertices, false);
			Orange_setattrDictionary((TPyOrange *)pysubgraph, strItems, selection, false);
		}
	}
	Py_DECREF(strItems);
	return pysubgraph;
	PyCATCH
}


PyObject *Graph_getSubGraphMergeCluster(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertices) -> list of [v1, v2, ..., vn]")
{
	PyTRY
		CAST_TO(TGraph, graph);

		PyObject *verticesWithout;
		PyObject *vertices = PyList_New(0);

		if (!PyArg_ParseTuple(args, "O:Graph.getSubGraphMergeCluster", &verticesWithout))
			return PYNULL;

		// create an array of vertices to be in a new graph
		int i;
		vector<int> neighbours;
		for (i = 0; i < graph->nVertices; i++)
		{
      if (PySequence_Contains(verticesWithout, PyInt_FromLong(i)) == 0)
			{
        PyObject *nel = Py_BuildValue("i", i);
			  PyList_Append(vertices, nel);
			  Py_DECREF(nel);
      }
    }

		// create new graph without cluster
		Py_ssize_t size = PyList_Size(vertices);
		PyList_Sort(vertices);

		TGraph *subgraph = new TGraphAsList(size + 1, graph->nEdgeTypes, graph->directed);
		PGraph wsubgraph = subgraph;

		for (i = 0; i < size; i++)
		{
			int vertex = PyInt_AsLong(PyList_GetItem(vertices, i));

			graph->getNeighboursFrom_Single(vertex, neighbours);
			ITERATE(vector<int>, ni, neighbours)
			{
				if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1)
				{
					int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

					if (index != -1)
					{
						double* w = subgraph->getOrCreateEdge(i, index);
						*w = 1.0;
					}
				}
			}
		}
		// connect new meta-node with all verties
		int sizeWithout = PyList_Size(verticesWithout);
		for (i = 0; i < sizeWithout; i++)
		{
			int vertex = PyInt_AsLong(PyList_GetItem(verticesWithout, i));

			graph->getNeighbours(vertex, neighbours);
			ITERATE(vector<int>, ni, neighbours)
			{
				if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1)
				{
					int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

					if (index != -1)
					{
						double* w = subgraph->getOrCreateEdge(size, index);
						*w = 1.0;
					}
				}
			}
		}

		PyObject *pysubgraph = WrapOrange(wsubgraph);

		// set graphs attribut items of type ExampleTable to subgraph
    PyObject *strItems = PyString_FromString("items");

		if (PyObject_HasAttr(self, strItems) == 1)
		{
			PyObject* items = PyObject_GetAttr(self, strItems);
      PyObject* selection = multipleSelectLow((TPyOrange *)items, vertices, false);

      PExampleTable graph_table = PyOrange_AsExampleTable(selection);
      TExample *example = new TExample(graph_table->domain, true);
      graph_table->push_back(example);
      Orange_setattrDictionary((TPyOrange *)pysubgraph, strItems, selection, false);
    }

	  Py_DECREF(strItems);

		return pysubgraph;
	PyCATCH
}


PyObject *Graph_getSubGraphMergeClusters(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "List of (vertices) -> list of [v1, v2, ..., vn]")
{
	PyTRY
		CAST_TO(TGraph, graph);

		set<int> verticesWithout;
		PyObject *fullGraphs;
		PyObject *vertices = PyList_New(0);

		if (!PyArg_ParseTuple(args, "O:Graph.getSubGraphMergeClusters", &fullGraphs))
			return PYNULL;

		// create an array of vertices to remove from the graph
		Py_ssize_t sizeFullGraphs = PyList_Size(fullGraphs);
		int i;
		for (i = 0; i < sizeFullGraphs; i++)
		{
			PyObject *fullGraph = PyList_GetItem(fullGraphs, i);
			Py_ssize_t sizeFullGraph = PyList_Size(fullGraph);

			int j;
			for (j = 0; j < sizeFullGraph; j++)
			{
				int vertex = PyInt_AsLong(PyList_GetItem(fullGraph, j));
				verticesWithout.insert(vertex);
			}
		}

		vector<int> neighbours;
		// create an array of vertices to be in a new graph
		for (i = 0; i < graph->nVertices; i++)
		{
			set<int>::iterator it = verticesWithout.find(i);
			if (it == verticesWithout.end())
			{
        PyObject *nel = Py_BuildValue("i", i);
			  PyList_Append(vertices, nel);
			  Py_DECREF(nel);
      }
    }

		// create new graph without cluster
		Py_ssize_t size = PyList_Size(vertices);
		PyList_Sort(vertices);

		TGraph *subgraph = new TGraphAsList(size + sizeFullGraphs, graph->nEdgeTypes, graph->directed);
		PGraph wsubgraph = subgraph;

		for (i = 0; i < size; i++)
		{
			int vertex = PyInt_AsLong(PyList_GetItem(vertices, i));

			graph->getNeighboursFrom_Single(vertex, neighbours);
			ITERATE(vector<int>, ni, neighbours)
			{
				if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1)
				{
					int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

					if (index != -1)
					{
						double* w = subgraph->getOrCreateEdge(i, index);
						*w = 1.0;
					}
				}
			}
		}
		// connect new meta-node with all verties
		for (i = 0; i < sizeFullGraphs; i++)
		{
			PyObject *fullGraph = PyList_GetItem(fullGraphs, i);
			Py_ssize_t sizeFullGraph = PyList_Size(fullGraph);
			int j;
			for (j = 0; j < sizeFullGraph; j++)
			{
				int vertex = PyInt_AsLong(PyList_GetItem(fullGraph, j));
				graph->getNeighbours(vertex, neighbours);

				// connect with old neighbours
				ITERATE(vector<int>, ni, neighbours)
				{
					if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1)
					{
						// vertex to connect with is in new graph
						int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

						if (index != -1)
						{
							double* w = subgraph->getOrCreateEdge(size + i, index);
							*w = 1.0;
						}
					}
					else
					{
						// vertex to connect with is a new meta node
						int k;
						for (k = 0; k < sizeFullGraphs; k++)
						{
							PyObject *fullGraph = PyList_GetItem(fullGraphs, k);

							if (PySequence_Contains(fullGraph, PyInt_FromLong(*ni)) == 1)
							{
								if (k != i)
								{
									double* w = subgraph->getOrCreateEdge(size + i, size + k);
									*w = 1.0;
								}
								break;
							}
						}
					}
				}
			}
		}

		PyObject *pysubgraph = WrapOrange(wsubgraph);

		// set graphs attribut items of type ExampleTable to subgraph
    PyObject *strItems = PyString_FromString("items");

		if (PyObject_HasAttr(self, strItems) == 1)
		{
			PyObject* items = PyObject_GetAttr(self, strItems);
      PyObject* selection = multipleSelectLow((TPyOrange *)items, vertices, false);

      PExampleTable graph_table = PyOrange_AsExampleTable(selection);
			for (i = 0; i < sizeFullGraphs; i++)
			{
				TExample *example = new TExample(graph_table->domain, true);
				graph_table->push_back(example);
			}
      Orange_setattrDictionary((TPyOrange *)pysubgraph, strItems, selection, false);
    }

	  Py_DECREF(strItems);

		return pysubgraph;
	PyCATCH
}


PyObject *Graph_getSubGraphWithout(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertices) -> list of [v1, v2, ..., vn]")
{
	PyTRY
		CAST_TO(TGraph, graph);

		PyObject *verticesWithout;
		PyObject *vertices = PyList_New(0);

		if (!PyArg_ParseTuple(args, "O:Graph.getSubGraphWithout", &verticesWithout))
			return PYNULL;

    int i;
    for (i = 0; i < graph->nVertices; i++)
		{
      if (PySequence_Contains(verticesWithout, PyInt_FromLong(i)) == 0)
			{
        PyObject *nel = Py_BuildValue("i", i);
			  PyList_Append(vertices, nel);
			  Py_DECREF(nel);
      }
    }

		Py_ssize_t size = PyList_Size(vertices);
		PyList_Sort(vertices);

		TGraph *subgraph = new TGraphAsList(size, graph->nEdgeTypes, graph->directed);
		PGraph wsubgraph = subgraph;

		vector<int> neighbours;
		for (i = 0; i < size; i++)
		{
			int vertex = PyInt_AsLong(PyList_GetItem(vertices, i));

			graph->getNeighboursFrom_Single(vertex, neighbours);
			ITERATE(vector<int>, ni, neighbours)
			{
				if (PySequence_Contains(vertices, PyInt_FromLong(*ni)) == 1)
				{
					int index = PySequence_Index(vertices, PyInt_FromLong(*ni));

					if (index != -1)
					{
						double* w = subgraph->getOrCreateEdge(i, index);
						*w = 1.0;
					}
				}
			}
		}

		// set graphs attribut items of type ExampleTable to subgraph
		/*
		TExampleTable *table;
		PExampleTable wtable;

		if (PyObject_HasAttr(self, PyString_FromString("items")) == 1)
		{
			PyObject* items = PyObject_GetAttr(self, PyString_FromString("items"));

			PExampleTable graph_table;
			if (PyArg_ParseTuple(PyTuple_Pack(1,items), "O", &graph_table))
			{

				table = new TExampleTable(graph_table->domain);
				wtable = table;

				for (i = 0; i < size; i++)
				{
					int vertex = PyInt_AsLong(PyList_GetItem(vertices, i));

					graph_table.
				}

				//PyObject_SetAttr((PyObject *)subgraph, PyString_FromString("items"), Py_BuildValue("N", WrapOrange(wtable)));
			}
		}

		return Py_BuildValue("NN", WrapOrange(wsubgraph), WrapOrange(wtable));
		/**/
		return Py_BuildValue("N", WrapOrange(wsubgraph));
	PyCATCH
}


PyObject *Graph_getHubs(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(n) -> HubList")
{
  PyTRY
		int n;

		if (!PyArg_ParseTuple(args, "n:Graph.getHubs", &n))
			return NULL;

		CAST_TO(TGraph, graph);

		int *vertexDegree = new int[graph->nVertices];

		int i;
		for (i=0; i < graph->nVertices; i++)
		{
			vertexDegree[i] = 0;
		}


		vector<int> neighbours;
		for(i = 0; i < graph->nVertices; i++)
		{
			graph->getNeighboursFrom_Single(i, neighbours);

			ITERATE(vector<int>, ni, neighbours)
			{
				vertexDegree[i]++;
				vertexDegree[*ni]++;
			}
		}

		PyObject* hubList = PyList_New(n);

		for (i=0; i < n; i++)
		{
			int j;
			int ndx_max = -1;
			int max = 0;
			for (j=0; j < graph->nVertices; j++)
			{
				if (vertexDegree[j] > max)
				{
					ndx_max = j;
					max = vertexDegree[j];
				}
			}
			//cout << "pow: " << vertexPower[ndx_max] << " ndx: " << ndx_max << endl;

			vertexDegree[ndx_max] = -2;
			PyList_SetItem(hubList, i, PyInt_FromLong(ndx_max));
		}

		delete [] vertexDegree;
		return hubList;
  PyCATCH
}


PyObject *Graph_getEdgesTo(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(vertex[, edgeType])")
{
	PyTRY
		CAST_TO(TGraph, graph);

	PyObject *pyv;
	int vertex, edgeType = -1;
	if (   !PyArg_ParseTuple(args, "O|i:Graph.getNeighbours", &pyv, &edgeType)
		|| ((vertex = Graph_getindex(graph, pyv)) < 0))
		return PYNULL;

	vector<int> neighbours;
	if (PyTuple_Size(args) == 1)
		graph->getNeighboursTo(vertex, neighbours);
	else
		graph->getNeighboursTo(vertex, edgeType, neighbours);

	return Graph_nodesToObjects(graph, neighbours);
	PyCATCH
}


PyObject *Graph_getEdges(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "([edgetype]) -> list of (v1, v2, weights)")
{
	PyTRY
		CAST_TO(TGraph, graph);

	int edgeType = -1;
	if (!PyArg_ParseTuple(args, "|i:Graph.getEdges", &edgeType))
		return PYNULL;

	bool hasType = (PyTuple_Size(args) != 0);
	if (hasType && (edgeType<0) || (edgeType >= graph->nEdgeTypes)) {
		PyErr_Format(PyExc_IndexError, "edge type out of range 0-%i", graph->nEdgeTypes);
		return PYNULL;
	}

	PyObject *res = PyList_New(0);
	vector<int> neighbours;

	for(int v1 = 0; v1 < graph->nVertices; v1++) {
		neighbours.clear();
		if (hasType)
			if (graph->directed) {
				graph->getNeighboursFrom(v1, edgeType, neighbours);
			}
			else {
				graph->getNeighboursFrom_Single(v1, edgeType, neighbours);
			}
		else
			if (graph->directed) {
				graph->getNeighboursFrom(v1, neighbours);
			}
			else {
				graph->getNeighboursFrom_Single(v1, neighbours);
			}

		ITERATE(vector<int>, ni, neighbours) {
			PyObject *nel = Py_BuildValue("ii", v1, *ni);
			PyList_Append(res, nel);
			Py_DECREF(nel);
		}
	}

	return res;
	PyCATCH
}

PyObject *Graph_getNodes(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "neighbours -> list of (v1, v2, weights)")
{
	PyTRY
		CAST_TO(TGraph, graph);

	int noOfNeighbours = -1;
	if (!PyArg_ParseTuple(args, "i:Graph.getNodes", &noOfNeighbours))
		return PYNULL;

	PyObject *res = PyList_New(0);
	vector<int> neighbours;

	for(int v1 = 0; v1 < graph->nVertices; v1++) {
			graph->getNeighbours(v1, neighbours);

      if (neighbours.size() == noOfNeighbours)
      {
			  PyObject *nel = Py_BuildValue("i", v1);
			  PyList_Append(res, nel);
			  Py_DECREF(nel);
		  }
	}

	return res;
	PyCATCH
}

PyObject *Graph_getShortestPaths(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(u, v) -> list of [v1, v2, ..., vn]")
{
	PyTRY
		CAST_TO(TGraph, graph);

	int u, v;
	u = v = -1;
	if (!PyArg_ParseTuple(args, "ii:Graph.getShortestPaths", &u, &v))
		return PYNULL;

	vector<int> path = graph->getShortestPaths(u, v);

	//cout << "vector size: " << neighbours.size() << endl;
	PyObject *res = PyList_New(0);

	ITERATE(vector<int>, ni, path) {
		PyObject *nel = Py_BuildValue("i", *ni);
		PyList_Append(res, nel);
		Py_DECREF(nel);
	}

	return res;
	PyCATCH
}

PyObject *Graph_getDistance(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(u, v) -> distance")
{
	PyTRY
		CAST_TO(TGraph, graph);

	int u, v;
	u = v = -1;
	if (!PyArg_ParseTuple(args, "ii:Graph.getDistance", &u, &v))
		return PYNULL;

	vector<int> path = graph->getShortestPaths(u, v);

	return Py_BuildValue("i", path.size() - 1);
	PyCATCH
}

PyObject *Graph_getDiameter(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "None -> diameter")
{
	PyTRY
		CAST_TO(TGraph, graph);

	  return Py_BuildValue("i", graph->getDiameter());
	PyCATCH
}

PyObject *Graph_getClusters(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "None -> list of clusters")
{
  PyTRY
    //cout << "clustering C++" << endl;
	  /*
	  if (!PyArg_ParseTuple(args, ":NetworkOptimization.getClusters", ))
		  return NULL;
	  */
	  CAST_TO(TGraph, graph);

	  graph->getClusters();
	  //return Py_BuildValue("id", ndx, sqrt(min));
	  RETURN_NONE;
  PyCATCH
}

PyObject *Graph_getClusteringCoefficient(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "None -> clustering_coefficient")
{
  PyTRY
  CAST_TO(TGraph, graph);

  double coef = graph->getClusteringCoefficient();
  return Py_BuildValue("d", coef);

  PyCATCH
}

PyObject *Graph_getLargestFullGraphs(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "None -> list of subgraphs")
{
  PyTRY
	  CAST_TO(TGraph, graph);
    int i;
    vector<int> largestFullgraph;
		vector<vector<int> > fullgraphs;
    for (i = 0; i < graph->nVertices; i++)
    {
      vector<int> nodes;
      vector<int> neighbours;
      nodes.push_back(i);
      graph->getNeighbours(i, neighbours);
	    vector<int> fullgraph = graph->getLargestFullGraphs(nodes, neighbours);

      if (largestFullgraph.size() < fullgraph.size())
      {
        largestFullgraph = fullgraph;
      }

      if (fullgraph.size() > 3)
      {
				fullgraphs.push_back(fullgraph);
      }
    }

		if (fullgraphs.size() == 0)
		{
			fullgraphs.push_back(largestFullgraph);
		}

		sort(fullgraphs.begin(), fullgraphs.end(), moreLength);

		PyObject *pyFullgraphs = PyList_New(0);
    ITERATE(vector<vector<int> >, fg, fullgraphs) {
			PyObject *pyFullgraph = PyList_New(0);

			ITERATE(vector<int>, node, *fg) {
				PyObject *nel = Py_BuildValue("i", *node);
				PyList_Append(pyFullgraph, nel);
				Py_DECREF(nel);
			}

			PyList_Append(pyFullgraphs, pyFullgraph);
		  Py_DECREF(pyFullgraph);
	  }

	  return pyFullgraphs;
  PyCATCH
}
int Graph_len(PyObject *self)
{
  return SELF_AS(TGraph).nVertices;
}

PyObject *GraphAsMatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Graph - Orange.network.GraphAsMatrix, "(nVertices, directed[, nEdgeTypes])")
{
	PyTRY
		int nVertices, directed, nEdgeTypes = 1;
	if (!PyArg_ParseTuple(args, "ii|i", &nVertices, &directed, &nEdgeTypes))
		PYERROR(PyExc_TypeError, "Graph.__new__: number of vertices directedness and optionaly, number of edge types expected", PYNULL);

	return WrapNewOrange(mlnew TGraphAsMatrix(nVertices, nEdgeTypes, directed != 0), type);
	PyCATCH
}


PyObject *GraphAsMatrix__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TGraphAsMatrix, graph)

		return Py_BuildValue("O(Oiiis#)N", getExportedFunction("__pickleLoaderGraphAsMatrix"),
		self->ob_type,
		graph->nVertices,
		graph->nEdgeTypes,
		graph->directed ? 1 : 0,
		graph->edges, graph->msize * sizeof(double),
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderGraphAsMatrix(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, nVertices, nEdgeTypes, directed, packed_edges)")
{
	PyTRY
		PyTypeObject *type;
	int nv, ne, di;
	char *buf;
	int bufSize;

	if (!PyArg_ParseTuple(args, "Oiiis#:__pickleLoaderGraphAsMatrix", &type, &nv, &ne, &di, &buf, &bufSize))
		return NULL;

	TGraphAsMatrix *graph = new TGraphAsMatrix(nv, ne, di != 0);
	memcpy(graph->edges, buf, bufSize);
	return WrapNewOrange(graph, type);
	PyCATCH
}


int GraphAsMatrix_traverse(PyObject *self, visitproc visit, void *arg)
{
	int err = Orange_traverse((TPyOrange *)self, visit, arg);
	if (err || !hasObjectsOnEdges(self))
		return err;

	CAST_TO_err(TGraphAsMatrix, graph, -1);
	for(double *ei = graph->edges, *ee = ei + graph->msize; ei != ee; ei++)
		if (CONNECTED(*ei)) {
			err = visit(DOUBLE_AS_PYOBJECT(*ei), arg);
			if (err)
				return err;
		}

		return 0;
}


void decrefGraph(TGraphAsMatrix &graph)
{
	for(double *ei = graph.edges, *ee = ei + graph.msize; ei != ee; ei++)
		if (CONNECTED(*ei)) {
			Py_DECREF(DOUBLE_AS_PYOBJECT(*ei));
			DISCONNECT(*ei);
		}
}

int GraphAsMatrix_clear(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsMatrix));

	return Orange_clear((TPyOrange *)self);
}


void GraphAsMatrix_dealloc(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsMatrix));

	Orange_dealloc((TPyOrange *)self);
}





PyObject *GraphAsList_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Graph - Orange.network.GraphAsList, "(nVertices, directed[, nEdgeTypes])")
{
	PyTRY
		int nVertices, directed, nEdgeTypes = 1;
	if (!PyArg_ParseTuple(args, "ii|i", &nVertices, &directed, &nEdgeTypes))
		PYERROR(PyExc_TypeError, "Graph.__new__: number of vertices directedness and optionaly, number of edge types expected", PYNULL);

	return WrapNewOrange(mlnew TGraphAsList(nVertices, nEdgeTypes, directed != 0), type);
	PyCATCH
}


PyObject *GraphAsList__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TGraphAsList, graph)

		TCharBuffer buf(1024);

	const int ebs = graph->nEdgeTypes * sizeof(double);
	for(TGraphAsList::TEdge **ei = graph->edges, **ee = graph->edges + graph->nVertices; ei != ee; ei++) {
		for(TGraphAsList::TEdge *eei = *ei; eei; eei = eei->next) {
			buf.writeInt(eei->vertex);
			buf.writeBuf(&(eei->weights), ebs);
		}
		buf.writeInt(-1);
	}

	return Py_BuildValue("O(Oiiis#)N", getExportedFunction("__pickleLoaderGraphAsList"),
		self->ob_type,
		graph->nVertices,
		graph->nEdgeTypes,
		graph->directed ? 1 : 0,
		buf.buf, buf.length(),
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderGraphAsList(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, nVertices, nEdgeTypes, directed, packed_edges)")
{
	PyTRY
		PyTypeObject *type;
	int nv, ne, di;
	char *pbuf;
	int bufSize;

	if (!PyArg_ParseTuple(args, "Oiiis#:__pickleLoaderGraphAsList", &type, &nv, &ne, &di, &pbuf, &bufSize))
		return NULL;

	TCharBuffer buf(pbuf);

	TGraphAsList *graph = new TGraphAsList(nv, ne, di != 0);
	const int ebs = graph->nEdgeTypes * sizeof(double);

	for(TGraphAsList::TEdge **ei = graph->edges, **ee = graph->edges + graph->nVertices; ei != ee; ei++) {
		TGraphAsList::TEdge **last = ei;
		for(int vertex = buf.readInt(); vertex != -1; vertex = buf.readInt()) {
			*last = graph->createEdge(NULL, vertex);
			buf.readBuf(&(*last)->weights, ebs);
			last = &(*last)->next;
		}
	}

	return WrapNewOrange(graph, type);
	PyCATCH
}


int GraphAsList_traverse(PyObject *self, visitproc visit, void *arg)
{
	int err = Orange_traverse((TPyOrange *)self, visit, arg);
	if (err || !hasObjectsOnEdges(self))
		return err;

	CAST_TO_err(TGraphAsList, graph, -1);
	for(TGraphAsList::TEdge **ei = graph->edges, **ee = ei + graph->nVertices; ei != ee; ei++)
		for(TGraphAsList::TEdge *e = *ei; e; e = e->next)
			for(double *w = &e->weights, *we = w + graph->nEdgeTypes; w != we; w++)
				if (CONNECTED(*w)) {
					err = visit(DOUBLE_AS_PYOBJECT(*w), arg);
					if (err)
						return err;
				}

				return 0;
}


void decrefGraph(TGraphAsList &graph)
{
	for(TGraphAsList::TEdge **ei = graph.edges, **ee = ei + graph.nVertices; ei != ee; ei++)
		for(TGraphAsList::TEdge *e = *ei; e; e = e->next)
			for(double *w = &e->weights, *we = w + graph.nEdgeTypes; w != we; w++)
				if (CONNECTED(*w)) {
					Py_DECREF(DOUBLE_AS_PYOBJECT(*w));
					DISCONNECT(*w);
				}
}

int GraphAsList_clear(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsList));

	return Orange_clear((TPyOrange *)self);
}


void GraphAsList_dealloc(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsList));

	Orange_dealloc((TPyOrange *)self);
}






PyObject *GraphAsTree_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Graph - Orange.network.GraphAsTree, "(nVertices, directed[, nEdgeTypes])")
{
	PyTRY
		int nVertices, directed, nEdgeTypes = 1;
	if (!PyArg_ParseTuple(args, "ii|i", &nVertices, &directed, &nEdgeTypes))
		PYERROR(PyExc_TypeError, "Graph.__new__: number of vertices directedness and optionaly, number of edge types expected", PYNULL);

	return WrapNewOrange(mlnew TGraphAsTree(nVertices, nEdgeTypes, directed != 0), type);
	PyCATCH
}


void reduceTree(const TGraphAsTree::TEdge *edge, TCharBuffer &buf, const int &ebs)
{
	if (edge) {
		buf.writeChar(1);
		buf.writeInt(edge->vertex);
		buf.writeBuf(&edge->weights, ebs);
		reduceTree(edge->left, buf, ebs);
		reduceTree(edge->right, buf, ebs);
	}
	else
		buf.writeChar(0);
}


TGraphAsTree::TEdge *readTree(TCharBuffer &buf, const int &ebs, const TGraphAsTree &graph)
{
	if (buf.readChar()) {
		TGraphAsTree::TEdge *edge = graph.createEdge(buf.readInt());
		buf.readBuf(&(edge)->weights, ebs);
		edge->left = readTree(buf, ebs, graph);
		edge->right = readTree(buf, ebs, graph);
		return edge;
	}
	else
		return NULL;
}


PyObject *GraphAsTree__reduce__(PyObject *self)
{
	PyTRY
		CAST_TO(TGraphAsTree, graph)

		TCharBuffer buf(1024);

	const int ebs = graph->nEdgeTypes * sizeof(double);
	for(TGraphAsTree::TEdge **ei = graph->edges, **ee = graph->edges + graph->nVertices; ei != ee; ei++)
		reduceTree(*ei, buf, ebs);

	return Py_BuildValue("O(Oiiis#)N", getExportedFunction("__pickleLoaderGraphAsTree"),
		self->ob_type,
		graph->nVertices,
		graph->nEdgeTypes,
		graph->directed ? 1 : 0,
		buf.buf, buf.length(),
		packOrangeDictionary(self));
	PyCATCH
}


PyObject *__pickleLoaderGraphAsTree(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, nVertices, nEdgeTypes, directed, packed_edges)")
{
	PyTRY
		PyTypeObject *type;
	int nv, ne, di;
	char *pbuf;
	int bufSize;

	if (!PyArg_ParseTuple(args, "Oiiis#:__pickleLoaderGraphAsTree", &type, &nv, &ne, &di, &pbuf, &bufSize))
		return NULL;

	TCharBuffer buf(pbuf);

	TGraphAsTree *graph = new TGraphAsTree(nv, ne, di != 0);
	const int ebs = graph->nEdgeTypes * sizeof(double);

	for(TGraphAsTree::TEdge **ei = graph->edges, **ee = graph->edges + graph->nVertices; ei != ee; ei++)
		*ei = readTree(buf, ebs, *graph);

	return WrapNewOrange(graph, type);
	PyCATCH
}



int traverse(TGraphAsTree::TEdge *edge, visitproc visit, void *arg, const int nEdgeTypes)
{
	int err;

	for(double *w = &edge->weights, *we = w + nEdgeTypes; w != we; w++)
		if (CONNECTED(*w)) {
			err = visit(DOUBLE_AS_PYOBJECT(*w), arg);
			if (err)
				return err;
		}

		err = edge->left ? traverse(edge->left, visit, arg, nEdgeTypes) : 0;
		if (!err)
			err = edge->right ? traverse(edge->right, visit, arg, nEdgeTypes) : 0;

		return err;
}

int GraphAsTree_traverse(PyObject *self, visitproc visit, void *arg)
{
	int err = Orange_traverse((TPyOrange *)self, visit, arg);
	if (err || !hasObjectsOnEdges(self))
		return err;

	CAST_TO_err(TGraphAsTree, graph, -1);
	for(TGraphAsTree::TEdge **ei = graph->edges, **ee = ei + graph->nVertices; ei != ee; ei++)
		if (*ei) {
			err = traverse(*ei, visit, arg, graph->nEdgeTypes);
			if (err)
				return err;
		}

		return 0;
}


void decrefGraph(TGraphAsTree::TEdge *edge, const int &nEdgeTypes)
{
	for(double *w = &edge->weights, *we = w + nEdgeTypes; w != we; w++)
		if (CONNECTED(*w)) {
			Py_DECREF(DOUBLE_AS_PYOBJECT(*w));
			DISCONNECT(*w);
		}

		if (edge->left)
			decrefGraph(edge->left, nEdgeTypes);
		if (edge->right)
			decrefGraph(edge->right, nEdgeTypes);
}


void decrefGraph(TGraphAsTree &graph)
{
	for(TGraphAsTree::TEdge **ei = graph.edges, **ee = ei + graph.nVertices; ei != ee; ei++)
		if (*ei)
			decrefGraph(*ei, graph.nEdgeTypes);
}

int GraphAsTree_clear(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsTree));

	return Orange_clear((TPyOrange *)self);
}


void GraphAsTree_dealloc(PyObject *self)
{
	if (hasObjectsOnEdges(self))
		decrefGraph(SELF_AS(TGraphAsTree));

	Orange_dealloc((TPyOrange *)  self);
}


#include "lib_components.px"
