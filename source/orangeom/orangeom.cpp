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

#include "orangeom_globals.hpp"
#include "orange_api.hpp"
#include "cls_orange.hpp"
#include "orvector.hpp"
#include "vectortemplates.hpp"


#include "px/externs.px"

/*********************SOM**********************/

#include "som.hpp"

C_CALL(SOMLearner, Learner, "([examples[, weight=]]) -/-> Classifier")
C_NAMED(SOMClassifier, Classifier, "")
C_NAMED(SOMMap, Orange, "")
C_NAMED(SOMNode, Orange, "")

PyObject *SOMNode_getDistance(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->float")
{
	PyTRY
	PExample ex;
	if(!PyArg_ParseTuple(args, "O&:getDistance", cc_Example, &ex))
		return NULL;
	float res=SELF_AS(TSOMNode).getDistance(ex.getReference());
	return Py_BuildValue("f", res);
	PyCATCH
}

PyObject *SOMClassifier_getWinner(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->SOMNode")
{
	PyTRY
	PExample ex;
	if(!PyArg_ParseTuple(args, "O&:getWinner", cc_Example, &ex))
		return NULL;
	return WrapOrange(SELF_AS(TSOMClassifier).getWinner(ex.getReference()));
	PyCATCH
}

PyObject *SOMClassifier_getError(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples)->float")
{
	PyTRY
	PExampleGenerator egen;
	if(!PyArg_ParseTuple(args, "O&", cc_ExampleGenerator, &egen))
		return NULL;
	float res=SELF_AS(TSOMClassifier).getError(egen);
	return Py_BuildValue("f", res);
	PyCATCH
}

PyObject *SOMMap_getWinner(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->SOMNode")
{
	PyTRY
	PExample ex;
	if(!PyArg_ParseTuple(args, "O&:getWinner", cc_Example, &ex))
		return NULL;
	return WrapOrange(SELF_AS(TSOMMap).getWinner(ex.getReference()));
	PyCATCH
}

extern ORANGEOM_API TOrangeType PyOrSOMNode_Type;

PSOMNodeList PSOMNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::P_FromArguments(arg); }
PyObject *SOMNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_FromArguments(type, arg); }
PyObject *SOMNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of SOMNode>)") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_new(type, arg, kwds); }
PyObject *SOMNodeList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_getitem(self, index); }
int       SOMNodeList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_setitem(self, index, item); }
PyObject *SOMNodeList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_getslice(self, start, stop); }
int       SOMNodeList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_setslice(self, start, stop, item); }
int       SOMNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_len(self); }
PyObject *SOMNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_richcmp(self, object, op); }
PyObject *SOMNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_concat(self, obj); }
PyObject *SOMNodeList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_repeat(self, times); }
PyObject *SOMNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_str(self); }
PyObject *SOMNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_str(self); }
int       SOMNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_contains(self, obj); }
PyObject *SOMNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(SOMNode) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_append(self, item); }
PyObject *SOMNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> int") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_count(self, obj); }
PyObject *SOMNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> SOMNodeList") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_filter(self, args); }
PyObject *SOMNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> int") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_index(self, obj); }
PyObject *SOMNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_insert(self, args); }
PyObject *SOMNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_native(self); }
PyObject *SOMNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> SOMNode") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_pop(self, args); }
PyObject *SOMNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_remove(self, obj); }
PyObject *SOMNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_reverse(self); }
PyObject *SOMNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_sort(self, args); }



/*************** MDS ***************/

#include "mds.hpp"

C_NAMED(MDS, Orange, "(distanceMatrix [dim, points])->MDS")
BASED_ON(StressFunc, Orange)
C_CALL(KruskalStress, StressFunc, "(float, float[,float])->float")
C_CALL(SammonStress, StressFunc, "(float, float[,float])->float")
C_CALL(SgnSammonStress, StressFunc, "(float, float[,float])->float")
C_CALL(SgnRelStress, StressFunc, "(float, float[,float])->float")
C_CALL(StressFunc_Python, StressFunc,"")

PyObject *mysetCallbackFunction(PyObject *self, PyObject *args)
{ PyObject *func;
  if (!PyArg_ParseTuple(args, "O", &func)) {
    PyErr_Format(PyExc_TypeError, "callback function for '%s' expected", self->ob_type->tp_name);
    Py_DECREF(self);
    return PYNULL;
  }
  else if (!PyCallable_Check(func)) {
    PyErr_Format(PyExc_TypeError, "'%s' object is not callable", func->ob_type->tp_name);
    Py_DECREF(self);
    return PYNULL;
  }

  PyObject_SetAttrString(self, "__callback", func);
  return self;
}

PyObject* StressFunc_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrStressFunc_Type)
    return mysetCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(), type), args);
  else
    return WrapNewOrange(mlnew TStressFunc_Python(), type);
}

PyObject *MDS_new(PyTypeObject *type, PyObject *args) BASED_ON(Orange, "(dissMatrix[, dim, points])")
{
    PyTRY
    int dim=2;
    PSymMatrix matrix;
    PFloatListList points;
    if(!PyArg_ParseTuple(args, "O&|iO&", cc_SymMatrix, &matrix, &dim, cc_FloatListList, &points))
        return NULL;
    
    PMDS mds=mlnew TMDS(matrix, dim);
    if(points && points->size()==matrix->dim)
        mds->points=points;
    else{
        PRandomGenerator rg=mlnew TRandomGenerator();
        for(int i=0;i<mds->n; i++)
            for(int j=0; j<mds->dim; j++)
                mds->points->at(i)->at(j)=rg->randfloat();
    }

    return WrapOrange(mds);
    PyCATCH
}

PyObject *MDS_SMACOFstep(PyTypeObject  *self) PYARGS(METH_NOARGS, "()")
{
    PyTRY
    SELF_AS(TMDS).SMACOFstep();
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_getDistance(PyTypeObject *self) PYARGS(METH_NOARGS, "()")
{
    PyTRY
    SELF_AS(TMDS).getDistances();
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_getStress(PyTypeObject *self, PyObject *args) PYARGS(METH_VARARGS, "([stressFunc=SgnRelStress])")
{
    PyTRY
    PStressFunc sf;
    PyObject *callback=NULL;
    if(PyTuple_Size(args)==1){
		/*
        if(!PyArg_ParseTuple(args, "O&", cc_StressFunc, &sf))
            if(!(PyArg_ParseTuple(args, "O", &callback) &&
                (sf=PyOrange_AsStressFunc(mysetCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(),
                (PyTypeObject*)&PyOrStressFunc_Type), args)))))
                return NULL;
				*/
		sf=PyOrange_AsStressFunc(StressFunc_new((PyTypeObject*)&PyOrStressFunc_Type, args, NULL));
        SELF_AS(TMDS).getStress(sf);
    }else
        SELF_AS(TMDS).getStress(mlnew TSgnRelStress());
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_optimize(PyObject* self, PyObject* args, PyObject* kwds) PYARGS(METH_VARARGS, "(numSteps[, stressFunc=orangemds.SgnRelStress, progressCallback=None])->None")
{
    PyTRY
    int iter;
    float eps=1e-3f;
    PProgressCallback callback;
    PStressFunc stress;
    PyObject *pyStress=NULL;
    if(!PyArg_ParseTuple(args, "i|O&f", &iter, cc_StressFunc, &stress, &eps))
        if(PyArg_ParseTuple(args, "i|Of", &iter, &pyStress, &eps) && pyStress){
            PyObject *arg=Py_BuildValue("(O)", pyStress);
            stress=PyOrange_AsStressFunc(StressFunc_new((PyTypeObject*)&PyOrStressFunc_Type, arg, NULL));
        } else
            return NULL;
            
    SELF_AS(TMDS).optimize(iter, stress, eps);
    RETURN_NONE;
    PyCATCH
}
PyObject *KruskalStress_call(PyTypeObject *self, PyObject *args)
{
    PyTRY
    float cur, cor, w;
    if(!PyArg_ParseTuple(args, "ff|f:KruskalStress.__call__", &cur, &cor, &w))
        return NULL;
    if(PyTuple_Size(args)==2)
        return Py_BuildValue("f",SELF_AS(TKruskalStress).operator ()(cur,cor));
    else
        return Py_BuildValue("f",SELF_AS(TKruskalStress).operator ()(cur, cor, w));
    PyCATCH
}

PyObject *SammonStress_call(PyTypeObject *self, PyObject *args)
{
    PyTRY
    float cur, cor, w;
    if(!PyArg_ParseTuple(args, "ff|f:SammonStress.__call__", &cur, &cor, &w))
        return NULL;
    if(PyTuple_Size(args)==2)
        return Py_BuildValue("f",SELF_AS(TSammonStress).operator ()(cur,cor));
    else
        return Py_BuildValue("f",SELF_AS(TSammonStress).operator ()(cur, cor, w));
    PyCATCH
}

PyObject *SgnSammonStress_call(PyTypeObject *self, PyObject *args)
{
    PyTRY
    float cur, cor, w;
    if(!PyArg_ParseTuple(args, "ff|f:SgnSammonStress.__call__", &cur, &cor, &w))
        return NULL;
    if(PyTuple_Size(args)==2)
        return Py_BuildValue("f",SELF_AS(TSgnSammonStress).operator ()(cur,cor));
    else
        return Py_BuildValue("f",SELF_AS(TSgnSammonStress).operator ()(cur, cor, w));
    PyCATCH
}

PyObject *SgnRelStress_call(PyTypeObject *self, PyObject *args)
{
    PyTRY
    float cur, cor, w;
    if(!PyArg_ParseTuple(args, "ff|f:SgnRelStress.__call__", &cur, &cor, &w))
        return NULL;
    if(PyTuple_Size(args)==2)
        return Py_BuildValue("f",SELF_AS(TSgnRelStress).operator ()(cur,cor));
    else
        return Py_BuildValue("f",SELF_AS(TSgnRelStress).operator ()(cur, cor, w));
    PyCATCH
}

extern ORANGEOM_API TOrangeType PyFloatList_Type;

PFloatListList PFloatListList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::P_FromArguments(arg); }
PyObject *FloatListList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_FromArguments(type, arg); }
PyObject *FloatListList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of FloatList>)") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_new(type, arg, kwds); }
PyObject *FloatListList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_getitem(self, index); }
int       FloatListList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_setitem(self, index, item); }
PyObject *FloatListList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_getslice(self, start, stop); }
int       FloatListList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_setslice(self, start, stop, item); }
int       FloatListList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_len(self); }
PyObject *FloatListList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_richcmp(self, object, op); }
PyObject *FloatListList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_concat(self, obj); }
PyObject *FloatListList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_repeat(self, times); }
PyObject *FloatListList_str(TPyOrange *self) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_str(self); }
PyObject *FloatListList_repr(TPyOrange *self) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_str(self); }
int       FloatListList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_contains(self, obj); }
PyObject *FloatListList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(FloatList) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_append(self, item); }
PyObject *FloatListList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> int") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_count(self, obj); }
PyObject *FloatListList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FloatListList") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_filter(self, args); }
PyObject *FloatListList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> int") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_index(self, obj); }
PyObject *FloatListList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_insert(self, args); }
PyObject *FloatListList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_native(self); }
PyObject *FloatListList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> FloatList") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_pop(self, args); }
PyObject *FloatListList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_remove(self, obj); }
PyObject *FloatListList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_reverse(self); }
PyObject *FloatListList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_sort(self, args); }


bool initorangeomExceptions()
{ return true; }

void gcorangeomUnsafeStaticInitialization()
{}

#include "px/externs.px"

#include "px/orangeom.px"

#include "px/initialization.px"
