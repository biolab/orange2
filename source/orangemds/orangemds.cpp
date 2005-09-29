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

#ifndef ORANGEMDS_CPP
#define ORANGEMDS_CPP


#include "orangemds_globals.hpp"
#include "mds.hpp"
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define MAX(a,b) (a>b)? a: b
#include<memory.h>

#include "px/externs.px"
#include "vectortemplates.hpp"
#include "orvector.hpp"
#include "cls_orange.hpp"
#include "random.hpp"


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
	
    //cout <<"MDS Create 1"<<endl;
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

PyObject *MDS_getDistances(PyTypeObject *self) PYARGS(METH_NOARGS, "()")
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
        if(!PyArg_ParseTuple(args, "O&", cc_StressFunc, &sf))
            if(!(PyArg_ParseTuple(args, "O", &callback) &&
				(sf=PyOrange_AsStressFunc(mysetCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(),
				(PyTypeObject*)&PyOrStressFunc_Type), args)))))
				return NULL;
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
			stress=PyOrange_AsStressFunc(mysetCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(),
			(PyTypeObject*)&PyOrStressFunc_Type), arg));
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


	
double *getArray(PyObject *obj, int *dim1, int *dim2)
{
    (*dim1)=PySequence_Size(obj);
    PyObject *b,*a=PySequence_GetItem(obj,0);
    if(!PySequence_Check(a)){
        PYERROR(PyExc_TypeError, "Not a sequence", false);
        return NULL;
    }
    (*dim2)=PySequence_Size(a);
    Py_DECREF(a);
    double *arr=NULL;
    for(int i=0;i<10000;i++)
        arr=NULL;
    //cout <<"dim:"<<(*dim1)<<" "<<(*dim2)<<endl;
    arr=(double*) malloc((*dim1)*(*dim2)*sizeof(double));
    //cout <<"allocated"<<endl;
    if(!arr){
        PYERROR(PyExc_MemoryError,"Memory to low", false);
        return NULL;
    }

    for(i=0;i<(*dim1);i++){
        ////cout << i<<"a ";
        a=PySequence_GetItem(obj,i);
        if(PySequence_Size(a)!=(*dim2)){
            PYERROR(PyExc_TypeError, "Sequence to short",false);
            Py_DECREF(a);
            free(arr);
            return NULL;
        }
        ////cout << i<<"b ";
        for(int j=0;j<(*dim2);j++){
            b=PySequence_GetItem(a,j);
            if(!PyFloat_Check(b)){
                PYERROR(PyExc_TypeError, "Float expected", false);
                Py_DECREF(b);
                Py_DECREF(a);
                free(arr);
                return NULL;
            }
            arr[i*(*dim2)+j]=PyFloat_AS_DOUBLE(b);
            Py_DECREF(b);
        }
        Py_DECREF(a);
    }
    //cout << "array got "<<arr<<endl;
    return arr;
}

void setArray(double* arr, PyObject *obj, int dim1, int dim2)
{
    PyObject *b, *f;
    for(int i=0;i<dim1;i++){
        b=PySequence_GetItem(obj,i);
        for(int j=0;j<dim2;j++){
            f=PyFloat_FromDouble(arr[i*dim2+j]);
            ////cout<<"set "<<i<<" "<<j<<" "<<arr[i*dim2+j]<<endl;
            PySequence_SetItem(b, j, f);
            Py_DECREF(f);
        }
        Py_DECREF(b);
    }
    //cout<<"array set "<<arr<<endl;
}

PyObject *getDistance(PyObject*, PyObject *args) PYARGS(METH_VARARGS, "(points, dist)")
{
    PyTRY
    PyObject *points, *dist;
    int dim;
    int len,len1, lenPoints;

    if(!PyArg_ParseTuple(args, "OO:getDistances", &points, &dist))
        return NULL;
    if(!PySequence_Check(points) && !PySequence_Check(dist)){
        PYERROR(PyExc_TypeError, "points and distances shoud be given as a sequence", false);
        return NULL;
    }
    double *arr=getArray(points, &lenPoints, &dim);
    if(!arr)
        return NULL;

    double *arrDist=getArray(dist, &len, &len1);
    if(!arrDist)
        return NULL;

    //cout <<"dim a:"<<len<<" "<<len1<<endl;
    if(!(len==lenPoints && len==len1)){
        PYERROR(PyExc_TypeError, "points and distances not of equal length", false);
        free(arr);
        free(arrDist);
        return NULL;
    }

    double d;
    for(int i=0;i<len;i++){
        arrDist[i*len+i]=0.0;
        for(int j=0;j<i;j++){
            d=0;
            for(int k=0;k<dim;k++){
                ////cout<<"d "<<dim<<endl;
                d+=pow(arr[i*dim+k]-arr[j*dim+k],2);
            }
            d=pow(d,0.5);
            ////cout << i<<" "<<j<<" "<<d<<endl;
            arrDist[i*len+j]=d;
            ////cout << "i "<<i*len+j<<endl;
            arrDist[j*len+i]=d;
            ////cout << "j "<<j*len+i<<endl;
		}
	}


    setArray(arrDist,dist, len, len);
    free(arr);
    free(arrDist);
    Py_INCREF(dist);
    //cout << "out getDistance"<<endl;
    return dist;
	PyCATCH
}

PyObject *getStress(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(diss ,dist, stress [,stressFunction])->stressList")
{
    PyTRY
    PyObject *diss, *dist, *pystress;
    int len1, len2, len3, len4,len5,len6, stressFunc=3;
    if(!PyArg_ParseTuple(args, "OOO|i:getStress", &diss, &dist, &pystress, &stressFunc))
        return NULL;
    if(!PySequence_Check(dist) && !PySequence_Check(diss) && PySequence_Check(pystress)){
        PYERROR(PyExc_TypeError, "dist, diss and stress must be sequences", false);
        return NULL;
    }

    double *arrDiss=getArray(diss, &len1, &len2);
    if(!arrDiss)
        return NULL;

    double *arrDist=getArray(dist, &len3, &len4);
    if(!arrDist){
        free(arrDiss);
        return NULL;
    }

    double *arrStress=getArray(dist, &len5, &len6);
    if(!arrDist){
        free(arrDiss);
        free(arrDist);
        return NULL;
    }

    if(!(len1==len2 && len1==len3 && len1==len4 && len1==len5 && len1==len6)){
        PYERROR(PyExc_TypeError, "diss, dist, and stress not of equal lenght", false);
        free(arrDiss);
        free(arrDist);
        free(arrStress);
        return NULL;
    }

    PyObject *listStress=PyList_New((len1-1)*(len1)/2);
    double stress=0;
	int index=0;
    for(int i=0;i<len1;i++)
        for(int j=0;j<i;j++){
            switch(stressFunc){
                case 0:stress=(arrDist[i*len1+j]-arrDiss[i*len1+j])*(arrDist[i*len1+j]-arrDiss[i*len1+j]);
                break;
                case 1:stress=(arrDist[i*len1+j]-arrDiss[i*len1+j])*(arrDist[i*len1+j]-arrDiss[i*len1+j])/(MAX(1e-6,arrDist[i*len1+j]));
                break;
                case 2:stress=(arrDist[i*len1+j]-arrDiss[i*len1+j])/(MAX(1e-6,arrDist[i*len1+j]));
                break;
                case 3:stress=(arrDist[i*len1+j]-arrDiss[i*len1+j])/(MAX(1e-6,arrDiss[i*len1+j]));
                break;
                default: stress=(arrDist[i*len1+j]-arrDiss[i*len1+j])/(MAX(1e-6,arrDiss[i*len1+j]));
            }
            ////cout<<stress<<" "<<stressFunc<<endl;
            PyObject *o=Py_BuildValue("[f(ii)]",stress, i, j);
            PyList_SET_ITEM(listStress, index++,o);
            //Py_DECREF(o);
            arrStress[i*len1+j]=stress;
        }

    setArray(arrStress, pystress, len1, len1);
	PyList_Sort(listStress);

    free(arrDist);
    free(arrDiss);
    free(arrStress);
 
	//Py_INCREF(listStress);
    //cout << "out getStress"<<endl;
    return listStress;
	PyCATCH
}

PyObject *SMACOFStep(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(dist, diss, points)")
{
    PyTRY
    PyObject *dist, *diss, *points;
    int len,len2,len3,len4, len5, dim;

    if(!PyArg_ParseTuple(args, "OOO:SMACOFStep", &dist, &diss, &points))
        return NULL;

    if(!PySequence_Check(dist) && !PySequence_Check(diss) && ! PySequence_Check(points)){
        PYERROR(PyExc_TypeError, "Not a sequence", false);
        return NULL;
	}

    double *arrDist=getArray(dist, &len, &len2);
    if(!arrDist)
        return NULL;

    double *arrDiss=getArray(diss, &len3, &len4);
    if(!arrDiss){
        free(arrDist);
        return NULL;
	}

    double *arrPoints=getArray(points, &len5, &dim);
    if(!arrPoints){
        free(arrDist);
        free(arrDiss);
        return NULL;
    }

    if(!(len==len2 && len==len3 && len==len4 && len==len5)){
        PYERROR(PyExc_TypeError, "Sequences not of equal length", false);
        free(arrDist);
        free(arrDiss);
        free(arrPoints);
        return NULL;
    }
	double *R=(double*) malloc(len*len*sizeof(double));
	if(!R){
		PYERROR(PyExc_MemoryError, "Memory low", false);
		free(arrDiss);
		free(arrDist);
		free(arrPoints);
		return NULL;
	}
	memset(R,0,len*len*sizeof(double));

	double *newPoints=(double*) malloc(len*dim*sizeof(double));
	if(!newPoints){
		PYERROR(PyExc_MemoryError, "Memory low", false);
		free(arrDiss);
		free(arrDist);
		free(arrPoints);
		free(R);
		return NULL;
	}
	memset(R,0,len*len*sizeof(double));
    //cout << "alloc SMACOF"<<endl;
	double s,t,sum=0;
	for(int i=0;i<len;i++){
		sum=0.0;
		for(int j=0;j<len;j++){
			if(j!=i){
				if(arrDist[i*len+j]>1e-6)
					s=1.0/arrDist[i*len+j];
				else
					s=0.0;
				t=arrDiss[i*len+j]*s;
				R[i*len+j]=-t;
                ////cout<<"R/ "<< -t<<endl;
				sum+=t;
			}
		}
		R[i*len+i]=sum;
        ////cout<<"R "<< sum<<endl;
	}

    //cout<<"R"<<endl;

	for(i=0;i<len;i++)
		for(int j=0;j<dim;j++){
			sum=0.0;
			for(int k=0;k<len;k++)
				sum+=R[i*len+k]*arrPoints[k*dim+j];
			newPoints[i*dim+j]=sum/len;
            ////cout <<"Sum: "<<sum<<endl;
		}
    //cout << "sssss"<<endl;
	setArray(newPoints, points, len, dim);
	free(arrDiss);
	free(arrDist);
	free(arrPoints);
	free(R);
	free(newPoints);

    Py_INCREF(points);
    //cout << "out SMACOF"<<endl;
    return points;
	PyCATCH
}


bool initorangemdsExceptions()
{ return true; }

void gcorangemdsUnsafeStaticInitialization()
{
}

#include "px/externs.px"

#include "px/orangemds.px"

#include "px/initialization.px"

#endif
