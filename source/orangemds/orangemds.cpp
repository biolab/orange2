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

#include "orange_api.hpp"
#ifdef _MSC_VER
  #ifndef ORANGEMDS_EXPORTS
    #define ORANGEMDS_API __declspec(dllexport)
  #else
    #define ORANGEMDS_API __declspec(dllimport)
  #endif
#else
  #define ORANGEMDS_API
#endif

#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define MAX(a,b) (a>b)? a: b
#include<memory.h>

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

    for(int i=0;i<(*dim1);i++){
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

	for(int i=0;i<len;i++)
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

#include "px/initialization.px"

#endif
