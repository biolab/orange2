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



#include "orange_api.hpp"

PyObject *dist(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(p, t)")
{
  PyObject *l1 = PyTuple_GET_ITEM(args, 0);
  PyObject *l2 = PyTuple_GET_ITEM(args, 1);
  double dist = 0;
  for(Py_ssize_t i = 0, e = PyList_Size(l1); i != e; i++) {
    const double d = PyFloat_AsDouble(PyList_GET_ITEM(l1, i)) - PyFloat_AsDouble(PyList_GET_ITEM(l2, i));
    dist += d*d;
  }
  return PyFloat_FromDouble(sqrt(dist));
}


PyObject *star(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(t, tri)")
{
  int t;
  PyObject *tri;
  PyArg_ParseTuple(args, "iO", &t, &tri);

  PyObject *res = PyList_New(0);
  for(Py_ssize_t i = 0, e = PyList_GET_SIZE(tri); i != e; i++) {
    const PyObject *lel = PyList_GET_ITEM(tri, i);
    Py_ssize_t j, je;
    for (j = 0, je = PyList_GET_SIZE(lel);
         (j != je) && (PyInt_AS_LONG(PyList_GET_ITEM(lel, j)) != t);
         j++);
    if (j != je)
      PyList_Append(res, const_cast<PyObject *>(lel));
  }

  return res;
}


#include "numeric_interface.hpp"
extern "C" {
#include "../qhull/qhull.h"
#include "../qhull/qset.h"		/* for FOREACHneighbor_() */
#include "../qhull/poly.h"		/* for qh_vertexneighbors() */
}

PyObject *qhull(PyObject *, PyObject *arg) PYARGS(METH_O, "(array) -> ?")
{
  if (!isSomeNumeric_wPrecheck(arg))
    PYERROR(PyExc_AttributeError, "numeric array expected", PYNULL);

  PyArrayObject *array = (PyArrayObject *)(arg);
  if (array->nd != 2)
    PYERROR(PyExc_AttributeError, "two-dimensional array expected", NULL);

  char a = getArrayType(array);
  if (getArrayType(array) != 'd')
    PYERROR(PyExc_AttributeError, "an array of doubles expected", NULL);

  if (!moduleNumpy)
    PYERROR(PyExc_SystemError, "this function need module Numeric", NULL);
    
  PyObject *moduleDict = PyModule_GetDict(moduleNumpy);
  PyObject *mzeros = PyDict_GetItemString(moduleDict, "zeros");
  if (!mzeros)
    PYERROR(PyExc_AttributeError, "numeric module has no function 'zeros'", PYNULL);

  const int &npoints = array->dimensions[0];
  const int &dimension = array->dimensions[1];
  const int &strideRow = array->strides[0];
  const int &strideCol = array->strides[1];
    
	coordT *points = new coordT[npoints*dimension];
	coordT *pointsi = points;
	char *rowPtr = (char *)array->data;
	for(int pi = npoints; pi--; rowPtr += strideRow) {
	  char *elPtr = rowPtr;
	  for(int di = dimension; di--; pointsi++, elPtr += strideCol)
	    *pointsi = *(double *)elPtr;
	}
	
	boolT ismalloc = False;	/* True if qhull should free points in qh_freeqhull() or reallocation */
	int exitcode = qh_new_qhull(dimension, npoints, points, ismalloc, "qhull d Qbb QJ", stdout, stdout);

	delete points;
	
	if (exitcode)
	  switch(exitcode) {
	    case qh_ERRinput: PyErr_BadInternalCall (); return NULL;
      case qh_ERRsingular: PYERROR(PyExc_ArithmeticError, "qhull singular input data", PYNULL);
      case qh_ERRprec: PYERROR(PyExc_ArithmeticError, "qhull precision error", PYNULL);
      case qh_ERRmem: PyErr_NoMemory(); return NULL;
      case qh_ERRqhull: PYERROR(PyExc_StandardError, "qhull internal error", PYNULL);
      default: PYERROR(PyExc_StandardError, "unidentified error in qhull", PYNULL);
    }
  
 	facetT *facet;
  vertexT *vertex, **vertexp;
  int nFacets = 0;
  FORALLfacets
    nFacets++;
  
  PyObject *facets = PyObject_CallFunction(mzeros, "(ii)s", nFacets, dimension+1, "i");
  if (!facets)
    return NULL;

  int *facetsdi = (int *)((PyArrayObject *)facets)->data;
  FORALLfacets
  	FOREACHvertex_(facet->vertices)
		  *facetsdi++ = qh_pointid(vertex->point);

	qh_freeqhull(qh_ALL);
	
	return facets;
}



void qing_f(int ndimension, // dimension of the space
            int npoints,    // number of points
            double *points, // points in which the function is sampled, flat array of size npoints*ndimension
            double *values, // function values in the above points
            int nsimplices, // number of simplices (is this the correct plural? :)
            int *simplices, // list of simplices, flat array of size nsimplices*(ndimension+1)
            
            int &npairs,    // number of simplex pairs
            int *&pairs,    // list of indices of pairs, array of size 2*npairs; MUST BE ALLOCATED in the function
            int &ncritical, // number of critical simplices
            int *&critical, // list of critical simplices of size ncritical; MUST BE ALLOCATED in the function
            double *corrected // corrected function values, size npoints; ALREADY ALLOCATED BY THE CALLER
           )
{
  npairs = ncritical = 2;

  pairs =  new int[4];
  memcpy(pairs, simplices, 4*sizeof(int));

  critical = new int[2];
  critical[0] = critical[1] = 42;
  
  double *vi = values, *ci = corrected;
  for(int i = npoints; i--; *ci++ = *vi++ +1);
}

PyObject *qing(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(points, func_values, simplices) -> (pairs, critical, corrected)")
{
  PyObject *pypoints, *pyvalues, *pysimplices;
 
  if (   !PyArg_UnpackTuple(args, "orangeom.qing", 3, 3, &pypoints, &pyvalues, &pysimplices)
      || !isSomeNumeric_wPrecheck(pypoints)
      || !isSomeNumeric_wPrecheck(pyvalues)
      || !isSomeNumeric_wPrecheck(pysimplices))
    PYERROR(PyExc_AttributeError, "three numeric arrays expected", NULL);
    
  PyArrayObject *points = (PyArrayObject *)pypoints;
  PyArrayObject *values = (PyArrayObject *)pyvalues;
  PyArrayObject *simplices = (PyArrayObject *)pysimplices;
  
  if ((getArrayType(points) != 'd') || (points->nd != 2))
    PYERROR(PyExc_AttributeError, "the first argument (points) must be a two-dimensional array of doubles", NULL);

  if ((getArrayType(values) != 'd') || (values->nd != 1))
    PYERROR(PyExc_AttributeError, "the second argument (values) must be a vector of doubles", NULL);
    
  if ((getArrayType(simplices) != 'i') || (simplices->nd != 2))
    PYERROR(PyExc_AttributeError, "the third argument must be a two-dimensional array of integers", NULL);
    
  const int npoints = points->dimensions[0];
  const int dimension = points->dimensions[1];
  const int nsimplices = simplices->dimensions[0];
  
  if (npoints != values->dimensions[0])
    PYERROR(PyExc_AttributeError, "the number of function values mismatches the number of arguments", NULL);

  if (simplices->dimensions[1] != dimension + 1)
    PYERROR(PyExc_AttributeError, "the number of function arguments does not match the dimensionality of simplices", NULL);
    
  PyObject *moduleDict = PyModule_GetDict(moduleNumpy);
  PyObject *mzeros = PyDict_GetItemString(moduleDict, "zeros");
  PyObject *pycorrected = PyObject_CallFunction(mzeros, "(i)s", values->dimensions[0], "d");
  if (!pycorrected)
    return NULL;

  points = PyArray_GETCONTIGUOUS(points);
  values = PyArray_GETCONTIGUOUS(values);
  simplices = PyArray_GETCONTIGUOUS(simplices);

  int npairs, *pairs;
  int ncritical, *critical;
  qing_f(dimension, npoints, (double *)points->data, (double *)values->data,
         nsimplices, (int *)simplices->data,
         npairs, pairs, ncritical, critical, (double *)((PyArrayObject *)pycorrected)->data);

  Py_DECREF(points);
  Py_DECREF(values);
  Py_DECREF(simplices);
    
  int i;
  
  PyObject *pypairs = PyList_New(npairs);
  for(i = 0; i < npairs; i++)
    PyList_SetItem(pypairs, i, Py_BuildValue("ii", pairs[2*i], pairs[2*i+1]));
    
  PyObject *pycritical = PyList_New(ncritical);
  for(i = 0; i < ncritical; i++)
    PyList_SetItem(pycritical, i, PyInt_FromLong(critical[i]));
    
  delete pairs;
  delete critical;
  
  return Py_BuildValue("OOO", pypairs, pycritical, pycorrected);
}
  
  
#include "px/triangulate.px"
