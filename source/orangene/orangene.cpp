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

#include "orange.hpp"
#include "heatmap.hpp"
#include "px/externs.px"
#include "cls_orange.hpp"
#include "slist.hpp"
#include "vectortemplates.hpp"

PyObject *HeatmapConstructor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Orange, "(ExampleTable[, baseHeatmap=None [, disregardClass=0]])")
{
  PyTRY
    PExampleTable table;
    PHeatmapConstructor baseHeatmap;
    int disregardClass = 0;
    if (!PyArg_ParseTuple(args, "O&|O&i:HeatmapConstructor.__new__", cc_ExampleTable, &table, ccn_HeatmapConstructor, &baseHeatmap, &disregardClass))
      return NULL;
    return WrapNewOrange(mlnew THeatmapConstructor(table, baseHeatmap, (PyTuple_Size(args)==2) && !baseHeatmap, (disregardClass!=0)), type);
  PyCATCH
}


PyObject *HeatmapConstructor__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(THeatmapConstructor, hmc);
    const int nColumns = hmc->nColumns;

    TCharBuffer buf(0);

    buf.writeInt(nColumns);
    buf.writeInt(hmc->nRows);
    buf.writeInt(hmc->nClasses);

    ITERATE(vector<float *>, fmi, hmc->floatMap)
      buf.writeBuf(*fmi, nColumns * sizeof(float));

    buf.writeIntVector(hmc->classBoundaries);
    buf.writeFloatVector(hmc->lineCenters);
    buf.writeFloatVector(hmc->lineAverages);
    buf.writeIntVector(hmc->sortIndices);

    return Py_BuildValue("O(ONs#)N", getExportedFunction("__pickleLoaderHeatmapConstructor"),
                                     self->ob_type,
                                     WrapOrange(hmc->sortedExamples),
                                     buf.buf, buf.length(),
                                     packOrangeDictionary(self));
  PyCATCH
}


PyObject *__pickleLoaderHeatmapConstructor(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, sortedExamples, packed-data)")
{
  PyTRY
    PyTypeObject *type;
    PExampleTable sortedExamples;
    char *pbuf;
    int bufSize;

    if (!PyArg_ParseTuple(args, "OO&s#:__pickleLoaderHeatmapConstructor", &type, ccn_ExampleTable, &sortedExamples, &pbuf, &bufSize))
      return NULL;

    TCharBuffer buf(pbuf);

    THeatmapConstructor *hmc = new THeatmapConstructor();
    hmc->sortedExamples = sortedExamples;

    const int nColumns = hmc->nColumns = buf.readInt();
    int rows = hmc->nRows = buf.readInt();
    hmc->nClasses = buf.readInt();

    vector<float *> &floatMap = hmc->floatMap;
    floatMap.reserve(rows);
    while(rows--) {
      float *arow = new float[nColumns];
      buf.readBuf(arow, nColumns * sizeof(float));
      floatMap.push_back(arow);
    }

    buf.readIntVector(hmc->classBoundaries);
    buf.readFloatVector(hmc->lineCenters);
    buf.readFloatVector(hmc->lineAverages);
    buf.readIntVector(hmc->sortIndices);

    return WrapNewOrange(hmc, type);
  PyCATCH
}



PyObject *HeatmapConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(squeeze) -> HeatmapList")
{
  PyTRY
    float squeeze;
    if (!PyArg_ParseTuple(args, "f:HeatmapConstructor.__call__", &squeeze))
      return NULL;

    float absLow, absHigh;
    PHeatmapList hml = SELF_AS(THeatmapConstructor).call(squeeze, absLow, absHigh);
    return Py_BuildValue("Nff", WrapOrange(hml), absLow, absHigh);
  PyCATCH
}


PyObject *HeatmapConstructor_getLegend(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(width, height, gamma) -> bitmap")
{ 
  PyTRY
    int width, height;
    float gamma;
    if (!PyArg_ParseTuple(args, "iif:HeatmapConstructor.getLegend", &width, &height, &gamma))
      return NULL;

    long size;
    unsigned char *bitmap = SELF_AS(THeatmapConstructor).getLegend(width, height, gamma, size);
    PyObject *res = PyString_FromStringAndSize((const char *)bitmap, (Py_ssize_t)size);
    delete bitmap;
    return res;
  PyCATCH
}

BASED_ON(Heatmap, Orange)


PyObject *Heatmap__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(THeatmap, hm)

    TCharBuffer buf(3 * sizeof(int) + hm->height * (1 + hm->width) * sizeof(float));
    buf.writeInt(hm->height);
    buf.writeInt(hm->width);
    buf.writeBuf(hm->cells, hm->height * hm->width * sizeof(float));
    buf.writeBuf(hm->averages, hm->height * sizeof(float));

    return Py_BuildValue("O(Os#NN)N", getExportedFunction("__pickleLoaderHeatmap"),
                                     self->ob_type,
                                     buf.buf, buf.length(),
                                     WrapOrange(hm->examples),
                                     WrapOrange(hm->exampleIndices),
                                     packOrangeDictionary(self));
  PyCATCH
}


PyObject *__pickleLoaderHeatmap(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed-data, examples, exampleIndices)")
{
  PyTRY
    PyTypeObject *type;
    char *pbuf;
    int bufSize;
    PExampleTable examples;
    PIntList exampleIndices;
    if (!PyArg_ParseTuple(args, "Os#O&O&:__pickleLoaderHeatmap", &type, &pbuf, &bufSize, ccn_ExampleTable, &examples, ccn_IntList, &exampleIndices))
      return NULL;

    TCharBuffer buf(pbuf);
    const int height = buf.readInt();
    const int width = buf.readInt();

    THeatmap *hm = new THeatmap(height, width, examples);
    hm->exampleIndices = exampleIndices;
    buf.readBuf(hm->cells, height * width * sizeof(float));
    buf.readBuf(hm->averages, height * sizeof(float));

    return WrapNewOrange(hm, type);
  PyCATCH
}


PyObject *Heatmap_getBitmap(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(cell_width, cell_height, lowerBound, upperBound, gamma[, grid, firstRow, nRows]) -> bitmap")
{
  PyTRY
    int cellWidth, cellHeight;
    float absLow, absHigh, gamma;
    int grid = 0;
    int firstRow = -1, nRows = -1;
    if (!PyArg_ParseTuple(args, "iifff|iii:Heatmap.getBitmap", &cellWidth, &cellHeight, &absLow, &absHigh, &gamma, &grid, &firstRow, &nRows))
      return NULL;

    CAST_TO(THeatmap, hm)

    if (firstRow < 0) {
      firstRow = 0;
      nRows = hm->height;
    }

    long size;
    unsigned char *bitmap = hm->heatmap2string(cellWidth, cellHeight, firstRow, nRows, absLow, absHigh, gamma, grid!=0, size);
    PyObject *res = Py_BuildValue("s#ii", (const char *)bitmap, (Py_ssize_t)size, cellWidth * hm->width, cellHeight * nRows);
    delete bitmap;
    return res;
  PyCATCH
}


PyObject *Heatmap_getAverages(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(cell_width, cell_height, lowerBound, upperBound, gamma[, grid, firstRow, nRows]) -> bitmap")
{
  PyTRY
    int width, height;
    float absLow, absHigh, gamma;
    int grid = 0;
    int firstRow = -1, nRows = -1;
    if (!PyArg_ParseTuple(args, "iifff|iii:Heatmap.getAverageBitmap", &width, &height, &absLow, &absHigh, &gamma, &grid, &firstRow, &nRows))
      return NULL;

    if (grid && height<3)
      grid = 0;

    CAST_TO(THeatmap, hm)

    if (firstRow < 0) {
      firstRow = 0;
      nRows = hm->height;
    }

    long size;
    unsigned char *bitmap = hm->averages2string(width, height, firstRow, nRows, absLow, absHigh, gamma, grid!=0, size);
    PyObject *res = Py_BuildValue("s#ii", (const char *)bitmap, (Py_ssize_t)size, width, height * hm->height);
    delete bitmap;
    return res;
  PyCATCH
}


PyObject *Heatmap_getCellIntensity(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(row, column) -> float")
{
  PyTRY
    int row, column;
    if (!PyArg_ParseTuple(args, "ii:Heatmap.getCellIntensity", &row, &column))
      return NULL;

    const float ci = SELF_AS(THeatmap).getCellIntensity(row, column);
    if (ci == UNKNOWN_F)
      RETURN_NONE;

    return PyFloat_FromDouble(ci);
  PyCATCH
}


PyObject *Heatmap_getRowIntensity(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(row) -> float")
{
  PyTRY
    int row;
    if (!PyArg_ParseTuple(args, "i:Heatmap.getRowIntensity", &row))
      return NULL;

    const float ri = SELF_AS(THeatmap).getRowIntensity(row);
    if (ri == UNKNOWN_F)
      RETURN_NONE;

    return PyFloat_FromDouble(ri);
  PyCATCH
}


PyObject *Heatmap_getPercentileInterval(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(lower_percentile, upper_percentile) -> (min, max)")
{
  PyTRY
    float lowperc, highperc;
    if (!PyArg_ParseTuple(args, "ff:Heatmap_percentileInterval", &lowperc, &highperc))
      return PYNULL;

    float minv, maxv;
    SELF_AS(THeatmap).getPercentileInterval(lowperc, highperc, minv, maxv);
    return Py_BuildValue("ff", minv, maxv);
  PyCATCH
}


extern ORANGENE_API TOrangeType PyOrHeatmap_Type;

PHeatmapList PHeatmapList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::P_FromArguments(arg); }
PyObject *HeatmapList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_FromArguments(type, arg); }
PyObject *HeatmapList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Heatmap>)")  ALLOWS_EMPTY { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_new(type, arg, kwds); }
PyObject *HeatmapList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_getitem(self, index); }
int       HeatmapList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_setitem(self, index, item); }
PyObject *HeatmapList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_getslice(self, start, stop); }
int       HeatmapList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       HeatmapList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_len(self); }
PyObject *HeatmapList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_richcmp(self, object, op); }
PyObject *HeatmapList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_concat(self, obj); }
PyObject *HeatmapList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_repeat(self, times); }
PyObject *HeatmapList_str(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_str(self); }
PyObject *HeatmapList_repr(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_str(self); }
int       HeatmapList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_contains(self, obj); }
PyObject *HeatmapList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_append(self, item); }
PyObject *HeatmapList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_extend(self, obj); }
PyObject *HeatmapList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_count(self, obj); }
PyObject *HeatmapList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> HeatmapList") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_filter(self, args); }
PyObject *HeatmapList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_index(self, obj); }
PyObject *HeatmapList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_insert(self, args); }
PyObject *HeatmapList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_native(self); }
PyObject *HeatmapList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Heatmap") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_pop(self, args); }
PyObject *HeatmapList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_remove(self, obj); }
PyObject *HeatmapList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_reverse(self); }
PyObject *HeatmapList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_sort(self, args); }
PyObject *HeatmapList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_reduce(self); }


bool initorangeneExceptions()
{ return true; }

void gcorangeneUnsafeStaticInitialization()
{}

ORANGENE_API PyObject *orangeneModule;

#include "px/initialization.px"

#include "px/orangene.px"


