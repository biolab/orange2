#include "orange.hpp"
#include "heatmap.hpp"
#include "px/externs.px"
#include "cls_orange.hpp"
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

    int size;
    unsigned char *bitmap = SELF_AS(THeatmapConstructor).getLegend(width, height, gamma, size);
    PyObject *res = PyString_FromStringAndSize((const char *)bitmap, size);
    delete bitmap;
    return res;
  PyCATCH
}

BASED_ON(Heatmap, Orange)

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

    int size;
    unsigned char *bitmap = hm->heatmap2string(cellWidth, cellHeight, firstRow, nRows, absLow, absHigh, gamma, grid!=0, size);
    PyObject *res = Py_BuildValue("Nii", PyString_FromStringAndSize((const char *)bitmap, size), cellWidth * hm->width, cellHeight * nRows);
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

    int size;
    unsigned char *bitmap = hm->averages2string(width, height, firstRow, nRows, absLow, absHigh, gamma, grid!=0, size);
    PyObject *res = Py_BuildValue("Nii", PyString_FromStringAndSize((const char *)bitmap, size), width, height * hm->height);
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
PyObject *HeatmapList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Heatmap>)") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_new(type, arg, kwds); }
PyObject *HeatmapList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_getitem(self, index); }
int       HeatmapList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_setitem(self, index, item); }
PyObject *HeatmapList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_getslice(self, start, stop); }
int       HeatmapList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_setslice(self, start, stop, item); }
int       HeatmapList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_len(self); }
PyObject *HeatmapList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_richcmp(self, object, op); }
PyObject *HeatmapList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_concat(self, obj); }
PyObject *HeatmapList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_repeat(self, times); }
PyObject *HeatmapList_str(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_str(self); }
PyObject *HeatmapList_repr(TPyOrange *self) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_str(self); }
int       HeatmapList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_contains(self, obj); }
PyObject *HeatmapList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_append(self, item); }
PyObject *HeatmapList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_count(self, obj); }
PyObject *HeatmapList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> HeatmapList") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_filter(self, args); }
PyObject *HeatmapList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> int") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_index(self, obj); }
PyObject *HeatmapList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_insert(self, args); }
PyObject *HeatmapList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_native(self); }
PyObject *HeatmapList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Heatmap") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_pop(self, args); }
PyObject *HeatmapList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Heatmap) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_remove(self, obj); }
PyObject *HeatmapList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_reverse(self); }
PyObject *HeatmapList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PHeatmapList, THeatmapList, PHeatmap, &PyOrHeatmap_Type>::_sort(self, args); }


bool initExceptions()
{ return true; }

void gcUnsafeStaticInitialization()
{}

#include "px/initialization.px"

#include "px/orangene.px"


