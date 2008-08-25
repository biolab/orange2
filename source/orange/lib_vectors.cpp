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


#include "orvector.hpp"
#include "cls_orange.hpp"
#include "vectortemplates.hpp"
#include "externs.px"

#include "distance_dtw.hpp"

bool convertFromPython(PyObject *, bool &);
PyObject *convertToPython(const bool &);

PBoolList PBoolList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::P_FromArguments(arg); }
PyObject *BoolList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_FromArguments(type, arg); }
PyObject *BoolList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of bool>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_new(type, arg, kwds); }
PyObject *BoolList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_getitem(self, index); }
int       BoolList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_setitem(self, index, item); }
PyObject *BoolList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_getslice(self, start, stop); }
int       BoolList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_setslice(self, start, stop, item); }
int       BoolList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_len(self); }
PyObject *BoolList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_richcmp(self, object, op); }
PyObject *BoolList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_concat(self, obj); }
PyObject *BoolList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_repeat(self, times); }
PyObject *BoolList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_str(self); }
PyObject *BoolList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_str(self); }
int       BoolList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_contains(self, obj); }
PyObject *BoolList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(bool) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_append(self, item); }
PyObject *BoolList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_extend(self, obj); }
PyObject *BoolList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> int") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_count(self, obj); }
PyObject *BoolList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> BoolList") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_filter(self, args); }
PyObject *BoolList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> int") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_index(self, obj); }
PyObject *BoolList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_insert(self, args); }
PyObject *BoolList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_native(self); }
PyObject *BoolList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> bool") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_pop(self, args); }
PyObject *BoolList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_remove(self, obj); }
PyObject *BoolList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_reverse(self); }
PyObject *BoolList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_sort(self, args); }
PyObject *BoolList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_reduce(self); }



bool convertFromPython(PyObject *, int &);
PyObject *convertToPython(const int &);

PIntList PIntList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::P_FromArguments(arg); }
PyObject *IntList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_FromArguments(type, arg); }
PyObject *IntList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of int>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_new(type, arg, kwds); }
PyObject *IntList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_getitem(self, index); }
int       IntList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_setitem(self, index, item); }
PyObject *IntList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_getslice(self, start, stop); }
int       IntList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_setslice(self, start, stop, item); }
int       IntList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_len(self); }
PyObject *IntList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_richcmp(self, object, op); }
PyObject *IntList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_concat(self, obj); }
PyObject *IntList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_repeat(self, times); }
PyObject *IntList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_str(self); }
PyObject *IntList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_str(self); }
int       IntList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_contains(self, obj); }
PyObject *IntList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_append(self, item); }
PyObject *IntList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_extend(self, obj); }
PyObject *IntList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_count(self, obj); }
PyObject *IntList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> IntList") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_filter(self, args); }
PyObject *IntList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_index(self, obj); }
PyObject *IntList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_insert(self, args); }
PyObject *IntList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_native(self); }
PyObject *IntList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_pop(self, args); }
PyObject *IntList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_remove(self, obj); }
PyObject *IntList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_reverse(self); }
PyObject *IntList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_sort(self, args); }
PyObject *IntList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_reduce(self); }



bool convertFromPython(PyObject *, float &);
PyObject *convertToPython(const float &);

PFloatList PFloatList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::P_FromArguments(arg); }
PyObject *FloatList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_FromArguments(type, arg); }
PyObject *FloatList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of float>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_new(type, arg, kwds); }
PyObject *FloatList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_getitem(self, index); }
int       FloatList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_setitem(self, index, item); }
PyObject *FloatList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_getslice(self, start, stop); }
int       FloatList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_setslice(self, start, stop, item); }
int       FloatList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_len(self); }
PyObject *FloatList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_richcmp(self, object, op); }
PyObject *FloatList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_concat(self, obj); }
PyObject *FloatList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_repeat(self, times); }
PyObject *FloatList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_str(self); }
PyObject *FloatList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_str(self); }
int       FloatList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_contains(self, obj); }
PyObject *FloatList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(float) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_append(self, item); }
PyObject *FloatList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_extend(self, obj); }
PyObject *FloatList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> int") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_count(self, obj); }
PyObject *FloatList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FloatList") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_filter(self, args); }
PyObject *FloatList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> int") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_index(self, obj); }
PyObject *FloatList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_insert(self, args); }
PyObject *FloatList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_native(self); }
PyObject *FloatList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> float") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_pop(self, args); }
PyObject *FloatList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_remove(self, obj); }
PyObject *FloatList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_reverse(self); }
PyObject *FloatList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_sort(self, args); }
PyObject *FloatList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_reduce(self); }



extern PyTypeObject PyOrFloatList_Type_inh;

PFloatListList PFloatListList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::P_FromArguments(arg); }
PyObject *FloatListList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_FromArguments(type, arg); }
PyObject *FloatListList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of FloatList>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_new(type, arg, kwds); }
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
PyObject *FloatListList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_extend(self, obj); }
PyObject *FloatListList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> int") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_count(self, obj); }
PyObject *FloatListList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FloatListList") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_filter(self, args); }
PyObject *FloatListList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> int") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_index(self, obj); }
PyObject *FloatListList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_insert(self, args); }
PyObject *FloatListList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_native(self); }
PyObject *FloatListList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> FloatList") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_pop(self, args); }
PyObject *FloatListList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(FloatList) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_remove(self, obj); }
PyObject *FloatListList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_reverse(self); }
PyObject *FloatListList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_sort(self, args); }
PyObject *FloatListList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PFloatListList, TFloatListList, PFloatList, &PyOrFloatList_Type>::_reduce(self); }



bool convertFromPython(PyObject *, string &);
PyObject *convertToPython(const string &);

PStringList PStringList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::P_FromArguments(arg); }
PyObject *StringList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_FromArguments(type, arg); }
PyObject *StringList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of string>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_new(type, arg, kwds); }
PyObject *StringList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_getitem(self, index); }
int       StringList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_setitem(self, index, item); }
PyObject *StringList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_getslice(self, start, stop); }
int       StringList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_setslice(self, start, stop, item); }
int       StringList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_len(self); }
PyObject *StringList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_richcmp(self, object, op); }
PyObject *StringList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_concat(self, obj); }
PyObject *StringList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_repeat(self, times); }
PyObject *StringList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_str(self); }
PyObject *StringList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_str(self); }
int       StringList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_contains(self, obj); }
PyObject *StringList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(string) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_append(self, item); }
PyObject *StringList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_extend(self, obj); }
PyObject *StringList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> int") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_count(self, obj); }
PyObject *StringList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> StringList") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_filter(self, args); }
PyObject *StringList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> int") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_index(self, obj); }
PyObject *StringList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_insert(self, args); }
PyObject *StringList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_native(self); }
PyObject *StringList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> string") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_pop(self, args); }
PyObject *StringList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_remove(self, obj); }
PyObject *StringList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_reverse(self); }
PyObject *StringList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_sort(self, args); }
PyObject *StringList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_reduce(self); }



bool convertFromPython(PyObject *, long &);
PyObject *convertToPython(const long &);

PLongList PLongList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::P_FromArguments(arg); }
PyObject *LongList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_FromArguments(type, arg); }
PyObject *LongList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of int>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_new(type, arg, kwds); }
PyObject *LongList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_getitem(self, index); }
int       LongList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_setitem(self, index, item); }
PyObject *LongList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_getslice(self, start, stop); }
int       LongList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_setslice(self, start, stop, item); }
int       LongList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_len(self); }
PyObject *LongList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_richcmp(self, object, op); }
PyObject *LongList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_concat(self, obj); }
PyObject *LongList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_repeat(self, times); }
PyObject *LongList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_str(self); }
PyObject *LongList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_str(self); }
int       LongList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_contains(self, obj); }
PyObject *LongList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_append(self, item); }
PyObject *LongList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_extend(self, obj); }
PyObject *LongList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_count(self, obj); }
PyObject *LongList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> LongList") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_filter(self, args); }
PyObject *LongList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_index(self, obj); }
PyObject *LongList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_insert(self, args); }
PyObject *LongList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_native(self); }
PyObject *LongList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_pop(self, args); }
PyObject *LongList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_remove(self, obj); }
PyObject *LongList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_reverse(self); }
PyObject *LongList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_sort(self, args); }
PyObject *LongList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_reduce(self); }



bool convertFromPython(PyObject *, TAlignment &);
PyObject *convertToPython(const TAlignment &);

PAlignmentList PAlignmentList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::P_FromArguments(arg); }
PyObject *AlignmentList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_FromArguments(type, arg); }
PyObject *AlignmentList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Alignment>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_new(type, arg, kwds); }
PyObject *AlignmentList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_getitem(self, index); }
int       AlignmentList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_setitem(self, index, item); }
PyObject *AlignmentList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_getslice(self, start, stop); }
int       AlignmentList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_setslice(self, start, stop, item); }
int       AlignmentList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_len(self); }
PyObject *AlignmentList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_richcmp(self, object, op); }
PyObject *AlignmentList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_concat(self, obj); }
PyObject *AlignmentList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_repeat(self, times); }
PyObject *AlignmentList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_str(self); }
PyObject *AlignmentList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_str(self); }
int       AlignmentList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_contains(self, obj); }
PyObject *AlignmentList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Alignment) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_append(self, item); }
PyObject *AlignmentList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_extend(self, obj); }
PyObject *AlignmentList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> int") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_count(self, obj); }
PyObject *AlignmentList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> AlignmentList") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_filter(self, args); }
PyObject *AlignmentList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> int") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_index(self, obj); }
PyObject *AlignmentList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_insert(self, args); }
PyObject *AlignmentList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_native(self); }
PyObject *AlignmentList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Alignment") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_pop(self, args); }
PyObject *AlignmentList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_remove(self, obj); }
PyObject *AlignmentList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_reverse(self); }
PyObject *AlignmentList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_sort(self, args); }
PyObject *AlignmentList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_reduce(self); }



bool convertFromPython(PyObject *, pair<int, float> &);
PyObject *convertToPython(const pair<int, float> &);

PIntFloatList PIntFloatList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::P_FromArguments(arg); }
PyObject *IntFloatList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_FromArguments(type, arg); }
PyObject *IntFloatList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of tuple(int, float)>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_new(type, arg, kwds); }
PyObject *IntFloatList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_getitem(self, index); }
int       IntFloatList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_setitem(self, index, item); }
PyObject *IntFloatList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_getslice(self, start, stop); }
int       IntFloatList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_setslice(self, start, stop, item); }
int       IntFloatList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_len(self); }
PyObject *IntFloatList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_richcmp(self, object, op); }
PyObject *IntFloatList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_concat(self, obj); }
PyObject *IntFloatList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_repeat(self, times); }
PyObject *IntFloatList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_str(self); }
PyObject *IntFloatList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_str(self); }
int       IntFloatList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_contains(self, obj); }
PyObject *IntFloatList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(tuple(int, float)) -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_append(self, item); }
PyObject *IntFloatList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_extend(self, obj); }
PyObject *IntFloatList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(int, float)) -> int") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_count(self, obj); }
PyObject *IntFloatList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> IntFloatList") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_filter(self, args); }
PyObject *IntFloatList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(int, float)) -> int") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_index(self, obj); }
PyObject *IntFloatList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_insert(self, args); }
PyObject *IntFloatList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_native(self); }
PyObject *IntFloatList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> tuple(int, float)") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_pop(self, args); }
PyObject *IntFloatList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(int, float)) -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_remove(self, obj); }
PyObject *IntFloatList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_reverse(self); }
PyObject *IntFloatList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_sort(self, args); }
PyObject *IntFloatList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PIntFloatList, TIntFloatList, pair<int, float>>::_reduce(self); }



bool convertFromPython(PyObject *, pair<float, float> &);
PyObject *convertToPython(const pair<float, float> &);

PFloatFloatList PFloatFloatList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::P_FromArguments(arg); }
PyObject *FloatFloatList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_FromArguments(type, arg); }
PyObject *FloatFloatList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of tuple(float, float)>)") ALLOWS_EMPTY { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_new(type, arg, kwds); }
PyObject *FloatFloatList_getitem_sq(TPyOrange *self, int index) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_getitem(self, index); }
int       FloatFloatList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_setitem(self, index, item); }
PyObject *FloatFloatList_getslice(TPyOrange *self, int start, int stop) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_getslice(self, start, stop); }
int       FloatFloatList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_setslice(self, start, stop, item); }
int       FloatFloatList_len_sq(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_len(self); }
PyObject *FloatFloatList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_richcmp(self, object, op); }
PyObject *FloatFloatList_concat(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_concat(self, obj); }
PyObject *FloatFloatList_repeat(TPyOrange *self, int times) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_repeat(self, times); }
PyObject *FloatFloatList_str(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_str(self); }
PyObject *FloatFloatList_repr(TPyOrange *self) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_str(self); }
int       FloatFloatList_contains(TPyOrange *self, PyObject *obj) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_contains(self, obj); }
PyObject *FloatFloatList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(tuple(float, float)) -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_append(self, item); }
PyObject *FloatFloatList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_extend(self, obj); }
PyObject *FloatFloatList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(float, float)) -> int") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_count(self, obj); }
PyObject *FloatFloatList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FloatFloatList") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_filter(self, args); }
PyObject *FloatFloatList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(float, float)) -> int") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_index(self, obj); }
PyObject *FloatFloatList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_insert(self, args); }
PyObject *FloatFloatList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_native(self); }
PyObject *FloatFloatList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> tuple(float, float)") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_pop(self, args); }
PyObject *FloatFloatList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(tuple(float, float)) -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_remove(self, obj); }
PyObject *FloatFloatList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_reverse(self); }
PyObject *FloatFloatList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_sort(self, args); }
PyObject *FloatFloatList__reduce__(TPyOrange *self, PyObject *) { return ListOfUnwrappedMethods<PFloatFloatList, TFloatFloatList, pair<float, float>>::_reduce(self); }


#include "lib_vectors.px"
