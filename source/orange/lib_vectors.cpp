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
PyObject *BoolList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of bool>)") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_new(type, arg, kwds); }
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
PyObject *BoolList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> int") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_count(self, obj); }
PyObject *BoolList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> BoolList") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_filter(self, args); }
PyObject *BoolList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> int") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_index(self, obj); }
PyObject *BoolList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_insert(self, args); }
PyObject *BoolList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_native(self); }
PyObject *BoolList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> bool") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_pop(self, args); }
PyObject *BoolList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(bool) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_remove(self, obj); }
PyObject *BoolList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_reverse(self); }
PyObject *BoolList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PBoolList, TBoolList, bool>::_sort(self, args); }



bool convertFromPython(PyObject *, int &);
PyObject *convertToPython(const int &);

PIntList PIntList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::P_FromArguments(arg); }
PyObject *IntList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_FromArguments(type, arg); }
PyObject *IntList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of int>)") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_new(type, arg, kwds); }
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
PyObject *IntList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_count(self, obj); }
PyObject *IntList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> IntList") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_filter(self, args); }
PyObject *IntList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_index(self, obj); }
PyObject *IntList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_insert(self, args); }
PyObject *IntList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_native(self); }
PyObject *IntList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> int") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_pop(self, args); }
PyObject *IntList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_remove(self, obj); }
PyObject *IntList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_reverse(self); }
PyObject *IntList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PIntList, TIntList, int>::_sort(self, args); }



bool convertFromPython(PyObject *, float &);
PyObject *convertToPython(const float &);

PFloatList PFloatList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::P_FromArguments(arg); }
PyObject *FloatList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_FromArguments(type, arg); }
PyObject *FloatList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of float>)") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_new(type, arg, kwds); }
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
PyObject *FloatList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> int") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_count(self, obj); }
PyObject *FloatList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> FloatList") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_filter(self, args); }
PyObject *FloatList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> int") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_index(self, obj); }
PyObject *FloatList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_insert(self, args); }
PyObject *FloatList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_native(self); }
PyObject *FloatList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> float") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_pop(self, args); }
PyObject *FloatList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(float) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_remove(self, obj); }
PyObject *FloatList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_reverse(self); }
PyObject *FloatList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PFloatList, TFloatList, float>::_sort(self, args); }

int pt_FloatList(PyObject *args, void *floatlist)
{
  *(PFloatList *)(floatlist) = PFloatList_FromArguments(args);
  return PyErr_Occurred() ? -1 : 0;
}

bool convertFromPython(PyObject *, string &);
PyObject *convertToPython(const string &);

PStringList PStringList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::P_FromArguments(arg); }
PyObject *StringList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_FromArguments(type, arg); }
PyObject *StringList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of string>)") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_new(type, arg, kwds); }
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
PyObject *StringList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> int") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_count(self, obj); }
PyObject *StringList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> StringList") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_filter(self, args); }
PyObject *StringList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> int") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_index(self, obj); }
PyObject *StringList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_insert(self, args); }
PyObject *StringList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_native(self); }
PyObject *StringList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> string") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_pop(self, args); }
PyObject *StringList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(string) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_remove(self, obj); }
PyObject *StringList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_reverse(self); }
PyObject *StringList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PStringList, TStringList, string>::_sort(self, args); }



bool convertFromPython(PyObject *, long &);
PyObject *convertToPython(const long &);

PLongList PLongList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::P_FromArguments(arg); }
PyObject *LongList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_FromArguments(type, arg); }
PyObject *LongList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of int>)") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_new(type, arg, kwds); }
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
PyObject *LongList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_count(self, obj); }
PyObject *LongList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> LongList") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_filter(self, args); }
PyObject *LongList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_index(self, obj); }
PyObject *LongList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_insert(self, args); }
PyObject *LongList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_native(self); }
PyObject *LongList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> int") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_pop(self, args); }
PyObject *LongList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(int) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_remove(self, obj); }
PyObject *LongList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_reverse(self); }
PyObject *LongList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PLongList, TLongList, long>::_sort(self, args); }



bool convertFromPython(PyObject *, TAlignment &);
PyObject *convertToPython(const TAlignment &);
#define TAlignmentList TWarpPath
#define PAlignmentList PWarpPath

PAlignmentList PAlignmentList_FromArguments(PyObject *arg) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::P_FromArguments(arg); }
PyObject *AlignmentList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_FromArguments(type, arg); }
PyObject *AlignmentList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Alignment>)") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_new(type, arg, kwds); }
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
PyObject *AlignmentList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> int") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_count(self, obj); }
PyObject *AlignmentList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> AlignmentList") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_filter(self, args); }
PyObject *AlignmentList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> int") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_index(self, obj); }
PyObject *AlignmentList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_insert(self, args); }
PyObject *AlignmentList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_native(self); }
PyObject *AlignmentList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Alignment") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_pop(self, args); }
PyObject *AlignmentList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Alignment) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_remove(self, obj); }
PyObject *AlignmentList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_reverse(self); }
PyObject *AlignmentList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfUnwrappedMethods<PAlignmentList, TAlignmentList, TAlignment>::_sort(self, args); }


#include "lib_vectors.px"
