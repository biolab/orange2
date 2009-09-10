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

#ifndef __VALUELISTTEMPLATE_HPP
#define __VALUELISTTEMPLATE_HPP

class TValueListMethods : public ListOfUnwrappedMethods<PValueList, TValueList, TValue> {
public:

  static PyObject *_CreateEmptyList(PyTypeObject *type, PVariable var = PVariable());
  static PValueList P_FromArguments(PyObject *arg, PVariable var = PVariable());
  static PyObject *_FromArguments(PyTypeObject *type, PyObject *arg, PVariable var = PVariable());
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *);
  static PyObject *_getitem(TPyOrange *self, Py_ssize_t index);
  static int _setitem(TPyOrange *self, Py_ssize_t, PyObject *item);
  static int _cmp(TPyOrange *self, PyObject *other);
  static PyObject *_str(TPyOrange *self);
  static PyObject *_append(TPyOrange *self, PyObject *item);
  static PyObject *_count(TPyOrange *self, PyObject *item);
  static int _contains(TPyOrange *self, PyObject *item);
  static PyObject *_filter(TPyOrange *self, PyObject *args);
  static PyObject *_index(TPyOrange *self, PyObject *item);
  static PyObject *_insert(TPyOrange *self, PyObject *args);
  static PyObject *_native(TPyOrange *self);
  static PyObject *_pop(TPyOrange *self, PyObject *args);
  static PyObject *_remove(TPyOrange *self, PyObject *item);

  class TCmpByCallback
  { public:
      PyObject *cmpfunc;
      PVariable variable;

      TCmpByCallback(PVariable var, PyObject *func);
      TCmpByCallback(const TCmpByCallback &other);
      ~TCmpByCallback();

      bool operator()(const TValue &x, const TValue &y) const;
  };

  static PyObject *_sort(TPyOrange *self, PyObject *args);
};

#endif
