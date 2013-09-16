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
