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


#include <limits>
#include <stdlib.h>

#include "random.hpp"

#include "domain.hpp"
#include "classify.hpp"
#include "pythonvars.ppp"

#include "cls_orange.hpp"
#include "cls_example.hpp"
#include "cls_value.hpp"
#include "externs.px"


PyObject *pickleFunction = NULL;
PyObject *unpickleFunction = NULL;

void loadPickleModule()
{
  PyObject *pickleModule = PyImport_ImportModule("cPickle");
  if (!pickleModule)
    throw pyexception();

  PyObject *moduleDict = PyModule_GetDict(pickleModule);
  pickleFunction = PyDict_GetItemString(moduleDict, "dumps");
  unpickleFunction = PyDict_GetItemString(moduleDict, "loads");

  if (!pickleFunction || !unpickleFunction)
    raiseErrorWho("PythonVariable", "invalid cPickle module");
}


TPythonValue::TPythonValue()
: value(Py_None)
{ Py_INCREF(Py_None); }


TPythonValue::TPythonValue(PyObject *pyvalue)
: value(pyvalue)
{ Py_INCREF(pyvalue); }


TPythonValue::TPythonValue(const TPythonValue &other)
: TSomeValue(other),
  value(other.value)
{
  Py_INCREF(value);
}


TPythonValue::~TPythonValue()
{
  Py_DECREF(value);
}


TPythonValue &TPythonValue::operator =(const TPythonValue &other)
{
  Py_INCREF(other.value);
  Py_XDECREF(value);
  value = other.value;
  return *this;
}  


int TPythonValue::compare(const TSomeValue &v) const
{ 
  const TPythonValue *other = dynamic_cast<const TPythonValue *>(&v);
  if (!other)
    raiseError("cannot compare 'PythonValue' with '%s'", TYPENAME(typeid(v)));

  if (value == Py_None)
    return other->value == Py_None ? 0 : 1;
  if (other->value == Py_None)
    return -1;

  int res = PyObject_Compare(value, other->value);
  if (PyErr_Occurred())
    throw pyexception();

  return res;
}


bool TPythonValue::compatible(const TSomeValue &v) const
{ 
  const TPythonValue *other = dynamic_cast<const TPythonValue *>(&v);
  if (!other)
    raiseError("cannot compare 'PythonValue' with '%s'", TYPENAME(typeid(v)));

  if ((value == Py_None) || (other->value == Py_None))
    return true;

  int res = PyObject_Compare(value, other->value);
  if (PyErr_Occurred())
    throw pyexception();

  return res == 0;
}




TPythonVariable::TPythonVariable()
: usePickle(false),
  useSomeValue(true)
{
  varType = PYTHONVAR; 

  DC_value = TValue(varType, valueDC);
  DK_value = TValue(varType, valueDK);

  DC_somevalue = TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, valueDC);
  DK_somevalue = TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, valueDK);
}



TPythonVariable::TPythonVariable(const string &aname)
: TVariable(aname),
  usePickle(false),
  useSomeValue(true)
{
  varType = PYTHONVAR; 
 
  DC_value = TValue(varType, valueDC);
  DK_value = TValue(varType, valueDK);

  DC_somevalue = TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, valueDC);
  DK_somevalue = TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, valueDK);
}


PyObject *TPythonVariable::toPyObject(const TValue &valu) const
{
  if (useSomeValue) {
    const TPythonValue *sv = dynamic_cast<const TPythonValue *>(valu.svalV.getUnwrappedPtr());
    if (!sv->value)
      raiseError("invalid PythonValue");
    PyObject *res = const_cast<PyObject *>(sv->value);
    Py_INCREF(res);
    return res;
  }
  else
    return Value_FromVariableValue(const_cast<TPythonVariable *>(this), valu);
}


// steals a reference!
TValue TPythonVariable::toValue(PyObject *pyvalue) const
{ 
  if (!pyvalue)
    throw pyexception();

  // the special format
  if (PyOrPythonValueSpecial_Check(pyvalue)) {
    int vtype = PyOrange_AsPythonValueSpecial(pyvalue)->valueType;
    Py_DECREF(pyvalue);

    if (!vtype)
      raiseError("invalid value type for special value");

    return TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, vtype);
  }

  if(!useSomeValue || !PyOrValue_Check(pyvalue))
    return TValue(PSomeValue(mlnew TPythonValue(pyvalue)), PYTHONVAR);
  else
    return PyValue_AS_Value(pyvalue);
}


TValue TPythonVariable::toNoneValue(const signed char &valueType) const
{ 
  if (valueType == valueDK)
    return DK();

  if (valueType == valueDC)
    return DC();

  if (useSomeValue)
    return TValue(PSomeValue(mlnew TPythonValue(Py_None)), PYTHONVAR, valueType);
  else
    return TValue(PYTHONVAR, valueType);
}


void TPythonVariable::toValue(PyObject *pyvalue, TValue &val) const
{
  if (!pyvalue)
    throw pyexception();

  if (useSomeValue || !PyOrValue_Check(pyvalue))
    val.svalV = mlnew TPythonValue(pyvalue);
  else 
    val = PyValue_AS_Value(pyvalue);
}


#define MYSELF ((PyObject *)myWrapper)

bool TPythonVariable::isOverloaded(char *method) const
{
  PyObject *pymethod = PyObject_GetAttrString(MYSELF, method);
  if (!pymethod) {
    PyErr_Clear();
    return false;
  }
  
  // can DECREF, won't go away...
  Py_DECREF(pymethod);
  return PyMethod_Check(pymethod);
}


const TValue &TPythonVariable::DC() const
{
  return useSomeValue ? DC_somevalue : DC_value;
}


const TValue &TPythonVariable::DK() const
{
  return useSomeValue ? DK_somevalue : DK_value;
}


TValue TPythonVariable::specialValue(int stype) const
{
  return toNoneValue(stype);
}
    

void TPythonVariable::val2str(const TValue &val, string &str) const
{
  static char const *val2strS[3] = {"val2str", "Value", "__str__"};
  
  if (special2str(val, str))
    return;

  PyObject *pyvalue = toPyObject(val);
  PyObject *reprs = NULL;
  char const *cls, *meth;

  if (isOverloaded("val2str")) {
    reprs = PyObject_CallMethod(MYSELF, "val2str", "O", pyvalue);
    cls = MYSELF->ob_type->tp_name;
    meth = val2strS[0];
  }
  else {
    reprs = PyObject_Str(pyvalue);
    cls = val2strS[1];
    meth = val2strS[2];
  }
  
  Py_DECREF(pyvalue);
  
  if (!reprs)
    throw pyexception();
    
  if (!PyString_Check(reprs)) {
    Py_DECREF(reprs);
    raiseError("%s.%s should return a 'string', not '%s'", cls, meth, reprs->ob_type->tp_name);
  }

  str = PyString_AsString(reprs);
  Py_DECREF(reprs);
}


void TPythonVariable::str2val(const string &valname, TValue &valu)
{
  if (str2special(valname, valu))
    return;

  if (isOverloaded("str2val"))
    valu = toValue(PyObject_CallMethod(MYSELF, "str2val", "s", valname.c_str()));
  else
    valu = toValue(PyString_FromString(valname.c_str()));
}

    
void TPythonVariable::str2val_add(const string &valname, TValue &valu)
{
  if (str2special(valname, valu))
    return;

  if (isOverloaded("str2val_add"))
    valu = toValue(PyObject_CallMethod(MYSELF, "str2val_add", "s", valname.c_str()));
  else
    str2val(valname, valu);
}


void TPythonVariable::val2filestr(const TValue &val, string &str, const TExample &example) const
{
  if (special2str(val, str))
    return;

  if (isOverloaded("val2filestr")) {
    PyObject *pyvalue = toPyObject(val);
    PyObject *reprs = PyObject_CallMethod(MYSELF, "val2filestr", "ON", pyvalue, Example_FromWrappedExample(PExample(const_cast<TExample &>(example))));
    Py_DECREF(pyvalue);
    if (!reprs)
      throw pyexception();

    str = PyString_AsString(reprs);
    Py_DECREF(reprs);
  }

  else {
    if (usePickle) {
      if (!pickleFunction)
        loadPickleModule();

      PyObject *pyvalue = toPyObject(val);
      PyObject *pickled = PyObject_CallFunctionObjArgs(pickleFunction, pyvalue, NULL);
      Py_DECREF(pyvalue);
      if (!pickled)
        throw pyexception();
      if (!PyString_Check(pickled)) {
        Py_DECREF(pickled);
        raiseError("cPickle.dumps returned a non-string(?!)");
      }

      char *pickleds = PyString_AsString(pickled);
      int newlines = 1;
      char *pi = pickleds, *ei;
      while(*pi)
        if (*pi++ == '\n')
          newlines++;

      char *escaped = new char[pi-pickleds];
      pi = pickleds;
      ei = escaped;
      while(*pi) {
        if (*pi=='\n') {
          *(ei++) = '\\';
          *(ei++) = 'n';
          pi++;
        }
        else
          *ei++ = *pi++;
      }
      *ei = 0;

      Py_DECREF(pickled);
      str = escaped;
    }
    else
      val2str(val, str);
  }
}


void TPythonVariable::filestr2val(const string &valname, TValue &valu, TExample &ex)
{
  if (str2special(valname, valu))
    return;

  if (isOverloaded("filestr2val")) {
    valu = toValue(PyObject_CallMethod(MYSELF, "filestr2val", "sN", valname.c_str(), Example_FromWrappedExample(PExample(ex))));
    return;
  }

  if (!usePickle && isOverloaded("str2val")) {
    valu = toValue(PyObject_CallMethod(MYSELF, "str2val", "s", valname.c_str()));
    return;
  }

  PyObject *res = NULL;

  if (!unpickleFunction)
    loadPickleModule();

  char *unescaped = mlnew char[valname.size()];
  char *ui = unescaped;
  for(const char *vi = valname.c_str(); *vi; vi++, ui++)
    if ((*vi == '\\') && (vi[1] == 'n')) {
      *ui = '\n';
      vi++;
    }
    else
      *ui = *vi;
  *ui = 0;

  res = PyObject_CallFunction(unpickleFunction, "s", unescaped);
  if (res) {
    valu = toValue(res);
    return;
  }

  PyErr_Clear();

  PyObject *globals = PyEval_GetGlobals();
  PyObject *locals = PyEval_GetLocals();

  PyObject *fdoms = PyString_FromString("__fileExample");
  PyObject *wo = Example_FromWrappedExample(PExample(ex));
  PyDict_SetItem(locals, fdoms, wo);
  Py_DECREF(wo);
  res = PyRun_String(const_cast<char *>(valname.c_str()), Py_eval_input, globals, locals);
  PyDict_DelItem(locals, fdoms);
  Py_DECREF(fdoms);

  if (!res) {
    PyErr_Clear();
    raiseError("cannot read the attribute value");
  }

  valu = toValue(res);
}


bool TPythonVariable::firstValue(TValue &val) const
{
  if (isOverloaded("firstvalue")) {
    val = toValue(PyObject_CallMethod(MYSELF, "firstvalue", NULL));
    return true;
  }

  else
    return TVariable::firstValue(val);
}


bool TPythonVariable::nextValue(TValue &val) const
{
  if (isOverloaded("nextvalue")) {
    PyObject *pyvalue = toPyObject(val);
    toValue(PyObject_CallMethod(MYSELF, "nextvalue", "O", pyvalue), val);
    Py_DECREF(pyvalue);
    return true;
  }

  else
    return TVariable::nextValue(val);
}

    
TValue TPythonVariable::randomValue(const int &rand)
{
  if (isOverloaded("randomvalue")) {
    return toValue(PyObject_CallMethod(MYSELF, "randomvalue", "i", rand));
  }

  else
    return TVariable::randomValue(rand);
}


int TPythonVariable::noOfValues() const
{
  if (isOverloaded("__len__")) {
    PyObject *pynvalues = PyObject_CallMethod(MYSELF, "__len__", NULL);
    if (!pynvalues)
      throw pyexception();

    if (!PyInt_Check(pynvalues))
      raiseError("PythonVariable.__len__ should return an integer");

    int res = PyInt_AsLong(pynvalues);
    Py_DECREF(pynvalues);
    return res;
  }

  else
    return -1;
}
