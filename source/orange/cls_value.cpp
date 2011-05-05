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


#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "cls_value.hpp"
#include "cls_orange.hpp"
#include "vars.hpp"
#include "stringvars.hpp"
#include "pythonvars.hpp"
#include "values.hpp"

#include "vectortemplates.hpp"

#include "externs.px"


#define CHECK_VARIABLE \
  if (!self->variable) PYERROR(PyExc_TypeError, "'variable' not set", PYNULL);

#define CHECK_SPECIAL_OTHER \
  if (self->value.isSpecial()) \
    PYERROR(PyExc_TypeError, "attribute value unknown", PYNULL); \
  if ((self->value.varType!=TValue::INTVAR) && (self->value.varType!=TValue::FLOATVAR)) \
    PYERROR(PyExc_TypeError, "attribute is not an ordinary discrete or continuous", PYNULL);



DATASTRUCTURE(Value - Orange.data.Value, TPyValue, 0)
ABSTRACT(SomeValue - Orange.core.SomeValue, Orange)

/* Converts a value into an appropriate python variable.
   Behaves as convertToPythonNative(const TValue &, PVariable)
   when the variable is not given. */

PyObject *convertToPythonNative(const TValue &val)
{ return convertToPythonNative(val, PVariable()); }



/* Converts a value into an appropriate python variable.
   Behaves as convertToPythonNative(const TValue &, PVariable);
   variable can be there or not. */

PyObject *convertToPythonNative(const TPyValue *value)
{ return convertToPythonNative(value->value, value->variable); }



/* Converts a value into an appropriate python variable.
   If value is known (e.g. not DC, DK...)
    - continuous values are returned as ordinary python floats
    - discrete are returned as strings (variable is required)
    - string values are returned as strings
    - other values are return as ordinary orange objects
   If value is special 
    - if the variable is given, its val2str is used to get a string
    - if the variable is not given, '?', '~' and '.' are returned
      for DK, DC and other, respectively.

   FAILS if the value is discrete and variable is not given
*/

PyObject *convertToPythonNative(const TValue &val, PVariable var)
{
  if ((val.varType==TValue::FLOATVAR) && !val.isSpecial())
    return PyFloat_FromDouble(double(val.floatV));

  if (val.varType == PYTHONVAR) {
    PyObject *res = val.svalV ? ((TPythonValue &)(val.svalV.getReference())).value : Py_None;
    Py_INCREF(res);
    return res;
  }

  if ((val.varType == STRINGVAR) && val.svalV) {
    string s;
    val.svalV.AS(TStringValue)->val2str(s);
    return PyString_FromString(s.c_str());
  }

  if ((val.varType!=TValue::INTVAR) && val.svalV)
    return WrapOrange((const_cast<TValue &>(val)).svalV);

  if (var) { // && (val.varType == TValue::INTVAR || val.isSpecial)
    string vs;
    var->val2str(val, vs);
    return PyString_FromString(vs.c_str());
  }

  if (val.isSpecial())
    if (val.isDK())
      return PyString_FromString("?");
    else if (val.isDC()) 
      return PyString_FromString("~");
    else
      return PyString_FromString(".");

  PYERROR(PyExc_TypeError, "unknown value type", PYNULL);
}


/* The main routine for converting values from python to TValue.
   If arguments are given as a
   - Value, it is simply copied.
       The variable is checked if given.
   - SomeValue, it is copied as such.
       If the variable is discrete or continuous, SomeValue must
       be DiscDistribution or ContDistribution.
   - string, we convert it to a value
       The variable must be given unless the string is '?', '~'
       (in this case INTVAR is ocnstructed)
       We could return a StringValue here, but if user passes a
       string without descriptor it is more probable that he just
       forgot it. I doubt that many would construct StringValues.)
   - int - if variable is given and is discrete, an integer value
           is constructed. If the variable is derived from
           EnumVariable the range is also tested
         - if variable is given and is continuous, a continuous
           value is constructed
         - if variable is given and is of other type, an error is
           raised
         - if the variable is not given, an integer value is constructed
   - float - a continuous value is constructed.
       If the variable is given, it is checked that it is continuous
   - other types: if it can be converted to float and the variable is
       given and is continuous, a continuous value is constructed.
       Otherwise, an exception is raised.         
*/

bool convertFromPython(PyObject *args, TValue &value, PVariable var)
{
  if (PyOrValue_Check(args)) {
    if (var && PyValue_AS_Variable(args) && (PyValue_AS_Variable(args)!=var)) {
      PyErr_Format(PyExc_TypeError, "wrong attribute value (expected value of '%s', got value of '%s')", var->get_name().c_str(), PyValue_AS_Variable(args)->get_name().c_str());
      return false;
    }
    else
      value = PyValue_AS_Value(args);
    return true;
  }

  if (PyOrSomeValue_Check(args)) {
    if (var) {
      if ((var->varType==TValue::INTVAR) && !PyOrDiscDistribution_Check(args)) {
        PyErr_Format(PyExc_TypeError, "attribute '%s' expects DiscDistribution, '%s' given", var->get_name().c_str(), args->ob_type->tp_name);
        return false;
      }
      if ((var->varType==TValue::FLOATVAR) && !PyOrContDistribution_Check(args)) {
        PyErr_Format(PyExc_TypeError, "attribute '%s' expects ContDistribution, '%s' given", var->get_name().c_str(), args->ob_type->tp_name);
        return false;
      }
    }
    int vtype;
    if (PyOrDiscDistribution_Check(args))
      vtype = TValue::INTVAR;
    else if (PyOrContDistribution_Check(args))
      vtype = TValue::FLOATVAR;
    else if (PyOrStringValue_Check(args))
      vtype = STRINGVAR;
    else if (PyOrPythonValue_Check(args))
      vtype = PYTHONVAR;
    else
      raiseError("unknovn variable type");

    value = TValue(PyOrange_AsSomeValue(args), vtype);
    return true;
  }
  
  if (PyString_Check(args)) {
    char *str = PyString_AsString(args);
    if (var)
      var->str2val(str, value);
    else
      if (!strcmp(str, "?"))
        value = TValue(TValue::INTVAR, valueDK);
      else if (!strcmp(str, "~"))
        value = TValue(TValue::INTVAR, valueDC);
      else {
        PyErr_Format(PyExc_TypeError, "cannot convert '%s' to a value of an unknown attribute", str);
        return false;
      }
    return true;
  }
  
  if (var && var->varType == PYTHONVAR) {
    value = TValue(mlnew TPythonValue(args), PYTHONVAR);
    return true;
  }
  
  if (args == Py_None) {
    value = var ? var->DK() : TValue(TValue::INTVAR, valueDK);
    return true;
  }

  if (PyInt_Check(args)) {
    int ii = int(PyInt_AsLong(args));

    if (var) {
      if (var->varType == TValue::FLOATVAR) {
        value = TValue(float(ii));
        return true;
      }

      if (var->varType == TValue::INTVAR) {
        if (var.is_derived_from(TEnumVariable)) {
          int nv = var.AS(TEnumVariable)->noOfValues();
          if (ii >= nv) {
            PyErr_Format(PyExc_TypeError, "value index %i out of range (0 - %i)", ii, nv-1);
            return false;
          }
        }

        value = TValue(ii);
        return true;
      }

      PyErr_Format(PyExc_TypeError,  "cannot convert an integer to a value of attribute '%s'", var->get_name().c_str());
      return false;
    }

    value = TValue(ii);
    return true;
  }

  if (PyFloat_Check(args)) {
    if (var && (var->varType != TValue::FLOATVAR)) {
      PyErr_Format(PyExc_TypeError,  "cannot convert a float to a value of attribute '%s'", var->get_name().c_str());
      return false;
    }

    value = TValue(float(PyFloat_AsDouble(args)));
    return true;
  }

  if (var && (var->varType == TValue::FLOATVAR)) {
    PyObject *pyfloat = PyNumber_Float(args);
    if (!pyfloat) {
      PyErr_Format(PyExc_TypeError, "cannot convert an object of type '%s' to value of attribute '%s'", args->ob_type->tp_name, var->get_name().c_str());
      return false;
    }

    value = TValue(float(PyFloat_AsDouble(pyfloat)));
    Py_DECREF(pyfloat);
    return true;
  }

  if (var)
    PyErr_Format(PyExc_TypeError,  "cannot convert an object of type '%s' to value of attribute '%s'", args->ob_type->tp_name, var->get_name().c_str());
  else
    PyErr_Format(PyExc_TypeError,  "cannot convert an object of type '%s' to value of attribute", args->ob_type->tp_name);

  return false;
}


/* Builds a TPyValue from arguments given in Python.
   See Value_FromArguments for details. */

bool convertFromPython(PyObject *args, TPyValue *&value)
{
  value = (TPyValue *)Value_FromArguments((PyTypeObject *)&PyOrValue_Type, args);
  return value!=NULL;
}



/* The main constructor for TPyValue.
   Gets a value and descriptor, allocates the memory and assigns fields. */

PyObject *Value_FromVariableValueType(PyTypeObject *type, PVariable var, const TValue &val)
{ 
  TPyValue *value = PyObject_GC_New(TPyValue, type);
  if (!value)
    return PYNULL;

  /* The below is needed since 'value' was allocated in C code, so it's
     constructor has never been called and the below fields (wrapped pointers)
     contain random data, which would lead to crash when trying to deallocate
     them. */
  value->value.svalV.init();
  value->variable.init();

  value->value = val;
  value->variable = var;

  PyObject_GC_Track(value);

  return (PyObject *)value;
}



/* Constructs a value from arguments in Python. Arguments must be given as a tuple
   with at least one element.
   - If the single element is a variable, a DK() value for that attribute is returned
   - Otherwise, it is converted using convertFromPython, without descriptor given

   If there are two elements
   - If one is variable, convertFromPython is used, passing the variable and the other
   - Otherwise, both must be integers and are used for varType and valueType.
*/

PyObject *Value_FromArguments(PyTypeObject *type, PyObject *args)
{   
  PyTRY
    PyObject *obj1;
    PyObject *obj2 = NULL;

    if (!PyArg_ParseTuple(args, "O|O:Value", &obj1, &obj2))
      return PYNULL;

    if (!obj2) {
      if (PyOrVariable_Check(obj1))
        return Value_FromVariableType(type, PyOrange_AsVariable(obj1));
      else {
        TValue val;
        return convertFromPython(obj1, val) ? Value_FromValueType(type, val) : PYNULL;
      }
    }

    TValue val;
    if (PyOrVariable_Check(obj1)) {
      const PVariable &var = PyOrange_AsVariable(obj1);
      return convertFromPython(obj2, val, var) ? Value_FromVariableValueType(type, var, val) : PYNULL;
    }
    else if (PyOrVariable_Check(obj2)) {
      const PVariable &var = PyOrange_AsVariable(obj2);
      return convertFromPython(obj1, val, var) ? Value_FromVariableValueType(type, var, val) : PYNULL;
    }
    else if (PyInt_Check(obj1) && PyInt_Check(obj2)) {
      int vartype = int(PyInt_AsLong(obj1));

      if (vartype == STRINGVAR)
        return Value_FromValueType(type, TValue(STRINGVAR, (signed char)PyInt_AsLong(obj2)));

      if (vartype > TValue::FLOATVAR) {
        PyErr_Format(PyExc_IndexError, "invalid value type (%i)", vartype);
        return PYNULL;
      }
        
      return Value_FromValueType(type, TValue((char)vartype, (signed char)PyInt_AsLong(obj2)));
    }

    PYERROR(PyExc_TypeError, "Value(): invalid arguments", PYNULL);
  PyCATCH
}




PyObject *Value_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(ROOT, "([Variable], [int | float | Value | ...])")
{ return Value_FromArguments(type, args); }


void Value_dealloc(TPyValue *self)
{ self->variable = PVariable();
  self->value.~TValue();

  if (PyObject_IsPointer(self)) {
    PyObject_GC_UnTrack((PyObject *)self);
    self->ob_type->tp_free((PyObject *)self); 
  }
}


int Value_traverse(TPyValue *self, visitproc visit, void *arg)
{ PVISIT(self->variable);
  PVISIT(self->value.svalV);
  return 0;
}


void Value_clear(TPyValue *self)
{ self->variable=PVariable();
  self->value.~TValue();
}


/* Returns a string representations for a value.
   - If descriptor is given, its val2str should take care of everything
   - If the value is special, we know that to do
   - If value is 
     - FLOATVAR, convert a floatV
     - INTVAR, print a intV in brackets
     - else if svalV is given, it should take care of itself
     - else, we return "###"
*/

char *pvs = NULL;
const char *TPyValue2string(TPyValue *self)
{ if (self->variable) {
    string str;
    self->variable->val2str(self->value, str);
    pvs = (char *)realloc(pvs, str.size()+1);
    strcpy(pvs, str.c_str());
  }
  else {
    if (self->value.isDK())
      return "?";
    if (self->value.isDC())
      return "~";
    if (self->value.isSpecial())
      return ".";

    pvs = (char *)realloc(pvs, 16);
    if (self->value.varType==TValue::FLOATVAR)
      sprintf(pvs, "%f", self->value.floatV);
    else if (self->value.varType==TValue::INTVAR)
      sprintf(pvs, "<%i>", self->value.intV);
    else if (self->value.svalV) {
      string str;
      self->value.svalV->val2str(str);
      pvs = (char *)realloc(pvs, str.size()+1);
      strcpy(pvs, str.c_str());
    }
    else
      return "###";
  }

  return pvs;
}



/* Compares two values. The first is always TPyValue.
   Comparisons of discrete are based on intV not on string representations
   If both are TPyValue, the values must be of same type
     - If both are special, they are equal/different if the valueType is
       equal/different. Operators >, <, <= and >= are not defined.
     - If only one is special, it's an error
     - If they are discrete and descriptors are known but different,
       each value's string representation is compared to the other's,
       both comparisons are made and must give the same result.
       If not, it's an error.
     - Otherwise, intV's r floatV's are compared
   If the other is an integer, it can be compared with discrete and
     continuous attributes
   If the other is a float, it can be compared with continuous attrs.
   If the first value is special and the other is string "~" or "?",
     they are compared as described above.
   Otherwise, the descriptor for the first value must be known and is
     used to convert the second value (if possible). The values are
     then compared by the same rules as if both were PyValues
     (except that both obviously have the same descriptor).
*/

#define errUndefinedIf(cond) if (cond) PYERROR(PyExc_TypeError, "Value.compare: cannot compare with undefined values", PYNULL);

PyObject *richcmp_from_sign(const int &i, const int &op)
{ int cmp;
  switch (op) {
		case Py_LT: cmp = (i<0); break;
		case Py_LE: cmp = (i<=0); break;
		case Py_EQ: cmp = (i==0); break;
		case Py_NE: cmp = (i!=0); break;
		case Py_GT: cmp = (i>0); break;
		case Py_GE: cmp = (i>=0); break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  
  PyObject *res;
  if (cmp)
    res = Py_True;
  else
    res = Py_False;
  Py_INCREF(res);
  return res;
}


PyObject *Value_richcmp(TPyValue *i, PyObject *j, int op)
{ 
  PyTRY

    const TValue &val1 = i->value;

    if (PyOrValue_Check(j)) {
      const TValue &val2 = PyValue_AS_Value(j);

      if (val1.varType != val2.varType)
        PYERROR(PyExc_TypeError, "Value.compare: can't compare values of different types", PYNULL)

      if (val1.isSpecial() || val2.isSpecial())
        if ((op==Py_EQ) || (op==Py_NE)) {
          PyObject *res = (val1.valueType==val2.valueType) == (op==Py_EQ) ? Py_True : Py_False;
          Py_INCREF(res);
          return res;
        }
        else {
          Py_INCREF(Py_NotImplemented);
          return Py_NotImplemented;
        }

      // Nominal values of different attributes are treated separately
      PVariable &var1 = i->variable;
      PVariable &var2 = PyValue_AS_Variable(j);
      if ((val1.varType==TValue::INTVAR) && var1 && var2 && (var1 != var2)) {
        TValue tempval;
        string tempstr;

        var2->val2str(val2, tempstr);
        if (var1->str2val_try(tempstr, tempval)) {
          int cmp1 = val1.compare(tempval);

          var1->val2str(val1, tempstr);
          if (var2->str2val_try(tempstr, tempval)) {
            int cmp2 = tempval.compare(val2);
            bool err = true;
            switch (op) {
              case Py_LE:
              case Py_GE: err = ((cmp1*cmp2) == -1); break;
              case Py_LT:
              case Py_GT: err = (cmp1!=cmp2); break;
              case Py_EQ:
              case Py_NE: err = ((cmp1==0) != (cmp2==0)); break;
            }

            if (err)
              PYERROR(PyExc_TypeError, "Value.compare: values are of different types and have different orders", PYNULL);
          }

          return richcmp_from_sign(cmp1, op);
        }

        var1->val2str(val1, tempstr);
        if (var2->str2val_try(tempstr, tempval))
          return richcmp_from_sign(tempval.compare(val2), op);

        PYERROR(PyExc_TypeError, "Value.compare: values are of different types and cannot be compared", PYNULL);
      }

      // Not nominal OR both values or of the same attribute
      return richcmp_from_sign(val1.compare(val2), op);
    }


    if (PyInt_Check(j)) {
      errUndefinedIf(val1.isSpecial());

      if (val1.varType==TValue::INTVAR)
        return richcmp_from_sign(val1.intV - (int)PyInt_AsLong(j), op);
      else if (val1.varType==TValue::FLOATVAR)
        return richcmp_from_sign(sign(val1.floatV - (int)PyInt_AsLong(j)), op);
    }

    else if (PyFloat_Check(j)) {
      errUndefinedIf(val1.isSpecial());
      if (val1.varType==TValue::FLOATVAR)
        return richcmp_from_sign(sign(val1.floatV - (float)PyFloat_AsDouble(j)), op);
    }

    else if (PyString_Check(j) && val1.isSpecial() && ((op==Py_EQ) || (op==Py_NE))) {
      char *s = PyString_AsString(j);
      PyObject *res = NULL;
      if (!strcmp(s, "~"))
        res = (val1.valueType==valueDC) == (op==Py_EQ) ? Py_True : Py_False;
      else if (!strcmp(s, "?"))
        res = (val1.valueType==valueDK) == (op==Py_EQ) ? Py_True : Py_False;
      if (res) {
        Py_INCREF(res);
        return res;
      }
    }

    if (i->variable) {
      TValue val2;
      if (!convertFromPython(j, val2, i->variable))
        return PYNULL;

      if (val1.isSpecial() || val2.isSpecial())
        if ((op==Py_EQ) || (op==Py_NE)) {
          PyObject *res = (val1.valueType==val2.valueType) == (op==Py_EQ) ? Py_True : Py_False;
          Py_INCREF(res);
          return res;
        }
        else {
          Py_INCREF(Py_NotImplemented);
          return Py_NotImplemented;
        }

      return richcmp_from_sign(val1.compare(val2), op);
    }
      
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;

  PyCATCH
}
#undef errUndefinedIf



PyObject *Value_str(TPyValue *self)
{ PyTRY
    return PyString_FromString(TPyValue2string(self)); 
  PyCATCH
}


PyObject *Value_repr(TPyValue *self)
{ PyTRY
    if (self->variable)
      return PyString_FromFormat("<orange.Value '%s'='%s'>", self->variable->get_name().c_str(), TPyValue2string(self));
    else
      return PyString_FromFormat("<orange.Value '%s'>", TPyValue2string(self)); 
  PyCATCH
}


bool checkSpecial(TPyValue *self, char *casttype)
{
  if (self->value.isSpecial()) {
    if (self->variable && self->variable->get_name().length())
      PyErr_Format(PyExc_TypeError, "value of '%s' is unknown and cannot be %s", self->variable->get_name().c_str(), casttype);
    else
      PyErr_Format(PyExc_TypeError, "attribute value is unknown and cannot be %s", casttype);
    return false;
  }

  return true;
}

PyObject *Value_int(TPyValue *self)
{ PyTRY
    if (!checkSpecial(self, "cast to an integer"))
      return PYNULL;

    return Py_BuildValue("i", (self->value.varType==TValue::INTVAR) ? self->value.intV : int(self->value.floatV)); 
  PyCATCH
}


PyObject *Value_long(TPyValue *self)
{ PyTRY
    if (!checkSpecial(self, "cast to a long integer"))
      return PYNULL;

    return Py_BuildValue("l", (self->value.varType==TValue::INTVAR) ? long(self->value.intV) : long(self->value.floatV)); 
  PyCATCH
}


PyObject *Value_float(TPyValue *self)
{ PyTRY
    if (!checkSpecial(self, "cast to a float"))
      return PYNULL;

    return Py_BuildValue("f", (self->value.varType==TValue::INTVAR) ? float(self->value.intV) : self->value.floatV); 
  PyCATCH
}


inline bool checkForNumerical(const TValue &val1, const TValue &val2, const char *op)
{
  if (val1.isSpecial() || val2.isSpecial())
    PYERROR(PyExc_TypeError, "cannot %s unknown values", false);
  if ((val1.varType!=TValue::FLOATVAR) || (val2.varType!=TValue::FLOATVAR))
    PYERROR(PyExc_TypeError, "cannot %s non-continuous values", false);
  return true;
}


#define VALUEOP(opname,FUN,opverb) \
PyObject *Value_##opname(TPyValue *self, PyObject *other) \
{ PyTRY \
    const TValue &val1 = self->value; \
\
    if (PyOrValue_Check(other)) { \
      const TValue &val2 = PyValue_AS_Value(other); \
      return checkForNumerical(val1, val2, opverb) ? PyFloat_FromDouble(val1.floatV FUN val2.floatV) : PYNULL; \
    } \
\
    TValue val2; \
    return convertFromPython(other, val2, self->variable) && checkForNumerical(val1, val2, opverb) ? PyFloat_FromDouble(val1.floatV FUN val2.floatV) : PYNULL; \
  PyCATCH \
}


PyObject *Value_add(TPyValue *self, PyObject *other);
PyObject *Value_sub(TPyValue *self, PyObject *other);
PyObject *Value_mul(TPyValue *self, PyObject *other);
PyObject *Value_div(TPyValue *self, PyObject *other);

VALUEOP(add,+,"sum")
VALUEOP(sub,-,"subtract")
VALUEOP(mul,*,"multiply")
VALUEOP(div,/,"divide")


PyObject *Value_pow(TPyValue *self, PyObject *other, PyObject *)
{ PyTRY
    const TValue &val1 = self->value;
    
    if (!val1.isSpecial() && (val1.varType==TValue::FLOATVAR) && (val1.floatV<=0))
      PYERROR(PyExc_TypeError, "negative base value", false);

    if (PyOrValue_Check(other)) { 
      const TValue &val2 = PyValue_AS_Value(other); 
      return checkForNumerical(val1, val2, "add") ? PyFloat_FromDouble(exp(val2.floatV*log(val1.floatV))) : PYNULL;
    }
    else {
      TValue val2; 
      return    convertFromPython(other, val2, self->variable)
             && checkForNumerical(val1, val2, "add")
           ? PyFloat_FromDouble(exp(val2.floatV*log(val1.floatV)))
           : PYNULL;
    }
  PyCATCH 
}


PyObject *Value_neg(TPyValue *self)
{ PyTRY
    if (!checkSpecial(self, "negated"))
      return PYNULL;

    const TValue &val1 = self->value;
    if (val1.varType!=TValue::FLOATVAR)
      PYERROR(PyExc_TypeError, "cannot negate non-continuous value", false);
    return PyFloat_FromDouble(-val1.floatV);
  PyCATCH
}


PyObject *Value_abs(TPyValue *self)
{ PyTRY
    if (self->value.isSpecial())
      if (self->variable && self->variable->get_name().length()) {
        PyErr_Format(PyExc_TypeError, "cannot compute an absolute value of '%s' since its value is unknown", self->variable->get_name().c_str());
        return PYNULL;
      }
      else
        PYERROR(PyExc_TypeError, "cannot compute an absolute value of attribute since its value is unknown", PYNULL);

    const TValue &val1 = self->value;
    if (val1.varType!=TValue::FLOATVAR)
      PYERROR(PyExc_TypeError, "cannot compute abs of non-continuous value", false);
    return PyFloat_FromDouble(fabs(val1.floatV));
  PyCATCH
}


int Value_nonzero(TPyValue *i)
{ PyTRY
    return !i->value.isSpecial();
  PyCATCH_1
}


int Value_coerce(PyObject **i, PyObject **obj)
{ PyTRY
    if (PyString_Check(*obj)) {
      *i = Value_str(*(TPyValue **)i);
      if (!*i)
        return -1;
      Py_INCREF(*obj);
      return 0;
    }

    if (PyInt_Check(*obj)) {
      TPyValue *val = *(TPyValue **)i;
      if (val->value.varType==TValue::INTVAR) {
        *i = Value_int(val);
        if (!*i)
          return -1;
        Py_INCREF(*obj);
        return 0;
      }
      else if (val->value.varType==TValue::FLOATVAR) {
        *i = Value_float(val);
        if (!*i)
          return -1;
        double x = PyFloat_AsDouble(*obj);
		    *obj = PyFloat_FromDouble(x);
        return 0;
      }
      else
        return -1;
    }

    if (PyFloat_Check(*obj)) {
      *i = Value_float(*(TPyValue **)i);
      if (!*i)
        return -1;
      Py_INCREF(*obj);
      return 0;
    }

    if (PyLong_Check(*obj)) {
      *i = Value_long(*(TPyValue **)i);
      if (!*i)
        return -1;
      Py_INCREF(*obj);
      return 0;
    }

    return -1;
  PyCATCH_1
}



PyObject *Value_get_svalue(TPyValue *self)
{ PyTRY
    if (self->value.varType == PYTHONVAR) {
      PyObject *res = self->value.svalV ? ((TPythonValue &)(self->value.svalV.getReference())).value : Py_None;
      Py_INCREF(res);
      return res;
    }

    return WrapOrange(self->value.svalV);
  PyCATCH
}


int Value_set_svalue(TPyValue *self, PyObject *arg)
{ PyTRY

    if (arg == Py_None) {
      self->value.svalV = PSomeValue();
      return 0;
    }

    if (PyOrSomeValue_Check(arg)) {
      self->value.svalV = PyOrange_AsSomeValue(arg);
      return 0;
    }

    self->value.svalV = mlnew TPythonValue(arg);
    return 0;

  PyCATCH_1
}


PyObject *Value_get_value(TPyValue *self)
{ PyTRY
    return convertToPythonNative(self);
  PyCATCH
}


int Value_set_value(TPyValue *self, PyObject *arg)
{ PyTRY
    return convertFromPython(arg, self->value, self->variable) ? 0 : -1;
  PyCATCH_1
}


PyObject *PyValue_Type_FromLong(long);

PyObject *Value_get_valueType(TPyValue *self)
{ return PyValue_Type_FromLong((long)self->value.valueType); }


PyObject *Value_get_variable(TPyValue *self)
{ return WrapOrange(self->variable); }


int Value_set_variable(TPyValue *self, PyObject *arg)
{ PyTRY
    if (arg == Py_None) {
      self->variable = PVariable();
      return 0;
    }
    if (!PyOrVariable_Check(arg))
      PYERROR(PyExc_TypeError, "invalid argument for attribute 'variable'", -1)
    else {
      self->variable = PyOrange_AsVariable(arg);
      return 0;
    }
  PyCATCH_1
}


PyObject *PyVariable_Type_FromLong(long);

PyObject *Value_get_varType(TPyValue *self)
{ return PyVariable_Type_FromLong((long)self->value.varType); }




char *value_underscores[][2] = {
    {"firstvalue", "first_value"},
    {"nextvalue", "next_value"},
    {"randomvalue", "random_value"},
    {"isDC", "is_DC"},
    {"isDK", "is_DK"},
    {"isSpecial", "is_special"},
    {"varType", "var_type"},
    {"valueType", "value_type"},    
    {NULL, NULL}
};

PyObject *Value_getattr(PyObject *self, PyObject *name)
{
  char *orig = PyString_AsString(name);
  for(char *(*ei)[2] = value_underscores; **ei; ei++) {
      if (!strcmp(orig, **ei)) {
          PyObject *trans = PyString_FromString((*ei)[1]);
          PyObject *value = PyObject_GenericGetAttr((PyObject *)self, trans);
          Py_DECREF(trans);
          return value;
      }
  }

  return PyObject_GenericGetAttr((PyObject *)self, name);

}

PyObject *Value_random_value(TPyValue *self) PYARGS(METH_NOARGS, "(); Sets the value to a random")
{ PyTRY
    CHECK_VARIABLE
    self->value = self->variable->randomValue();
    RETURN_NONE
  PyCATCH
}


PyObject *Value_first_value(TPyValue *self)  PYARGS(METH_NOARGS, "() -> bool; Sets the value to the first value")
{ PyTRY
    CHECK_VARIABLE
    return PyInt_FromLong(self->variable->firstValue(self->value) ? 1 : 0);
  PyCATCH
}


PyObject *Value_next_value(TPyValue *self)  PYARGS(METH_NOARGS, "() -> bool; Increases the value (if possible)")
{ PyTRY
    CHECK_VARIABLE
    return PyInt_FromLong(self->variable->nextValue(self->value) ? 1 : 0);
  PyCATCH
}


PyObject *Value_isSpecial(TPyValue *self)  PYARGS(METH_NOARGS, "() -> bool; Returns true if value is DK, DC...")
{ return PyInt_FromLong(self->value.isSpecial() ? 1 : 0); }


PyObject *Value_isDK(TPyValue *self)  PYARGS(METH_NOARGS, "() -> bool; Returns true if value is DK")
{ return PyInt_FromLong(self->value.isDK() ? 1 : 0); }


PyObject *Value_isDC(TPyValue *self)  PYARGS(METH_NOARGS, "() -> bool; Returns true if value is DC")
{ return PyInt_FromLong(self->value.isDC() ? 1 : 0); }


PyObject *Value_native(TPyValue *self)   PYARGS(METH_NOARGS, "() -> bool; Converts the value into string or float")
{ PyTRY
    return convertToPythonNative(self);
  PyCATCH
}


#include "slist.hpp"

bool Value_pack(const TValue &value, TCharBuffer &buf, PyObject *&otherValues)
{
  const char svalFlag = value.svalV ? 1 << 5 : 0;
  if (svalFlag) {
    if (!otherValues)
      otherValues = PyList_New(0);
    PyObject *sv = WrapOrange(value.svalV);
    PyList_Append(otherValues, sv);
    Py_DECREF(sv);
  }

  if (value.valueType) {
    buf.writeChar(svalFlag | (value.valueType & 0x3f));
    return true;
  }

  if (value.varType == TValue::INTVAR) {
    if (value.intV < (1 << (sizeof(char) << 3))) {
      buf.writeChar((1 << 6) | svalFlag);
      buf.writeChar(char(value.intV));
    }

    else if (value.intV < (1 << (sizeof(short) << 3))) {
      buf.writeChar((2 << 6) | svalFlag);
      buf.writeShort((unsigned short)(value.intV));
    }

    else {
      buf.writeChar((3 << 6) | svalFlag);
      buf.writeInt(value.intV);
    }

    return true;
  }

  else if (value.varType == TValue::FLOATVAR) {
    buf.writeChar(svalFlag);
    buf.writeFloat(value.floatV);
  }

  else
    buf.writeChar(svalFlag);

  return true;
}


bool Value_unpack(TValue &value, TCharBuffer &buf, PyObject *otherValues, int &otherValuesIndex)
{
  char flags = buf.readChar();

  if (flags & (1 << 5))
    value.svalV = PyOrange_AsSomeValue(PyList_GetItem(otherValues, otherValuesIndex++));

  value.valueType = flags & 0x1f;

  if (value.valueType) {
    value.floatV = numeric_limits<float>::quiet_NaN();
    value.intV = numeric_limits<int>::max();
    return true;
  }

  if (value.varType == TValue::INTVAR) {
    flags >>= 6;
    if (flags == 1)
      value.intV = buf.readChar();
    else if (flags == 2)
      value.intV = buf.readShort();
    else if (flags == 3)
      value.intV = buf.readInt();
    value.floatV = numeric_limits<float>::quiet_NaN();
  }

  else if (value.varType == TValue::FLOATVAR) {
    value.floatV = buf.readFloat();
    value.intV = numeric_limits<int>::max();
  }
    
  return true;
}

PyObject *Value__reduce__(PyObject *self)
{
  PyTRY
    TCharBuffer buf(16);
    PyObject *otherValues = NULL;
    buf.writeChar(PyValue_AS_Value(self).varType);
    Value_pack(PyValue_AS_Value(self), buf, otherValues);
    if (!otherValues) {
      otherValues = Py_None;
      Py_INCREF(otherValues);
    }
    
    return Py_BuildValue("O(Ns#N)", getExportedFunction("__pickleLoaderValue"),
                                   WrapOrange(PyValue_AS_Variable(self)),
                                   buf.buf, buf.length(),
                                   otherValues);
  PyCATCH
}


PyObject *__pickleLoaderValue(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(variable, packed_values, other_values)")
{
  PyTRY
    PVariable var;
    char *pbuf;
    int bufSize;
    PyObject *otherValues;
    if (!PyArg_ParseTuple(args, "O&s#O:__pickleLoaderValue", ccn_Variable, &var, &pbuf, &bufSize, &otherValues))
      return PYNULL;

    TCharBuffer buf(pbuf);
    int otherValuesIndex = 0;
    TValue val((const unsigned char &)(buf.readChar()));
    Value_unpack(val, buf, otherValues, otherValuesIndex);
    return Value_FromVariableValue(var, val);
  PyCATCH
}

#undef CHECK_VARIABLE
#undef CHECK_SPECIAL_OTHER


// This is in a separate file to avoid scanning by pyxtract
#include "valuelisttemplate.hpp"

// Modified new and related stuff, removed rich_cmp (might be added later, but needs to be programmed specifically)
PValueList PValueList_FromArguments(PyObject *arg, PVariable var = PVariable())
{ return TValueListMethods::P_FromArguments(arg, var); }


PyObject *ValueList_FromArguments(PyTypeObject *type, PyObject *arg, PVariable var = PVariable())
{ return TValueListMethods::_FromArguments(type, arg, var); }


PyObject *ValueList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Value>)")  ALLOWS_EMPTY
{ return TValueListMethods::_new(type, arg, kwds); }


PyObject *ValueList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return TValueListMethods::_getitem(self, index); }
int       ValueList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return TValueListMethods::_setitem(self, index, item); }
PyObject *ValueList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return TValueListMethods::_getslice(self, start, stop); }
int       ValueList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return TValueListMethods::_setslice(self, start, stop, item); }
Py_ssize_t ValueList_len_sq(TPyOrange *self) { return TValueListMethods::_len(self); }
PyObject *ValueList_concat(TPyOrange *self, PyObject *obj) { return TValueListMethods::_concat(self, obj); }
PyObject *ValueList_repeat(TPyOrange *self, Py_ssize_t times) { return TValueListMethods::_repeat(self, times); }
PyObject *ValueList_str(TPyOrange *self) { return TValueListMethods::_str(self); }
PyObject *ValueList_repr(TPyOrange *self) { return TValueListMethods::_str(self); }
int       ValueList_contains(TPyOrange *self, PyObject *obj) { return TValueListMethods::_contains(self, obj); }
PyObject *ValueList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Value) -> None") { return TValueListMethods::_append(self, item); }
PyObject *ValueList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return TValueListMethods::_extend(self, obj); }
PyObject *ValueList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Value) -> int") { return TValueListMethods::_count(self, obj); }
PyObject *ValueList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> ValueList") { return TValueListMethods::_filter(self, args); }
PyObject *ValueList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Value) -> int") { return TValueListMethods::_index(self, obj); }
PyObject *ValueList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return TValueListMethods::_insert(self, args); }
PyObject *ValueList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return TValueListMethods::_native(self); }
PyObject *ValueList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Value") { return TValueListMethods::_pop(self, args); }
PyObject *ValueList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Value) -> None") { return TValueListMethods::_remove(self, obj); }
PyObject *ValueList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return TValueListMethods::_reverse(self); }
PyObject *ValueList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return TValueListMethods::_sort(self, args); }
PyObject *ValueList__reduce__(TPyOrange *self, PyObject *) { return TValueListMethods::_reduce(self); }



PyObject *VarTypes()
{ PyObject *vartypes=PyModule_New("Orange.data.Type");
  PyModule_AddObject(vartypes, "None", PyVariable_Type_FromLong((int)TValue::NONE));
  PyModule_AddObject(vartypes, "Discrete", PyVariable_Type_FromLong((int)TValue::INTVAR));
  PyModule_AddObject(vartypes, "Continuous", PyVariable_Type_FromLong((int)TValue::FLOATVAR));
  PyModule_AddObject(vartypes, "Other", PyVariable_Type_FromLong((int)TValue::FLOATVAR+1)); // for compatibility; don't use!
  PyModule_AddObject(vartypes, "String", PyVariable_Type_FromLong((int)STRINGVAR));
  return vartypes;
}

/* Left here for compatibility */

PYCONSTANTFUNC(VarTypes, VarTypes)

/* This cannot be done in the header since Value is not derived from Orange */

TNamedConstantsDef Value_Type_values[] = {{"Regular", 0}, {"DC", valueDC}, {"DK", valueDK}, {0, 0}};

PYXTRACT_IGNORE static PyObject *Value_Type_repr(PyObject *self)
{
  return stringFromList(self, Value_Type_values);
}

PYXTRACT_IGNORE PyObject *Value_Type__reduce__(PyObject *self);
PyMethodDef Value_Type_methods[] = { {"__reduce__", (binaryfunc)Value_Type__reduce__, METH_NOARGS, "reduce"}, {NULL, NULL}};
PyTypeObject PyValue_Type_Type = {PyObject_HEAD_INIT(&PyType_Type) 0, "Value.Type", sizeof(PyIntObject), 0, 0, 0, 0, 0, 0, (reprfunc)Value_Type_repr, 0, 0, 0, 0, 0, (reprfunc)Value_Type_repr, 0, 0, 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES, 0, 0, 0, 0, 0, 0, 0, Value_Type_methods, 0, 0, &PyInt_Type};

PyObject *PyValue_Type_FromLong(long ok)
{ PyIntObject *r = PyObject_New(PyIntObject, &PyValue_Type_Type);
  r->ob_ival = ok;
  return (PyObject *)r;
}

void *PTValue_Type(void *l)
{ return PyValue_Type_FromLong(*(long *)l); }


PYXTRACT_IGNORE PyObject *Value_Type__reduce__(PyObject *self)
{ return Py_BuildValue("O(i)", getExportedFunction("__pickleLoaderValueType"), ((PyIntObject *)(self))->ob_ival); }

PyObject *__pickleLoaderValueType(PyObject *, PyObject *args) PYARGS(METH_O, "")
{ return PyValue_Type_FromLong(PyInt_AsLong(args)); }

/* Left for backward compatibility; also used the opportunity to initialize the type */

PyObject *ValueTypes()
{ PyType_Ready(&PyValue_Type_Type);
  PyValue_Type_Type.tp_print = 0;
  PyObject *valuetypes=PyModule_New("ValueTypes");
  PyModule_AddObject(valuetypes, "Regular", PyValue_Type_FromLong(valueRegular));
  PyModule_AddObject(valuetypes, "DC", PyValue_Type_FromLong(valueDC));
  PyModule_AddObject(valuetypes, "DK", PyValue_Type_FromLong(valueDK));
  return valuetypes;
}

PYCONSTANTFUNC(ValueTypes, ValueTypes)

PYCLASSCONSTANT(Value, Regular, PyValue_Type_FromLong(valueRegular))
PYCLASSCONSTANT(Value, DC, PyValue_Type_FromLong(valueDC))
PYCLASSCONSTANT(Value, DK, PyValue_Type_FromLong(valueDK))

#include "cls_value.px"
