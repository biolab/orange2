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

class TMLClassDefinition;
extern TMLClassDefinition MLDef_Domain;
extern TMLClassDefinition MLDef_Variable;

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"

#include "cls_value.hpp"
#include "cls_example.hpp"
#include "cls_orange.hpp"
#include "lib_kernel.hpp"
#include "converts.hpp"

#include "externs.px"


DATASTRUCTURE(Example - Orange.data.Instance, TPyExample, 0)


bool convertFromPythonExisting(PyObject *lst, TExample &example)
{
  PDomain dom=example.domain;

  if (PyOrExample_Check(lst)) {
    const TExample &orex = PyExample_AS_ExampleReference(lst);
    if (orex.domain != dom)
      dom->convert(example, orex);
    else
      example = orex;
    return true;
  }

  if (!PyList_Check(lst)) {
    PyErr_Format(PyExc_TypeError, "invalid argument type (expected list, got '%s)", lst ? lst->ob_type->tp_name : "None");
    return false;
  }

  if (Py_ssize_t(dom->variables->size()) != PyList_Size(lst)) {
    PyErr_Format(PyExc_IndexError, "invalid list size (%i items expected)", dom->variables->size());
    return false;
  }

  Py_ssize_t pos = 0;
  TExample::iterator ei(example.begin());
  PITERATE(TVarList, vi, dom->variables) {
    PyObject *li=PyList_GetItem(lst, pos++);
    if (!li)
      PYERROR(PyExc_SystemError, "can't read the list", false);

    if (PyOrValue_Check(li))
      if (PyValue_AS_Variable(li) ? (PyValue_AS_Variable(li) != *vi) : (PyValue_AS_Value(li).varType=!(*vi)->varType) ) {
        PyErr_Format(PyExc_TypeError, "wrong value type for attribute no. %i (%s)", pos, (*vi)->get_name().c_str());
        return false;
      }
      else
        *(ei++)=PyValue_AS_Value(li);

    else {
      if (li == Py_None) {
        *(ei++) = (*vi)->DK();
      }
        
      else if (PyString_Check(li))
          (*vi)->str2val(string(PyString_AsString(li)), *(ei++));

      else if ((*vi)->varType==TValue::INTVAR) {
        if (PyInt_Check(li))
          *(ei++)=TValue(int(PyInt_AsLong(li)));
        else {
          PyErr_Format(PyExc_TypeError, "attribute no. %i (%s) is ordinal, string value expected", pos, (*vi)->get_name().c_str());
          return false;
        }
      }
      else if ((*vi)->varType==TValue::FLOATVAR) {
        float f;
        if (PyNumber_ToFloat(li, f))
          *(ei++) = TValue(f);
        else {
          PyErr_Format(PyExc_TypeError, "attribute no. %i (%s) is continuous, float value expected", pos, (*vi)->get_name().c_str());
          return false;
        }
      }
      else
        ei++;
    }
  }

  return true;
}


bool convertFromPython(PyObject *lst, TExample &example, PDomain dom)
{ example=TExample(dom);
  return convertFromPythonExisting(lst, example);
}


int cc_Example(PyObject *obj, void *ptr)
{ if (!PyOrExample_Check(obj))
    return 0;
  *(PExample *)(ptr) = PyExample_AS_Example(obj);
  return 1;
}

int ccn_Example(PyObject *obj, void *ptr)
{ if (obj == Py_None) {
    *(PExample *)(ptr) = PExample();
    return 1;
  }
  else
    return cc_Example(obj, ptr);
}


int ptr_Example(PyObject *obj, void *ptr)
{ if (!PyOrExample_Check(obj))
    return 0;
  *(TExample **)(ptr) = PyExample_AS_Example(obj).getUnwrappedPtr();
  return 1;
}

int ptrn_Example(PyObject *obj, void *ptr)
{ if (obj == Py_None) {
    *(TExample **)(ptr) = NULL;
    return 1;
  }
  else
    return ptr_Example(obj, ptr);
}


PyObject *Example_FromExample(PyTypeObject *type, PExample example, POrange lock)
{ TPyExample *self=PyObject_GC_New(TPyExample, type);
  self->example.init();
  self->lock.init();
  self->example = example;
  self->lock = lock;
  PyObject_GC_Track(self);
  return (PyObject *)self;
}


void Example_dealloc(TPyExample *self)
{ self->lock.~POrange();
  self->example.~PExample();
  /* Should not call tp_free if it is a reference 
     Destructor is also called by exit proc, not by wrapped */
  if (PyObject_IsPointer(self)) {
    PyObject_GC_UnTrack((PyObject *)self);
    self->ob_type->tp_free((PyObject *)self); 
  }
}


int Example_traverse(TPyExample *self, visitproc visit, void *arg)
{ PVISIT(self->lock)
  if (!self->lock) // don't visit if it's a reference!
    PVISIT(self->example);

  return 0;
}

int Example_clear(TPyExample *self)
{ self->lock=POrange();
  self->example=PExample();
  return 0;
}


bool readBoolFlag(PyObject *keywords, char *flag);

CONSTRUCTOR_KEYWORDS(Example, "filterMetas filter_metas")


PyObject *Example_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(ROOT, "(domain, [list of values])")
{ PyTRY
    PyObject *list=PYNULL;
    PDomain dom;

    if (PyArg_ParseTuple(args, "O&|O", cc_Domain, &dom, &list)) {
      if (list && PyOrExample_Check(list)) {
        PExample ex = mlnew TExample(dom, PyExample_AS_Example(list).getReference(), readBoolFlag(keywords, "filterMetas") || readBoolFlag(keywords, "filter_metas"));
        return Example_FromWrappedExample(ex);
      }

      PyObject *example = Example_FromDomain(dom);
      
      if (list) {
        if (PyList_Check(list) && PyList_Size(list) && PyOrExample_Check(PyList_GET_ITEM(list, 0))) {
          TExampleList elist;
          PyObject *iterator = PyObject_GetIter(list);
          PyObject *item = PyIter_Next(iterator);
          for(; item; item = PyIter_Next(iterator)) {
            if (!PyOrExample_Check(item)) {
              Py_DECREF(item);
              break;
            }
            elist.push_back(PyExample_AS_Example(item));
            Py_DECREF(item);
          }
          Py_DECREF(iterator);
          if (item)
            raiseError("invalid elements in list for example join");
          else {
            PExample ex = mlnew TExample(dom, PExampleList(elist));
            return Example_FromWrappedExample(ex);
          }
        }
        else if (!convertFromPythonExisting(list, PyExample_AS_ExampleReference(example))) {
          Py_DECREF(example);
          return PYNULL;
        }
      }

      return example;
    }

    PyErr_Clear();

    PExample example;
    int keepId = 0;
    if (PyArg_ParseTuple(args, "O&|i", cc_Example, &example, &keepId)) {
      PExample ex = mlnew TExample(example.getReference());
      if (!keepId)
        ex->id = getExampleId();
      return Example_FromWrappedExample(ex);
    }
      
    PYERROR(PyExc_TypeError, "domain and (optionally) list arguments accepted", PYNULL);
  PyCATCH
}



int getMetaIdFromPy(PExample example, PyObject *index, PVariable &var)
{ if (PyInt_Check(index)) {
    int idx=PyInt_AsLong(index);
    var=example->domain->getMetaVar(idx, false); // it may also be NULL
    return idx;
  }
  else if (PyString_Check(index)) {
    TMetaDescriptor const *desc=example->domain->metas[string(PyString_AsString(index))];
    if (!desc) {
      PyErr_Format(PyExc_IndexError, "invalid meta variable name '%s'", PyString_AsString(index));
      return 0;
    }
    var=desc->variable;
    return desc->id;
  }
  else if (PyOrVariable_Check(index)) {
    var = PyOrange_AsVariable(index);
    int idx = example->domain->getMetaNum(var, false);
    if (idx == ILLEGAL_INT)
      PYERROR(PyExc_IndexError, "invalid meta variable", 0);
    return idx;
  }

  PYERROR(PyExc_IndexError, "invalid meta variable", 0);
}


int weightIndex(const TExample &example, PyObject *pyindex)
{
  if (pyindex == Py_None)
    return 0;

  if (PyInt_Check(pyindex))
    return (int)PyInt_AsLong(pyindex);

  PVariable var = varFromArg_byDomain(pyindex, example.domain);
  if (!var) 
    PYERROR(PyExc_TypeError, "invalid arguments or unknown attribute", ILLEGAL_INT);

  return example.domain->getVarNum(var);
}


PyObject *Example_reference(TPyExample *pex) PYARGS(METH_NOARGS, "unique reference (pointer to) the object")
{
  return PyInt_FromLong(int(&PyExample_AS_ExampleReference(pex)));
}


PyObject *Example_get_weight(TPyExample *pex, PyObject *pyindex) PYARGS(METH_O, "(id) -> weight; Returns example's weight")
{
  PyTRY
    const TExample &example = PyExample_AS_ExampleReference(pex);
    int index = weightIndex(example, pyindex);
    if (index == ILLEGAL_INT)
      return PYNULL;

    if (!index)
      return PyFloat_FromDouble(1.0);

    TValue val = example.getMeta(index);
    if (val.isSpecial() || (val.varType!=TValue::FLOATVAR))
      PYERROR(PyExc_TypeError, "invalid weight", PYNULL);

    return PyFloat_FromDouble((double)val.floatV);
  PyCATCH
}


PyObject *Example_set_weight(TPyExample *pex, PyObject *args) PYARGS(METH_VARARGS, "(id[, weight]); Sets example's weight to given value")
{ PyTRY
    PyObject *pyindex;
    float weight = 1;

    if (!PyArg_ParseTuple(args, "O|f:setweight", &pyindex, &weight))
      return PYNULL;

    TExample &example = PyExample_AS_ExampleReference(pex);
    int index = weightIndex(example, pyindex);

    if (index == ILLEGAL_INT)
      return PYNULL;

    if (index>0)
      PYERROR(PyExc_IndexError, "Example.setweight: invalid weight id", PYNULL);      

    if (index)
      example.setMeta(index, TValue(weight));

    RETURN_NONE;
  PyCATCH
}


PyObject *Example_remove_weight(TPyExample *pex, PyObject *pyindex) PYARGS(METH_O, "(id); Removes examples's weight")
{ PyTRY
    TExample &example = PyExample_AS_ExampleReference(pex);
    int index = weightIndex(example, pyindex);

    if (index == ILLEGAL_INT)
      return PYNULL;

    if (index>0)
      PYERROR(PyExc_IndexError, "Example.setweight: invalid weight id", PYNULL);      

    if (index)
      example.removeMeta(index);

    RETURN_NONE;
  PyCATCH
}


PyObject *Example_get_meta(TPyExample *pex, PyObject *index) PYARGS(METH_O, "(id | var) -> Value; Gets a meta-value")
{ PyTRY
    PVariable var;
    int idx = getMetaIdFromPy(PyExample_AS_Example(pex), index, var);
    if (!idx)
      return PYNULL; 

    return convertToPythonNative(PyExample_AS_Example(pex)->getMeta(idx), var);

    // This could be better, but I wouldn't dare to change this
    // return Value_FromVariableValue(var, PyExample_AS_Example(pex)->getMeta(idx));
  PyCATCH
}


PyObject *Example_get_metas(TPyExample *pex, PyObject *args) PYARGS(METH_VARARGS, "([key-type]) -> dictionary with a copy of example's meta attributes")
{
  PyTRY
    PyObject *pyoptional = NULL;
    PyTypeObject *keytype = NULL;
    int optional = ILLEGAL_INT;
    if (!PyArg_ParseTuple(args, "|OO:Example.getmetas", &pyoptional, &keytype))
      return NULL;

    if (!keytype) {
      if (pyoptional && PyType_Check(pyoptional)) {
        keytype = (PyTypeObject *)pyoptional;
        pyoptional = NULL;
      }
      else {
        keytype = &PyInt_Type;
      }
    }
    
    if (pyoptional) {
      if (!PyInt_Check(pyoptional)) {
        PyErr_Format(PyExc_TypeError, "invalid type for argument 'optional' (expected int, got '%s')", pyoptional->ob_type->tp_name);
        return NULL;
      }
      optional = PyInt_AsLong(pyoptional);
    }
    
    if ((keytype != &PyInt_Type) && (keytype != &PyString_Type) && (keytype != (PyTypeObject *)&PyOrVariable_Type))
      PYERROR(PyExc_TypeError, "invalid key type (should be nothing, int, str, or orange.Variable)", NULL);
      
    PExample ex = PyExample_AS_Example(pex);
    const TDomain &dom = ex->domain.getReference();

    PyObject *res = PyDict_New();

    try {
      const_ITERATE(TMetaValues, mi, ex->meta) {
        const TMetaDescriptor *md = dom.getMetaDescriptor(mi->first, false);
        
        if ((optional != ILLEGAL_INT) && (!md || (md->optional != optional)))
          continue;
        
        PyObject *key;
        PVariable variable = md ? md->variable : PVariable();
        
        if (keytype == &PyInt_Type)
          key = PyInt_FromLong(mi->first);
        else {
          if (!variable)
            continue;
          if (keytype == &PyString_Type)
            key = PyString_FromString(variable->get_name().c_str());
          else
            key = WrapOrange(variable);
        }

        PyObject *value = Value_FromVariableValue(variable, mi->second);
        PyDict_SetItem(res, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
      }

      return res;
    }
    catch (...) {
      Py_DECREF(res);
      throw;
    }

  PyCATCH

}


PyObject *Example_has_meta(TPyExample *pex, PyObject *index) PYARGS(METH_O, "(id | var) -> bool")
{ PyTRY
    PVariable var;
    int idx = getMetaIdFromPy(PyExample_AS_Example(pex), index, var);
    PyErr_Clear();
    return PyInt_FromLong(idx && PyExample_AS_Example(pex)->hasMeta(idx) ? 1 : 0);
  PyCATCH
}


PyObject *Example_set_value(TPyExample *pex, PyObject *vala) PYARGS(METH_O, "(Value) -> None")
{ PyTRY
    if (!PyOrValue_Check(vala))
      PYERROR(PyExc_TypeError, "Example.setvalue: orange.Value expected", PYNULL);

    PVariable var = PyValue_AS_Variable(vala);
    if (!var)
      PYERROR(PyExc_TypeError, "Example.setvalue: values 'variable' should not be None", PYNULL);

    PExample example=PyExample_AS_Example(pex);
    int idx = example->domain->getVarNum(var);

    if (idx>=0)
      example->operator[](idx) = PyValue_AS_Value(vala);
    else
      example->setMeta(idx, PyValue_AS_Value(vala));

    RETURN_NONE;
  PyCATCH
}


PyObject *Example_set_meta(TPyExample *pex, PyObject *args) PYARGS(METH_VARARGS, "(Value, int) | (variable, value); Sets a meta-value")
{ PyTRY
    PExample example=PyExample_AS_Example(pex);
  
    PyObject *par1, *par2=PYNULL;
    if (!PyArg_ParseTuple(args, "O|O", &par1, &par2))
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

    int idx=0;

    if (PyOrValue_Check(par1)) {
      // first parameter is a PyValue
      // second parameter is an index and is accepted iff variable not among domain's meta variables
      TMetaDescriptor *desc=PyValue_AS_Variable(par1) ? example->domain->metas[PyValue_AS_Variable(par1)] : (TMetaDescriptor *)NULL;
      if (desc) 
        if (par2)
          PYERROR(PyExc_TypeError, "second argument (index) not expected", PYNULL)
        else
          idx=desc->id;
      else
       if (!par2)
         PYERROR(PyExc_TypeError, "second argument (index) needed", PYNULL)
       else
         if (!PyInt_Check(par2))
           PYERROR(PyExc_TypeError, "invalid index type (int expected)", PYNULL)
         else {
           idx = int(PyInt_AsLong(par2));
           if (idx>=0)
             PYERROR(PyExc_TypeError, "invalid meta-id index (negative integer expected)", PYNULL);
         }

      example->setMeta(idx, PyValue_AS_Value(par1));
    }

    else if (!par2)
      PYERROR(PyExc_TypeError, "invalid arguments (second argument missing or the first is of wrong type)", PYNULL)

    else if (PyOrVariable_Check(par1) || PyString_Check(par1) || PyInt_Check(par1)) {
      // first parameter will denote the variable, the second the value
      int idx;
      PVariable var;

      if (PyInt_Check(par1)) {
        idx = PyInt_AsLong(par1);
        TMetaDescriptor *desc=example->domain->metas[idx];
        if (desc)
          var = desc->variable;
      }

      else {
        TMetaDescriptor *desc=example->domain->metas[
            PyOrVariable_Check(par1) ? PyOrange_AsVariable(par1)->get_name()
                                     : string(PyString_AsString(par1))];
        if (!desc)
          PYERROR(PyExc_TypeError, "invalid variable", PYNULL);
        idx = desc->id;
        var = desc->variable;
      }

      if (idx>=0)
        PYERROR(PyExc_TypeError, "invalid meta-id index (negative integer expected)", PYNULL);

      TValue val;
      if (!convertFromPython(par2, val, var))
        return PYNULL;
      example->setMeta(idx, val);
    }

    else
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL)

    RETURN_NONE;
  PyCATCH
}
  

PyObject *Example_remove_meta(TPyExample *pex, PyObject *index) PYARGS(METH_O, "(id); Removes a meta-value")
{ PyTRY
    PVariable var;
    int idx = getMetaIdFromPy(PyExample_AS_Example(pex), index, var);
    if (!idx)
      return PYNULL; 

    PyExample_AS_Example(pex)->removeMeta(idx);
    RETURN_NONE;
  PyCATCH
}



PyObject *Example_get_class(TPyExample *pex) PYARGS(METH_NOARGS, "()  -> Value; Returns example's class")
{ PyTRY
      const TExample &example = PyExample_AS_ExampleReference(pex);
      const PVariable &classVar = example.domain->classVar;

      if (!classVar)
        raiseError("class-less domain");

      return Value_FromVariableValue(classVar, example.getClass());
  PyCATCH
}


PyObject *Example_get_classes(TPyExample *pex) PYARGS(METH_NOARGS, "()  -> [Values]; Returns example's class")
{ PyTRY
      const TExample &example = PyExample_AS_ExampleReference(pex);
      PyObject *list=PyList_New(0);
      TExample::const_iterator ei=example.values_end;
      const_PITERATE(TVarList, vi, example.domain->classVars) {
          PyObject *valo = Value_FromVariableValue(*vi, *ei++);
          PyList_Append(list, valo);
          Py_DECREF(valo);
      }
      return list;
  PyCATCH
}


PyObject *Example_set_class(TPyExample *pex, PyObject *val) PYARGS(METH_O, "(value); Sets example's class")
{ PyTRY
    PExample &example=PyExample_AS_Example(pex);
    PVariable &classVar = example->domain->classVar;

    if (!classVar)
      PYERROR(PyExc_SystemError, "classless domain", PYNULL);

    TValue value;
    if (!convertFromPython(val, value, classVar)) 
      return PYNULL;
    example->setClass(value);

    RETURN_NONE;
  PyCATCH
}


PyObject *Example_set_classes(TPyExample *pex, PyObject *val) PYARGS(METH_O, "(list-of-Values); Returns example's class")
{ PyTRY
      const TExample &example = PyExample_AS_ExampleReference(pex);
      if (!PyList_Check(val)) {
          PYERROR(PyExc_TypeError, "list of values expected", PYNULL);
      }
      if (PyList_Size(val) != example.domain->classVars->size()) {
          PyErr_Format(PyExc_IndexError, "expected %i values, got %i", example.domain->classVars->size(), PyList_Size(val));
          return NULL;
      }
      TExample::iterator ei=example.values_end;
      int pos = 0;
      const_PITERATE(TVarList, vi, example.domain->classVars) {
          if (!convertFromPython(PyList_GET_ITEM(val, pos++), *ei++, *vi))
              return PYNULL;
      }
      RETURN_NONE;
  PyCATCH
}



PyObject *Example_compatible(TPyExample *pex, PyObject *args) PYARGS(METH_VARARGS, "(example[, ignoreClass]); Returns true if examples are compatible")
{ PyTRY
    PExample example;
    int ic = 0;
    if (!PyArg_ParseTuple(args, "O&|i", cc_Example, &example, &ic))
      PYERROR(PyExc_TypeError, "example and, optionally, a flag for ignoring the class expected", PYNULL)

    return PyInt_FromLong(PyExample_AS_Example(pex)->compatible(example.getReference(), ic != 0) ? 1 : 0);
  PyCATCH
}


PyObject *PyExampleIter_New(TPyExample *);

PyObject *Example_iter(TPyExample *pex)
{ return PyExampleIter_New(pex);
}


int getAttributeIndex(PDomain domain, PyObject *vara)
{
    if (PyInt_Check(vara)) {
      int ind = int(PyInt_AsLong(vara));
      if (ind >= (int)(domain->variables->size())) {
        PyErr_Format(PyExc_IndexError, "index %i to large (> %i)", ind, domain->variables->size()-1);
        return ILLEGAL_INT;
      }

      // Exception: example[-1] gives class value
      return ind==-1 ? domain->variables->size()-1 : ind;
    }

    PVariable var=varFromArg_byDomain(vara, domain);
    if (!var) 
      PYERROR(PyExc_TypeError, "invalid arguments or unknown attribute name", ILLEGAL_INT);

    return domain->getVarNum(var);
}


PyObject *Example_getitem(TPyExample *pex, PyObject *vara)
{ PyTRY
    PExample example = PyExample_AS_Example(pex);

    int ind = getAttributeIndex(example->domain, vara);
    if (ind==ILLEGAL_INT)
      return PYNULL;

    if ((ind < 0) && !example->hasMeta(ind)) {
      TMetaDescriptor *metadesc = example->domain->metas[ind];
      if (metadesc && metadesc->optional)
        return Value_FromVariableValue(metadesc->variable, metadesc->variable->DK());
    }
    
    /* getVar will return NULL if ind is meta-attribute not registered with the domain.
       That's OK - we don't need PVariable (Value_FromValue would do exactly the same).
       operator[] will raise an exception if meta-value is requested and the example
       doesn't have it. */
    return Value_FromVariableValue(example->domain->getVar(ind, false), example->operator[](ind));
  PyCATCH
}


int Example_setitem(TPyExample *pex, PyObject *vara, PyObject *vala)
{ PyTRY
    PExample example = PyExample_AS_Example(pex);
    const int ind = getAttributeIndex(example->domain, vara);
    if (ind==ILLEGAL_INT)
      return -1;

    PVariable var = example->domain->getVar(ind, false);

    if (PyOrValue_Check(vala)) {
      if (PyValue_AS_Variable(vala) && var && (PyValue_AS_Variable(vala)!=var)) {
          string vals;
          PyValue_AS_Variable(vala)->val2str(PyValue_AS_Value(vala), vals);
          if (ind>=0)
            var->str2val(vals, example->operator[](ind));
          else {
            TValue val;
            var->str2val(vals, val);
            example->setMeta(ind, val);
          }
        }
      else {
        if (ind>=0)
          example->operator[](ind) = PyValue_AS_Value(vala);
        else
          example->setMeta(ind, PyValue_AS_Value(vala));
      }
    }

    else {
      TValue value;
      if (!convertFromPython(vala, value, var)) 
        return -1;
      if (ind>=0)
        example->operator[](ind) = value;
      else
        example->setMeta(ind, value);
    }

    return 0;

  PyCATCH_1
}



inline PyObject *toValue(const TValue &val, PVariable var, int natvt, PyObject *forDK, PyObject *forDC, PyObject *forSpecial)
{ switch (natvt) {
    case -1:
      return (val.varType==TValue::INTVAR)
            ? PyInt_FromLong(long(val.intV))
            : PyFloat_FromDouble(double(val.floatV));

    case  0:
      if (val.isSpecial()) {
        PyObject *res;
        if (val.isDC())
          res = forDC;
        else if (val.isDK())
          res = forDK;
        else
          res = forSpecial;
        Py_INCREF(res);
        return res;
      }   
      else
        return convertToPythonNative(val, var);

    default:
      return Value_FromVariableValue(var, val);
  }
}

PyObject *convertToPythonNative(const TExample &example, int natvt, bool tuples, PyObject *forDK, PyObject *forDC, PyObject *forSpecial)
{
  if (forDK)
    Py_INCREF(forDK);
  else
    forDK = PyString_FromString("?");

  if (forDC)
    Py_INCREF(forDC);
  else
    forDC = PyString_FromString("~");

  if (forSpecial)
    Py_INCREF(forSpecial);
  else
    forSpecial = PyString_FromString(".");

  PyObject *list=PyList_New(0);
  TExample::const_iterator ei=example.begin();
  const_PITERATE(TVarList, vi, example.domain->attributes) {
    PyObject *valo = toValue(*(ei++), *vi, natvt, forDK, forDC, forSpecial);
    PyList_Append(list, valo);
    Py_DECREF(valo);
  }

  PyObject *res;

  if (example.domain->classVar) {
    PyObject *pyclass = toValue(example.getClass(), example.domain->classVar, natvt, forDK, forDC, forSpecial);
    if (tuples)
      res = Py_BuildValue("NN", list, pyclass);
    else {
      PyList_Append(list, pyclass);
      Py_DECREF(pyclass);
      res = list;
    }
  }
    
  else {
    if (tuples)
      res = Py_BuildValue("NO", list, Py_None);
    else
      res = list;
  }

  Py_DECREF(forDK);
  Py_DECREF(forDC);
  Py_DECREF(forSpecial);

  return res;
}


PyObject *Example_native(TPyExample *pex, PyObject *args, PyObject *keyws) PYARGS(METH_VARARGS | METH_KEYWORDS, "([nativity, tuple=, substitute_DC=, substitute_DK=, substitute_Other=])  -> list; Converts an example to a list")
{ PyTRY
    int natvt=1;
    if (args && !PyArg_ParseTuple(args, "|i", &natvt))
      PYERROR(PyExc_TypeError, "invalid arguments (no arguments or an integer expected)", PYNULL);

    bool tuples = false;
    PyObject *pytuples = keyws ? PyDict_GetItemString(keyws, "tuple") : PYNULL;
    tuples = pytuples && (PyObject_IsTrue(pytuples) != 0);

    PyObject *forDC = NULL, *forDK = NULL, *forSpecial = NULL;
    if (keyws) {
        forDC = PyDict_GetItemString(keyws, "substitute_DC");
        if (!forDC) {
            forDC = PyDict_GetItemString(keyws, "substituteDC");
        }
        forDC = PyDict_GetItemString(keyws, "substitute_DK");
        if (!forDC) {
            forDC = PyDict_GetItemString(keyws, "substituteDK");
        }
        forDC = PyDict_GetItemString(keyws, "substitute_other");
        if (!forDC) {
            forDC = PyDict_GetItemString(keyws, "substituteOther");
        }
    }
    return convertToPythonNative(PyExample_AS_ExampleReference(pex), natvt, tuples, forDK, forDC, forSpecial);
  PyCATCH
}


PyObject *Example_checksum(TPyExample *pex, PyObject *) PYARGS(METH_NOARGS, "() -> crc")
{ return PyInt_FromLong(PyExample_AS_ExampleReference(pex).sumValues()); }


#include "slist.hpp"

void Example_pack(const TExample &example, TCharBuffer &buf, PyObject *&otherValues)
{
  for(TValue *vali = example.values; vali != example.classes_end; vali++)
    Value_pack(*vali, buf, otherValues);

  buf.writeInt(example.meta.size() | (example.name ? 1<<31 : 0));
  buf.writeLong(example.id);

  if (example.name) {
    if (!otherValues)
      otherValues = PyList_New(0);
    PyObject *pyname = PyString_FromString(example.name->c_str());
    PyList_Append(otherValues, pyname);
    Py_DECREF(pyname);
  }

  const_ITERATE(TMetaValues, mi, example.meta) {
    buf.writeInt((int)(mi->first));
    buf.writeChar(mi->second.varType);
    Value_pack(mi->second, buf, otherValues);
  }
}


void Example_unpack(TExample &example, TCharBuffer &buf, PyObject *&otherValues, int &otherValuesIndex)
{
  TVarList::const_iterator vi;
  TValue *vali = example.values;
  for(vi = example.domain->variables->begin(); vali != example.values_end; vali++, vi++) {
    vali->varType = (*vi)->varType;
    Value_unpack(*vali, buf, otherValues, otherValuesIndex);
  }
  for(vi = example.domain->classVars->begin(); vali != example.classes_end; vali++, vi++) {
    vali->varType = (*vi)->varType;
    Value_unpack(*vali, buf, otherValues, otherValuesIndex);
  }

  int nMetas = buf.readInt();
  example.id = buf.readLong();

  if (nMetas & (1<<31)) {
    PyObject *pyname = PyList_GetItem(otherValues, otherValuesIndex++);
    example.name = new string(PyString_AsString(pyname));
  }

  for(int i = nMetas & 0x7fffffff; i--; ) {
    const int id = buf.readInt();
    TValue val((const unsigned char &)(buf.readChar()));
    Value_unpack(val, buf, otherValues, otherValuesIndex);
    example.meta.setValue(id, val);
  }
}


PyObject *Example__reduce__(TPyExample *pex)
{
  PyTRY
    if (pex->lock)
      PYERROR(PyExc_TypeError, "examples that reference tables cannot be pickled", PYNULL);

    TExample &example = PyExample_AS_ExampleReference(pex);

    TCharBuffer buf(1024);
    PyObject *otherValues = NULL;
    Example_pack(example, buf, otherValues);
    if (!otherValues) {
      otherValues = Py_None;
      Py_INCREF(otherValues);
    }
    
    if (pex->example.counter->orange_dict)
      return Py_BuildValue("O(Ns#N)O", getExportedFunction("__pickleLoaderExample"),
                                      WrapOrange(example.domain),
                                      buf.buf, buf.length(),
                                      otherValues,
                                      pex->example.counter->orange_dict);
    else
      return Py_BuildValue("O(Ns#N)", getExportedFunction("__pickleLoaderExample"),
                                      WrapOrange(example.domain),
                                      buf.buf, buf.length(),
                                      otherValues);
  PyCATCH
}


PyObject *__pickleLoaderExample(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(domain, packed_values, other_values)")
{
  PyTRY
    PDomain domain;
    char *pbuf;
    int bufSize;
    PyObject *otherValues;

    if (!PyArg_ParseTuple(args, "O&s#O:__pickleLoaderExample", cc_Domain, &domain, &pbuf, &bufSize, &otherValues))
      return NULL;

    TExample *newEx = new TExample(domain);
    PExample wex = newEx;
    int otherValuesIndex = 0;

    TCharBuffer buf(pbuf);
    Example_unpack(*newEx, buf, otherValues, otherValuesIndex);
    return Example_FromWrappedExample(wex);
  PyCATCH
}

inline void addValue(string &res, const TValue &val, PVariable var)
{ string str;
  var->val2str(val, str);
  if (var->varType!=TValue::FLOATVAR) res+="'"+str+"'";
  else res+=str;
}

string TPyExample2string(TPyExample *pex)
{ PExample example = PyExample_AS_Example(pex);
  string res("[");
  TVarList::iterator vi(example->domain->variables->begin());
  TVarList::iterator ve(example->domain->variables->end());
  TExample::const_iterator ei = example->values;
  for(; vi != ve; vi++, ei++) {
    if (ei != example->values)
      res+=", ";
    addValue(res, *ei, *vi);
  }
  res+="]";

  vi = example->domain->classVars->begin();
  ve = example->domain->classVars->end();
  if (vi != ve) {
      res += " (";
      for(; vi != ve; vi++, ei++) {
          if (ei != example->values_end)
              res += ", ";
          addValue(res, *ei, *vi);
      }
      res += ")";
  }

  int madded=0;
  ITERATE(TMetaValues, mi, example->meta) {
    res+= (madded++) ? ", " : ", {";
    
    TMetaDescriptor *desc=example->domain->metas[(*mi).first];
    if (desc) {
      res+="\""+desc->variable->get_name()+"\":";
      addValue(res, (*mi).second, desc->variable);
    }
    else
      if ((*mi).second.varType==TValue::FLOATVAR) {
        char buf[128];
        sprintf(buf, "%i:%.2f", int((*mi).first), (*mi).second.floatV);
        res += buf;
      }
      else
        res+="???";
  }

  if (madded) res+="}";

  return res;
}

PyObject *Example_repr(TPyExample *pex)
{ PyTRY
    return PyString_FromString(TPyExample2string(pex).c_str()); 
  PyCATCH
}

PyObject *Example_str(TPyExample *pex)
{ PyTRY
    return PyString_FromString(TPyExample2string(pex).c_str()); 
  PyCATCH
}

PyObject *Example_get_domain(TPyExample *self)
{ PyTRY
    return WrapOrange(PyExample_AS_Example(self)->domain);
  PyCATCH
}

PyObject *Example_get_id(TPyExample *self)
{ PyTRY
    return PyInt_FromLong(PyExample_AS_Example(self)->id);
  PyCATCH
}

PyObject *Example_newId(TPyExample *self)
{
  PyTRY
    PyExample_AS_Example(self)->id = getExampleId();
    RETURN_NONE;
  PyCATCH
}

int Example_cmp(TPyExample *one, TPyExample *another)
{ PyTRY
    return PyExample_AS_Example(one)->compare(PyExample_AS_ExampleReference(another));
  PyCATCH_1
}


int Example_hash(TPyExample *ex)
{ PyTRY
    return PyExample_AS_Example(ex)->sumValues();
  PyCATCH_1
}


int Example_len(TPyExample *pex)
{ PyTRY
    return PyExample_AS_Example(pex)->domain->variables->size();
  PyCATCH_1
}


char *example_underscores[][2] = {
    {"getweight", "get_weight"},
    {"setweight", "set_weight"},
    {"removeweight", "remove_weight"},
    {"getmeta", "get_meta"},
    {"getmetas", "get_metas"},
    {"hasmetas", "has_meta"},
    {"setvalue", "set_value"},
    {"setmeta", "set_meta"},
    {"removemeta", "remove_meta"},
    {"getclass", "get_class"},
    {"setclass", "set_class"},
    {NULL, NULL}
};

PyObject *Example_getattr(TPyExample *self, PyObject *name)
{
  char *orig = PyString_AsString(name);
  for(char *(*ei)[2] = example_underscores; **ei; ei++) {
      if (!strcmp(orig, **ei)) {
          PyObject *trans = PyString_FromString((*ei)[1]);
          PyObject *value = PyObject_GenericGetAttr((PyObject *)self, trans);
          Py_DECREF(trans);
          return value;
      }
  }

  if (!PyString_Check(name) || strcmp(orig, "name"))
    return PyObject_GenericGetAttr((PyObject *)self, name);

  const string *ename = self->example->name;
  return PyString_FromString(ename ? ename->c_str() : "");
}


int Example_setattr(TPyExample *self, PyObject *name, PyObject *text)
{
  if (!PyString_Check(name) || strcmp(PyString_AsString(name), "name"))
    return PyObject_GenericSetAttr((PyObject *)self, name, text);

  string *&ename = self->example->name;

  if (text == Py_None) {
    if (ename) {
      delete ename;
      ename = NULL;
    }
    return 0;
  }

  if (PyString_Check(text)) {
    if (ename)
      delete ename;
    ename = new string(PyString_AsString(text));
    return 0;
  }

  PYERROR(PyExc_AttributeError, "Example.name must be a string", -1);
}


extern PyTypeObject PyExampleIter_Type;

class TPyExampleIter {
public:
  PyObject_HEAD

  long index;
  TPyExample *example; /* Set to NULL when iterator is exhausted */
};

PyObject *PyExampleIter_New(TPyExample *ex)
{
  TPyExampleIter *self = PyObject_GC_New(TPyExampleIter, &PyExampleIter_Type);
  if (self == NULL)
    return NULL;

  self->index = 0;

  Py_INCREF(ex);
	self->example = ex;
	PyObject_GC_Track(self);
	return (PyObject *)self;
}

static void PyExampleIter_Dealloc(TPyExampleIter *self)
{
  PyObject_GC_UnTrack(self);
  Py_XDECREF(self->example);
  PyObject_GC_Del(self);
}

static int PyExampleIter_Traverse(TPyExampleIter *self, visitproc visit, void *arg)
{
	return self->example ? visit((PyObject *)(self->example), arg) : 0;
}


int PyExampleIter_Clear(TPyExampleIter *self)
{ Py_XDECREF((PyObject *)(self->example));
  self->example = NULL;
  return 0;
}


static PyObject *PyExampleIter_Iternext(TPyExampleIter *self)
{
  if (!self->example)
    return NULL;

  if (self->index >= PyExample_AS_ExampleReference(self->example).domain->variables->size()) {
    Py_DECREF(self->example);
    self->example = NULL;
    PYERROR(PyExc_StopIteration, "", PYNULL);
  }

  TExample &ex = PyExample_AS_ExampleReference(self->example);
  PyObject *result = Value_FromVariableValue(ex.domain->getVar(self->index), ex[self->index]);
  self->index++;
  return result;
}


PyObject *PyExampleIter_Iter(PyObject *self)
{ Py_INCREF(self);
  return self;
}


PyTypeObject PyExampleIter_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,
	"orange.Example iterator",
	sizeof(TPyExampleIter),
	0,
	(destructor)PyExampleIter_Dealloc,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	PyObject_GenericGetAttr,
	0, 0,
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
 	0,
 	(traverseproc)PyExampleIter_Traverse,
 	(inquiry)PyExampleIter_Clear, 0, 0,
	PyExampleIter_Iter,
	(iternextfunc)PyExampleIter_Iternext,
	0, 0, 0, 0, 0, 0, 0,
};


#include "cls_example.px"
