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


#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "classify.hpp"
#include "estimateprob.hpp"
#include "distvars.hpp"
#include "distance.hpp"

#include "cls_orange.hpp"
#include "cls_value.hpp"
#include "cls_example.hpp"
#include "lib_kernel.hpp"
#include "vectortemplates.hpp"
#include "maptemplates.hpp"

#include "converts.hpp"
#include "slist.hpp"

#include "externs.px"


/* ************ DISCRETIZATION ************ */

#include "discretize.hpp"


ABSTRACT(Discretizer, TransformValue)
C_NAMED(EquiDistDiscretizer, Discretizer, "([numberOfIntervals=, firstCut=, step=])")
C_NAMED(IntervalDiscretizer, Discretizer, "([points=])")
C_NAMED(ThresholdDiscretizer, Discretizer, "([threshold=])")
C_NAMED(BiModalDiscretizer, Discretizer, "([low=, high=])")

ABSTRACT(Discretization, Orange)
C_CALL (EquiDistDiscretization, Discretization, "() | (attribute, examples[, weight, numberOfIntervals=]) -/-> Variable")
C_CALL (   EquiNDiscretization, Discretization, "() | (attribute, examples[, weight, numberOfIntervals=]) -/-> Variable")
C_CALL ( EntropyDiscretization, Discretization, "() | (attribute, examples[, weight]) -/-> Variable")
C_CALL ( BiModalDiscretization, Discretization, "() | (attribute, examples[, weight]) -/-> Variable")


PyObject *Discretization_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(attribute, examples[, weight]) -> Variable")
{
  PyTRY
    NO_KEYWORDS

    PyObject *variable;
    PExampleGenerator egen;
    int weightID=0;
    if (!PyArg_ParseTuple(args, "OO&|O&", &variable, pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID)) 
      PYERROR(PyExc_SystemError, "invalid parameters", PYNULL);

    PVariable toDiscretize = varFromArg_byDomain(variable, egen->domain);
    if (!toDiscretize)
      return PYNULL; // varFromArg_byDomain has already set the error message

    PVariable discr = SELF_AS(TDiscretization)(egen, toDiscretize, weightID);
    if (!discr)
      PYERROR(PyExc_SystemError, "discretization construction failed", PYNULL);

    return WrapOrange(discr);
  PyCATCH
}


PyObject *Discretizer_constructVariable(PyObject *self, PyObject *var) PYARGS(METH_O, "(variable) -> variable")
{ PyTRY
    if (!PyOrVariable_Check(var))
      PYERROR(PyExc_TypeError, "invalid parameters (variable expected)", PYNULL);

    return WrapOrange(PyOrange_AsDiscretizer(self)->constructVar(PyOrange_AsVariable(var)));
  PyCATCH
}


PyObject *EquiDistDiscretizer_get_points(PyObject *self)
{ PyTRY
   CAST_TO(TEquiDistDiscretizer, edd);
    int nint = edd->numberOfIntervals - 1;
    PyObject *res = PyList_New(nint);
    for(Py_ssize_t i = 0; i < nint; i++)
      PyList_SetItem(res, i, PyFloat_FromDouble(edd->firstCut + i*edd->step));
    return res;
  PyCATCH
}


/* ************ FILTERS FOR REGRESSION ************** */

#include "transval.hpp"

C_NAMED(MapIntValue, TransformValue, "([mapping=])")
C_NAMED(Discrete2Continuous, TransformValue, "([value=])")
C_NAMED(Ordinal2Continuous, TransformValue, "([nvalues=])")
C_NAMED(NormalizeContinuous, TransformValue, "([average=, span=])")

C_NAMED(DomainContinuizer, Orange, "(domain|examples, convertClass=, invertClass=, zeroBased=, normalizeContinuous=, baseValueSelection=) -/-> Domain")

int getTargetClass(PVariable classVar, PyObject *pyval)
{
  if (pyval) {
    if (!classVar)
      PYERROR(PyExc_TypeError, "cannot set target class value for class-less domain", -2);
    if (classVar->varType != TValue::INTVAR)
      PYERROR(PyExc_TypeError, "cannot set target value for non-discrete class", -2);

    TValue targetValue;
    if (!convertFromPython(pyval, targetValue, classVar))
      return -2;
    if (targetValue.isSpecial())
      PYERROR(PyExc_TypeError, "unknown value passed as class target", -2)
    else
      return targetValue.intV;
  }
  return -1; // not an error, but undefined!
}

PyObject *DomainContinuizer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(domain[, targetClass] | examples[, weightID, targetClass]) -> domain")
{ 
  PyTRY
    NO_KEYWORDS

    if (args && (PyTuple_GET_SIZE(args)<=2) && PyOrDomain_Check(PyTuple_GET_ITEM(args, 0))) {
      PDomain domain;
      PyObject *pyval = PYNULL;
      if (!PyArg_ParseTuple(args, "O&|O", cc_Domain, &domain, &pyval))
        return PYNULL;
      int targetClass = getTargetClass(domain->classVar, pyval);
      if (targetClass == -2)
        return PYNULL;
     
      return WrapOrange(SELF_AS(TDomainContinuizer)(domain, targetClass));
    }

    PExampleGenerator egen;
    int weightID = 0;
    PyObject *pyval = PYNULL;
    if (!PyArg_ParseTuple(args, "O&|O&O", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID, &pyval))
      PYERROR(PyExc_AttributeError, "DomainContinuizer.__call__: domain or examples (and, optionally, weight attribute) expected", PYNULL);

    int targetClass = getTargetClass(egen->domain->classVar, pyval);
    if (targetClass == -2)
      return PYNULL;

    //printf("%p-%p\n", self, ((TPyOrange *)self)->ptr);
    return WrapOrange(SELF_AS(TDomainContinuizer)(egen, weightID, targetClass));

  PyCATCH
}

/* ************ REDUNDANCIES ************ */

#include "redundancy.hpp"

ABSTRACT(RemoveRedundant, Orange)

C_CALL(RemoveRedundantByInduction, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")
C_CALL(RemoveRedundantByQuality, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")
C_CALL(RemoveRedundantOneValue, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")

C_CALL3(RemoveUnusedValues, RemoveUnusedValues, Orange, "([[attribute, ]examples[, weightId]]) -/-> attribute")

PyObject *RemoveRedundant_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([examples[, weightID][, suspicious]) -/-> Domain")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *suspiciousList=NULL;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&|OO&:RemoveRedundant.call", pt_ExampleGenerator, &egen, &suspiciousList, pt_weightByGen(egen), &weight))
      return PYNULL;

    TVarList suspiciousset;
    if (suspiciousList)
      if (!varListFromDomain(suspiciousList, egen->domain, suspiciousset))
        return PYNULL;

    PDomain newdomain = SELF_AS(TRemoveRedundant)(egen, suspiciousList ? &suspiciousset : NULL, NULL, weight);
    return WrapOrange(newdomain);
  PyCATCH
}


PyObject *RemoveRedundantOneValue_hasAtLeastTwoValues(PyObject *, PyObject *args) PYARGS(METH_VARARGS | METH_STATIC, "(attribute, examples) -> bool")
{
  PyTRY
    PExampleGenerator gen;
    PyObject *var;
    if (!PyArg_ParseTuple(args, "O&O:RemoveRedundantOneValue.hasAtLeastTwoValues", pt_ExampleGenerator, &gen, &var))
      return NULL;
    int varIdx;
    if (!varNumFromVarDom(var, gen->domain, varIdx))
      PYERROR(PyExc_AttributeError, "RemoveRedundantOneValue.hasAtLeastTwoValues: invalid attribute", NULL);
    return PyBool_FromLong(TRemoveRedundantOneValue::hasAtLeastTwo(gen, varIdx) ? 1 : 0);
  PyCATCH
}
    
PyObject *RemoveUnusedValues_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(attribute, examples[, weightId]) -> attribute")
{
  PyTRY
    NO_KEYWORDS
    CAST_TO(TRemoveUnusedValues, ruv);
    bool storeOv = ruv->removeOneValued;

    PExampleGenerator egen;
    PVariable var;
    int weightID = 0;
    int removeOneValued = -1;
    int checkClass = 0;

    if (PyArg_ParseTuple(args, "O&O&|O&i:RemoveUnusedValues.call", cc_Variable, &var, pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID, &removeOneValued)) {
      if (removeOneValued >= 0)
        ruv->removeOneValued = removeOneValued != 0;
      PyObject *res = WrapOrange(ruv->call(var, egen, weightID));
      ruv->removeOneValued = storeOv;
      return res;
    }

    PyErr_Clear();

    if (PyArg_ParseTuple(args, "O&|O&ii:RemoveUnusedValues.call", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weightID, &removeOneValued, &checkClass)) {
      if (removeOneValued >= 0)
        ruv->removeOneValued = removeOneValued != 0;
      PyObject *res = WrapOrange(ruv->call(egen, weightID, checkClass != 0));
      ruv->removeOneValued = storeOv;
      return res;
    }

    PYERROR(PyExc_AttributeError, "RemoveUnusedValues.__call__: invalid arguments", PYNULL);

  PyCATCH
}


/* ************ PREPROCESSORS ************ */

#include "preprocessors.hpp"

ABSTRACT(Preprocessor, Orange)

C_CALL(Preprocessor_select, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_ignore, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")

C_CALL(Preprocessor_take, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_drop, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_removeDuplicates, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_takeMissing, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_dropMissing, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_takeMissingClasses, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_dropMissingClasses, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")

C_CALL(Preprocessor_shuffle, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")

C_CALL(Preprocessor_addMissing, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addMissingClasses, Preprocessor, "([examples[, weightID]] [classMissing=<float>]) -/-> ExampleTable")
C_CALL(Preprocessor_addNoise, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addClassNoise, Preprocessor, "([examples[, weightID]] [proportion=<float>]) -/-> ExampleTable")
C_CALL(Preprocessor_addGaussianNoise, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addGaussianClassNoise, Preprocessor, "([examples[, weightID]] [deviation=<float>]) -/-> ExampleTable")

C_CALL(Preprocessor_addClassWeight, Preprocessor, "([examples[, weightID]] [equalize=, classWeights=) -/-> ExampleTable")
C_CALL(Preprocessor_addCensorWeight, Preprocessor, "([examples[, weightID]] [method=0-km, 1-nmr, 2-linear, outcomeVar=, eventValue=, timeID=, maxTime=]) -/-> ExampleTable")

C_CALL(Preprocessor_filter, Preprocessor, "([examples[, weightID]] [filter=]) -/-> ExampleTable")
C_CALL(Preprocessor_imputeByLearner, Preprocessor, "([examples[, weightID]] [learner=]) -/-> ExampleTable")
C_CALL(Preprocessor_discretize, Preprocessor, "([examples[, weightID]] [notClass=, method=, attributes=<list-of-strings>]) -/-> ExampleTable")

C_NAMED(ImputeClassifier, Classifier, "([classifierFromVar=][imputer=])")

PyObject *Preprocessor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> ExampleTable")
{ 
  PyTRY
    NO_KEYWORDS

    int weightID=0;
    PExampleGenerator egen = exampleGenFromArgs(args, weightID);
    if (!egen)
      PYERROR(PyExc_TypeError, "attribute error (example generator expected)", PYNULL);
    bool weightGiven = (weightID!=0);

    int newWeight;
    PExampleGenerator res = SELF_AS(TPreprocessor)(egen, weightID, newWeight);
    PyObject *wrappedGen=WrapOrange(res);
    return weightGiven || newWeight ? Py_BuildValue("Ni", wrappedGen, newWeight) : wrappedGen;
  PyCATCH
}


PyObject *Preprocessor_selectionVector(PyObject *self, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(examples[, weightID])")
{
  PyTRY
    int weightID = 0;
    PExampleGenerator egen = exampleGenFromArgs(args, weightID);
    if (!egen)
      PYERROR(PyExc_TypeError, "attribute error (example generator expected)", PYNULL);

    return WrapOrange(SELF_AS(TPreprocessor).selectionVector(egen, weightID));
  PyCATCH
}


#include "stringvars.hpp"

typedef MapMethods<PVariableFilterMap, TVariableFilterMap, PVariable, PValueFilter> TMM_VariableFilterMap;

int VariableFilterMap_setitemlow(TVariableFilterMap *aMap, PVariable var, PyObject *pyvalue)
{
  PValueFilter value;
  if (TMM_VariableFilterMap::_valueFromPython(pyvalue, value)) {
    aMap->__ormap[var] = value;
    return 0;
  }

  PyErr_Clear();

  if (var->varType == TValue::FLOATVAR) {
    float min, max;
    if (!PyArg_ParseTuple(pyvalue, "ff:VariableFilterMap.__setitem__", &min, &max))
      return -1;

    aMap->__ormap[var] = (min<=max) ? mlnew TValueFilter_continuous(ILLEGAL_INT, min, max)
                                   : mlnew TValueFilter_continuous(ILLEGAL_INT, max, min, true);
    return 0;
  }

  if (var->varType == TValue::INTVAR) {
    TValueFilter_discrete *vfilter = mlnew TValueFilter_discrete(ILLEGAL_INT, var);
    PValueFilter wvfilter = vfilter;
    TValueList &valueList = vfilter->values.getReference();

    if (PyTuple_Check(pyvalue) || PyList_Check(pyvalue)) {
      PyObject *iterator = PyObject_GetIter(pyvalue);
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
        TValue value;
        if (!convertFromPython(item, value, var)) {
          Py_DECREF(item);
          Py_DECREF(iterator);
          return -1;
        }
        Py_DECREF(item);
        if (value.isSpecial())
          vfilter->acceptSpecial = 1;
        else
          valueList.push_back(value);
      }
      Py_DECREF(iterator);
    }
    else {
      TValue value;
      if (!convertFromPython(pyvalue, value, var))
        return -1;
      if (value.isSpecial())
        vfilter->acceptSpecial = 1;
      else
        valueList.push_back(value);
    }

    aMap->__ormap[var] = wvfilter;
    return 0;
  }

  if (var.is_derived_from(TStringVariable)) {
    TValueFilter_stringList *vfilter = mlnew TValueFilter_stringList(ILLEGAL_INT, mlnew TStringList());
    PValueFilter wvfilter = vfilter;
    TStringList &values = vfilter->values.getReference();

    if (PyTuple_Check(pyvalue) || PyList_Check(pyvalue)) {
      PyObject *iterator = PyObject_GetIter(pyvalue);
      int i = 0;
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator), i++) {
        if (!PyString_Check(item)) {
          PyErr_Format(PyExc_TypeError, "error at index %i, string expected", i);
          Py_DECREF(item);
          Py_DECREF(iterator);
          return -1;
        }
        Py_DECREF(item);
        values.push_back(PyString_AsString(item));
      }
      Py_DECREF(iterator);
    }
    else if (PyString_Check(pyvalue))
      values.push_back(PyString_AsString(pyvalue));
    else
      PyErr_Format(PyExc_TypeError, "string or a list of strings expected", -1);

    aMap->__ormap[var] = wvfilter;
    return 0;
  }
    
  PYERROR(PyExc_TypeError, "VariableFilterMap.__setitem__: unrecognized item type", -1);
}


template<>
int TMM_VariableFilterMap::_setitemlow(TVariableFilterMap *aMap, PyObject *pykey, PyObject *pyvalue)
{ PyTRY
    PVariable var;
    return TMM_VariableFilterMap::_keyFromPython(pykey, var) ? VariableFilterMap_setitemlow(aMap, var, pyvalue) : -1;
  PyCATCH_1
}


template<>
PyObject *TMM_VariableFilterMap::_setdefault(TPyOrange *self, PyObject *args)
{ PyObject *pykey;
  PyObject *deflt = Py_None;
  if (!PyArg_ParseTuple(args, "O|O:get", &pykey, &deflt))
    return PYNULL;

  PVariable var;
  if (!TMM_VariableFilterMap::_keyFromPython(pykey, var))
    return PYNULL;

  TVariableFilterMap *aMap = const_cast<TVariableFilterMap *>(PyOrange_AsVariableFilterMap(self).getUnwrappedPtr());
  
  iterator fi = aMap->find(var);
  if (fi==aMap->end()) {
    if (VariableFilterMap_setitemlow(aMap, var, deflt)<0)
      return PYNULL;

    // cannot return deflt here, since it is probably a string or tuple which was converted to ValueFilter
    // we just reinitialize fi and let the function finish :)
    fi = aMap->find(var);
  }

  return convertValueToPython((*fi).second);
}


PDistribution kaplanMeier(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID);

PyObject *kaplanMeier(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(examples, outcome attribute, fail value, time attribute[, weightID]) -> survival curve")
{ PExampleGenerator egen;
  PyObject *outcomevar, *timevar;
  PyObject *pyfailvalue;
  TValue failvalue;
  int weightID = 0;
  if (!PyArg_ParseTuple(args, "O&OOOO&:kaplanMeier", pt_ExampleGenerator, &egen, &outcomevar, &pyfailvalue, &timevar, pt_weightByGen(egen), &weightID))

    return PYNULL;

  int outcomeIndex, timeIndex;
  if (outcomevar) {
    if (!varNumFromVarDom(outcomevar, egen->domain, outcomeIndex)) 
      PYERROR(PyExc_AttributeError, "outcome variable not found in domain", PYNULL);
  }
  else
    if (egen->domain->classVar)
      outcomeIndex = egen->domain->attributes->size();
    else
      PYERROR(PyExc_AttributeError, "'outcomeVar' not set and the domain is class-less", PYNULL);

  PVariable ovar = egen->domain->getVar(outcomeIndex);

  if (   !convertFromPython(pyfailvalue, failvalue, ovar)
      || failvalue.isSpecial()
      || (failvalue.varType != TValue::INTVAR))
    PYERROR(PyExc_AttributeError, "invalid value for failure", PYNULL);

  return WrapOrange(kaplanMeier(egen, outcomeIndex, failvalue, timeIndex, weightID));
}


// modified setitem to accept intervals/names of values
INITIALIZE_MAPMETHODS(TMM_VariableFilterMap, &PyOrVariable_Type, &PyOrValueFilter_Type, _orangeValueFromPython<PVariable>, _orangeValueFromPython<PValueFilter>, _orangeValueToPython<PVariable>, _orangeValueToPython<PValueFilter>)

PVariableFilterMap PVariableFilterMap_FromArguments(PyObject *arg) { return TMM_VariableFilterMap::P_FromArguments(arg); }
PyObject *VariableFilterMap_FromArguments(PyTypeObject *type, PyObject *arg) { return TMM_VariableFilterMap::_FromArguments(type, arg); }
PyObject *VariableFilterMap_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(items)") ALLOWS_EMPTY { return TMM_VariableFilterMap::_new(type, arg, kwds); }
PyObject *VariableFilterMap_str(TPyOrange *self) { return TMM_VariableFilterMap::_str(self); }
PyObject *VariableFilterMap_repr(TPyOrange *self) { return TMM_VariableFilterMap::_str(self); }
PyObject *VariableFilterMap_getitem(TPyOrange *self, PyObject *key) { return TMM_VariableFilterMap::_getitem(self, key); }
int       VariableFilterMap_setitem(TPyOrange *self, PyObject *key, PyObject *value) { return TMM_VariableFilterMap::_setitem(self, key, value); }
Py_ssize_t       VariableFilterMap_len(TPyOrange *self) { return TMM_VariableFilterMap::_len(self); }
int       VariableFilterMap_contains(TPyOrange *self, PyObject *key) { return TMM_VariableFilterMap::_contains(self, key); }

PyObject *VariableFilterMap_has_key(TPyOrange *self, PyObject *key) PYARGS(METH_O, "(key) -> None") { return TMM_VariableFilterMap::_has_key(self, key); }
PyObject *VariableFilterMap_get(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFilterMap::_get(self, args); }
PyObject *VariableFilterMap_setdefault(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFilterMap::_setdefault(self, args); }
PyObject *VariableFilterMap_clear(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> None") { return TMM_VariableFilterMap::_clear(self); }
PyObject *VariableFilterMap_keys(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> keys") { return TMM_VariableFilterMap::_keys(self); }
PyObject *VariableFilterMap_values(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> values") { return TMM_VariableFilterMap::_values(self); }
PyObject *VariableFilterMap_items(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> items") { return TMM_VariableFilterMap::_items(self); }
PyObject *VariableFilterMap_update(TPyOrange *self, PyObject *args) PYARGS(METH_O, "(items) -> None") { return TMM_VariableFilterMap::_update(self, args); }
PyObject *VariableFilterMap__reduce__(TPyOrange *self, PyObject *) { return TMM_VariableFilterMap::_reduce(self); }


typedef MapMethods<PVariableFloatMap, TVariableFloatMap, PVariable, float> TMM_VariableFloatMap;
INITIALIZE_MAPMETHODS(TMM_VariableFloatMap, &PyOrVariable_Type, NULL, _orangeValueFromPython<PVariable>, _nonOrangeValueFromPython<float>, _orangeValueToPython<PVariable>, _nonOrangeValueToPython<float>);

PVariableFloatMap PVariableFloatMap_FromArguments(PyObject *arg) { return TMM_VariableFloatMap::P_FromArguments(arg); }
PyObject *VariableFloatMap_FromArguments(PyTypeObject *type, PyObject *arg) { return TMM_VariableFloatMap::_FromArguments(type, arg); }
PyObject *VariableFloatMap_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(items)") ALLOWS_EMPTY { return TMM_VariableFloatMap::_new(type, arg, kwds); } 
PyObject *VariableFloatMap_str(TPyOrange *self) { return TMM_VariableFloatMap::_str(self); }
PyObject *VariableFloatMap_repr(TPyOrange *self) { return TMM_VariableFloatMap::_str(self); }
PyObject *VariableFloatMap_getitem(TPyOrange *self, PyObject *key) { return TMM_VariableFloatMap::_getitem(self, key); }
int       VariableFloatMap_setitem(TPyOrange *self, PyObject *key, PyObject *value) { return TMM_VariableFloatMap::_setitem(self, key, value); }
Py_ssize_t       VariableFloatMap_len(TPyOrange *self) { return TMM_VariableFloatMap::_len(self); }
int       VariableFloatMap_contains(TPyOrange *self, PyObject *key) { return TMM_VariableFloatMap::_contains(self, key); }

PyObject *VariableFloatMap_has_key(TPyOrange *self, PyObject *key) PYARGS(METH_O, "(key) -> None") { return TMM_VariableFloatMap::_has_key(self, key); }
PyObject *VariableFloatMap_get(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFloatMap::_get(self, args); }
PyObject *VariableFloatMap_setdefault(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFloatMap::_setdefault(self, args); }
PyObject *VariableFloatMap_clear(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> None") { return TMM_VariableFloatMap::_clear(self); }
PyObject *VariableFloatMap_keys(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> keys") { return TMM_VariableFloatMap::_keys(self); }
PyObject *VariableFloatMap_values(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> values") { return TMM_VariableFloatMap::_values(self); }
PyObject *VariableFloatMap_items(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> items") { return TMM_VariableFloatMap::_items(self); }
PyObject *VariableFloatMap_update(TPyOrange *self, PyObject *args) PYARGS(METH_O, "(items) -> None") { return TMM_VariableFloatMap::_update(self, args); }
PyObject *VariableFloatMap__reduce__(TPyOrange *self, PyObject *) { return TMM_VariableFloatMap::_reduce(self); }


C_CALL3(TableAverager, TableAverager, Orange, "(list-of-example-generators) -/-> ExampleTable")

PExampleGeneratorList PExampleGeneratorList_FromArguments(PyObject *arg);

PyObject *TableAverager_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(list-of-example-generators) --> ExampleTable")
{
  PyTRY
    NO_KEYWORDS
    if (!args || (PyTuple_Size(args) != 1))
      PYERROR(PyExc_TypeError, "TableAverager expects a list of example generators", PYNULL);
    PExampleGeneratorList tables = PExampleGeneratorList_FromArguments(PyTuple_GET_ITEM(args, 0));
    if (!tables)
      return PYNULL;
    return WrapOrange(SELF_AS(TTableAverager)(tables));
  PyCATCH
}

/* ************ INDUCE ************ */

#include "induce.hpp"
#include "subsets.hpp"

ABSTRACT(FeatureInducer, Orange)

ABSTRACT(SubsetsGenerator, Orange)
C_NAMED(SubsetsGenerator_withRestrictions, SubsetsGenerator, "([subGenerator=])")

ABSTRACT(SubsetsGenerator_iterator, Orange)
C_NAMED(SubsetsGenerator_constant_iterator, SubsetsGenerator_iterator, "")
BASED_ON(SubsetsGenerator_constSize_iterator, SubsetsGenerator_iterator)
BASED_ON(SubsetsGenerator_minMaxSize_iterator, SubsetsGenerator_iterator)
C_NAMED(SubsetsGenerator_withRestrictions_iterator, SubsetsGenerator_iterator, "")

PyObject *FeatureInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples, bound-attrs, new-name, weightID) -> (Variable, float)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *boundList;
    char *name;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&Os|O&", pt_ExampleGenerator, &egen, &boundList, &name, pt_weightByGen(egen), &weight))
      PYERROR(PyExc_TypeError, "invalid arguments", PYNULL);

    TVarList boundset;
    if (!varListFromDomain(boundList, egen->domain, boundset))
      return PYNULL;

    float meas;
    PVariable newvar = SELF_AS(TFeatureInducer)(egen, boundset, name, meas, weight);
    return Py_BuildValue("Nf", WrapOrange(newvar), meas);
  PyCATCH
}




PVarList PVarList_FromArguments(PyObject *arg);

PVarList varListForReset(PyObject *vars)
{
  if (PyOrDomain_Check(vars))
    return PyOrange_AsDomain(vars)->attributes;

  PVarList variables = PVarList_FromArguments(vars);
  if (!variables)
    PYERROR(PyExc_TypeError, "SubsetsGenerator.reset: invalid arguments", NULL);

  return variables;
}


PyObject *SubsetsGenerator_reset(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([[var0, var1, ...]]) -> int")
{ PyTRY
    PyObject *vars = PYNULL;
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator.reset", &vars))
      return PYNULL;

    if (!vars)
      PYERROR(PyExc_TypeError, "SubsetsGenerator.reset does not reset the generator (as it used to)", false);

    PVarList varList = varListForReset(vars);
    if (!varList)
      return NULL;

    SELF_AS(TSubsetsGenerator).varList = varList;
    RETURN_NONE;
  PyCATCH
}


PyObject *SubsetsGenerator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([var0, var1] | domain |) -> SubsetsGenerator")
{ PyTRY
    NO_KEYWORDS

    if (args && PyTuple_Size(args) && !SubsetsGenerator_reset(self, args))
      return NULL;

    Py_INCREF(self);
    return self;
  PyCATCH
}


PyObject *SubsetsGenerator_iter(PyObject *self)
{ PyTRY
    return WrapOrange(SELF_AS(TSubsetsGenerator).call());
  PyCATCH
}


PyObject *SubsetsGenerator_iterator_iternext(PyObject *self)
{ PyTRY
    TVarList vl;
    if (!SELF_AS(TSubsetsGenerator_iterator).call(vl))
      return PYNULL;

    PyObject *list=PyTuple_New(vl.size());
    Py_ssize_t i=0;
    ITERATE(TVarList, vi, vl)
      PyTuple_SetItem(list, i++, WrapOrange(*vi));
    return list;
  PyCATCH
}


PyObject *SubsetsGenerator_iterator_next(PyObject *self)
{ Py_INCREF(self);
  return self;
}


void packCounter(const TCounter &cnt, TCharBuffer &buf)
{
  buf.writeInt(cnt.limit);
  buf.writeInt(cnt.size());
  const_ITERATE(TCounter, ci, cnt)
    buf.writeInt(*ci);
}


void unpackCounter(TCharBuffer &buf, TCounter &cnt)
{
  cnt.limit = buf.readInt();

  int size = buf.readInt();
  cnt.resize(size);
  for(TCounter::iterator ci(cnt.begin()); size--; *ci++ = buf.readInt());
}


PyObject *SubsetsGenerator_constSize_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SubsetsGenerator, "(size)") ALLOWS_EMPTY
{ PyTRY
    int B = 2;
    PyObject *varlist = NULL;
    PyObject *res;

    // This is for compatibility ...
    if (PyArg_ParseTuple(args, "|iO:SubsetsGenerator_constSize.__new__", &B, &varlist)) {
      TSubsetsGenerator *ssg = mlnew TSubsetsGenerator_constSize(B);
      res = WrapNewOrange(ssg, type);
      if (varlist) {
        SubsetsGenerator_reset(res, varlist);
      }
      return res;
    }
    PyErr_Clear();

    // ... and this if for real
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator_constSize.__new__", &varlist))
      return PYNULL;

    TSubsetsGenerator *gen = mlnew TSubsetsGenerator_constSize(B);
    if (varlist && !(gen->varList = varListForReset(varlist))) {
      delete gen;
      return NULL;
    }

    return WrapNewOrange(gen, type);
  PyCATCH
}

PyObject *SubsetsGenerator_constSize_iterator__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TSubsetsGenerator_constSize_iterator, gen);

    TCharBuffer buf((gen->counter.size() + 4) * sizeof(int));
    packCounter(gen->counter, buf);
    buf.writeChar(gen->moreToCome ? 1 : 0);

    return Py_BuildValue("O(OOs#)N", getExportedFunction("__pickleLoaderSubsetsGeneratorConstSizeIterator"),
                                    self->ob_type,
                                    WrapOrange(gen->varList),
                                    buf.buf, buf.length(),
                                    packOrangeDictionary(self));
 PyCATCH
}


PyObject *__pickleLoaderSubsetsGeneratorConstSizeIterator(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_counter)")
{
  PyTRY
    PyTypeObject *type;
    PVarList varList;
    char *pbuf;
    int bufSize;
    if (!PyArg_ParseTuple(args, "OOs#:__pickleLoaderSubsetsGenerator_constSizeIterator", &type, ccn_VarList, &varList, &pbuf, &bufSize))
      return NULL;

    TCharBuffer buf(pbuf);

    TSubsetsGenerator_constSize_iterator *gen = new TSubsetsGenerator_constSize_iterator(varList, buf.readInt());
    unpackCounter(buf, gen->counter);
    gen->moreToCome = buf.readChar() != 0;

    return WrapNewOrange(gen, type);
  PyCATCH
}



PyObject *SubsetsGenerator_minMaxSize_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SubsetsGenerator, "([min=, max=][, varList=])") ALLOWS_EMPTY
{ PyTRY
    int min = 2, max = 3;
    PyObject *varlist = NULL;

    // This is for compatibility ...
    if (args && PyArg_ParseTuple(args, "|iiO", &min, &max, &varlist)) {
      PyObject *res = WrapNewOrange(mlnew TSubsetsGenerator_minMaxSize(min, max), type);
      if (varlist)
        SubsetsGenerator_reset(res, varlist);

      return res;
    }
    PyErr_Clear();

    // ... and this if for real
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator_minMaxSize.__new__", &varlist))
      return PYNULL;
      
    TSubsetsGenerator *gen = mlnew TSubsetsGenerator_minMaxSize(min, max);
    if (varlist && !(gen->varList = varListForReset(varlist))) {
      delete gen;
      return NULL;
    }

    return WrapNewOrange(gen, type);
  PyCATCH
}


PyObject *SubsetsGenerator_minMaxSize_iterator__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TSubsetsGenerator_minMaxSize_iterator, gen);

    TCharBuffer buf((gen->counter.size() + 5) * sizeof(int));
    buf.writeInt(gen->B);
    buf.writeInt(gen->max);
    packCounter(gen->counter, buf);
    buf.writeChar(gen->moreToCome ? 1 : 0);

    return Py_BuildValue("O(OOs#)N", getExportedFunction("__pickleLoaderSubsetsGeneratorMinMaxSizeIterator"),
                                    self->ob_type,
                                    WrapOrange(gen->varList),
                                    buf.buf, buf.length(),
                                    packOrangeDictionary(self));
 PyCATCH
}


PyObject *__pickleLoaderSubsetsGeneratorMinMaxSizeIterator(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, varList, packed_counter)")
{
  PyTRY
    PyTypeObject *type;
    PVarList varList;
    char *pbuf;
    int bufSize;
    if (!PyArg_ParseTuple(args, "OO&s#:__pickleLoaderSubsetsGenerator_minMaxSizeIterator", &type, ccn_VarList, &varList, &pbuf, &bufSize))
      return NULL;

    TCharBuffer buf(pbuf);

    const int B = buf.readInt();
    const int max = buf.readInt();
    TSubsetsGenerator_minMaxSize_iterator *gen = new TSubsetsGenerator_minMaxSize_iterator(varList, B, max);
    unpackCounter(buf, gen->counter);
    gen->moreToCome = buf.readChar() != 0;

    return WrapNewOrange(gen, type);
  PyCATCH
}




PyObject *SubsetsGenerator_constant_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SubsetsGenerator, "([constant=])") ALLOWS_EMPTY
{ PyTRY
    PyObject *varlist = NULL;

    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator_constant.__new__", &varlist))
      return PYNULL;

    TSubsetsGenerator_constant *gen = mlnew TSubsetsGenerator_constant();
    if (varlist && !(gen->varList = varListForReset(varlist))) {
      delete gen;
      return NULL;
    }

    gen->constant = CLONE(TVarList, gen->varList);
    return WrapNewOrange(gen, type);
  PyCATCH
}
/* ************ MINIMAL COMPLEXITY ************ */

#include "minimal_complexity.hpp"

ABSTRACT(IGConstructor, Orange)
C_CALL(IGBySorting, IGConstructor, "([examples, bound-attrs]) -/-> IG")

ABSTRACT(ColorIG, Orange)
C_CALL(ColorIG_MCF, ColorIG, "([IG]) -/-> ColoredIG")

C_CALL(FeatureByMinComplexity, FeatureInducer, "([examples, bound-attrs, name] [IGConstructor=, classifierFromIG=) -/-> Variable")

C_NAMED(ColoredIG, GeneralExampleClustering, "(ig=, colors=)")


bool convertFromPython(PyObject *args, TIGNode &ign)
{ PyTRY
    PDiscDistribution inco, co;
    TExample *example;
    if (!PyArg_ParseTuple(args, "O&|O&O&:convertFromPython(IG)", ptr_Example, &example, ccn_DiscDistribution, &inco, ccn_DiscDistribution, &co))
      return false;

    ign.example = PExample(mlnew TExample(*example));

    if (inco)
      ign.incompatibility = inco.getReference();
    if (co)
      ign.compatibility = co.getReference();
    return true;
  PyCATCH_r(false);
}
      

bool convertFromPython(PyObject *args, PIG &ig)
{ if (!PyList_Check(args))
    PYERROR(PyExc_AttributeError, "invalid arguments (list expected)", false);

  ig=PIG(mlnew TIG());
  for(Py_ssize_t i=0; i<PyList_Size(args); i++) {
    ig->nodes.push_back(TIGNode());
    if (!convertFromPython(PyList_GetItem(args, i), ig->nodes.back())) {
      ig=PIG();
      PYERROR(PyExc_AttributeError, "invalid list argument", false);
    }
  }

  return true;
}


PyObject *IG_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "<see the manual>")
{ PyTRY
    PyObject *pyig;
    PIG ig;
    return PyArg_ParseTuple(args, "O:IG.new", &pyig) && convertFromPython(pyig, ig) ? WrapOrange(ig) : PYNULL;
  PyCATCH
}


PyObject *IG_native(PyObject *self) PYARGS(0, "() -> [(Example, [incompatibility-float], [compatibility-float])]")
{ PyTRY
    CAST_TO(TIG, graph);
  
    PyObject *result=PyList_New(graph->nodes.size());
    Py_ssize_t i=0;
    ITERATE(vector<TIGNode>, ni, graph->nodes)
      PyList_SetItem(result, i++, Py_BuildValue("NNN", 
        Example_FromWrappedExample((*ni).example),
         // it's OK to wrap a reference - we're just copying it
        WrapNewOrange(mlnew TDiscDistribution((*ni).incompatibility), (PyTypeObject *)&PyOrDiscDistribution_Type),
        WrapNewOrange(mlnew TDiscDistribution((*ni).compatibility), (PyTypeObject *)&PyOrDiscDistribution_Type)
      ));

    return result;
  PyCATCH
}


PyObject *IG__reduce__(PyObject *self)
{
  PyTRY
    return Py_BuildValue("O(N)N", self->ob_type, IG_native(self), packOrangeDictionary(self));
  PyCATCH
}


PyObject *IG_normalize(PyObject *self) PYARGS(0, "() -> None")
{ PyTRY
    SELF_AS(TIG).normalize();
    RETURN_NONE;
  PyCATCH
}


PyObject *IG_make0or1(PyObject *self) PYARGS(0, "() -> None")
{ PyTRY
    SELF_AS(TIG).make0or1();
    RETURN_NONE;
  PyCATCH
}


PyObject *IG_complete(PyObject *self) PYARGS(0, "() -> None")
{ PyTRY
    SELF_AS(TIG).complete();
    RETURN_NONE;
  PyCATCH
}


PyObject *IG_removeEmpty(PyObject *self) PYARGS(0, "() -> None")
{ PyTRY
    SELF_AS(TIG).complete();
    RETURN_NONE;
  PyCATCH
}



PyObject *IGConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples, bound-attrs) -> IG")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *boundList;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&O|O&", pt_ExampleGenerator, &egen, &boundList, pt_weightByGen(egen), &weight))
      PYERROR(PyExc_TypeError, "attribute error", PYNULL);

    TVarList boundset;
    if (!varListFromDomain(boundList, egen->domain, boundset))
      return PYNULL;

    return WrapOrange(SELF_AS(TIGConstructor)(egen, boundset, weight));
  PyCATCH
}



PyObject *ColorIG_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(ig) -> [int]")
{
  PyTRY
    NO_KEYWORDS

    PIG graph;
    return PyArg_ParseTuple(args, "O&:ColorIG.__call__", cc_IG, &graph) ? WrapOrange(SELF_AS(TColorIG)(graph)) : PYNULL;
  PyCATCH
}


/* ************ MINIMAL ERROR ******** */

#include "minimal_error.hpp"

C_CALL(FeatureByIM, FeatureInducer, "([examples, bound-attrs, name] [constructIM=, classifierFromIM=]) -/-> Variable")

ABSTRACT(IMConstructor, Orange)
C_CALL(IMBySorting, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")
C_CALL(IMByIMByRows, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")
C_CALL(IMByRelief, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")

ABSTRACT(ClustersFromIM, Orange)
C_CALL(ClustersFromIMByAssessor, ClustersFromIM, "([IM] [minProfitProportion=, columnAssessor=, stopCriterion=]) -/-> IMClustering")

C_NAMED(IMClustering, Orange, "([im= clusters=, maxCluster=])")

BASED_ON(IMByRows, Orange)
NO_PICKLE(IMByRows)

ABSTRACT(IMByRowsConstructor, Orange)
C_CALL(IMByRowsBySorting, IMByRowsConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IMByRows")
C_CALL(IMByRowsByRelief, IMByRowsConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IMByRows")

ABSTRACT(IMByRowsPreprocessor, Orange)
C_CALL(IMBlurer, IMByRowsPreprocessor, "([IMByRows]) -> None")

C_CALL3(AssessIMQuality, AssessIMQuality, Orange, "([IM] -/-> float)")

ABSTRACT(StopIMClusteringByAssessor, Orange)
C_NAMED(StopIMClusteringByAssessor_noProfit, StopIMClusteringByAssessor, "([minProfitProportion=])")
C_NAMED(StopIMClusteringByAssessor_binary, StopIMClusteringByAssessor, "()")
C_NAMED(StopIMClusteringByAssessor_n, StopIMClusteringByAssessor, "(n=)")
C_NAMED(StopIMClusteringByAssessor_noBigChange, StopIMClusteringByAssessor, "()")

ABSTRACT(ColumnAssessor, Orange)
C_NAMED(ColumnAssessor_m, ColumnAssessor, "([m=])")
C_NAMED(ColumnAssessor_Laplace, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_mf, ColumnAssessor, "([m=])")
C_NAMED(ColumnAssessor_N, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_Relief, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_Measure, ColumnAssessor, "(measure=)")
C_NAMED(ColumnAssessor_Kramer, ColumnAssessor, "()")

C_CALL(MeasureAttribute_IM, MeasureAttribute, "(constructIM=, columnAssessor=) | (attr, examples[, apriori] [,weightID]) -/-> (float, meas-type)")


PyObject *IMConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example, bound-attrs[, weightID]) -> IM")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *boundList;
    int weightID = 0;
    if (PyArg_ParseTuple(args, "O&O|O&", pt_ExampleGenerator, &egen, &boundList, pt_weightByGen(egen), &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      PIM im=SELF_AS(TIMConstructor)(egen, boundset, weightID);
      return WrapOrange(im);
    }

    PyErr_Clear();

    PyObject *freeList;
    if (PyArg_ParseTuple(args, "O&OO|O&", pt_ExampleGenerator, &egen, &boundList, &freeList, pt_weightByGen(egen), &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      TVarList freeset;
      if (!varListFromDomain(freeList, egen->domain, freeset))
        return PYNULL;

      PIM im = SELF_AS(TIMConstructor)(egen, boundset, freeset, weightID);
      return WrapOrange(im);
    }

    PyErr_Clear();

    PIMByRows imbr;
    if (PyArg_ParseTuple(args, "O&", cc_IMByRows, &imbr))
      return WrapOrange(SELF_AS(TIMConstructor)(imbr));

    PYERROR(PyExc_TypeError, "invalid arguments -- examples, boundset and optional freeset and weight expected", PYNULL);
  PyCATCH
}



PyObject *IMByRowsConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example, bound-attrs[, weightID]) -> IM")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *boundList;
    int weightID=0;
    if (PyArg_ParseTuple(args, "O&O|O&", pt_ExampleGenerator, &egen, &boundList, pt_weightByGen(egen), &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      PIMByRows im=SELF_AS(TIMByRowsConstructor)(egen, boundset, weightID);
      return WrapOrange(im);
    }

    PyErr_Clear();

    PyObject *freeList;
    if (PyArg_ParseTuple(args, "O&OO|O&", pt_ExampleGenerator, &egen, &boundList, &freeList, pt_weightByGen(egen), &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      TVarList freeset;
      if (!varListFromDomain(freeList, egen->domain, freeset))
        return PYNULL;

      PIMByRows im=SELF_AS(TIMByRowsConstructor)(egen, boundset, freeset, weightID);
      return WrapOrange(im);
    }

    PYERROR(PyExc_TypeError, "invalid arguments -- examples, boundset and optional freeset and weight expected", PYNULL);
  PyCATCH
}




PyObject *IMByRowsPreprocessor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(IMByRows) -> None")
{ 
  PyTRY
    NO_KEYWORDS
    
    PIMByRows pimbr;
    if (!PyArg_ParseTuple(args, "O&", cc_IMByRows, &pimbr))
      PYERROR(PyExc_TypeError, "IMByRows expected", PYNULL)

    SELF_AS(TIMByRowsPreprocessor)(pimbr);
    RETURN_NONE;
  PyCATCH
}


PyObject *Float2List(float *f, int size)
{ PyObject *dlist = PyList_New(size);
  for(int i = 0; i < size; i++)
    PyList_SetItem(dlist, Py_ssize_t(i), PyFloat_FromDouble((double)*(f++)));
  return dlist;
}


bool List2Float(PyObject *l, float *&f, int &size)
{ if (!PyList_Check(l))
    PYERROR(PyExc_TypeError, "invalid type (list expected)", false);

  size = PyList_Size(l);
  float *fi = f = mlnew float[size];

  for(int s = 0; s<size; s++) {
    PyObject *flt = PyNumber_Float(PyList_GetItem(l, s));
    if (!flt) {
      PyErr_Format(PyExc_TypeError, "invalid list element at index '%i'", s);
      mldelete f;
      return false;
    }
    *fi = (float)PyFloat_AsDouble(flt);
    Py_DECREF(flt);
  }

  return true;
}


PyObject *convertToPython(const T_ExampleIMColumnNode &eicn)
{ PyObject *column=PyList_New(0);

  if (eicn.column) {
    bool discrete = dynamic_cast<TDIMColumnNode *>(eicn.column) != NULL;

    for(TIMColumnNode *node=eicn.column; node; node=node->next) {
      PyObject *pycnode = PYNULL;
      if (discrete) {
        TDIMColumnNode *dnode=dynamic_cast<TDIMColumnNode *>(node);
        pycnode=Py_BuildValue("ifN", dnode->index, dnode->nodeQuality, Float2List(dnode->distribution, dnode->noOfValues)); 
      }
      else {
        TFIMColumnNode *fnode=dynamic_cast<TFIMColumnNode *>(node);
        if (fnode)
          pycnode=Py_BuildValue("iffff", fnode->index, fnode->nodeQuality,
                                         fnode->sum, fnode->sum2, fnode->N);
      }

      if (!pycnode)
        PYERROR(PyExc_TypeError, "invalid IMColumnNode", PYNULL);

      PyList_Append(column, pycnode);
      Py_DECREF(pycnode);
    }
  }

  return Py_BuildValue("NN", Example_FromWrappedExample(eicn.example), column);
}


bool convertFromPython(PyObject *args, T_ExampleIMColumnNode &eicn)
{ PyObject *column;
  TExample *example;
  if (   !PyArg_ParseTuple(args, "O&O", ptr_Example, &example, &column)
      || !PyTuple_Check(column))
    PYERROR(PyExc_TypeError, "convertFromPython(T_ExampleIMColumnNode): invalid arguments", false);

  bool discrete = PyTuple_Size(column)==3;

  eicn.example = mlnew TExample(*example);
  eicn.column = NULL;
  TIMColumnNode **nodeptr = &eicn.column;

  for(Py_ssize_t i=0; i<PyList_Size(column); i++) {
    PyObject *item=PyList_GetItem(column, i);
    if (discrete) {
      *nodeptr=mlnew TDIMColumnNode(0, 0);
      PyObject *distr;
      TDIMColumnNode *dimcn = dynamic_cast<TDIMColumnNode *>(*nodeptr);
      if (   !PyArg_ParseTuple(item, "ifO", &(*nodeptr)->index, &(*nodeptr)->nodeQuality, &distr)
          || !List2Float(distr, dimcn->distribution, dimcn->noOfValues)) {
        mldelete eicn.column;
        PYERROR(PyExc_TypeError, "invalid column node", false);
      }
    }
    else {
      *nodeptr=mlnew TFIMColumnNode(0);
      if (!PyArg_ParseTuple(item, "iffff", &(*nodeptr)->index, &(*nodeptr)->nodeQuality,
                                           &dynamic_cast<TFIMColumnNode *>(*nodeptr)->sum, 
                                           &dynamic_cast<TFIMColumnNode *>(*nodeptr)->sum2, 
                                           &dynamic_cast<TFIMColumnNode *>(*nodeptr)->N)) {
        mldelete eicn.column;
        PYERROR(PyExc_TypeError, "invalid column node", false);
      }
    }
  }
  return true;
}


PyObject *convertToPython(const PIM &im)
{ PyObject *result=PyList_New(0);
  const_ITERATE(vector<T_ExampleIMColumnNode>, ici, im->columns) {
    PyObject *item=convertToPython(*ici);
    if (!item) {
      PyMem_DEL(result);
      PYERROR(PyExc_SystemError, "out of memory", PYNULL);
    }
    PyList_Append(result, item);
    Py_DECREF(item);
  }
  return result;
}
      
bool convertFromPython(PyObject *args, PIM &im)
{ im=PIM();
  if (!PyList_Check(args) || !PyList_Size(args))
    PYERROR(PyExc_TypeError, "invalid incompatibility matrix", false);

  // This is just to determine the type...
  int varType = -1;
  T_ExampleIMColumnNode testcolumn;
  if (!convertFromPython(PyList_GetItem(args, 0), testcolumn))
    return false;

  varType = dynamic_cast<TDIMColumnNode *>(testcolumn.column) ? TValue::INTVAR : TValue::FLOATVAR;
  const type_info &tinfo = typeid(*testcolumn.column);

  im=PIM(mlnew TIM(varType));
  im->columns=vector<T_ExampleIMColumnNode>();
  for(Py_ssize_t i=0; i<PyList_Size(args); i++) {
    PyObject *item=PyList_GetItem(args, i);
    im->columns.push_back(T_ExampleIMColumnNode());
    if (!convertFromPython(item, im->columns.back())) {
      im=PIM();
      return false;
    }
    if (tinfo == typeid(im->columns.back().column))
      PYERROR(PyExc_TypeError, "invalid incompatibility matrix (mixed discrete and continuous classes)", false)
  }

  return true;
}


PyObject *IM_native(PyObject *self) PYARGS(0, "() -> [[index, quality, distribution, c]] | [[index, quality, sum, sum2, N]]")
{ PyTRY
    return convertToPython(PyOrange_AsIM(self)); 
  PyCATCH
}

PyObject *IM_fuzzy(PyObject *self) PYARGS(0, "() -> boolean")
{ PyTRY
    return PyInt_FromLong(SELF_AS(TIM).fuzzy() ? 1L : 0L); 
  PyCATCH
}


PyObject *IM_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "<see the manual>")
{ PyTRY
    PIM im;
    PyObject *pyim;
    return PyArg_ParseTuple(args, "O:IM.new", &pyim) && convertFromPython(pyim, im) ? WrapOrange(im) : PYNULL;
  PyCATCH
}


PyObject *IM__reduce__(PyObject *self)
{
  PyTRY
    return Py_BuildValue("O(N)N", self->ob_type, IM_native(self), packOrangeDictionary(self));
  PyCATCH
}


PyObject *convertToPython(const TDIMRow &row)
{ PyObject *pyrow=PyList_New(row.nodes.size());
  Py_ssize_t i = 0;
  const int &noval = row.noOfValues;
  const_ITERATE(vector<float *>, ii, row.nodes)
    PyList_SetItem(pyrow, i++, Float2List(*ii, noval));

  return Py_BuildValue("NN", Example_FromWrappedExample(row.example), pyrow);
}


PyObject *convertToPython(const PIMByRows &im)
{ PyObject *result=PyList_New(im->rows.size());
  Py_ssize_t i=0;
  const_ITERATE(vector<TDIMRow>, ri, im->rows)
    PyList_SetItem(result, i++, convertToPython(*ri));
  return result;  
}

PyObject *IMByRows_native(PyObject *self) PYARGS(0, "() -> [example, [distributions]]")
{ PyTRY
    return convertToPython(PyOrange_AsIMByRows(self));
  PyCATCH
}

PyObject *IMByRows_get_columnExamples(PyObject *self) PYDOC("Values of bound attributes for each column")
{ PyTRY
    CAST_TO(TIMByRows, pimr);
    PyObject *result=PyList_New(pimr->columnExamples.size());
    Py_ssize_t i=0;
    ITERATE(vector<PExample>, ei, pimr->columnExamples)
      PyList_SetItem(result, i++, Example_FromWrappedExample(*ei));
    return result;
  PyCATCH
}

PyObject *ClustersFromIM_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(im) -> IMClustering")
{ 
  PyTRY
    NO_KEYWORDS

    PIM im;
    if (!PyArg_ParseTuple(args, "O&:ClustersFromIM.__call__", cc_IM, &im))
      return PYNULL;

    return WrapOrange(SELF_AS(TClustersFromIM)(im));
  PyCATCH
}





PyObject *AssessIMQuality_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(im) -> float")
{ 
  PyTRY
    NO_KEYWORDS

    PIM im;
    if (!PyArg_ParseTuple(args, "O&:AssessIMQuality.__call__", cc_IM, &im))
      return PYNULL;

    return PyFloat_FromDouble((double)SELF_AS(TAssessIMQuality)(im));
  PyCATCH
}



/* ************ FEATURE CONSTRUCTION BY CLUSTERING ******** */

#include "dist_clustering.hpp"

ABSTRACT(ExampleDistConstructor, Orange)
C_CALL(ExampleDistBySorting, ExampleDistConstructor, "([examples, bound-attrs[, weightID]]) -/-> ExampleDistVector")
BASED_ON(ExampleDistVector, Orange)
ABSTRACT(ClustersFromDistributions, Orange)
C_CALL(ClustersFromDistributionsByAssessor, ClustersFromDistributions, "([example-dist-vector] [minProfitProportion=, distributionAssessor=, stopCriterion=]) -/-> DistClustering")
C_CALL(FeatureByDistributions, FeatureInducer, "() | ([examples, bound-attrs, name], [constructExampleDist=, completion=]) -/-> Variable")

ABSTRACT(DistributionAssessor, Orange)
C_NAMED(DistributionAssessor_Laplace, DistributionAssessor, "()")
C_NAMED(DistributionAssessor_m, DistributionAssessor, "([m=])")
C_NAMED(DistributionAssessor_mf, DistributionAssessor, "([m=])")
C_NAMED(DistributionAssessor_Relief, DistributionAssessor, "()")
C_NAMED(DistributionAssessor_Measure, DistributionAssessor, "([measure=])")
C_NAMED(DistributionAssessor_Kramer, DistributionAssessor, "()")

ABSTRACT(StopDistributionClustering, Orange)
C_NAMED(StopDistributionClustering_noProfit, StopDistributionClustering, "([minProfitProportion=])")
C_NAMED(StopDistributionClustering_binary, StopDistributionClustering, "()")
C_NAMED(StopDistributionClustering_n, StopDistributionClustering, "([n=])")
C_NAMED(StopDistributionClustering_noBigChange, StopDistributionClustering, "()")


PyObject *ExampleDistConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples, bound-attrs[, weightID]) -> ExampleDistVector")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator egen;
    PyObject *boundList;
    int weightID=0;
    if (!PyArg_ParseTuple(args, "O&O|O&:ExampleDistConstructor.__call__", pt_ExampleGenerator, &egen, &boundList, pt_weightByGen(egen), &weightID))
      return PYNULL;

    TVarList boundset;
    if (!varListFromDomain(boundList, egen->domain, boundset))
      return PYNULL;

    PExampleDistVector edv = SELF_AS(TExampleDistConstructor)(egen, boundset, weightID);
    return WrapOrange(edv);
  PyCATCH
}





PyObject *convertToPython(const T_ExampleDist &ed)
{ return Py_BuildValue("NN", Example_FromWrappedExample(ed.example), WrapOrange(const_cast<GCPtr<TDistribution> &>(ed.distribution))); }



PyObject *convertToPython(const PExampleDistVector &edv)
{ PyObject *result=PyList_New(0);
  const_ITERATE(vector<T_ExampleDist>, ici, edv->values) {
    PyObject *item=convertToPython(*ici);
    if (!item) {
      PyMem_DEL(result);
      PYERROR(PyExc_SystemError, "out of memory", PYNULL);
    }
    PyList_Append(result, item);
    Py_DECREF(item);
  }
  return result;
}
      

PyObject *ExampleDistVector__reduce__(PyObject *self)
{
  PyTRY
    vector<T_ExampleDist> &values = SELF_AS(TExampleDistVector).values;

    PyObject *pyvalues = PyList_New(values.size() * 2);
    Py_ssize_t i = 0;
    ITERATE(vector<T_ExampleDist>, edi, values) {
      PyList_SetItem(pyvalues, i++, Example_FromWrappedExample(edi->example));
      PyList_SetItem(pyvalues, i++, WrapOrange(edi->distribution));
    }
     
    return Py_BuildValue("O(ON)N", getExportedFunction("__pickleLoaderExampleDistVector"),
                                   self->ob_type,
                                   pyvalues,
                                   packOrangeDictionary(self));

  PyCATCH
}


PyObject *__pickleLoaderExampleDistVector(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, values)")
{
  PyTRY
    PyTypeObject *type;
    PyObject *pyvalues;
    if (!PyArg_ParseTuple(args, "OO:__pickleLoaderExampleDistVector", &type, &pyvalues))
      return NULL;

    TExampleDistVector *ed = new TExampleDistVector();

    try {
      Py_ssize_t i = 0, e = PyList_Size(pyvalues);
      ed->values.reserve(e>>1);
      while(i < e) {
        PExample ex = PyExample_AS_Example(PyList_GetItem(pyvalues, i++));
        PDistribution dist = PyOrange_AsDistribution(PyList_GetItem(pyvalues, i++));
        ed->values.push_back(T_ExampleDist(ex, dist));
      }

      return WrapNewOrange(ed, type);
    }
    catch (...) {
      delete ed;
      throw;
    }
  PyCATCH
}


PyObject *ExampleDistVector_native(PyObject *self) PYARGS(0, "() -> [[[float]]] | [[{float: float}]]")
{ PyTRY
    return convertToPython(PyOrange_AsExampleDistVector(self));
  PyCATCH
}



PyObject *ClustersFromDistributions_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example-dist-vector) -> DistClustering")
{ 
  PyTRY
    NO_KEYWORDS

    PExampleDistVector edv;
    if (!PyArg_ParseTuple(args, "O&:ClustersFromDistributions.__call__", cc_ExampleDistVector, &edv))
      return PYNULL;

    return WrapOrange(SELF_AS(TClustersFromDistributions)(edv));
  PyCATCH
}


#include "lib_preprocess.px"
