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


/********************************

This file includes constructors and specialized methods for ML* object, defined in project Preprocess

*********************************/

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

#include "externs.px"


/* ************ DISCRETIZATION ************ */

#include "discretize.hpp"


C_NAMED(EquiDistDiscretizer, TransformValue, "([numberOfIntervals=, firstVal=, step=])")
C_NAMED(IntervalDiscretizer, TransformValue, "([points=])")
C_NAMED(ThresholdDiscretizer, TransformValue, "([threshold=])")

BASED_ON(Discretization, Orange)
BASED_ON(DiscretizedDomain, Domain)
C_CALL (EquiDistDiscretization, Discretization, "() | (attribute, examples[, weight, numberOfIntervals=]) -/-> Variable")
C_CALL (   EquiNDiscretization, Discretization, "() | (attribute, examples[, weight, numberOfIntervals=]) -/-> Variable")
C_CALL ( EntropyDiscretization, Discretization, "() | (attribute, examples[, weight]) -/-> Variable")
C_CALL ( BiModalDiscretization, Discretization, "() | (attribute, examples[, weight]) -/-> Variable")


PyObject *Discretization_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(attribute, examples[, weight]) -> Variable")
{
  PyTRY
    SETATTRIBUTES 
    PyObject *variable;
    PExampleGenerator egen;
    int weightID=0;
    if (!PyArg_ParseTuple(args, "OO&|i", &variable, pt_ExampleGenerator, &egen, &weightID)) 
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


PyObject *IntervalDiscretizer_constructVariable(PyObject *self, PyObject *var) PYARGS(METH_O, "(variable) -> variable")
{ PyTRY
    if (!PyOrIntervalDiscretizer_Check(self) || !PyOrVariable_Check(var))
      PYERROR(PyExc_TypeError, "invalid parameters (variable expected)", PYNULL);

    return WrapOrange(TIntervalDiscretizer::constructVar(PyOrange_AsVariable(var), PyOrange_AsIntervalDiscretizer(self)));
  PyCATCH
}

/* ************ SVM FILTERS ************** */

#include "svm_filtering.hpp"

C_NAMED(Discrete2Continuous, TransformValue, "([value=])")
C_NAMED(NormalizeContinuous, TransformValue, "([average=, span=])")

PyObject *Domain4SVM(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "([domain | examples]) -> domain")
{ PyTRY
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg))
      PYERROR(PyExc_TypeError, "Domain4SVM: domain or examples expected", PYNULL);

    PDomain res;

    if (PyOrDomain_Check(arg))
      res=domain4SVM(PyOrange_AsDomain(arg));
    else {
      PExampleGenerator egen=exampleGenFromParsedArgs(arg);
      if (!egen)
        PYERROR(PyExc_TypeError, "Domain4SVM: domain or examples expected", PYNULL);
      res=domain4SVM(egen);
    }

    return WrapOrange(res);
  PyCATCH
}

/* ************ REDUNDANCIES ************ */

#include "redundancy.hpp"

BASED_ON(RemoveRedundant, Orange)

C_CALL(RemoveRedundantByInduction, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")
C_CALL(RemoveRedundantByQuality, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")
C_CALL(RemoveRedundantOneValue, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")
C_CALL(RemoveNonexistentValues, RemoveRedundant, "([examples[, weightID][, suspicious]) -/-> Domain")


PyObject *RemoveRedundant_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([examples[, weightID][, suspicious]) -/-> Domain")
{
  PyTRY
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *suspiciousList=NULL;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&|Oi", pt_ExampleGenerator, &egen, &suspiciousList, &weight))
      PYERROR(PyExc_TypeError, "attribute error", PYNULL);

    TVarList suspiciousset;
    if (suspiciousList)
      if (!varListFromDomain(suspiciousList, egen->domain, suspiciousset))
        return PYNULL;

    PDomain newdomain = SELF_AS(TRemoveRedundant)(egen, suspiciousList ? &suspiciousset : NULL, NULL, weight);
    return WrapOrange(newdomain);
  PyCATCH
}



/* ************ PREPROCESSORS ************ */

#include "preprocessors.hpp"

BASED_ON(Preprocessor, Orange)

C_CALL(Preprocessor_select, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_ignore, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")

C_CALL(Preprocessor_take, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_drop, Preprocessor, "([examples[, weightID]] [attributes=<list-of-strings>]) -/-> ExampleTable")
C_CALL(Preprocessor_removeDuplicates, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_takeMissing, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_dropMissing, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_takeMissingClasses, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")
C_CALL(Preprocessor_dropMissingClasses, Preprocessor, "([examples[, weightID]]) -/-> ExampleTable")

C_CALL(Preprocessor_addMissing, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addMissingClasses, Preprocessor, "([examples[, weightID]] [classMissing=<float>]) -/-> ExampleTable")
C_CALL(Preprocessor_addNoise, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addClassNoise, Preprocessor, "([examples[, weightID]] [classNoise=<float>]) -/-> ExampleTable")
C_CALL(Preprocessor_addGaussianNoise, Preprocessor, "([examples[, weightID]] [<see the manual>]) -/-> ExampleTable")
C_CALL(Preprocessor_addGaussianClassNoise, Preprocessor, "([examples[, weightID]] [classDeviation=<float>]) -/-> ExampleTable")

C_CALL(Preprocessor_addClassWeight, Preprocessor, "([examples[, weightID]] [equalize=, classWeights=) -/-> ExampleTable")
C_CALL(Preprocessor_addCensorWeight, Preprocessor, "([examples[, weightID]] [method=0-km, 1-nmr, 2-linear, outcomeVar=, eventValue=, timeID=, maxTime=]) -/-> ExampleTable")

C_CALL(Preprocessor_filter, Preprocessor, "([examples[, weightID]] [filter=]) -/-> ExampleTable")
C_CALL(Preprocessor_discretize, Preprocessor, "([examples[, weightID]] [noOfIntervals=, notClass=, method=, attributes=<list-of-strings>]) -/-> ExampleTable")

PYCLASSCONSTANT_INT(Preprocessor_addCensorWeight, KM, TPreprocessor_addCensorWeight::km)
PYCLASSCONSTANT_INT(Preprocessor_addCensorWeight, Linear, TPreprocessor_addCensorWeight::linear)
PYCLASSCONSTANT_INT(Preprocessor_addCensorWeight, Bayes, TPreprocessor_addCensorWeight::bayes)

PyObject *Preprocessor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> ExampleTable")
{ 
  PyTRY
    SETATTRIBUTES
    long weightID=0;
    PExampleGenerator egen=exampleGenFromArgs(args, &weightID);
    if (!egen)
      PYERROR(PyExc_TypeError, "attribute error (example generator expected)", PYNULL);
    bool weightGiven=(weightID!=0);

    int newWeight;
    PExampleGenerator res = SELF_AS(TPreprocessor)(egen, weightID, newWeight);
    PyObject *wrappedGen=WrapOrange(res);
    return weightGiven || newWeight ? Py_BuildValue("Ni", wrappedGen, newWeight) : wrappedGen;
  PyCATCH
}



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

    aMap->__ormap[var] = (min<max) ? mlnew TValueFilter_continuous(min, max)
                                   : mlnew TValueFilter_continuous(max, min, true);
    return 0;
  }

  if (var->varType == TValue::INTVAR) {
    TValueFilter_discrete *vfilter = mlnew TValueFilter_discrete(var);
    PValueFilter wvfilter = vfilter;
    TValueList &valueList = vfilter->acceptableValues.getReference();

    if (PyTuple_Check(pyvalue) || PyList_Check(pyvalue)) {
      PyObject *iterator = PyObject_GetIter(pyvalue);
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
        TValue value;
        if (!convertFromPython(item, value, var)) {
          Py_DECREF(item);
          Py_DECREF(iterator);
          return -1;
        }
        valueList.push_back(value);
      }
      Py_DECREF(iterator);
    }
    else {
      TValue value;
      if (!convertFromPython(pyvalue, value, var))
        return -1;
      valueList.push_back(value);
    }

    aMap->__ormap[var] = wvfilter;
    return 0;
  }

  PYERROR(PyExc_TypeError, "VariableFilterMap.__setitem__: unrecognized item type", -1);
}


int TMM_VariableFilterMap::_setitemlow(TVariableFilterMap *aMap, PyObject *pykey, PyObject *pyvalue)
{ PyTRY
    PVariable var;
    return TMM_VariableFilterMap::_keyFromPython(pykey, var) ? VariableFilterMap_setitemlow(aMap, var, pyvalue) : -1;
  PyCATCH_1
}


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
  if (!PyArg_ParseTuple(args, "O&OOOi:kaplanMeier", pt_ExampleGenerator, &egen, &outcomevar, &pyfailvalue, &timevar, &weightID))

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
PyObject *VariableFilterMap_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(items)") { return TMM_VariableFilterMap::_new(type, arg, kwds); }
PyObject *VariableFilterMap_str(TPyOrange *self) { return TMM_VariableFilterMap::_str(self); }
PyObject *VariableFilterMap_repr(TPyOrange *self) { return TMM_VariableFilterMap::_str(self); }
PyObject *VariableFilterMap_getitem(TPyOrange *self, PyObject *key) { return TMM_VariableFilterMap::_getitem(self, key); }
int       VariableFilterMap_setitem(TPyOrange *self, PyObject *key, PyObject *value) { return TMM_VariableFilterMap::_setitem(self, key, value); }
int       VariableFilterMap_len(TPyOrange *self) { return TMM_VariableFilterMap::_len(self); }
int       VariableFilterMap_contains(TPyOrange *self, PyObject *key) { return TMM_VariableFilterMap::_contains(self, key); }

PyObject *VariableFilterMap_has_key(TPyOrange *self, PyObject *key) PYARGS(METH_O, "(key) -> None") { return TMM_VariableFilterMap::_has_key(self, key); }
PyObject *VariableFilterMap_get(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFilterMap::_get(self, args); }
PyObject *VariableFilterMap_setdefault(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFilterMap::_setdefault(self, args); }
PyObject *VariableFilterMap_clear(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> None") { return TMM_VariableFilterMap::_clear(self); }
PyObject *VariableFilterMap_keys(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> keys") { return TMM_VariableFilterMap::_keys(self); }
PyObject *VariableFilterMap_values(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> values") { return TMM_VariableFilterMap::_values(self); }
PyObject *VariableFilterMap_items(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> items") { return TMM_VariableFilterMap::_items(self); }
PyObject *VariableFilterMap_update(TPyOrange *self, PyObject *args) PYARGS(METH_O, "(items) -> None") { return TMM_VariableFilterMap::_update(self, args); }


typedef MapMethods<PVariableFloatMap, TVariableFloatMap, PVariable, float> TMM_VariableFloatMap;
INITIALIZE_MAPMETHODS(TMM_VariableFloatMap, &PyOrVariable_Type, NULL, _orangeValueFromPython<PVariable>, _nonOrangeValueFromPython<float>, _orangeValueToPython<PVariable>, _nonOrangeValueToPython<float>);

PVariableFloatMap PVariableFloatMap_FromArguments(PyObject *arg) { return TMM_VariableFloatMap::P_FromArguments(arg); }
PyObject *VariableFloatMap_FromArguments(PyTypeObject *type, PyObject *arg) { return TMM_VariableFloatMap::_FromArguments(type, arg); }
PyObject *VariableFloatMap_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(items)") { return TMM_VariableFloatMap::_new(type, arg, kwds); }
PyObject *VariableFloatMap_str(TPyOrange *self) { return TMM_VariableFloatMap::_str(self); }
PyObject *VariableFloatMap_repr(TPyOrange *self) { return TMM_VariableFloatMap::_str(self); }
PyObject *VariableFloatMap_getitem(TPyOrange *self, PyObject *key) { return TMM_VariableFloatMap::_getitem(self, key); }
int       VariableFloatMap_setitem(TPyOrange *self, PyObject *key, PyObject *value) { return TMM_VariableFloatMap::_setitem(self, key, value); }
int       VariableFloatMap_len(TPyOrange *self) { return TMM_VariableFloatMap::_len(self); }
int       VariableFloatMap_contains(TPyOrange *self, PyObject *key) { return TMM_VariableFloatMap::_contains(self, key); }

PyObject *VariableFloatMap_has_key(TPyOrange *self, PyObject *key) PYARGS(METH_O, "(key) -> None") { return TMM_VariableFloatMap::_has_key(self, key); }
PyObject *VariableFloatMap_get(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFloatMap::_get(self, args); }
PyObject *VariableFloatMap_setdefault(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(key[, default]) -> value") { return TMM_VariableFloatMap::_setdefault(self, args); }
PyObject *VariableFloatMap_clear(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> None") { return TMM_VariableFloatMap::_clear(self); }
PyObject *VariableFloatMap_keys(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> keys") { return TMM_VariableFloatMap::_keys(self); }
PyObject *VariableFloatMap_values(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> values") { return TMM_VariableFloatMap::_values(self); }
PyObject *VariableFloatMap_items(TPyOrange *self, PyObject *args) PYARGS(METH_NOARGS, "() -> items") { return TMM_VariableFloatMap::_items(self); }
PyObject *VariableFloatMap_update(TPyOrange *self, PyObject *args) PYARGS(METH_O, "(items) -> None") { return TMM_VariableFloatMap::_update(self, args); }


/* ************ INDUCE ************ */

#include "induce.hpp"
#include "subsets.hpp"

BASED_ON(FeatureInducer, Orange)

BASED_ON(SubsetsGenerator, Orange)
C_NAMED(SubsetsGenerator_constant, SubsetsGenerator, "()")
C_NAMED(SubsetsGenerator_withRestrictions, SubsetsGenerator, "([subGenerator=])")


PyObject *FeatureInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples, bound-attrs, new-name, weightID) -> (Variable, float)")
{
  PyTRY
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *boundList;
    char *name;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&Os|i", pt_ExampleGenerator, &egen, &boundList, &name, &weight))
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

int SubsetsGeneratorResetFromVars(PyObject *self, PyObject *vars)
{
  if (vars) {
    PVarList variables;
    if (PyOrDomain_Check(vars))
      variables = PyOrange_AsDomain(vars)->attributes;
    else
      variables = PVarList_FromArguments(vars);
    
    if (!variables)
      PYERROR(PyExc_TypeError, "SubsetsGenerator.reset: invalid arguments", -1);

    return SELF_AS(TSubsetsGenerator).reset(variables.getReference()) ? 1 : 0;
  }
  else
    return SELF_AS(TSubsetsGenerator).reset() ? 1 : 0;
}


PyObject *SubsetsGenerator_reset(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "([[var0, var1, ...]]) -> int")
{ PyTRY
    PyObject *vars = PYNULL;
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator.reset", &vars))
      return PYNULL;

    return PyInt_FromLong(SubsetsGeneratorResetFromVars(self, vars) ? 1L : 0L);
  PyCATCH
}


PyObject *SubsetsGenerator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([var0, var1] | domain |) -> SubsetsGenerator")
{ PyTRY
    SETATTRIBUTES
    SubsetsGenerator_reset(self, args);
    Py_INCREF(self);
    return self;
  PyCATCH
}


PyObject *SubsetsGenerator_iter(PyObject *self)
{ if (!SELF_AS(TSubsetsGenerator).reset())
    PYERROR(PyExc_SystemError, "'SubsetsGenerator.iter' cannot reset (check the arguments)", PYNULL);

  Py_INCREF(self);
  return self;
}


PyObject *SubsetsGenerator_iternext(PyObject *self)
{ PyTRY
    TVarList vl;
    if (!SELF_AS(TSubsetsGenerator).nextSubset(vl))
      return PYNULL;

    PyObject *list=PyTuple_New(vl.size());
    int i=0;
    ITERATE(TVarList, vi, vl)
      PyTuple_SetItem(list, i++, WrapOrange(*vi));
    return list;
  PyCATCH
}



PyObject *SubsetsGenerator_constSize_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SubsetsGenerator, "(size)")
{ PyTRY
    int B = 2;
    PyObject *varlist = NULL;

    // This is for compatibility ...
    if (PyArg_ParseTuple(args, "i|O:SubsetsGenerator_constSize.__new__", &B, &varlist)) {
      TSubsetsGenerator_constSize *ssg = mlnew TSubsetsGenerator_constSize(B);
      PyObject *res = WrapNewOrange(ssg, type);
      if (varlist) {
        ssg->varList = varlist;
        SubsetsGenerator_reset(res, varlist);
      return res;
    }
    PyErr_Clear();

    // ... and this if for real
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator_constSize.__new__", &varlist))
      return PYNULL;
      
    PyObject *res = WrapNewOrange(mlnew TSubsetsGenerator_constSize(B), type);
    if (SubsetsGeneratorResetFromVars(res, varlist) < 0)
      return PYNULL;
    return res;
  PyCATCH
}


PyObject *SubsetsGenerator_minMaxSize_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(SubsetsGenerator, "([min=, max=])")
{ PyTRY
    int min = 2, max = 3;
    PyObject *varlist = NULL;

    // This is for compatibility ...
    if (args && PyArg_ParseTuple(args, "i|iO", &min, &max, &varlist)) {
      PyObject *res = WrapNewOrange(mlnew TSubsetsGenerator_minMaxSize(min, max), type);
      if (varlist)
        SubsetsGenerator_reset(res, varlist);

      return res;
    }
    PyErr_Clear();

    // ... and this if for real
    if (!PyArg_ParseTuple(args, "|O:SubsetsGenerator_minMaxSize.__new__", &varlist))
      return PYNULL;
      
    PyObject *res = WrapNewOrange(mlnew TSubsetsGenerator_minMaxSize(min, max), type);
    if (varlist && SubsetsGeneratorResetFromVars(res, varlist) < 0)
      return PYNULL;
    return res;
  PyCATCH
}


/* ************ MINIMAL COMPLEXITY ************ */

#include "minimal_complexity.hpp"

BASED_ON(IGConstructor, Orange)
C_CALL(IGBySorting, IGConstructor, "([examples, bound-attrs]) -/-> IG")

BASED_ON(ColorIG, Orange)
C_CALL(ColorIG_MCF, ColorIG, "([IG]) -/-> ColoredIG")

C_CALL(FeatureByMinComplexity, FeatureInducer, "([examples, bound-attrs, name] [IGConstructor=, classifierFromIG=) -/-> Variable")

C_NAMED(ColoredIG, GeneralExampleClustering, "(ig=, colors=)")

PYCLASSCONSTANT_INT(FeatureByMinComplexity, NoCompletion, completion_no)
PYCLASSCONSTANT_INT(FeatureByMinComplexity, CompletionByDefault, completion_default)
PYCLASSCONSTANT_INT(FeatureByMinComplexity, CompletionByBayes, completion_bayes)

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
  for(int i=0; i<PyList_Size(args); i++) {
    ig->nodes.push_back(TIGNode());
    if (convertFromPython(PyList_GetItem(args, i), ig->nodes.back())) {
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
    int i=0;
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
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *boundList;
    int weight=0;
    if (!PyArg_ParseTuple(args, "O&O|i", pt_ExampleGenerator, &egen, &boundList, &weight))
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
    SETATTRIBUTES
    PIG graph;
    return PyArg_ParseTuple(args, "O&:ColorIG.__call__", cc_IG, &graph) ? WrapOrange(SELF_AS(TColorIG)(graph)) : PYNULL;
  PyCATCH
}


/* ************ MINIMAL ERROR ******** */

#include "minimal_error.hpp"

C_CALL(FeatureByIM, FeatureInducer, "([examples, bound-attrs, name] [constructIM=, classifierFromIM=]) -/-> Variable")
BASED_ON(IMConstructor, Orange)
C_CALL(IMBySorting, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")
C_CALL(IMByIMByRows, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")
C_CALL(IMByRelief, IMConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IM")
BASED_ON(ClustersFromIM, Orange)
C_CALL(ClustersFromIMByAssessor, ClustersFromIM, "([IM] [minProfitProportion=, columnAssessor=, stopCriterion=]) -/-> IMClustering")
C_NAMED(IMClustering, Orange, "([im= clusters=, maxCluster=])")

BASED_ON(IMByRows, Orange)

BASED_ON(IMByRowsConstructor, Orange)
C_CALL(IMByRowsBySorting, IMByRowsConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IMByRows")
C_CALL(IMByRowsByRelief, IMByRowsConstructor, "() | (examples, bound-attrs[[, free-attrs], weightID]) -/-> IMByRows")

BASED_ON(IMByRowsPreprocessor, Orange)
C_CALL(IMBlurer, IMByRowsPreprocessor, "([IMByRows]) -> None")

C_CALL3(AssessIMQuality, AssessIMQuality, Orange, "([IM] -/-> float)")

BASED_ON(StopIMClusteringByAssessor, Orange)
C_NAMED(StopIMClusteringByAssessor_noProfit, StopIMClusteringByAssessor, "([minProfitProportion=])")
C_NAMED(StopIMClusteringByAssessor_binary, StopIMClusteringByAssessor, "()")
C_NAMED(StopIMClusteringByAssessor_n, StopIMClusteringByAssessor, "(n=)")
C_NAMED(StopIMClusteringByAssessor_noBigChange, StopIMClusteringByAssessor, "()")

BASED_ON(ColumnAssessor, Orange)
C_NAMED(ColumnAssessor_m, ColumnAssessor, "([m=])")
C_NAMED(ColumnAssessor_Laplace, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_mf, ColumnAssessor, "([m=])")
C_NAMED(ColumnAssessor_N, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_Relief, ColumnAssessor, "()")
C_NAMED(ColumnAssessor_Measure, ColumnAssessor, "(measure=)")
C_NAMED(ColumnAssessor_Kramer, ColumnAssessor, "()")

C_CALL(MeasureAttribute_IM, MeasureAttribute, "(constructIM=, columnAssessor=) | (attr, examples[, apriori] [,weightID]) -/-> (float, meas-type)")


PYCLASSCONSTANT_INT(FeatureByIM, NoCompletion, completion_no)
PYCLASSCONSTANT_INT(FeatureByIM, CompletionByDefault, completion_default)
PYCLASSCONSTANT_INT(FeatureByIM, CompletionByBayes, completion_bayes)


PyObject *IMConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example, bound-attrs[, weightID]) -> IM")
{
  PyTRY
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *boundList;
    int weightID = 0;
    if (PyArg_ParseTuple(args, "O&O|i", pt_ExampleGenerator, &egen, &boundList, &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      PIM im=SELF_AS(TIMConstructor)(egen, boundset, weightID);
      return WrapOrange(im);
    }

    PyErr_Clear();

    PyObject *freeList;
    if (PyArg_ParseTuple(args, "O&OO|i", pt_ExampleGenerator, &egen, &boundList, &freeList, &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      TVarList freeset;
      if (!varListFromDomain(freeList, egen->domain, freeset))
        return PYNULL;

      PIM im=SELF_AS(TIMConstructor)(egen, boundset, freeset, weightID);
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
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *boundList;
    int weightID=0;
    if (PyArg_ParseTuple(args, "O&O|i", pt_ExampleGenerator, &egen, &boundList, &weightID)) {
      TVarList boundset;
      if (!varListFromDomain(boundList, egen->domain, boundset))
        return PYNULL;

      PIMByRows im=SELF_AS(TIMByRowsConstructor)(egen, boundset, weightID);
      return WrapOrange(im);
    }

    PyErr_Clear();

    PyObject *freeList;
    if (PyArg_ParseTuple(args, "O&OO|i", pt_ExampleGenerator, &egen, &boundList, &freeList, &weightID)) {
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
{ PyTRY
    SETATTRIBUTES
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
    PyList_SetItem(dlist, i, PyFloat_FromDouble((double)*(f++)));
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

  for(int i=0; i<PyList_Size(column); i++) {
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
  for(int i=0; i<PyList_Size(args); i++) {
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


PyObject *convertToPython(const TDIMRow &row)
{ PyObject *pyrow=PyList_New(row.nodes.size());
  int i = 0;
  const int &noval = row.noOfValues;
  const_ITERATE(vector<float *>, ii, row.nodes)
    PyList_SetItem(pyrow, i++, Float2List(*ii, noval));

  return Py_BuildValue("NN", Example_FromWrappedExample(row.example), pyrow);
}


PyObject *convertToPython(const PIMByRows &im)
{ PyObject *result=PyList_New(im->rows.size());
  int i=0;
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
    int i=0;
    ITERATE(vector<PExample>, ei, pimr->columnExamples)
      PyList_SetItem(result, i++, Example_FromWrappedExample(*ei));
    return result;
  PyCATCH
}

PyObject *ClustersFromIM_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(im) -> IMClustering")
{ 
  PyTRY
    SETATTRIBUTES
    PIM im;
    if (!PyArg_ParseTuple(args, "O&:ClustersFromIM.__call__", cc_IM, &im))
      return PYNULL;

    return WrapOrange(SELF_AS(TClustersFromIM)(im));
  PyCATCH
}





PyObject *AssessIMQuality_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(im) -> float")
{ 
  PyTRY
    SETATTRIBUTES
    PIM im;
    if (!PyArg_ParseTuple(args, "O&:AssessIMQuality.__call__", cc_IM, &im))
      return PYNULL;

    return PyFloat_FromDouble((double)SELF_AS(TAssessIMQuality)(im));
  PyCATCH
}



/* ************ FEATURE CONSTRUCTION BY CLUSTERING ******** */

#include "dist_clustering.hpp"

BASED_ON(ExampleDistConstructor, Orange)
C_CALL(ExampleDistBySorting, ExampleDistConstructor, "([examples, bound-attrs[, weightID]]) -/-> ExampleDistVector")
BASED_ON(ExampleDistVector, Orange)
BASED_ON(ClustersFromDistributions, Orange)
C_CALL(ClustersFromDistributionsByAssessor, ClustersFromDistributions, "([example-dist-vector] [minProfitProportion=, distributionAssessor=, stopCriterion=]) -/-> DistClustering")
C_CALL(FeatureByDistributions, FeatureInducer, "() | ([examples, bound-attrs, name], [constructExampleDist=, completion=]) -/-> Variable")

BASED_ON(DistributionAssessor, Orange)
C_NAMED(DistributionAssessor_Laplace, DistributionAssessor, "()")
C_NAMED(DistributionAssessor_m, DistributionAssessor, "([m=])")
C_NAMED(DistributionAssessor_mf, DistributionAssessor, "([m=])")
C_NAMED(DistributionAssessor_Relief, DistributionAssessor, "()")
C_NAMED(DistributionAssessor_Measure, DistributionAssessor, "([measure=])")
C_NAMED(DistributionAssessor_Kramer, DistributionAssessor, "()")

BASED_ON(StopDistributionClustering, Orange)
C_NAMED(StopDistributionClustering_noProfit, StopDistributionClustering, "([minProfitProportion=])")
C_NAMED(StopDistributionClustering_binary, StopDistributionClustering, "()")
C_NAMED(StopDistributionClustering_n, StopDistributionClustering, "([n=])")
C_NAMED(StopDistributionClustering_noBigChange, StopDistributionClustering, "()")


PYCLASSCONSTANT_INT(FeatureByDistributions, NoCompletion, completion_no)
PYCLASSCONSTANT_INT(FeatureByDistributions, CompletionByDefault, completion_default)
PYCLASSCONSTANT_INT(FeatureByDistributions, CompletionByBayes, completion_bayes)


PyObject *ExampleDistConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples, bound-attrs[, weightID]) -> ExampleDistVector")
{
  PyTRY
    SETATTRIBUTES
    PExampleGenerator egen;
    PyObject *boundList;
    int weightID=0;
    if (!PyArg_ParseTuple(args, "O&O|i:ExampleDistConstructor.__call__", pt_ExampleGenerator, &egen, &boundList, &weightID))
      return PYNULL;

    TVarList boundset;
    if (!varListFromDomain(boundList, egen->domain, boundset))
      return PYNULL;

    PExampleDistVector edv = SELF_AS(TExampleDistConstructor)(egen, boundset, weightID);
    return WrapOrange(edv);
  PyCATCH
}





PyObject *convertToPython(const T_ExampleDist &ed)
{ return Py_BuildValue("NN", Example_FromWrappedExample(ed.example), WrapOrange(ed.distribution)); }



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
      


PyObject *ExampleDistVector_native(PyObject *self) PYARGS(0, "() -> [[[float]]] | [[{float: float}]]")
{ PyTRY
    return convertToPython(PyOrange_AsExampleDistVector(self));
  PyCATCH
}



PyObject *ClustersFromDistributions_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(example-dist-vector) -> DistClustering")
{ 
  PyTRY
    SETATTRIBUTES
    PExampleDistVector edv;
    if (!PyArg_ParseTuple(args, "O&:ClustersFromDistributions.__call__", cc_ExampleDistVector, &edv))
      return PYNULL;

    return WrapOrange(SELF_AS(TClustersFromDistributions)(edv));
  PyCATCH
}


#include "lib_preprocess.px"
