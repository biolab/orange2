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


#include "values.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "table.hpp"
#include "contingency.hpp"
#include "distance.hpp"
#include "errors.hpp"

#include "cls_example.hpp"
#include "cls_value.hpp"
#include "cls_orange.hpp"

#include "lib_kernel.hpp"
#include "externs.px"

#include "callback.ppp"


inline PyObject *callCallback(PyObject *self, PyObject *args)
{ PyObject *result;
  
  if (PyObject_HasAttrString(self, "__callback")) {
    PyObject *callback = PyObject_GetAttrString(self, "__callback");
    result = PyObject_CallObject(callback, args);
    Py_DECREF(callback);
  }
  else 
    result = PyEval_CallObject(self, args);

  Py_DECREF(args);

  if (!result)
    throw pyexception();

  return result;
}


PyObject *setCallbackFunction(PyObject *self, PyObject *args)
{ PyObject *func;
  if (!PyArg_ParseTuple(args, "O", &func)) {
    PyErr_Format(PyExc_TypeError, "callback function for '%s' expected", self->ob_type->tp_name);
    Py_DECREF(self);
    return PYNULL;
  }
  else if (!PyCallable_Check(func)) {
    PyErr_Format(PyExc_TypeError, "'%s' object is not callable", func->ob_type->tp_name);
    Py_DECREF(self);
    return PYNULL;
  }

  PyObject_SetAttrString(self, "__callback", func);
  return self;
}


bool TFilter_Python::operator()(const TExample &ex)
{ PyObject *result = callCallback((PyObject *)myWrapper, Py_BuildValue("(N)", Example_FromExampleCopyRef(ex)));

  bool res = bool(PyObject_IsTrue(result)!=0);
  Py_DECREF(result);
  return res;
}


void TTransformValue_Python::transform(TValue &val)
{
  PyObject *result = callCallback((PyObject *)myWrapper, Py_BuildValue("(N)", Value_FromValue(val)));

  PVariable var;
  bool succ = convertFromPython(result, val, var);
  Py_DECREF(result);

  if (!succ)
    raiseError("TransformValue.__call__'s result cannot be converted to a Value");
}


TMeasureAttribute_Python::TMeasureAttribute_Python()
: TMeasureAttribute(TMeasureAttribute::DomainContingency, true, true)
{}


float TMeasureAttribute_Python::callMeasure(PyObject *args)
{
  PyObject *res = callCallback((PyObject *)myWrapper, args);
  PyObject *resf = PyNumber_Float(res);
  Py_DECREF(res);

  if (!resf)
    raiseError("invalid result from __call__");

  float mres = (float)PyFloat_AsDouble(resf);  
  Py_DECREF(resf);
  return mres;
}


float TMeasureAttribute_Python::operator()(PContingency cont, PDistribution classDistribution, PDistribution apriorClass)
{ return callMeasure(Py_BuildValue("(NNN)", WrapOrange(cont),
                                            WrapOrange(classDistribution),
                                            WrapOrange(apriorClass))); }


float TMeasureAttribute_Python::operator()(int attrNo, PDomainContingency dcont, PDistribution apriorClass)
{ return callMeasure(Py_BuildValue("(iNN)", attrNo, 
                                            WrapOrange(dcont),
                                            WrapOrange(apriorClass))); }



PClassifier TLearner_Python::operator()(PExampleGenerator eg, const int &weight)
{ if (!eg)
    raiseError("invalid example generator");

  PyObject *res = callCallback((PyObject *)myWrapper, Py_BuildValue("(Ni)", WrapOrange(POrange(eg)), weight));

  if (!PyOrClassifier_Check(res)) 
    raiseError("__call__ is expected to return something derived from Classifier");

  PClassifier clsf = PyOrange_AsClassifier(res);
  Py_DECREF(res);
  return clsf;
}


#include "vectortemplates.hpp"
#include "converts.hpp"
PFloatList TLogisticFitter_Python::operator()(PExampleGenerator eg, const int &weightID, PFloatList &beta_se, float &likelihood, int &status, PVariable &attribute)
{
  if (!eg)
    raiseError("invalid example generator");

  PyObject *res = callCallback((PyObject *)myWrapper, Py_BuildValue("(Ni)", WrapOrange(POrange(eg)), weightID));

  if (!PyTuple_Check(res) || (PyTuple_Size(res)<2) || !PyInt_Check(PyTuple_GET_ITEM(res, 0)))
    raiseError("invalid result from __call__");

  status = (int)PyInt_AsLong(PyTuple_GET_ITEM(res, 0));
  if (status <= TLogisticFitter::Divergence) {
    if (PyTuple_Size(res) != 4)
      raiseError("invalid result from __call__");

    PFloatList beta = ListOfUnwrappedMethods<PFloatList, TFloatList, float>::P_FromArguments(PyTuple_GET_ITEM(res, 1));
    beta_se = ListOfUnwrappedMethods<PFloatList, TFloatList, float>::P_FromArguments(PyTuple_GET_ITEM(res, 2));
    Py_DECREF(res);
    if (!beta || !beta_se || !PyNumber_ToFloat(PyTuple_GET_ITEM(res, 3), likelihood))
      throw pyexception();

    attribute = PVariable();
    return beta;
  }
  else {
    if (PyTuple_Size(res) != 2)
      raiseError("invalid result from __call__");

    attribute = PyOrange_AsVariable(PyTuple_GET_ITEM(res, 1));
    beta_se = PFloatList();
    return PFloatList();
  }
}


TValue TClassifier_Python::operator ()(const TExample &ex)
{ PyObject *result = callCallback((PyObject *)myWrapper, Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 0));

  if (result==Py_None) {
    Py_DECREF(result);
    return classVar ? classVar->DK() : TValue(TValue::INTVAR, valueDK);
  }
  
  TValue value;
  if (!convertFromPython(result, value, classVar)) {
    Py_DECREF(result);
    raiseError("invalid result from __call__");
  }

  Py_DECREF(result);
  return value;
}


PDistribution TClassifier_Python::classDistribution(const TExample &ex)
{ PyObject *result=callCallback((PyObject *)myWrapper, Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 1));

  if (result==Py_None) {
    Py_DECREF(result);
    return PDistribution(classVar);
  }

  if (PyOrDistribution_Check(result)) {
    PDistribution dist = PyOrange_AsDistribution(result);
    Py_DECREF(result);
    return dist;
  }

  raiseError("invalid result from __call__");
  return PDistribution();
}


void TClassifier_Python::predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &dist)
{ PyObject *result=callCallback((PyObject *)myWrapper, Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 2));

  if (result==Py_None) {
    Py_DECREF(result);
    if (classVar) {
      val = classVar->DK();
      dist = PDistribution(classVar);
    }
    else {
      val = TValue(TValue::INTVAR, valueDK);
      dist = PDistribution();
    }
    return;
  }

  PyObject *obj1;
  if (   !PyArg_ParseTuple(result, "OO&", &obj1, cc_Distribution, &dist)
      || !convertFromPython(obj1, val, classVar)) {
    Py_DECREF(result);
    raiseError("invalid result from __call__");
  }

  Py_DECREF(result);
}



PStringList PStringList_FromArguments(PyObject *arg);

PClassifier TTreeSplitConstructor_Python::operator()(
   PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,
   PExampleGenerator gen, const int &weightID, PDomainContingency dcont, PDistribution apriorClass, const vector<bool> &candidates, PClassifier nodeClassifier)

{ if (!gen)
    raiseError("invalid example generator");
  
  PyObject *pycandidates = PYNULL;
  if (candidates.size()) {
    pycandidates = PyList_New(candidates.size());
    int it = 0;
    const_ITERATE(vector<bool>, ci, candidates)
      PyList_SetItem(pycandidates, it++, PyInt_FromLong(*ci ? 1 : 0));
  }
  else {
    int as = gen->domain->attributes->size();
    pycandidates = PyList_New(as);
    while (as--)
      PyList_SetItem(pycandidates, as, PyInt_FromLong(1));
  }

  PyObject *res=callCallback((PyObject *)myWrapper,
     Py_BuildValue("(NiNNNN)", WrapOrange(gen), weightID, WrapOrange(dcont), WrapOrange(apriorClass), pycandidates, WrapOrange(nodeClassifier))
  );

  if (res==Py_None) {
    Py_DECREF(res);
    return PClassifier();
  }

  PClassifier classifier;
  PyObject *pydesc = NULL;
  spentAttribute = -1;
  quality = 0.0;
  subsetSizes = PDistribution();
  if (!PyArg_ParseTuple(res, "O&|OO&fi", ccn_Classifier, &classifier, &pydesc, ccn_DiscDistribution, &subsetSizes, &quality, &spentAttribute)) {
    Py_DECREF(res);
    throw pyexception();
  }
  Py_DECREF(res);
  descriptions = pydesc ? (PyOrStringList_Check(pydesc) ? PyOrange_AsStringList(pydesc) : PStringList_FromArguments(pydesc))
                        : PStringList();
  return classifier;
}



PExampleGeneratorList PExampleGeneratorList_FromArguments(PyObject *args);

PExampleGeneratorList TTreeExampleSplitter_Python::operator()(PTreeNode node, PExampleGenerator gen, const int &weightID, vector<int> &newWeights)
{ if (!gen)
    raiseError("invalid example generator");
  
  PyObject *res=callCallback((PyObject *)myWrapper, Py_BuildValue("(NNi)", WrapOrange(node), WrapOrange(gen), weightID));

  if (res == Py_None) {
    Py_DECREF(res);
    return PExampleGeneratorList();
  }

  PyObject *pygen;
  PyObject *pyweights = NULL;
  if (!PyArg_ParseTuple(res, "O|O", &pygen, &pyweights))
    raiseError("invalid result from __call__ (a list of list of examples and, optionally a list of weight ID's expected)");

  PExampleGeneratorList eglist = PExampleGeneratorList_FromArguments(pygen);
  if (!eglist)
    raiseError("invalid result from __call__ (a list of list of examples and, optionally a list of weight ID's expected)");

  if (pyweights && (pyweights!=Py_None)) {
    if (!PyList_Check(pyweights) || (PyList_Size(pyweights)!=(int)node->branches->size()))
      raiseError("invalid result from __call__ (length of weight list should equal the number of branches)");
      
    for (int sz = 0, se = PyList_Size(pyweights); sz<se; sz++) {
      PyObject *li = PyList_GetItem(pyweights, sz);
      if (!PyInt_Check(li))
        raiseError("invalid weight list (int's expected).");
      newWeights.push_back(int(PyInt_AsLong(li)));
    }  
  }
  else
    newWeights.clear();

  Py_DECREF(res);

  return eglist;
}




bool TTreeStopCriteria_Python::operator()(PExampleGenerator gen, const int &weightID, PDomainContingency dcont)
{ if (!gen)
    raiseError("invalid example generator");

  PyObject *result=callCallback((PyObject *)myWrapper, Py_BuildValue("(NiN)", WrapOrange(gen), weightID, WrapOrange(dcont)));

  bool res = bool(PyObject_IsTrue(result)!=0);
  Py_DECREF(result);
  return res;
}



int pt_DiscDistribution(PyObject *args, void *dist);

PTreeNode TTreeDescender_Python::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ PyObject *result=callCallback((PyObject *)myWrapper, Py_BuildValue("(NN)", WrapOrange(node), Example_FromExampleCopyRef(ex)));

  if (result == Py_None) {
    Py_DECREF(result);
    distr->clear();
    return PTreeNode();
  }
      
  PTreeNode newnode;
  distr = PDiscDistribution();
  if (PyOrTreeNode_Check(result))
    newnode = PyOrange_AsTreeNode(result);
  else if (!PyArg_ParseTuple(result, "O&|O&", cc_TreeNode, &newnode, pt_DiscDistribution, &distr)) {
    Py_DECREF(result);
    raiseError("invalid result from __call__");
  }

  Py_DECREF(result);
  return newnode;
}


/*
PIM TConstructIM_Python::operator()(PExampleGenerator gen, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID)
{ if (!gen)
    raiseError("invalid example generator");

  PyObject *boundList = PyList_New(0);
  PyObject *freeList = PyList_New(0);
  vector<bool>::const_iterator bi(bound.begin()), fi(free.begin());

  const_PITERATE(TVarList, vi, gen->domain->attributes) {
    PyObject *m = WrapOrange(*vi);
    if (*(bi++))
      PyList_Append(boundList, m);
    if (*(fi++))
      PyList_Append(freeList, m);
    Py_DECREF(m);
  }

  PyObject *res = callCallback((PyObject *)myWrapper, Py_BuildValue("(NNNi)", WrapOrange(gen), boundList, freeList, weightID));

  if (!PyOrIM_Check(res))
    raiseError("invalid result from __call__");

  PIM im = PyOrange_AsIM(res);
  Py_DECREF(res);
  return im;
}
*/
