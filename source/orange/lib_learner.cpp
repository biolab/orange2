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

This file includes constructors and specialized methods for ML* object, defined in project Learners

*********************************/

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include <string>

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "nearest.hpp"
#include "estimateprob.hpp"
#include "induce.hpp"
#include "cost.hpp"
#include "measures.hpp"
#include "distance.hpp"
#include "contingency.hpp"

#include "callback.hpp"

#include "cls_orange.hpp"
#include "cls_value.hpp"
#include "cls_example.hpp"
#include "lib_kernel.hpp"

#include "converts.hpp"

#include "vectortemplates.hpp"

#include "externs.px"

/* ************ MAJORITY AND COST ************ */

#include "majority.hpp"
C_CALL(MajorityLearner, Learner, "([examples] [, weight=, estimate=]) -/-> Classifier")
C_CALL(CostLearner, Learner, "([examples] [, weight=, estimate=, costs=]) -/-> Classifier")

//#include "linreg.hpp"
PYXTRACT_IGNORE C_CALL(LinRegLearner, Learner, "([examples] [, weight=]) -/-> Classifier")
PYXTRACT_IGNORE C_NAMED(LinRegClassifier, ClassifierFD, "([classifier=, costs=])")

#include "costwrapper.hpp"
C_CALL(CostWrapperLearner, Learner, "([examples] [, weight=, costs=]) -/-> Classifier")
C_NAMED(CostWrapperClassifier, Classifier, "([classifier=, costs=])")


/************* ASSOCIATION RULES ************/

#include "assoc.hpp"
C_CALL(AssociationLearner, Learner, "([examples] [, weight=, conf=, supp=, voteWeight=]) -/-> Classifier")
C_NAMED(AssociationClassifier, ClassifierFD, "([rules=, voteWeight=])")
C_CALL3(AssociationRulesInducer, AssociationRulesInducer, Orange, "([examples[, weightID]], confidence=, support=]) -/-> AssociationRules")
C_CALL3(AssociationRulesSparseInducer, AssociationRulesSparseInducer, Orange, "([examples[, weightID]], confidence=, support=]) -/-> AssociationRules")


bool operator < (const TAssociationRule &, const TAssociationRule &) { return false; }
bool operator > (const TAssociationRule &, const TAssociationRule &) { return false; }

PyObject *AssociationRulesInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> AssociationRules")
{
  PyTRY
    SETATTRIBUTES

    PExampleGenerator egen;
    int weightID = 0;
    if (!PyArg_ParseTuple(args, "O&|i:AssociationRulesInducer.call", pt_ExampleGenerator, &egen, &weightID))
      return PYNULL;

    return WrapOrange(SELF_AS(TAssociationRulesInducer)(egen, weightID));
  PyCATCH
}


PyObject *AssociationRulesSparseInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> AssociationRules")
{
  PyTRY
    SETATTRIBUTES

    PExampleGenerator egen;
    int weightID = 0;
    if (!PyArg_ParseTuple(args, "O&|i:AssociationRulesInducer.call", pt_ExampleGenerator, &egen, &weightID))
      return PYNULL;

    return WrapOrange(SELF_AS(TAssociationRulesSparseInducer)(egen, weightID));
  PyCATCH
}


bool convertFromPython(PyObject *, PAssociationRule &);

PyObject *AssociationRule_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange, "(left, right, support, confidence)")
{ PyTRY
    PAssociationRule rule;
    return  convertFromPython(args, rule) ? WrapOrange(rule) : PYNULL;
  PyCATCH
}


PyObject *AssociationRule_appliesLeft(PyObject *self, PyObject *arg, PyObject *) PYARGS(METH_O, "(example) -> bool")
{ PyTRY
    if (!PyOrExample_Check(arg))
      PYERROR(PyExc_TypeError, "attribute error (example expected)", PYNULL);
    
    CAST_TO(TAssociationRule, rule)
    return PyInt_FromLong(rule->appliesLeft(PyExample_AS_ExampleReference(arg)) ? 1 : 0);
  PyCATCH
}


PyObject *AssociationRule_appliesRight(PyObject *self, PyObject *arg, PyObject *) PYARGS(METH_O, "(example) -> bool")
{ PyTRY
    if (!PyOrExample_Check(arg))
      PYERROR(PyExc_TypeError, "attribute error (example expected)", PYNULL);
    
    CAST_TO(TAssociationRule, rule)
    return PyInt_FromLong(rule->appliesRight(PyExample_AS_ExampleReference(arg)) ? 1 : 0);
  PyCATCH
}


PyObject *AssociationRule_appliesBoth(PyObject *self, PyObject *arg, PyObject *) PYARGS(METH_O, "(example) -> bool")
{ PyTRY
    if (!PyOrExample_Check(arg))
      PYERROR(PyExc_TypeError, "attribute error (example expected)", PYNULL);
    
    CAST_TO(TAssociationRule, rule)
    return PyInt_FromLong(rule->appliesBoth(PyExample_AS_ExampleReference(arg)) ? 1 : 0);
  PyCATCH
}


PyObject *AssociationRule_native(PyObject *self)
{ PyTRY
    CAST_TO(TAssociationRule, rule)
    return Py_BuildValue("NNff", Example_FromWrappedExample(rule->left), Example_FromWrappedExample(rule->right), rule->support, rule->confidence);
  PyCATCH
}

bool convertFromPython(PyObject *obj, PAssociationRule &rule)
{ if (PyOrOrange_Check(obj))
    if (!PyOrange_AS_Orange(obj)) {
      rule = PAssociationRule();
      return true;
    }
    else if (PyOrAssociationRule_Check(obj)) {
      rule = PyOrange_AsAssociationRule(obj);
      return true;
    }

  TExample *le, *re;

  switch (PyTuple_Size(obj)) {
    case 6:
      float nAppliesLeft, nAppliesRight, nAppliesBoth, nExamples;
      if (PyArg_ParseTuple(obj, "O&O&ffff", ptr_Example, &le, ptr_Example, &re, &nAppliesLeft, &nAppliesRight, &nAppliesBoth, &nExamples)) {
        PExample nle = mlnew TExample(*le);
        PExample nre = mlnew TExample(*re);
        rule = mlnew TAssociationRule(nle, nre, nAppliesLeft, nAppliesRight, nAppliesBoth, nExamples);
        return true;
      }
      else
        break;

    case 4:
      float support, confidence;
      if (PyArg_ParseTuple(obj, "O&O&ff", ptr_Example, &le, ptr_Example, &re, &support, &confidence)) {
        PExample nle = mlnew TExample(*le);
        PExample nre = mlnew TExample(*re);
        rule = mlnew TAssociationRule(nle, nre);
        rule->support = support;
        rule->confidence = confidence;
        return true;
      }
      else
        break;

    case 1: 
      if (PyArg_ParseTuple(obj, "O&:convertFromPython(AssociationRule)", cc_AssociationRule, &rule))
        return true;
      else
        break;
  }
    
  PYERROR(PyExc_TypeError, "invalid arguments", false);
}


string side2string(PExample ex)
{ string res;

  if (ex->domain->variables->empty())
    ITERATE(TMetaValues, mi, ex->meta) {
      if (res.length())
        res += " ";
      res += ex->domain->getMetaVar((*mi).first)->name;
    }

  else {
    string val;

    TVarList::const_iterator vi(ex->domain->variables->begin());
    for(TExample::const_iterator ei(ex->begin()), ee(ex->end()); ei!=ee; ei++, vi++)
      if (!(*ei).isSpecial()) {
        if (res.length())
          res += " ";
        (*vi)->val2str(*ei, val);
        res += (*vi)->name + "=" + val;
      }
  }

  return res;
}

PyObject *AssociationRule_str(TPyOrange *self)
{ CAST_TO(TAssociationRule, rule);
  return PyString_FromFormat("%s -> %s", side2string(rule->left).c_str(), side2string(rule->right).c_str());
}


PyObject *AssociationRule_repr(TPyOrange *self)
{ CAST_TO(TAssociationRule, rule);
  return PyString_FromFormat("%s -> %s", side2string(rule->left).c_str(), side2string(rule->right).c_str());
}


PAssociationRules PAssociationRules_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::P_FromArguments(arg); }
PyObject *AssociationRules_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_FromArguments(type, arg); }
PyObject *AssociationRules_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of AssociationRule>)") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_new(type, arg, kwds); }
PyObject *AssociationRules_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_getitem(self, index); }
int       AssociationRules_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_setitem(self, index, item); }
PyObject *AssociationRules_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_getslice(self, start, stop); }
int       AssociationRules_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_setslice(self, start, stop, item); }
int       AssociationRules_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_len(self); }
PyObject *AssociationRules_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_richcmp(self, object, op); }
PyObject *AssociationRules_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_concat(self, obj); }
PyObject *AssociationRules_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_repeat(self, times); }
PyObject *AssociationRules_str(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_str(self); }
PyObject *AssociationRules_repr(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_str(self); }
int       AssociationRules_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_contains(self, obj); }
PyObject *AssociationRules_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(AssociationRule) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_append(self, item); }
PyObject *AssociationRules_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> int") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_count(self, obj); }
PyObject *AssociationRules_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> AssociationRules") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_filter(self, args); }
PyObject *AssociationRules_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> int") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_index(self, obj); }
PyObject *AssociationRules_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_insert(self, args); }
PyObject *AssociationRules_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_native(self); }
PyObject *AssociationRules_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> AssociationRule") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_pop(self, args); }
PyObject *AssociationRules_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_remove(self, obj); }
PyObject *AssociationRules_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_reverse(self); }
PyObject *AssociationRules_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, (PyTypeObject *)&PyOrAssociationRule_Type>::_sort(self, args); }

/************* CLASSIFICATION TREES ************/

#include "tdidt.hpp"
#include "tdidt_split.hpp"
#include "tdidt_stop.hpp"
#include "callback.hpp"

C_NAMED(MapIntValue, TransformValue, "([mapping=])")
 
C_CALL(TreeLearner, Learner, "([examples] [, weight=, split=, stop=, nodeLearner=, lookDownOnUnknown=]) -/-> Classifier")

C_NAMED(TreeNode, Orange, "([lookDownOnUnknown=, chooseBranch=, nodeClassifier=, branches=, contingency=])")
C_NAMED(TreeClassifier, ClassifierFD, "([domain=, tree=, descender=])")

C_NAMED(TreeStopCriteria_common, TreeStopCriteria, "([maxMajority=, minExamples=])")
HIDDEN(TreeStopCriteria_Python, TreeStopCriteria)

C_CALL(TreeSplitConstructor_Combined, TreeSplitConstructor, "([examples, [weight, domainContingency, apriorClass, candidates] [discreteTreeSplitConstructor=, continuousTreeSplitConstructor=]) -/-> (Classifier, descriptions, sizes, quality)")

BASED_ON(TreeSplitConstructor_Measure, TreeSplitConstructor)
C_CALL(TreeSplitConstructor_Attribute, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
C_CALL(TreeSplitConstructor_ExhaustiveBinary, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
C_CALL(TreeSplitConstructor_Threshold, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
PYXTRACT_IGNORE C_CALL(TreeSplitConstructor_LR, TreeSplitConstructor, "([minSubset=])")

BASED_ON(TreeExampleSplitter, Orange)

C_CALL(TreeExampleSplitter_IgnoreUnknowns, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToCommon, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToAll, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToRandom, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToBranch, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")

C_CALL(TreeExampleSplitter_UnknownsAsBranchSizes, TreeExampleSplitter, "([branchIndex, node, examples[, weight]]) -/-> (ExampleGenerator, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsAsSelector, TreeExampleSplitter, "([branchIndex, node, examples[, weight]]) -/-> (ExampleGenerator, [list of weight ID's])")

C_CALL(TreeDescender_UnknownToBranch, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownToCommonBranch, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownToCommonSelector, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownMergeAsBranchSizes, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownMergeAsSelector, TreeDescender, "(node, example) -/-> (node, {distribution | None})")

BASED_ON(TreePruner, Orange)
C_CALL (TreePruner_SameMajority, TreePruner, "([tree]) -/-> tree")
C_CALL (TreePruner_m, TreePruner, "([tree]) -/-> tree")


PyObject *TreeNode_treesize(PyObject *self, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> int")
{ PyTRY
    return PyInt_FromLong(PyOrange_AsTreeNode(self)->treeSize());
  PyCATCH
}


PyObject *TreeNode_removestoredinfo(PyObject *self, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> None")
{ PyTRY
    PyOrange_AsTreeNode(self)->removeStoredInfo();
    RETURN_NONE;
  PyCATCH
}


PyObject *TreeStopCriteria_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "()")
{ if (type == (PyTypeObject *)&PyOrTreeStopCriteria_Type) {
      PyObject *name=NULL;
      if (args && !PyArg_ParseTuple(args, "|O", &name))
        PYERROR(PyExc_TypeError, "TreeStopCriteria: invalid arguments - name or callback function expected", PYNULL);

      if (!args || !name || name && PyString_Check(name)) {
          PyObject *self = WrapOrange(mlnew TTreeStopCriteria());
          if (name)
            PyObject_SetAttrString(self, "name", name);
          return self;
      }
      // (args && name && !PyStringString_Check(name)

      return setCallbackFunction(WrapOrange(mlnew TTreeStopCriteria_Python()), args);
  }

  return WrapNewOrange(mlnew TTreeStopCriteria_Python(), type);
}



PyObject *TreeStopCriteria_lowcall(PyObject *self, PyObject *args, PyObject *keywords, bool allowPython)
{ 
  static TTreeStopCriteria _cbdefaultStop;
  PyTRY
    SETATTRIBUTES
    CAST_TO(TTreeStopCriteria, stop);
    if (!stop)
      PYERROR(PyExc_SystemError, "attribute error", PYNULL);

    PExampleGenerator egen;
    PDomainContingency dcont;
    int weight = 0;
    if (!PyArg_ParseTuple(args, "O&|iO&:TreeStopCriteria.__call__", pt_ExampleGenerator, &egen, &weight, pt_DomainContingency, &dcont))
      return PYNULL;

    bool res;

    if (allowPython || (stop->classDescription() != &TTreeStopCriteria_Python::st_classDescription))
      res = (*stop)(egen, weight, dcont);
    else
      res = _cbdefaultStop(egen, weight, dcont);

    return PyInt_FromLong(res ? 1 : 0);
  PyCATCH
}


PyObject *TreeStopCriteria_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([examples, [weight, domainContingency]) -> bool")
{ return TreeStopCriteria_lowcall(self, args, keywords, false); }


PyObject *TreeStopCriteria_Python_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([examples, [weight, domainContingency, apriorClass, candidates]) -/-> (Classifier, descriptions, sizes, quality)")
{ return TreeStopCriteria_lowcall(self, args, keywords, false); }



PyObject *TreeSplitConstructor_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeSplitConstructor_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TTreeSplitConstructor_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeSplitConstructor_Python(), type);
}


PyObject *TreeSplitConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weight, contingency, apriori class distribution, candidates, nodeClassifier]) -> (Classifier, descriptions, sizes, quality)")
{ PyTRY

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeSplitConstructor_Type) {
      PyErr_Format(PyExc_SystemError, "TreeSplitConstructor.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES
    PExampleGenerator gen;
    int weightID = 0;
    PDomainContingency dcont;
    PDistribution apriori;
    PyObject *pycandidates = PYNULL;
    PClassifier nodeClassifier;

    if (!PyArg_ParseTuple(args, "O&|iO&O&O&O:TreeSplitConstructor.call", pt_ExampleGenerator, &gen, &weightID, ccn_DomainContingency, &dcont, ccn_Distribution, &apriori, &pycandidates, ccn_Classifier, &nodeClassifier))
      return PYNULL;

    vector<bool> candidates;
    if (pycandidates) {
      PyObject *iterator = PyObject_GetIter(pycandidates);
      if (!iterator) {
        Py_DECREF(pycandidates);
        PYERROR(PyExc_SystemError, "TreeSplitConstructor.call: cannot iterate through candidates; a list exected", PYNULL);
      }
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
        candidates.push_back(PyObject_IsTrue(item) != 0);
        Py_DECREF(item);
      }
      
      Py_DECREF(iterator);
      Py_DECREF(pycandidates);

      if (PyErr_Occurred())
        return PYNULL;
    }
    
    PClassifier branchSelector;
    PStringList descriptions;
    PDiscDistribution subsetSizes;
    float quality;
    int spentAttribute;

    branchSelector = SELF_AS(TTreeSplitConstructor)(descriptions, subsetSizes, quality, spentAttribute,
                                                    gen, weightID, dcont, apriori, candidates, nodeClassifier);

    return Py_BuildValue("NNNfi", WrapOrange(branchSelector), WrapOrange(descriptions), WrapOrange(subsetSizes), quality, spentAttribute);
  PyCATCH
}


PyObject *TreeExampleSplitter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeExampleSplitter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TTreeExampleSplitter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeExampleSplitter_Python(), type);
}


PyObject *TreeExampleSplitter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(node, examples[, weight]) -/-> (ExampleGeneratorList, list of weight ID's")
{ PyTRY
    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeExampleSplitter_Type) {
      PyErr_Format(PyExc_SystemError, "TreeExampleSplitter.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES
    PTreeNode node;
    PExampleGenerator gen;
    int weightID = 0;

    if (!PyArg_ParseTuple(args, "O&O&|i:TreeExampleSplitter.call", cc_TreeNode, &node, pt_ExampleGenerator, &gen, &weightID))
      return PYNULL;

    vector<int> newWeights;
    PExampleGeneratorList egl = SELF_AS(TTreeExampleSplitter)(node, gen, weightID, newWeights);

    if (newWeights.size()) {
      PyObject *pyweights = PyList_New(newWeights.size());
      int i = 0;
      ITERATE(vector<int>, li, newWeights)
        PyList_SetItem(pyweights, i++, PyInt_FromLong(*li));

      return Py_BuildValue("NN", WrapOrange(egl), pyweights);
    }

    else {
      return Py_BuildValue("NO", WrapOrange(egl), Py_None);
    }

  PyCATCH 
}



PyObject *TreeDescender_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeDescender_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TMeasureAttribute_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeDescender_Python(), type);
}


PyObject *TreeDescender_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(node, example) -/-> (node, {distribution | None})")
{ PyTRY
    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeDescender_Type) {
      PyErr_Format(PyExc_SystemError, "TreeDescender.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    SETATTRIBUTES

    PTreeNode onode;
    TExample *example;
    if (!PyArg_ParseTuple(args, "O&O&", cc_TreeNode, &onode, ptr_Example, &example))
      PYERROR(PyExc_TypeError, "invalid parameters", PYNULL);

    PDiscDistribution distr;
    PTreeNode node = SELF_AS(TTreeDescender)(onode, *example, distr);
    return Py_BuildValue("NN", WrapOrange(node), WrapOrange(distr));
  PyCATCH
}


PyObject *TreePruner_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(tree) -> tree")
{ 
  PyTRY
    SETATTRIBUTES
    PyObject *obj;
    PTreeNode node;
    PTreeClassifier classifier;
    if (PyArg_ParseTuple(args, "O", &obj))
      if (PyOrTreeClassifier_Check(obj)) {
        classifier = PyOrange_AsClassifier(obj);
        node = classifier->tree;
      }
      else if (PyOrTreeNode_Check(obj))
        node = PyOrange_AsTreeNode(obj);

    if (!node)
      PYERROR(PyExc_TypeError, "invalid arguments (a classifier expected)", PYNULL);

    PTreeNode newRoot = SELF_AS(TTreePruner)(node);

    if (classifier) {
      PTreeClassifier newClassifier = CLONE(TTreeClassifier, classifier);
      newClassifier->tree = newRoot;
      return WrapOrange(newClassifier);
    }
    else
      return WrapOrange(newRoot);
  PyCATCH
}


PyObject *TreeClassifier_treesize(PyObject *self, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> size")
{ PyTRY
    CAST_TO(TTreeClassifier, me);
    if (!me->tree)
      PYERROR(PyExc_SystemError, "TreeClassifier: 'tree' not defined", PYNULL);

    return PyInt_FromLong(long(me->tree->treeSize()));
  PyCATCH
}


PTreeNodeList PTreeNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::P_FromArguments(arg); }
PyObject *TreeNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_FromArguments(type, arg); }
PyObject *TreeNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of TreeNode>)") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_new(type, arg, kwds); }
PyObject *TreeNodeList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_getitem(self, index); }
int       TreeNodeList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_setitem(self, index, item); }
PyObject *TreeNodeList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_getslice(self, start, stop); }
int       TreeNodeList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_setslice(self, start, stop, item); }
int       TreeNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_len(self); }
PyObject *TreeNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_richcmp(self, object, op); }
PyObject *TreeNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_concat(self, obj); }
PyObject *TreeNodeList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_repeat(self, times); }
PyObject *TreeNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_str(self); }
PyObject *TreeNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_str(self); }
int       TreeNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_contains(self, obj); }
PyObject *TreeNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(TreeNode) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_append(self, item); }
PyObject *TreeNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> int") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_count(self, obj); }
PyObject *TreeNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> TreeNodeList") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_filter(self, args); }
PyObject *TreeNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> int") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_index(self, obj); }
PyObject *TreeNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_insert(self, args); }
PyObject *TreeNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_native(self); }
PyObject *TreeNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> TreeNode") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_pop(self, args); }
PyObject *TreeNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_remove(self, obj); }
PyObject *TreeNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_reverse(self); }
PyObject *TreeNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, (PyTypeObject *)&PyOrTreeNode_Type>::_sort(self, args); }

 
/************* C45 ************/

#include "c4.5.hpp"

C_CALL(C45Learner, Learner, "([examples] [, weight=, gainRatio=, subset=, batch=, probThresh=, minObjs=, window=, increment=, cf=, trials=]) -/-> Classifier")
BASED_ON(C45Classifier, Classifier)

PyObject *C45Learner_commandline(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(line) -> None")
{ PyTRY
    char *line;
    if (!PyArg_ParseTuple(args, "s", &line))
      PYERROR(PyExc_TypeError, "C45Learner.commandline: string argument expected", NULL);

    SELF_AS(TC45Learner).parseCommandLine(string(line));
    RETURN_NONE;
  PyCATCH
}

C_NAMED(C45TreeNode, Orange, "")

PYCLASSCONSTANT_INT(C45TreeNode, Leaf, TC45TreeNode::Leaf)
PYCLASSCONSTANT_INT(C45TreeNode, Branch, TC45TreeNode::Branch)
PYCLASSCONSTANT_INT(C45TreeNode, Cut, TC45TreeNode::Cut)
PYCLASSCONSTANT_INT(C45TreeNode, Subset, TC45TreeNode::Subset)


PC45TreeNodeList PC45TreeNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::P_FromArguments(arg); }
PyObject *C45TreeNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_FromArguments(type, arg); }
PyObject *C45TreeNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of C45TreeNode>)") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_new(type, arg, kwds); }
PyObject *C45TreeNodeList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_getitem(self, index); }
int       C45TreeNodeList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_setitem(self, index, item); }
PyObject *C45TreeNodeList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_getslice(self, start, stop); }
int       C45TreeNodeList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_setslice(self, start, stop, item); }
int       C45TreeNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_len(self); }
PyObject *C45TreeNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_richcmp(self, object, op); }
PyObject *C45TreeNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_concat(self, obj); }
PyObject *C45TreeNodeList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_repeat(self, times); }
PyObject *C45TreeNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_str(self); }
PyObject *C45TreeNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_str(self); }
int       C45TreeNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_contains(self, obj); }
PyObject *C45TreeNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(C45TreeNode) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_append(self, item); }
PyObject *C45TreeNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> int") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_count(self, obj); }
PyObject *C45TreeNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> C45TreeNodeList") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_filter(self, args); }
PyObject *C45TreeNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> int") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_index(self, obj); }
PyObject *C45TreeNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_insert(self, args); }
PyObject *C45TreeNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_native(self); }
PyObject *C45TreeNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> C45TreeNode") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_pop(self, args); }
PyObject *C45TreeNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_remove(self, obj); }
PyObject *C45TreeNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_reverse(self); }
PyObject *C45TreeNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, (PyTypeObject *)&PyOrC45TreeNode_Type>::_sort(self, args); }

/************* kNN ************/

#include "knn.hpp"
C_CALL(kNNLearner, Learner, "([examples] [, weight=, k=] -/-> Classifier")
C_NAMED(kNNClassifier, ClassifierFD, "([k=, weightID=, findNearest=])")

/************* Logistic Regression ************/

#include "logistic.hpp"
C_CALL(LogisticLearner, Learner, "([examples[, weight=]]) -/-> Classifier")
C_NAMED(LogisticClassifier, ClassifierFD, "([probabilities=])")


PyObject *LogisticFitter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrLogisticFitter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TLogisticFitter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TLogisticFitter_Python(), type);
}

C_CALL(LogisticFitter_Cholesky, LogisticFitter, "([example[, weightID]]) -/-> (status, beta, beta_se, likelihood) | (status, attribute)")

PYCLASSCONSTANT_INT(LogisticFitter, OK, TLogisticFitter::OK)
PYCLASSCONSTANT_INT(LogisticFitter, Infinity, TLogisticFitter::Infinity)
PYCLASSCONSTANT_INT(LogisticFitter, Divergence, TLogisticFitter::Divergence)
PYCLASSCONSTANT_INT(LogisticFitter, Constant, TLogisticFitter::Constant)
PYCLASSCONSTANT_INT(LogisticFitter, Singularity, TLogisticFitter::Singularity)

PyObject *LogisticLearner_fitModel(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples[, weight])")
{
  PyTRY
      PExampleGenerator egen;
      PyObject *pyweight = NULL;
      if (!PyArg_ParseTuple(args, "O&|O:LogisticLearner", pt_ExampleGenerator, &egen, &pyweight))
	    return PYNULL;

      int weight;

      if (!pyweight || (pyweight == Py_None))
        weight = 0;
	  else if (PyInt_Check(pyweight))
		weight = (int)PyInt_AsLong(pyweight);
	  else {
		PVariable var = varFromArg_byDomain(pyweight, egen->domain);
		if (!var) 
		  PYERROR(PyExc_TypeError, "invalid or unknown weight attribute", PYNULL);

		weight = egen->domain->getVarNum(var);
	  }

	  CAST_TO(TLogisticLearner, loglearn)

	  int error;
	  PVariable variable;
	  PClassifier classifier;

	  classifier = loglearn->fitModel(egen, weight, error, variable);
	  if (error <= TLogisticFitter::Divergence)
		  return Py_BuildValue("N", WrapOrange(classifier));
	  else 
		  return Py_BuildValue("N", WrapOrange(variable));
  PyCATCH
}


PyObject *LogisticFitter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -/-> (status, beta, beta_se, likelihood) | (status, attribute)")
{
  PyTRY
    SETATTRIBUTES

    PExampleGenerator egen;
    PyObject *pyweight = NULL;
    if (!PyArg_ParseTuple(args, "O&|O:LogisticFitter.__call__", pt_ExampleGenerator, &egen, &pyweight))
      return PYNULL;

    int weight = 0;

    if (!pyweight || (pyweight == Py_None))
      weight = 0;
    else if (PyInt_Check(pyweight))
	    weight = (int)PyInt_AsLong(pyweight);
    else {
      PVariable var = varFromArg_byDomain(pyweight, egen->domain);
    if (!var) 
      PYERROR(PyExc_TypeError, "invalid or unknown weight attribute", PYNULL);
      weight = egen->domain->getVarNum(var);
    }

    CAST_TO(TLogisticFitter, fitter)

    PFloatList beta, beta_se;
    float likelihood;
    int error;
    PVariable attribute;
    
    beta = (*fitter)(egen, weight, beta_se, likelihood, error, attribute);

    if (error <= TLogisticFitter::Divergence)
      return Py_BuildValue("iNNf", error, WrapOrange(beta), WrapOrange(beta_se), likelihood);
    else
      return Py_BuildValue("iN", error, WrapOrange(attribute));

  PyCATCH
}

/************* SVM ************/

#include "svm.hpp"
C_CALL(SVMLearner, Learner, "([examples] -/-> Classifier)")
BASED_ON(SVMClassifier, Classifier)


/************* BAYES ************/

#include "bayes.hpp"
C_CALL(BayesLearner, Learner, "([examples], [weight=, estimate=] -/-> Classifier")
C_NAMED(BayesClassifier, ClassifierFD, "([probabilities=])")

PyObject *BayesClassifier_p(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(class, example) -> float")
{ PyTRY
    CAST_TO(TBayesClassifier, me);

    PyObject *pyvalue;
    TValue value;
    TExample *ex;
    if (   !PyArg_ParseTuple(args, "OO&:BayesClassifier.p", &pyvalue, ptr_Example, &ex)
        || !convertFromPython(pyvalue, value, me->domain->classVar))
      return PYNULL;
      
    return PyFloat_FromDouble((double)SELF_AS(TBayesClassifier).p(value, *ex));

  PyCATCH
}

#include "lib_learner.px"

