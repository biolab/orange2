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
#include "slist.hpp"

#include "externs.px"

/* ************ SIMPLE TREE LEARNER ************ */

#include "tdidt_simple.hpp"
C_CALL(SimpleTreeLearner - Orange.classification.tree.SimpleTreeLearner, Learner, "([examples], [maxMajority=, minExamples=, maxDepth=])")
C_NAMED(SimpleTreeClassifier - Orange.classification.tree.SimpleTreeClassifier, Classifier, "()")

/* ************ MAJORITY AND COST ************ */

#include "majority.hpp"
C_CALL(MajorityLearner - Orange.classification.majority.MajorityLearner, Learner, "([examples] [, weight=, estimate=]) -/-> Classifier")
C_CALL(CostLearner - Orange.wrappers.CostLearner, Learner, "([examples] [, weight=, estimate=, costs=]) -/-> Classifier")


#include "costwrapper.hpp"
C_CALL(CostWrapperLearner - Orange.wrappers.CostWrapperLearner, Learner, "([examples] [, weight=, costs=]) -/-> Classifier")
C_NAMED(CostWrapperClassifier - Orange.wrappers.CostWrapperClassifier, Classifier, "([classifier=, costs=])")


/************* ASSOCIATION RULES ************/

#include "assoc.hpp"
C_CALL(AssociationLearner - Orange.classification.rules.AssociationLearner, Learner, "([examples] [, weight=, conf=, supp=, voteWeight=]) -/-> Classifier")
C_NAMED(AssociationClassifier - Orange.classification.rules.AssociationClassifier, ClassifierFD, "([rules=, voteWeight=])")
C_CALL3(AssociationRulesInducer - Orange.associate.AssociationRulesInducer, AssociationRulesInducer, Orange, "([examples[, weightID]], confidence=, support=]) -/-> AssociationRules")
C_CALL3(AssociationRulesSparseInducer - Orange.associate.AssociationRulesSparseInducer, AssociationRulesSparseInducer, Orange, "([examples[, weightID]], confidence=, support=]) -/-> AssociationRules")
C_CALL3(ItemsetsSparseInducer - Orange.associate.ItemsSparseInducer, ItemsetsSparseInducer, Orange, "([examples[, weightID]], support=]) -/-> AssociationRules")

BASED_ON(ItemsetNodeProxy - Orange.associate.ItemsetNodeProxy, Orange)

bool operator < (const TAssociationRule &, const TAssociationRule &) { return false; }
bool operator > (const TAssociationRule &, const TAssociationRule &) { return false; }

PyObject *AssociationRulesInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> AssociationRules")
{
  PyTRY
    NO_KEYWORDS

    int weightID;
    PExampleGenerator egen = exampleGenFromArgs(args, weightID);
    if (!egen)
      return PYNULL;

    return WrapOrange(SELF_AS(TAssociationRulesInducer)(egen, weightID));
  PyCATCH
}

void gatherRules(TItemSetNode *node, vector<pair<int, int> > &itemsSoFar, PyObject *listOfItems, bool storeExamples)
{
  for(; node; node = node->nextAttribute) {
    itemsSoFar.push_back(make_pair(node->attrIndex, (int)0));
    ITERATE(vector<TItemSetValue>, isi, node->values) {
      itemsSoFar.back().second = (*isi).value;

      PyObject *itemset = PyTuple_New(itemsSoFar.size());
      int el = 0;
      vector<pair<int, int> >::const_iterator sfi(itemsSoFar.begin()), sfe(itemsSoFar.end());
      for(; sfi != sfe; sfi++, el++) {
        PyObject *vp = PyTuple_New(2);
        PyTuple_SET_ITEM(vp, 0, PyInt_FromLong((*sfi).first));
        PyTuple_SET_ITEM(vp, 1, PyInt_FromLong((*sfi).second));
        PyTuple_SET_ITEM(itemset, el, vp);
      }

      PyObject *examples;
      if (storeExamples) {
        examples = PyList_New((*isi).examples.size());
        Py_ssize_t ele = 0;
        ITERATE(TExampleSet, ei, (*isi).examples)
          PyList_SetItem(examples, ele++, PyInt_FromLong((*ei).example));
      }
      else {
        examples = Py_None;
        Py_INCREF(Py_None);
      }

      PyObject *rr = PyTuple_New(2);
      PyTuple_SET_ITEM(rr, 0, itemset);
      PyTuple_SET_ITEM(rr, 1, examples);

      PyList_Append(listOfItems, rr);
      Py_DECREF(rr);

      gatherRules((*isi).branch, itemsSoFar, listOfItems, storeExamples);
    }
    itemsSoFar.pop_back();
  }
}

PyObject *AssociationRulesInducer_getItemsets(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(examples[, weightID]) -> list-of-itemsets")
{
  PyTRY
    int weightID;
    PExampleGenerator egen = exampleGenFromArgs(args, weightID);
    if (!egen)
      return PYNULL;

    if (egen->domain->hasContinuousAttributes(true))
      PYERROR(PyExc_TypeError, "cannot induce rules with non-discrete attributes", NULL);

    TItemSetNode *tree = NULL;
    PyObject *listOfItemsets = NULL;
    try {
		int depth, nOfExamples;
		TDiscDistribution classDist;
		CAST_TO(TAssociationRulesInducer, inducer);
		inducer->buildTrees(egen, weightID, tree, depth, nOfExamples, classDist);

		listOfItemsets = PyList_New(0);
		vector<pair<int, int> > itemsSoFar;
		gatherRules(tree, itemsSoFar, listOfItemsets, inducer->storeExamples);
    }
    catch (...) {
    	if (tree)
    		delete tree;
    	throw;
    }

    delete tree;
    return listOfItemsets;
  PyCATCH
}


void gatherRules(TSparseItemsetNode *node, vector<int> &itemsSoFar, PyObject *listOfItems, bool storeExamples)
{
	if (itemsSoFar.size()) {
        PyObject *itemset = PyTuple_New(itemsSoFar.size());
        Py_ssize_t el = 0;
        vector<int>::const_iterator sfi(itemsSoFar.begin()), sfe(itemsSoFar.end());
        for(; sfi != sfe; sfi++, el++)
            PyTuple_SET_ITEM(itemset, el, PyInt_FromLong(*sfi));

		PyObject *examples;
		if (storeExamples) {
		  examples = PyList_New(node->exampleIds.size());
		  Py_ssize_t ele = 0;
		  ITERATE(vector<int>, ei, node->exampleIds)
			PyList_SetItem(examples, ele++, PyInt_FromLong(*ei));
		}
		else {
		  examples = Py_None;
		  Py_INCREF(Py_None);
		}

      PyObject *rr = PyTuple_New(2);
      PyTuple_SET_ITEM(rr, 0, itemset);
      PyTuple_SET_ITEM(rr, 1, examples);

      PyList_Append(listOfItems, rr);
      Py_DECREF(rr);
	}

    itemsSoFar.push_back(0);
    ITERATE(TSparseISubNodes, isi, node->subNode) {
    	itemsSoFar.back() = (*isi).first;
        gatherRules((*isi).second, itemsSoFar, listOfItems, storeExamples);
    }
    itemsSoFar.pop_back();
}


PyObject *AssociationRulesSparseInducer_getItemsets(PyObject *self, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(examples[, weightID]) -> list-of-itemsets")
{
  PyTRY
    int weightID;
    PExampleGenerator egen = exampleGenFromArgs(args, weightID);
    if (!egen)
      return PYNULL;

    CAST_TO(TAssociationRulesSparseInducer, inducer);
    long i;
    float fullWeight;
    TSparseItemsetTree *tree = NULL;
    PyObject *listOfItemsets = NULL;

    try {
    	  tree = inducer->buildTree(egen, weightID, i, fullWeight);
        listOfItemsets = PyList_New(0);
        vector<int> itemsSoFar;
        gatherRules(tree->root, itemsSoFar, listOfItemsets, inducer->storeExamples);
    }
    catch (...) {
    	if (tree)
    		delete tree;
    	throw;
    }

    delete tree;
    return listOfItemsets;
  PyCATCH
}

PyObject *AssociationRulesSparseInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> AssociationRules")
{
  PyTRY
    NO_KEYWORDS

    int weightID = 0;
    PExampleGenerator egen =  exampleGenFromArgs(args, weightID);
    if (!egen)
      return PYNULL;

    return WrapOrange(SELF_AS(TAssociationRulesSparseInducer)(egen, weightID));
  PyCATCH
}

class TItemsetNodeProxy : public TOrange {
public:
    const TSparseItemsetNode *node;
    PSparseItemsetTree tree;

    TItemsetNodeProxy(const TSparseItemsetNode *n, PSparseItemsetTree t)
    : node(n),
    tree(t)
    {}
};


PyObject *ItemsetsSparseInducer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -> AssociationRules")
{
  PyTRY
    NO_KEYWORDS

    int weightID = 0;
    PExampleGenerator egen =  exampleGenFromArgs(args, weightID);
    if (!egen)
      return PYNULL;

    PSparseItemsetTree tree = SELF_AS(TItemsetsSparseInducer)(egen, weightID);
    return WrapOrange(POrange(new TItemsetNodeProxy(tree->root, tree)));
  PyCATCH
}

PYXTRACT_IGNORE int Orange_traverse(TPyOrange *, visitproc, void *);
PYXTRACT_IGNORE int Orange_clear(TPyOrange *);

int ItemsetNodeProxy_traverse(PyObject *self, visitproc visit, void *arg)
{
	int err = Orange_traverse((TPyOrange *)self, visit, arg);
	if (err)
		return err;

	CAST_TO_err(TItemsetNodeProxy, node, -1);
	PVISIT(node->tree);
  PVISIT(node->tree->domain);
	return 0;
}

int ItemsetNodeProxy_clear(PyObject *self)
{
  SELF_AS(TItemsetNodeProxy).tree = PSparseItemsetTree();
	return Orange_clear((TPyOrange *)self);
}

PyObject *ItemsetNodeProxy_get_children(PyObject *self)
{
  PyTRY
    CAST_TO(TItemsetNodeProxy, nodeProxy);
    const TSparseItemsetNode *me = nodeProxy->node;
    PyObject *children = PyDict_New();
    const_ITERATE(TSparseISubNodes, ci, me->subNode)
      PyDict_SetItem(children, PyInt_FromLong(ci->first), WrapOrange(POrange(new TItemsetNodeProxy(ci->second, nodeProxy->tree))));
    return children;
  PyCATCH
}

PyObject *ItemsetNodeProxy_get_examples(PyObject *self)
{
  PyTRY
    const TSparseItemsetNode *me = SELF_AS(TItemsetNodeProxy).node;
    PyObject *examples = PyList_New(me->exampleIds.size());
    Py_ssize_t i = 0;
    const_ITERATE(vector<int>, ci, me->exampleIds)
      PyList_SetItem(examples, i++, PyInt_FromLong(*ci));
    return examples;
  PyCATCH
}

PyObject *ItemsetNodeProxy_get_support(PyObject *self)
{
  PyTRY
    return PyFloat_FromDouble(SELF_AS(TItemsetNodeProxy).node->weiSupp);
  PyCATCH
}

PyObject *ItemsetNodeProxy_get_itemId(PyObject *self)
{
  PyTRY
    return PyInt_FromLong(SELF_AS(TItemsetNodeProxy).node->value);
  PyCATCH
}



bool convertFromPython(PyObject *, PAssociationRule &);

PyObject *AssociationRule_new(PyTypeObject *type, PyObject *args, PyObject *) BASED_ON(Orange - Orange.associate.AssociationRule, "(left, right, support, confidence)")
{ PyTRY
    PAssociationRule rule;
    return  convertFromPython(args, rule) ? WrapOrange(rule) : PYNULL;
  PyCATCH
}

PyObject *AssociationRule__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TAssociationRule, arule);
    return Py_BuildValue("O(NN)N", self->ob_type,
                                   Example_FromWrappedExample(arule->left),
                                   Example_FromWrappedExample(arule->right),
                                   packOrangeDictionary(self));
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
      if (PyArg_ParseTuple(obj, "O&O&ffff:convertFromPython(AssociationRule)", ptr_Example, &le, ptr_Example, &re, &nAppliesLeft, &nAppliesRight, &nAppliesBoth, &nExamples)) {
        PExample nle = mlnew TExample(*le);
        PExample nre = mlnew TExample(*re);
        rule = mlnew TAssociationRule(nle, nre, nAppliesLeft, nAppliesRight, nAppliesBoth, nExamples);
        return true;
      }
      else
        break;

    case 2:
    case 3:
    case 4: {
      float support = -1, confidence = -1;
      if (PyArg_ParseTuple(obj, "O&O&|ff:convertFromPython(AssociationRule)", ptr_Example, &le, ptr_Example, &re, &support, &confidence)) {
        PExample nle = mlnew TExample(*le);
        PExample nre = mlnew TExample(*re);
        rule = mlnew TAssociationRule(nle, nre);
        rule->support = support;
        rule->confidence = confidence;
        return true;
      }
      else
        break;
    }

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
      res += ex->domain->getMetaVar((*mi).first)->get_name();
    }

  else {
    string val;

    TVarList::const_iterator vi(ex->domain->variables->begin());
    for(TExample::const_iterator ei(ex->begin()), ee(ex->end()); ei!=ee; ei++, vi++)
      if (!(*ei).isSpecial()) {
        if (res.length())
          res += " ";
        (*vi)->val2str(*ei, val);
        res += (*vi)->get_name() + "=" + val;
      }
  }

  return res;
}

PyObject *AssociationRule_str(TPyOrange *self)
{
  PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
  if (result)
    return result;

  CAST_TO(TAssociationRule, rule);
  return PyString_FromFormat("%s -> %s", side2string(rule->left).c_str(), side2string(rule->right).c_str());
}


PyObject *AssociationRule_repr(TPyOrange *self)
{
  PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "repr", "str");
  if (result)
    return result;

  CAST_TO(TAssociationRule, rule);
  return PyString_FromFormat("%s -> %s", side2string(rule->left).c_str(), side2string(rule->right).c_str());
}


PAssociationRules PAssociationRules_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::P_FromArguments(arg); }
PyObject *AssociationRules_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_FromArguments(type, arg); }
PyObject *AssociationRules_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange - Orange.associate.AssociationRules, "(<list of AssociationRule>)")  ALLOWS_EMPTY { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_new(type, arg, kwds); }
PyObject *AssociationRules_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_getitem(self, index); }
int       AssociationRules_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_setitem(self, index, item); }
PyObject *AssociationRules_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_getslice(self, start, stop); }
int       AssociationRules_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       AssociationRules_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_len(self); }
PyObject *AssociationRules_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_richcmp(self, object, op); }
PyObject *AssociationRules_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_concat(self, obj); }
PyObject *AssociationRules_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_repeat(self, times); }
PyObject *AssociationRules_str(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_str(self); }
PyObject *AssociationRules_repr(TPyOrange *self) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_str(self); }
int       AssociationRules_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_contains(self, obj); }
PyObject *AssociationRules_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(AssociationRule) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_append(self, item); }
PyObject *AssociationRules_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_extend(self, obj); }
PyObject *AssociationRules_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> int") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_count(self, obj); }
PyObject *AssociationRules_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> AssociationRules") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_filter(self, args); }
PyObject *AssociationRules_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> int") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_index(self, obj); }
PyObject *AssociationRules_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_insert(self, args); }
PyObject *AssociationRules_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_native(self); }
PyObject *AssociationRules_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> AssociationRule") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_pop(self, args); }
PyObject *AssociationRules_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(AssociationRule) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_remove(self, obj); }
PyObject *AssociationRules_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_reverse(self); }
PyObject *AssociationRules_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_sort(self, args); }
PyObject *AssociationRules__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PAssociationRules, TAssociationRules, PAssociationRule, &PyOrAssociationRule_Type>::_reduce(self); }

/************* CLASSIFICATION TREES ************/

#include "tdidt.hpp"
#include "tdidt_split.hpp"
#include "tdidt_stop.hpp"
#include "callback.hpp"

C_CALL(TreeLearner - Orange.core.TreeLearner, Learner, "([examples] [, weight=, split=, stop=, nodeLearner=, lookDownOnUnknown=]) -/-> Classifier")

C_NAMED(TreeNode - Orange.classification.tree.Node, Orange, "([lookDownOnUnknown=, branchSelector=, nodeClassifier=, branches=, contingency=])")
C_NAMED(TreeClassifier - Orange.classification.tree.TreeClassifier, ClassifierFD, "([domain=, tree=, descender=])")

C_NAMED(TreeStopCriteria_common - Orange.classification.tree.StopCriteria_common, TreeStopCriteria, "([maxMajority=, minExamples=])")
HIDDEN(TreeStopCriteria_Python - Orange.classification.tree.StopCriteria_Python, TreeStopCriteria)
NO_PICKLE(TreeStopCriteria_Python)

C_CALL(TreeSplitConstructor_Combined - Orange.classification.tree.SplitConstructor_Combined, TreeSplitConstructor, "([examples, [weight, domainContingency, apriorClass, candidates] [discreteTreeSplitConstructor=, continuousTreeSplitConstructor=]) -/-> (Classifier, descriptions, sizes, quality)")

ABSTRACT(TreeSplitConstructor_Measure - Orange.classification.tree.SplitConstructor_Measure, TreeSplitConstructor)
C_CALL(TreeSplitConstructor_Attribute - Orange.classification.tree.SplitConstructor_Attribute, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
C_CALL(TreeSplitConstructor_ExhaustiveBinary - Orange.classification.tree.SplitConstructor_ExhaustiveBinary, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
C_CALL(TreeSplitConstructor_OneAgainstOthers - Orange.classification.tree.SplitConstructor_OneAgainstOthers, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
C_CALL(TreeSplitConstructor_Threshold - Orange.classification.tree.SplitConstructor_Threshold, TreeSplitConstructor_Measure, "([measure=, worstAcceptable=, minSubset=])")
PYXTRACT_IGNORE C_CALL(TreeSplitConstructor_LR - Orange.classification.tree.SplitConstructor_LR, TreeSplitConstructor, "([minSubset=])")

BASED_ON(TreeExampleSplitter - Orange.classification.tree.Splitter, Orange)

C_CALL(TreeExampleSplitter_IgnoreUnknowns - Orange.classification.tree.Splitter_IgnoreUnknowns, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToCommon - Orange.classification.tree.Splitter_UnknownsToCommon, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToAll - Orange.classification.tree.Splitter_UnknownsToAll, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToRandom - Orange.classification.tree.Splitter_UnknownsToRandom, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsToBranch - Orange.classification.tree.Splitter_UnknownsToBranch, TreeExampleSplitter, "([node, examples[, weight]]) -/-> (ExampleGeneratorList, [list of weight ID's])")

C_CALL(TreeExampleSplitter_UnknownsAsBranchSizes - Orange.classification.tree.Splitter_UnknownsAsBranchSizes, TreeExampleSplitter, "([branchIndex, node, examples[, weight]]) -/-> (ExampleGenerator, [list of weight ID's])")
C_CALL(TreeExampleSplitter_UnknownsAsSelector - Orange.classification.tree.Splitter_UnknownsAsSelector, TreeExampleSplitter, "([branchIndex, node, examples[, weight]]) -/-> (ExampleGenerator, [list of weight ID's])")

C_CALL(TreeDescender_UnknownToBranch - Orange.classification.tree.Descender_Unknown, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownToCommonBranch - Orange.classification.tree.Descender_UnknownToCommonBranch, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownToCommonSelector - Orange.classification.tree.Descender_UnknownToCommonSelector, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownMergeAsBranchSizes - Orange.classification.tree.Descender_UnknownMergeAsBranchSizes, TreeDescender, "(node, example) -/-> (node, {distribution | None})")
C_CALL(TreeDescender_UnknownMergeAsSelector - Orange.classification.tree.Descender_UnknownMergeAsSelector, TreeDescender, "(node, example) -/-> (node, {distribution | None})")

ABSTRACT(TreePruner - Orange.classification.tree.Pruner, Orange)
C_CALL (TreePruner_SameMajority - Orange.classification.tree.Pruner_SameMajority, TreePruner, "([tree]) -/-> tree")
C_CALL (TreePruner_m - Orange.classification.tree.Pruner_m, TreePruner, "([tree]) -/-> tree")


PyObject *TreeNode_tree_size(PyObject *self, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> int")
{ PyTRY
    return PyInt_FromLong(PyOrange_AsTreeNode(self)->treeSize());
  PyCATCH
}


PyObject *TreeNode_remove_stored_info(PyObject *self, PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> None")
{ PyTRY
    PyOrange_AsTreeNode(self)->removeStoredInfo();
    RETURN_NONE;
  PyCATCH
}


PyObject *TreeStopCriteria_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.tree.StopCriteria, "()")
{ if (type == (PyTypeObject *)&PyOrTreeStopCriteria_Type) {
      PyObject *name=NULL;
      if (args && !PyArg_ParseTuple(args, "|O", &name))
        PYERROR(PyExc_TypeError, "TreeStopCriteria: invalid arguments - name or callback function expected", PYNULL);

      if (!args || !name || name && PyString_Check(name)) {
          PyObject *self = WrapNewOrange(mlnew TTreeStopCriteria(), type);
          if (name)
            PyObject_SetAttrString(self, "name", name);
          return self;
      }
      // (args && name && !PyStringString_Check(name)

      return setCallbackFunction(WrapNewOrange(mlnew TTreeStopCriteria_Python(), type), args);
  }

  return WrapNewOrange(mlnew TTreeStopCriteria_Python(), type);
}


/* This is all twisted: Python classes are derived from TreeStopCriteria;
   although the underlying C++ structure is TreeStopCriteria_Python,
   the Python base is always TreeStopCritera. We must therefore define
   TreeStopCriteria__reduce__ to handle both C++ objects, and need not
   define TreeStopCriteria_Python__reduce__
*/

PyObject *TreeStopCriteria__reduce__(PyObject *self)
{
  POrange orself = PyOrange_AS_Orange(self);

  if (orself.is_derived_from(TTreeStopCriteria_Python) && PyObject_HasAttrString(self, "__callback")) {
    PyObject *packed = packOrangeDictionary(self);
    PyObject *callback = PyDict_GetItemString(packed, "__callback");
    PyDict_DelItemString(packed, "__callback");
    return Py_BuildValue("O(O)N", self->ob_type, callback, packed);
  }

  /* This works for ordinary (not overloaded) TreeStopCriteria
     and for Python classes derived from TreeStopCriteria.
     The latter have different self->ob_type, so TreeStopCriteria_new will construct
     an instance of TreeStopCriteria_Python */
  return Py_BuildValue("O()N", self->ob_type, packOrangeDictionary(self));
}


PyObject *TreeStopCriteria_lowcall(PyObject *self, PyObject *args, PyObject *keywords, bool allowPython)
{
  static TTreeStopCriteria _cbdefaultStop;
  PyTRY
    NO_KEYWORDS

    CAST_TO(TTreeStopCriteria, stop);
    if (!stop)
      PYERROR(PyExc_SystemError, "attribute error", PYNULL);

    PExampleGenerator egen;
    PDomainContingency dcont;
    int weight = 0;
    if (!PyArg_ParseTuple(args, "O&|O&O&:TreeStopCriteria.__call__", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weight, ptn_DomainContingency, &dcont))
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
{  return TreeStopCriteria_lowcall(self, args, keywords, false); }


PyObject *TreeStopCriteria_Python_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("([examples, [weight, domainContingency, apriorClass, candidates]) -/-> (Classifier, descriptions, sizes, quality)")
{ return TreeStopCriteria_lowcall(self, args, keywords, false); }



PyObject *TreeSplitConstructor_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.tree.SplitConstructor, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeSplitConstructor_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TTreeSplitConstructor_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeSplitConstructor_Python(), type);
}


PyObject *TreeSplitConstructor__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrTreeSplitConstructor_Type);
}


PyObject *TreeSplitConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weight, contingency, apriori class distribution, candidates, nodeClassifier]) -> (Classifier, descriptions, sizes, quality)")
{ PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeSplitConstructor_Type) {
      PyErr_Format(PyExc_SystemError, "TreeSplitConstructor.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    PExampleGenerator gen;
    int weightID = 0;
    PDomainContingency dcont;
    PDistribution apriori;
    PyObject *pycandidates = PYNULL;
    PClassifier nodeClassifier;

    if (!PyArg_ParseTuple(args, "O&|O&O&O&OO&:TreeSplitConstructor.call", pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, ccn_DomainContingency, &dcont, ccn_Distribution, &apriori, &pycandidates, ccn_Classifier, &nodeClassifier))
      return PYNULL;

    vector<bool> candidates;
    if (pycandidates) {
      PyObject *iterator = PyObject_GetIter(pycandidates);
      if (!iterator)
        PYERROR(PyExc_SystemError, "TreeSplitConstructor.call: cannot iterate through candidates; a list exected", PYNULL);
      for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator)) {
        candidates.push_back(PyObject_IsTrue(item) != 0);
        Py_DECREF(item);
      }

      Py_DECREF(iterator);
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


PyObject *TreeExampleSplitter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.tree.Splitter, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeExampleSplitter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TTreeExampleSplitter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeExampleSplitter_Python(), type);
}


PyObject *TreeExampleSplitter__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrTreeExampleSplitter_Type);
}


PyObject *TreeExampleSplitter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(node, examples[, weight]) -/-> (ExampleGeneratorList, list of weight ID's")
{ PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeExampleSplitter_Type) {
      PyErr_Format(PyExc_SystemError, "TreeExampleSplitter.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    PTreeNode node;
    PExampleGenerator gen;
    int weightID = 0;

    if (!PyArg_ParseTuple(args, "O&O&|O&:TreeExampleSplitter.call", cc_TreeNode, &node, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;

    vector<int> newWeights;
    PExampleGeneratorList egl = SELF_AS(TTreeExampleSplitter)(node, gen, weightID, newWeights);

    if (newWeights.size()) {
      PyObject *pyweights = PyList_New(newWeights.size());
      Py_ssize_t i = 0;
      ITERATE(vector<int>, li, newWeights)
        PyList_SetItem(pyweights, i++, PyInt_FromLong(*li));

      return Py_BuildValue("NN", WrapOrange(egl), pyweights);
    }

    else {
      return Py_BuildValue("NO", WrapOrange(egl), Py_None);
    }

  PyCATCH
}



PyObject *TreeDescender_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.tree.Descender, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrTreeDescender_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TMeasureAttribute_Python(), type), args);
  else
    return WrapNewOrange(mlnew TTreeDescender_Python(), type);
}


PyObject *TreeDescender__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrTreeDescender_Type);
}


PyObject *TreeDescender_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(node, example) -/-> (node, {distribution | None})")
{ PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrTreeDescender_Type) {
      PyErr_Format(PyExc_SystemError, "TreeDescender.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

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
    NO_KEYWORDS

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


PTreeNodeList PTreeNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::P_FromArguments(arg); }
PyObject *TreeNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_FromArguments(type, arg); }
PyObject *TreeNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange - Orange.classification.tree.NodeList, "(<list of TreeNode>)")  ALLOWS_EMPTY { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_new(type, arg, kwds); }
PyObject *TreeNodeList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_getitem(self, index); }
int       TreeNodeList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_setitem(self, index, item); }
PyObject *TreeNodeList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_getslice(self, start, stop); }
int       TreeNodeList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       TreeNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_len(self); }
PyObject *TreeNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_richcmp(self, object, op); }
PyObject *TreeNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_concat(self, obj); }
PyObject *TreeNodeList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_repeat(self, times); }
PyObject *TreeNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_str(self); }
PyObject *TreeNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_str(self); }
int       TreeNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_contains(self, obj); }
PyObject *TreeNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(TreeNode) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_append(self, item); }
PyObject *TreeNodeList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_extend(self, obj); }
PyObject *TreeNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> int") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_count(self, obj); }
PyObject *TreeNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> TreeNodeList") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_filter(self, args); }
PyObject *TreeNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> int") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_index(self, obj); }
PyObject *TreeNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_insert(self, args); }
PyObject *TreeNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_native(self); }
PyObject *TreeNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> TreeNode") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_pop(self, args); }
PyObject *TreeNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(TreeNode) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_remove(self, obj); }
PyObject *TreeNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_reverse(self); }
PyObject *TreeNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_sort(self, args); }
PyObject *TreeNodeList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PTreeNodeList, TTreeNodeList, PTreeNode, &PyOrTreeNode_Type>::_reduce(self); }


/************* C45 ************/

#include "c4.5.hpp"

C_CALL(C45Learner - Orange.classification.tree.C45Learner, Learner, "([examples] [, weight=, gainRatio=, subset=, batch=, probThresh=, minObjs=, window=, increment=, cf=, trials=]) -/-> Classifier")
C_NAMED(C45Classifier - Orange.classification.tree.C45Classifier, Classifier, "()")

PyObject *C45Learner_command_line(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(line) -> None")
{ PyTRY
    char *line;
    if (!PyArg_ParseTuple(args, "s", &line))
      PYERROR(PyExc_TypeError, "C45Learner.commandline: string argument expected", NULL);

    SELF_AS(TC45Learner).parseCommandLine(string(line));
    RETURN_NONE;
  PyCATCH
}

C_NAMED(C45TreeNode - Orange.classification.tree.C45Node, Orange, "")

PC45TreeNodeList PC45TreeNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::P_FromArguments(arg); }
PyObject *C45TreeNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_FromArguments(type, arg); }
PyObject *C45TreeNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange - Orange.classification.tree.C45NodeList, "(<list of C45TreeNode>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_new(type, arg, kwds); }
PyObject *C45TreeNodeList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_getitem(self, index); }
int       C45TreeNodeList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_setitem(self, index, item); }
PyObject *C45TreeNodeList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_getslice(self, start, stop); }
int       C45TreeNodeList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       C45TreeNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_len(self); }
PyObject *C45TreeNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_richcmp(self, object, op); }
PyObject *C45TreeNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_concat(self, obj); }
PyObject *C45TreeNodeList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_repeat(self, times); }
PyObject *C45TreeNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_str(self); }
PyObject *C45TreeNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_str(self); }
int       C45TreeNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_contains(self, obj); }
PyObject *C45TreeNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(C45TreeNode) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_append(self, item); }
PyObject *C45TreeNodeList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_extend(self, obj); }
PyObject *C45TreeNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> int") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_count(self, obj); }
PyObject *C45TreeNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> C45TreeNodeList") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_filter(self, args); }
PyObject *C45TreeNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> int") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_index(self, obj); }
PyObject *C45TreeNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_insert(self, args); }
PyObject *C45TreeNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_native(self); }
PyObject *C45TreeNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> C45TreeNode") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_pop(self, args); }
PyObject *C45TreeNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(C45TreeNode) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_remove(self, obj); }
PyObject *C45TreeNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_reverse(self); }
PyObject *C45TreeNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_sort(self, args); }
PyObject *C45TreeNodeList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PC45TreeNodeList, TC45TreeNodeList, PC45TreeNode, &PyOrC45TreeNode_Type>::_reduce(self); }


/************* kNN ************/

#include "knn.hpp"
C_CALL(kNNLearner - Orange.classification.knn.kNNLearner, Learner, "([examples] [k=, weightID=, findNearest=] -/-> Classifier")
C_NAMED(kNNClassifier - Orange.classification.knn.kNNClassifier, ClassifierFD, "(example[, returnWhat]) -> prediction")


/************* PNN ************/

#include "numeric_interface.hpp"

#include "pnn.hpp"

PyObject *P2NN_new(PyTypeObject *type, PyObject *args, PyObject *keywords) BASED_ON(ClassifierFD - Orange.classification.knn.P2NN, "(examples, anchors[, domain]) -> PNN")
{
  PyTRY
    PDomain domain;
    PExampleGenerator examples;
    PyObject *pybases;
    int normalizeExamples = 1;
    if (PyArg_ParseTuple(args, "O&O|iO&:P2NN", pt_ExampleGenerator, &examples, &pybases, &normalizeExamples, cc_Domain, &domain)) {
      if (!domain)
        domain = examples->domain;

      if (!PyList_Check(pybases))
        PYERROR(PyExc_AttributeError, "the anchors should be given as a list", PYNULL);

      const Py_ssize_t nAnchors = PyList_Size(pybases);
      if (nAnchors != domain->attributes->size())
        PYERROR(PyExc_AttributeError, "the number of attributes does not match the number of anchors", PYNULL);

      TFloatList *basesX = mlnew TFloatList(nAnchors);
      TFloatList *basesY = mlnew TFloatList(nAnchors);
      PFloatList wbasesX = basesX, wbasesY = basesY;

      TFloatList::iterator xi(basesX->begin());
      TFloatList::iterator yi(basesY->begin());
      PyObject *foo;

      for(Py_ssize_t i = 0; i < nAnchors; i++)
        if (!PyArg_ParseTuple(PyList_GetItem(pybases, i), "ff|O", &*xi++, &*yi++, &foo)) {
          PyErr_Format(PyExc_TypeError, "anchor #%i is not a tuple of (at least) two elements", i);
          return PYNULL;
        }

      return WrapNewOrange(mlnew TP2NN(domain, examples, wbasesX, wbasesY, -1.0, normalizeExamples != 0), type);
    }

      PyErr_Clear();
      PyObject *matrix;
      PyObject *pyoffsets, *pynormalizers, *pyaverages;
      if (PyArg_ParseTuple(args, "O&OOOOO|i", cc_Domain, &domain, &matrix, &pybases, &pyoffsets, &pynormalizers, &pyaverages, &normalizeExamples)) {
        prepareNumeric();
  //      if (!PyArray_Check(matrix))
  //        PYERROR(PyExc_AttributeError, "the second argument (projection matrix) must a Numeric.array", PYNULL);

        const int nAttrs = domain->attributes->size();

        PyArrayObject *array = (PyArrayObject *)(matrix);
        if (array->nd != 2)
          PYERROR(PyExc_AttributeError, "two-dimensional array expected for matrix of projections", PYNULL);
        if (array->dimensions[1] != 3)
          PYERROR(PyExc_AttributeError, "the matrix of projections must have three columns", PYNULL);

        const char arrayType = getArrayType(array);
        if ((arrayType != 'f') && (arrayType != 'd'))
          PYERROR(PyExc_AttributeError, "elements of matrix of projections must be doubles or floats", PYNULL);

        const int nExamples = array->dimensions[0];

        double *projections = new double[3*nExamples];

        char *rowPtr = array->data;
        double *pi = projections;
        const int &strideRow = array->strides[0];
        const int &strideCol = array->strides[1];

        if (arrayType == 'f') {
          for(int row = 0, rowe = nExamples; row < rowe; row++, rowPtr += strideRow) {
            *pi++ = double(*(float *)(rowPtr));
            *pi++ = double(*(float *)(rowPtr+strideCol));
            *pi++ = double(*(float *)(rowPtr+2*strideCol));
          }
        }
        else {
          for(int row = 0, rowe = nExamples; row < rowe; row++, rowPtr += strideRow) {
            *pi++ = *(double *)(rowPtr);
            *pi++ = *(double *)(rowPtr+strideCol);
            *pi++ = *(double *)(rowPtr+2*strideCol);
          }
        }


        double *bases = NULL;
        PFloatList offsets, normalizers, averages;

        if (pybases == Py_None) {
          if ((pyoffsets != Py_None) || (pynormalizers != Py_None) || (pyaverages != Py_None))
            PYERROR(PyExc_AttributeError, "anchors, offsets, normalizers and averages must be either all given or all None", PYNULL);
        }

        else {
          if (!PyList_Check(pybases) || ((pybases != Py_None) && (PyList_Size(pybases) != nAttrs)))
            PYERROR(PyExc_AttributeError, "the third argument must be a list of anchors with length equal the number of attributes", PYNULL);


          #define LOADLIST(x) \
          x = ListOfUnwrappedMethods<PAttributedFloatList, TAttributedFloatList, float>::P_FromArguments(py##x); \
          if (!x) return PYNULL; \
          if (x->size() != nAttrs) PYERROR(PyExc_TypeError, "invalid size of "#x" list", PYNULL);

          LOADLIST(offsets)
          LOADLIST(normalizers)
          LOADLIST(averages)
          #undef LOADLIST

          bases = new double[2*nAttrs];
          double *bi = bases;
          PyObject *foo;

          for(int i = 0; i < nAttrs; i++, bi+=2)
            if (!PyArg_ParseTuple(PyList_GetItem(pybases, i), "dd|O", bi, bi+1, &foo)) {
              PyErr_Format(PyExc_TypeError, "anchor #%i is not a tuple of (at least) two elements", i);
              delete bases;
              return PYNULL;
            }
        }

        return WrapNewOrange(mlnew TP2NN(domain, projections, nExamples, bases, offsets, normalizers, averages, TP2NN::InverseSquare, normalizeExamples != 0), type);
      }

    PyErr_Clear();
    PYERROR(PyExc_TypeError, "P2NN.invalid arguments", PYNULL);

  PyCATCH;
}


PyObject *P2NN__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TP2NN, p2nn);

    if (!p2nn->offsets)
      PYERROR(PyExc_SystemError, "cannot pickle an invalid instance of P2NN (no offsets)", NULL);

    const int nAttrs = p2nn->offsets->size();
    const int nExamples = p2nn->nExamples;

    TCharBuffer buf(3 + 2 * sizeof(int) + (4 * nAttrs + 3 * nExamples + 2) * sizeof(double));

    buf.writeInt(nAttrs);
    buf.writeInt(nExamples);

    if (p2nn->bases) {
      buf.writeChar(1);
      buf.writeBuf(p2nn->bases, 2 * nAttrs * sizeof(double));
    }
    else
      buf.writeChar(0);

    if (p2nn->radii) {
      buf.writeChar(1);
      buf.writeBuf(p2nn->radii, 2 * nAttrs * sizeof(double));
    }
    else
      buf.writeChar(0);

    if (p2nn->projections) {
      buf.writeChar(1);
      buf.writeBuf(p2nn->projections, 3 * nExamples * sizeof(double));
    }
    else
      buf.writeChar(0);

    buf.writeDouble(p2nn->minClass);
    buf.writeDouble(p2nn->maxClass);

    return Py_BuildValue("O(Os#)N", getExportedFunction("__pickleLoaderP2NN"),
                                    self->ob_type,
                                    buf.buf, buf.length(),
                                    packOrangeDictionary(self));
  PyCATCH
}


PyObject *__pickleLoaderP2NN(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_data)")
{
  PyTRY
    PyTypeObject *type;
    char *pbuf;
    int bufSize;
    if (!PyArg_ParseTuple(args, "Os#:__pickleLoaderP2NN", &type, &pbuf, &bufSize))
      return NULL;

    TCharBuffer buf(pbuf);

    const int nAttrs = buf.readInt();
    const int nExamples = buf.readInt();

    TP2NN *p2nn = new TP2NN(nAttrs, nExamples);
    if (buf.readChar()) {
      buf.readBuf(p2nn->bases, 2 * nAttrs * sizeof(double));
    }
    else {
      delete p2nn->bases;
      p2nn->bases = NULL;
    }

    if (buf.readChar()) {
      buf.readBuf(p2nn->radii, 2 * nAttrs * sizeof(double));
    }
    else {
      delete p2nn->radii;
      p2nn->radii = NULL;
    }

    if (buf.readChar()) {
      buf.readBuf(p2nn->projections, 3 * nExamples * sizeof(double));
    }
    else {
      delete p2nn->projections;
      p2nn->projections = NULL;
    }

    p2nn->minClass = buf.readDouble();
    p2nn->maxClass = buf.readDouble();

    return WrapNewOrange(p2nn, type);
  PyCATCH
}


C_CALL(kNNLearner, Learner, "([examples] [, weight=, k=] -/-> Classifier")
C_NAMED(kNNClassifier, ClassifierFD, "([k=, weightID=, findNearest=])")


/************* Logistic Regression ************/

#include "logistic.hpp"
C_CALL(LogRegLearner - Orange.classification.logreg.LogRegLearner, Learner, "([examples[, weight=]]) -/-> Classifier")
C_NAMED(LogRegClassifier - Orange.classification.logreg.LogRegClassifier, ClassifierFD, "([probabilities=])")


PyObject *LogRegFitter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.logreg.LogRegFitter, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrLogRegFitter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TLogRegFitter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TLogRegFitter_Python(), type);
}

PyObject *LogRegFitter__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrLogRegFitter_Type);
}


C_CALL(LogRegFitter_Cholesky - Orange.classification.logreg.LogRegFitter_Cholesky, LogRegFitter, "([example[, weightID]]) -/-> (status, beta, beta_se, likelihood) | (status, attribute)")

PyObject *LogRegLearner_fitModel(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples[, weight])")
{
  PyTRY
      PExampleGenerator egen;
      int weight = 0;
      if (!PyArg_ParseTuple(args, "O&|O&:LogRegLearner", pt_ExampleGenerator, &egen, pt_weightByGen(egen), &weight))
  	    return PYNULL;

	  CAST_TO(TLogRegLearner, loglearn)

	  int error;
	  PVariable variable;
	  PClassifier classifier;

	  classifier = loglearn->fitModel(egen, weight, error, variable);
	  if (error <= TLogRegFitter::Divergence)
		  return Py_BuildValue("N", WrapOrange(classifier));
	  else
		  return Py_BuildValue("N", WrapOrange(variable));
  PyCATCH
}


PyObject *PyLogRegFitter_ErrorCode_FromLong(long);

PyObject *LogRegFitter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(examples[, weightID]) -/-> (status, beta, beta_se, likelihood) | (status, attribute)")
{
  PyTRY
    NO_KEYWORDS

    int weight;
    PExampleGenerator egen = exampleGenFromArgs(args, weight);
    if (!egen)
      return PYNULL;

    CAST_TO(TLogRegFitter, fitter)

    PAttributedFloatList beta, beta_se;
    float likelihood;
    int error;
    PVariable attribute;

    beta = (*fitter)(egen, weight, beta_se, likelihood, error, attribute);

    if (error <= TLogRegFitter::Divergence)
      return Py_BuildValue("NNNf", PyLogRegFitter_ErrorCode_FromLong(error), WrapOrange(beta), WrapOrange(beta_se), likelihood);
    else
      return Py_BuildValue("NN", PyLogRegFitter_ErrorCode_FromLong(error), WrapOrange(attribute));

  PyCATCH
}

/************ Linear **********/
#include "liblinear_interface.hpp"
C_CALL(LinearLearner - Orange.classification.svm.LinearLearner, Learner, "([examples] -/-> Classifier)")
C_NAMED(LinearClassifier - Orange.classification.svm.LinearClassifier, ClassifierFD, " ")



PyObject *LinearClassifier__reduce__(PyObject *self){
  PyTRY
	CAST_TO(TLinearClassifier, classifier);
	string buff;
	if (linear_save_model_alt(buff, classifier->getModel()) != 0)
		raiseError("Could not save the model");
	return Py_BuildValue("O(OOOs)N", getExportedFunction("__pickleLoaderLinearClassifier"),
									self->ob_type,
									WrapOrange(classifier->classVar),
									WrapOrange(classifier->examples),
									buff.c_str(),
									packOrangeDictionary(self));
  PyCATCH
}

PyObject *__pickleLoaderLinearClassifier(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, packed_data)")
{
  PyTRY
	PyTypeObject* type;
    PVariable var;
	PExampleTable examples;
	char *pBuff;
	if (!PyArg_ParseTuple(args, "OO&O&s", &type, cc_Variable, &var, cc_ExampleTable, &examples, &pBuff))
		return NULL;
	string buff(pBuff);
	model *model = linear_load_model_alt(buff);
	if (!model)
		raiseError("Could not load the model");
	return WrapNewOrange(mlnew TLinearClassifier(var, examples, model), (PyTypeObject*)&PyOrLinearClassifier_Type);
  PyCATCH
}

/************* LIBSVM ************/

#include "libsvm_interface.hpp"
C_CALL(SVMLearner - Orange.core.SVMLearner, Learner, "([examples] -/-> Classifier)")
C_CALL(SVMLearnerSparse - Orange.core.SVMLearnerSparse, SVMLearner, "([examples] -/-> Classifier)")
C_NAMED(SVMClassifier - Orange.classification.svm.SVMClassifier, ClassifierFD," ")
C_NAMED(SVMClassifierSparse - Orange.classification.svm.SVMClassifierSparse, SVMClassifier," ")

PyObject *SVMLearner_setWeights(PyObject *self, PyObject* args, PyObject *keywords) PYARGS(METH_VARARGS, "('list of tuple pairs') -> None")
{
	PyTRY

	PyObject *pyWeights;
	if (!PyArg_ParseTuple(args, "O:SVMLearner.setWeights", &pyWeights)) {
		//PyErr_Format(PyExc_TypeError, "SVMLearner.setWeights: an instance of Python List expected got '%s'", pyWeights->ob_type->tp_name);
		PYERROR(PyExc_TypeError, "SVMLearner.setWeights: Python List of attribute weights expected", PYNULL);
		return PYNULL;
	}

	CAST_TO(TSVMLearner, learner);

	Py_ssize_t size = PyList_Size(pyWeights);
	//cout << "n weights: " << size << endl;
	Py_ssize_t i;

	free(learner->weight_label);
	free(learner->weight);

	learner->nr_weight = size;
	learner->weight_label = NULL;
	learner->weight = NULL;

	if (size > 0) {
		learner->weight_label = (int *)malloc((size)*sizeof(int));
		learner->weight = (double *)malloc((size)*sizeof(double));
	}

	for (i = 0; i < size; i++) {
		int l;
		double w;
		PyArg_ParseTuple(PyList_GetItem(pyWeights, i), "id:SVMLearner.setWeights", &l, &w);
		learner->weight[i] = w;
		learner->weight_label[i] = l;
		//cout << "class: " << l << ", w: " << w << endl;
	}

	RETURN_NONE;
	PyCATCH
}

PyObject *KernelFunc_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.kernels.KernelFunc, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrKernelFunc_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TKernelFunc_Python(), type), args);
  else
    return WrapNewOrange(mlnew TKernelFunc_Python(), type);
}


PyObject *KernelFunc__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrKernelFunc_Type);
}


PyObject *KernelFunc_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(Example, Example) -> float")
{
  PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrKernelFunc_Type) {
      PyErr_Format(PyExc_SystemError, "KernelFunc.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    float f;
    PExample e1,e2;
	if (!PyArg_ParseTuple(args, "O&O&", cc_Example, &e1, cc_Example, &e2))
		return NULL;
	f=SELF_AS(TKernelFunc)(e1.getReference(),e2.getReference());
	return Py_BuildValue("f", f);
  PyCATCH
}

PyObject *SVMClassifier__reduce__(PyObject* self)
{
  PyTRY
    CAST_TO(TSVMClassifier, svm);
    string buf;
    if (svm_save_model_alt(buf, svm->getModel())){
    	raiseError("Error saving SVM model");
    }
    if(svm->kernelFunc)
        return Py_BuildValue("O(OOOsO)N", self->ob_type,
                                    WrapOrange(svm->classVar),
                                    WrapOrange(svm->examples),
                                    WrapOrange(svm->supportVectors),
                                    buf.c_str(),
                                    WrapOrange(svm->kernelFunc),
                                    packOrangeDictionary(self));
    else
        return Py_BuildValue("O(OOOs)N", self->ob_type,
                                    WrapOrange(svm->classVar),
                                    WrapOrange(svm->examples),
                                    WrapOrange(svm->supportVectors),
                                    buf.c_str(),
                                    packOrangeDictionary(self));
  PyCATCH
}


PyObject *SVMClassifierSparse__reduce__(PyObject* self)
{
  PyTRY
    CAST_TO(TSVMClassifierSparse, svm);
    string buf;
    if (svm_save_model_alt(buf, svm->getModel())){
        raiseError("Error saving SVM model.");
    }
    if(svm->kernelFunc)
        return Py_BuildValue("O(OOOsbO)N", self->ob_type,
                                    WrapOrange(svm->classVar),
                                    WrapOrange(svm->examples),
                                    WrapOrange(svm->supportVectors),
                                    buf.c_str(),
									(char)(svm->useNonMeta? 1: 0),
                                    WrapOrange(svm->kernelFunc),
                                    packOrangeDictionary(self));
    else
        return Py_BuildValue("O(OOOsb)N", self->ob_type,
                                    WrapOrange(svm->classVar),
                                    WrapOrange(svm->examples),
                                    WrapOrange(svm->supportVectors),
                                    buf.c_str(),
                                    (char)(svm->useNonMeta? 1: 0),
                                    packOrangeDictionary(self));
  PyCATCH
}


PyObject *SVMClassifier_getDecisionValues(PyObject *self, PyObject* args, PyObject *keywords) PYARGS(METH_VARARGS, "(Example) -> list of floats")
{PyTRY
	PExample example;
	if (!PyArg_ParseTuple(args, "O&", cc_Example, &example))
		return NULL;
	PFloatList f=SELF_AS(TSVMClassifier).getDecisionValues(example.getReference());
	return WrapOrange(f);
PyCATCH
}

PyObject *SVMClassifier_getModel(PyObject *self, PyObject* args, PyObject *keywords) PYARGS(METH_VARARGS, "() -> string")
{PyTRY
	string buf;
	svm_model* model = SELF_AS(TSVMClassifier).getModel();
	if (!model)
		raiseError("No model.");
	svm_save_model_alt(buf, model);
	return Py_BuildValue("s", buf.c_str());
PyCATCH
}


PyObject * SVMClassifier_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) BASED_ON(ClassifierFD, "(Variable, Examples, Examples, string, [kernelFunc]) -> SVMClassifier")
{PyTRY
	PVariable classVar;
	PExampleTable examples;
	PExampleTable supportVectors;
	char*  model_string;
	PKernelFunc kernel;
	if (PyArg_ParseTuple(args, ""))
		return WrapNewOrange(mlnew TSVMClassifier(), type);
	PyErr_Clear();
	
	if (!PyArg_ParseTuple(args, "O&O&O&s|O&:__new__", cc_Variable, &classVar, cc_ExampleTable, &examples, cc_ExampleTable, &supportVectors, &model_string, cc_KernelFunc, &kernel))
		return NULL;

	string buffer(model_string);
	svm_model* model = svm_load_model_alt(buffer);
	if (!model)
		raiseError("Error building LibSVM Model");
//	model->param.learner = NULL;
	PSVMClassifier svm = mlnew TSVMClassifier(classVar, examples, model, NULL, kernel);
//	svm->kernelFunc = kernel;
	svm->supportVectors = supportVectors;
	return WrapOrange(svm);
PyCATCH
}

PyObject * SVMClassifierSparse_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) BASED_ON(SVMClassifier, "(Variable, Examples, Examples, string, [useNonMeta, kernelFunc]) -> SVMClassifierSparse")
{PyTRY
	PVariable classVar;
	PExampleTable examples;
	PExampleTable supportVectors;
	char*  model_string;
	char useNonMeta;
	PKernelFunc kernel;
	if (PyArg_ParseTuple(args, ""))
		return WrapNewOrange(mlnew TSVMClassifierSparse(), type);
	PyErr_Clear();
	
	if (!PyArg_ParseTuple(args, "O&O&O&s|bO&:__new__", cc_Variable, &classVar, cc_ExampleTable, &examples, cc_ExampleTable, &supportVectors, &model_string, &useNonMeta, cc_KernelFunc, &kernel))
		return NULL;
	
	string buffer(model_string);
	svm_model* model = svm_load_model_alt(buffer);
	if (!model)
		raiseError("Error building LibSVM Model");
//	model->param.learner = NULL;
	PSVMClassifier svm = mlnew TSVMClassifierSparse(classVar, examples, model, NULL, useNonMeta != 0, kernel);
//	svm->kernelFunc = kernel;
	svm->supportVectors = supportVectors;
	return WrapOrange(svm);
	PyCATCH
}


/************ EARTH (MARS) ******/
#include "earth.hpp"

C_CALL(EarthLearner - Orange.core.EarthLearner, Learner, "([examples], [weight=] -/-> Classifier)")
C_NAMED(EarthClassifier - Orange.core.EarthClassifier, ClassifierFD, " ")

PyObject *EarthClassifier_formatEarth(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
	PyTRY
	CAST_TO(TEarthClassifier, classifier);
	classifier->format_earth();
	RETURN_NONE;
	PyCATCH
}

PyObject *EarthClassifier__reduce__(PyObject *self) PYARGS(METH_VARARGS, "")
{
	PyTRY
	CAST_TO(TEarthClassifier, classifier);
	TCharBuffer buffer(1024);
	classifier->save_model(buffer);
	return Py_BuildValue("O(s#)N", getExportedFunction("__pickleLoaderEarthClassifier"),
									buffer.buf, buffer.length(),
									packOrangeDictionary(self));
	PyCATCH
}

PyObject *__pickleLoaderEarthClassifier(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(buffer)")
{
	PyTRY
	char * cbuf = NULL;
	int buffer_size = 0;
	if (!PyArg_ParseTuple(args, "s#:__pickleLoaderEarthClassifier", &cbuf, &buffer_size))
		return NULL;
	TCharBuffer buffer(cbuf);
	PEarthClassifier classifier = mlnew TEarthClassifier();
	classifier->load_model(buffer);
	return WrapOrange(classifier);
	PyCATCH
}


	
/************* BAYES ************/

#include "bayes.hpp"
C_CALL(BayesLearner - Orange.core.BayesLearner, Learner, "([examples], [weight=, estimate=] -/-> Classifier")
C_NAMED(BayesClassifier - Orange.core.BayesClassifier, ClassifierFD, "([probabilities=])")

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



/************* RULES ************/

#include "rulelearner.hpp"

C_NAMED(Rule - Orange.classification.rules.Rule, Orange, "()")

C_NAMED(RuleValidator_LRS - Orange.classification.rules.RuleValidator_LRS, RuleValidator, "([alpha=0.05,min_coverage=0,max_rule_complexity=0,min_quality=numeric_limits<float>::min()])")

C_NAMED(RuleEvaluator_Entropy - Orange.classification.rules.RuleEvaluator_Entropy, RuleEvaluator, "()")
C_NAMED(RuleEvaluator_Laplace - Orange.classification.rules.RuleEvaluator_Laplace, RuleEvaluator, "()")
C_NAMED(RuleEvaluator_LRS - Orange.classification.rules.RuleEvaluator_LRS, RuleEvaluator, "()")
C_NAMED(RuleEvaluator_mEVC - Orange.classification.rules.RuleEvaluator_mEVC, RuleEvaluator, "(ruleAlpha=1.0,attributeAlpha=1.0)")

C_NAMED(EVDist, Orange, "()")
C_NAMED(EVDistGetter_Standard, EVDistGetter, "()")

C_NAMED(RuleBeamFinder - Orange.classification.rules.RuleBeamFinder, RuleFinder, "([validator=, evaluator=, initializer=, refiner=, candidateSelector=, ruleFilter=])")

C_NAMED(RuleBeamInitializer_Default - Orange.classification.rules.RuleBeamInitializer_Default, RuleBeamInitializer, "()")

C_NAMED(RuleBeamRefiner_Selector - Orange.classification.rules.RuleBeamRefiner_Selector, RuleBeamRefiner, "([discretization=])")

C_NAMED(RuleBeamCandidateSelector_TakeAll - Orange.classification.rules.RuleBeamCandidateSelector_TakeAll, RuleBeamCandidateSelector, "()")

C_NAMED(RuleBeamFilter_Width - Orange.classification.rules.RuleBeamFilter_Width, RuleBeamFilter, "([width=5])")

C_NAMED(RuleDataStoppingCriteria_NoPositives - Orange.classification.rules.RuleDataStoppingCriteria_NoPositives, RuleDataStoppingCriteria, "()")

C_NAMED(RuleCovererAndRemover_Default - Orange.classification.rules.RuleCovererAndRemover_Default, RuleCovererAndRemover, "()")

C_NAMED(RuleStoppingCriteria_NegativeDistribution - Orange.classification.rules.RuleStoppingCriteria_NegativeDistribution, RuleStoppingCriteria, "()")
C_CALL(RuleLearner - Orange.classification.rules.RuleLearner, Learner, "([examples[, weightID]]) -/-> Classifier")

ABSTRACT(RuleClassifier - Orange.classification.rules.RuleClassifier, Classifier)
C_NAMED(RuleClassifier_firstRule - Orange.classification.rules.RuleClassifier_firstRule, RuleClassifier, "([rules,examples[,weightID]])")
C_NAMED(RuleClassifier_logit - Orange.classification.rules.RuleClassifier_logit, RuleClassifier, "([rules,minSig,minBeta,examples[,weightID]])")

PyObject *Rule_call(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyTRY
    NO_KEYWORDS

    if (PyTuple_Size(args)==1) {
      PyObject *pyex = PyTuple_GET_ITEM(args, 0);
      if (PyOrExample_Check(pyex))
        return PyInt_FromLong(PyOrange_AsRule(self)->call(PyExample_AS_ExampleReference(pyex)) ? 1 : 0);
    }

    PExampleGenerator egen;
    int references = 1;
    int negate = 0;
    if (!PyArg_ParseTuple(args, "O&|ii:Rule.__call__", &pt_ExampleGenerator, &egen, &references, &negate))
      return PYNULL;

    CAST_TO(TRule, rule)
    PExampleTable res = (*rule)(egen,(references?true:false),(negate?true:false));
    return WrapOrange(res);
  PyCATCH
}

PyObject *Rule_filterAndStore(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples, weightID, targetClass)")
{
  PyTRY
    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;

    if (!PyArg_ParseTuple(args, "O&O&i:RuleEvaluator.call",  pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass))
      return PYNULL;

    CAST_TO(TRule, rule);
    rule->filterAndStore(gen,weightID,targetClass);
    RETURN_NONE;
 PyCATCH
}

PyObject *RuleEvaluator_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleEvaluator, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleEvaluator_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleEvaluator_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleEvaluator_Python(), type);
}

PyObject *RuleEvaluator__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleEvaluator_Type);
}


PyObject *RuleEvaluator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rule, table, weightID, targetClass, apriori) -/-> (quality)")
{
  PyTRY
    NO_KEYWORDS

    PRule rule;
    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;
    PDistribution apriori;

    if (!PyArg_ParseTuple(args, "O&O&O&iO&:RuleEvaluator.call", cc_Rule, &rule, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass, cc_Distribution, &apriori))
      return PYNULL;
    CAST_TO(TRuleEvaluator, evaluator)
    float quality;

    quality = (*evaluator)(rule, gen, weightID, targetClass, apriori);
    return PyFloat_FromDouble(quality);
  PyCATCH
}

PyObject *EVDistGetter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrEVDistGetter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TEVDistGetter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TEVDistGetter_Python(), type);
}

PyObject *EVDistGetter__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrEVDistGetter_Type);
}


PyObject *EVDistGetter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rule, length) -/-> (EVdist)")
{
  PyTRY
    NO_KEYWORDS

    PRule rule;
    int parentLength, rLength;

    if (!PyArg_ParseTuple(args, "O&ii:EVDistGetter.call", cc_Rule, &rule, &parentLength, &rLength))
      return PYNULL;
    CAST_TO(TEVDistGetter, getter)
    PEVDist dist = (*getter)(rule, parentLength, rLength);

    return WrapOrange(dist);
  PyCATCH
}

PyObject *RuleValidator_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleValidator, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleValidator_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleValidator_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleValidator_Python(), type);
}

PyObject *RuleValidator__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleValidator_Type);
}


PyObject *RuleValidator_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rule, table, weightID, targetClass, apriori) -/-> (quality)")
{

  PyTRY
    NO_KEYWORDS

    PRule rule;
    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;
    PDistribution apriori;

    if (!PyArg_ParseTuple(args, "O&O&O&iO&:RuleValidator.call", cc_Rule, &rule, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass, cc_Distribution, &apriori))
      return PYNULL;
    CAST_TO(TRuleValidator, validator)

    bool valid;
    valid = (*validator)(rule, gen, weightID, targetClass, apriori);
    return PyInt_FromLong(valid?1:0);
  PyCATCH
}

PyObject *RuleCovererAndRemover_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleCovererAndRemover, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleCovererAndRemover_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleCovererAndRemover_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleCovererAndRemover_Python(), type);
}

PyObject *RuleCovererAndRemover__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleCovererAndRemover_Type);
}


PyObject *RuleCovererAndRemover_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rule, table, weightID, targetClass) -/-> (table,newWeight)")
{
  PyTRY
    NO_KEYWORDS

    PRule rule;
    PExampleGenerator gen;
    int weightID = 0;
    int newWeightID = 0;
    int targetClass = -1;

    if (!PyArg_ParseTuple(args, "O&O&O&i:RuleCovererAndRemover.call", cc_Rule, &rule, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID,&targetClass))
      return PYNULL;
    CAST_TO(TRuleCovererAndRemover, covererAndRemover)

    PExampleTable res = (*covererAndRemover)(rule, gen, weightID, newWeightID, targetClass);
    return Py_BuildValue("Ni", WrapOrange(res),newWeightID);
  PyCATCH
}

PyObject *RuleStoppingCriteria_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleStoppingCriteria, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleStoppingCriteria_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleStoppingCriteria_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleStoppingCriteria_Python(), type);
}

PyObject *RuleStoppingCriteria__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleStoppingCriteria_Type);
}


PyObject *RuleStoppingCriteria_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rulelist, rule, table, weightID) -/-> (table)")
{
  PyTRY
    NO_KEYWORDS

    PRuleList ruleList;
    PRule rule;
    PExampleGenerator gen;
    int weightID = 0;

    if (!PyArg_ParseTuple(args, "O&O&O&O&:RuleStoppingCriteria.call", cc_RuleList, &ruleList, cc_Rule, &rule, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;
    CAST_TO(TRuleStoppingCriteria, ruleStopping)

    bool stop = (*ruleStopping)(ruleList, rule, gen, weightID);
    return PyInt_FromLong(stop?1:0);
  PyCATCH
}

PyObject *RuleDataStoppingCriteria_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleDataStoppingCriteria, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleDataStoppingCriteria_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleDataStoppingCriteria_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleDataStoppingCriteria_Python(), type);
}

PyObject *RuleDataStoppingCriteria__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleDataStoppingCriteria_Type);
}


PyObject *RuleDataStoppingCriteria_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(table, weightID, targetClass) -/-> (table)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;

    if (!PyArg_ParseTuple(args, "O&O&i:RuleDataStoppingCriteria.call", pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass))
      return PYNULL;
    CAST_TO(TRuleDataStoppingCriteria, dataStopping)

    bool stop = (*dataStopping)(gen, weightID, targetClass);
    return PyInt_FromLong(stop?1:0);
  PyCATCH
}

PyObject *RuleFinder_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleFinder, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleFinder_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleFinder_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleFinder_Python(), type);
}

PyObject *RuleFinder__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleFinder_Type);
}


PyObject *RuleFinder_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(table, weightID, targetClass, baseRules) -/-> (rule)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;
    PRuleList baseRules;

    if (!PyArg_ParseTuple(args, "O&O&iO&:RuleFinder.call", pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass, ccn_RuleList, &baseRules))
      return PYNULL;
    CAST_TO(TRuleFinder, finder)

    PRule res = (*finder)(gen, weightID, targetClass, baseRules);
    return WrapOrange(res);
  PyCATCH
}

PyObject *RuleBeamRefiner_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleBeamRefiner, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleBeamRefiner_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleBeamRefiner_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleBeamRefiner_Python(), type);
}

PyObject *RuleBeamRefiner__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleBeamRefiner_Type);
}


PyObject *RuleBeamRefiner_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rule, table, weightID, targetClass) -/-> (rules)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator gen;
    int weightID = 0;
    int targetClass = -1;
    PRule rule;

    if (!PyArg_ParseTuple(args, "O&O&O&i:RuleBeamRefiner.call", cc_Rule, &rule, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass))
      return PYNULL;
    CAST_TO(TRuleBeamRefiner, refiner)

    PRuleList res = (*refiner)(rule, gen, weightID, targetClass);
    return WrapOrange(res);
  PyCATCH
}

PyObject *RuleBeamInitializer_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleBeamInitializer, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleBeamInitializer_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleBeamInitializer_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleBeamInitializer_Python(), type);
}

PyObject *RuleBeamInitializer__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleBeamInitializer_Type);
}


PyObject *RuleBeamInitializer_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(table, weightID, targetClass, baseRules, evaluator, prior) -/-> (rules, bestRule)")
{
  PyTRY
     NO_KEYWORDS

    PExampleGenerator gen;
    PRuleList baseRules;
    PRuleEvaluator evaluator;
    PDistribution prior;
    PRule bestRule;
    int weightID = 0;
    int targetClass = -1;
    PRule rule;

    if (!PyArg_ParseTuple(args, "O&O&iO&O&O&:RuleBeamInitializer.call", pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &targetClass, ccn_RuleList, &baseRules, cc_RuleEvaluator, &evaluator, cc_Distribution, &prior))
      return PYNULL;
    CAST_TO(TRuleBeamInitializer, initializer)

    PRuleList res = (*initializer)(gen, weightID, targetClass, baseRules, evaluator, prior, bestRule);
    return Py_BuildValue("NN", WrapOrange(res), WrapOrange(bestRule));
  PyCATCH
}

PyObject *RuleBeamCandidateSelector_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleBeamCandidateSelector, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleBeamCandidateSelector_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleBeamCandidateSelector_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleBeamCandidateSelector_Python(), type);
}

PyObject *RuleBeamCandidateSelector__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleBeamCandidateSelector_Type);
}


PyObject *RuleBeamCandidateSelector_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(existingRules, table, weightID) -/-> (candidates, remainingRules)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator gen;
    PRuleList existingRules;
    int weightID = 0;
    PRuleList candidates;

    if (!PyArg_ParseTuple(args, "O&O&O&:RuleBeamCandidateSelector.call", cc_RuleList, &existingRules, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;
    CAST_TO(TRuleBeamCandidateSelector, selector)

    PRuleList res = (*selector)(existingRules, gen, weightID);
    return Py_BuildValue("NN", WrapOrange(res), WrapOrange(existingRules));
  PyCATCH
}

PyObject *RuleBeamFilter_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleBeamFilter, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleBeamFilter_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleBeamFilter_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleBeamFilter_Python(), type);
}

PyObject *RuleBeamFilter__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleBeamFilter_Type);
}


PyObject *RuleBeamFilter_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rules, table, weightID) -/-> (rules)")
{
  PyTRY
    NO_KEYWORDS

    PExampleGenerator gen;
    PRuleList rules;
    int weightID = 0;

    if (!PyArg_ParseTuple(args, "O&O&O&:RuleBeamFilter.call", cc_RuleList, &rules, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;
    CAST_TO(TRuleBeamFilter, filter)

    (*filter)(rules, gen, weightID);
    return WrapOrange(rules);
  PyCATCH
}

PyObject *RuleClassifierConstructor_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(Orange - Orange.classification.rules.RuleClassifierConstructor, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrRuleClassifierConstructor_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TRuleClassifierConstructor_Python(), type), args);
  else
    return WrapNewOrange(mlnew TRuleClassifierConstructor_Python(), type);
}


PyObject *RuleClassifierConstructor__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrRuleClassifierConstructor_Type);
}


PyObject *RuleClassifierConstructor_call(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rules, examples[, weight]) -> (RuleClassifier)")
{
  PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrRuleClassifierConstructor_Type) {
      PyErr_Format(PyExc_SystemError, "RuleClassifierConstructor.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    PExampleGenerator gen;
    int weightID = 0;
    PRuleList rules;

    if (!PyArg_ParseTuple(args, "O&O&|O&:RuleClassifierConstructor.call", cc_RuleList, &rules, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID))
      return PYNULL;

    PRuleClassifier ruleClassifier;
    ruleClassifier = SELF_AS(TRuleClassifierConstructor)(rules, gen, weightID);
    return WrapOrange(ruleClassifier);
  PyCATCH
}

PyObject *RuleClassifier_logit_new(PyObject *self, PyObject *args, PyObject *keywords) PYDOC("(rules, min_beta, examples[, weight])")
{
  PyTRY
    NO_KEYWORDS

    if (PyOrange_OrangeBaseClass(self->ob_type) == &PyOrRuleClassifier_Type) {
      PyErr_Format(PyExc_SystemError, "RuleClassifier.call called for '%s': this may lead to stack overflow", self->ob_type->tp_name);
      return PYNULL;
    }

    PExampleGenerator gen;
    int weightID = 0;
    float minSignificance = 0.5;
    float minBeta = 0.0;
    PRuleList rules;
    PDistributionList probList;
    PClassifier classifier;
    bool setPrefixRules;
    bool optimizeBetasFlag;

    if (!PyArg_ParseTuple(args, "O&ffO&|O&iiO&O&:RuleClassifier.call", cc_RuleList, &rules, &minSignificance, &minBeta, pt_ExampleGenerator, &gen, pt_weightByGen(gen), &weightID, &setPrefixRules, &optimizeBetasFlag, cc_Classifier, &classifier, cc_DistributionList, &probList))
      return PYNULL;

    TRuleClassifier *rc = new TRuleClassifier_logit(rules, minSignificance, minBeta, gen, weightID, classifier, probList,setPrefixRules, optimizeBetasFlag);
    PRuleClassifier ruleClassifier = rc;
//    ruleClassifier = new SELF_AS(TRuleClassifier)(rules, gen, weightID);
    return WrapOrange(ruleClassifier);
  PyCATCH
}

PRuleList PRuleList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::P_FromArguments(arg); }
PyObject *RuleList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_FromArguments(type, arg); }
PyObject *RuleList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of Rule>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_new(type, arg, kwds); }
PyObject *RuleList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_getitem(self, index); }
int       RuleList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_setitem(self, index, item); }
PyObject *RuleList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_getslice(self, start, stop); }
int       RuleList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       RuleList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_len(self); }
PyObject *RuleList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_richcmp(self, object, op); }
PyObject *RuleList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_concat(self, obj); }
PyObject *RuleList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_repeat(self, times); }
PyObject *RuleList_str(TPyOrange *self) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_str(self); }
PyObject *RuleList_repr(TPyOrange *self) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_str(self); }
int       RuleList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_contains(self, obj); }
PyObject *RuleList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(Rule) -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_append(self, item); }
PyObject *RuleList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_extend(self, obj); }
PyObject *RuleList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Rule) -> int") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_count(self, obj); }
PyObject *RuleList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> RuleList") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_filter(self, args); }
PyObject *RuleList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Rule) -> int") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_index(self, obj); }
PyObject *RuleList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_insert(self, args); }
PyObject *RuleList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_native(self); }
PyObject *RuleList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Rule") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_pop(self, args); }
PyObject *RuleList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(Rule) -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_remove(self, obj); }
PyObject *RuleList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_reverse(self); }
PyObject *RuleList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_sort(self, args); }
PyObject *RuleList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PRuleList, TRuleList, PRule, &PyOrRule_Type>::_reduce(self); }

PEVDistList PEVDistList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::P_FromArguments(arg); }
PyObject *EVDistList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_FromArguments(type, arg); }
PyObject *EVDistList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of EVDist>)") ALLOWS_EMPTY { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_new(type, arg, kwds); }
PyObject *EVDistList_getitem_sq(TPyOrange *self, Py_ssize_t index) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_getitem(self, index); }
int       EVDistList_setitem_sq(TPyOrange *self, Py_ssize_t index, PyObject *item) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_setitem(self, index, item); }
PyObject *EVDistList_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_getslice(self, start, stop); }
int       EVDistList_setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *item) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_setslice(self, start, stop, item); }
Py_ssize_t       EVDistList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_len(self); }
PyObject *EVDistList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_richcmp(self, object, op); }
PyObject *EVDistList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_concat(self, obj); }
PyObject *EVDistList_repeat(TPyOrange *self, Py_ssize_t times) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_repeat(self, times); }
PyObject *EVDistList_str(TPyOrange *self) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_str(self); }
PyObject *EVDistList_repr(TPyOrange *self) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_str(self); }
int       EVDistList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_contains(self, obj); }
PyObject *EVDistList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(EVDist) -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_append(self, item); }
PyObject *EVDistList_extend(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(sequence) -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_extend(self, obj); }
PyObject *EVDistList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(EVDist) -> int") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_count(self, obj); }
PyObject *EVDistList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> EVDistList") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_filter(self, args); }
PyObject *EVDistList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(EVDist) -> int") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_index(self, obj); }
PyObject *EVDistList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_insert(self, args); }
PyObject *EVDistList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_native(self); }
PyObject *EVDistList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> EVDist") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_pop(self, args); }
PyObject *EVDistList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(EVDist) -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_remove(self, obj); }
PyObject *EVDistList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_reverse(self); }
PyObject *EVDistList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_sort(self, args); }
PyObject *EVDistList__reduce__(TPyOrange *self, PyObject *) { return ListOfWrappedMethods<PEVDistList, TEVDistList, PEVDist, &PyOrEVDist_Type>::_reduce(self); }

#include "lib_learner.px"

