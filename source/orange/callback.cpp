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

#include "values.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "table.hpp"
#include "contingency.hpp"
#include "distance.hpp"

#include "cls_example.hpp"
#include "cls_value.hpp"
#include "cls_orange.hpp"

#include "lib_kernel.hpp"
#include "externs.px"

#include "callback.ppp"

PyObject *callCallback(PyObject *self, PyObject *args)
{
    PyObject *result;

    if (PyObject_HasAttrString(self, "__callback")) {
        PyObject *callback = PyObject_GetAttrString(self, "__callback");
        result = PyObject_CallObject(callback, args);
        Py_DECREF(callback);
    }
    else
        result = PyObject_CallObject(self, args);

    if (!result)
        throw pyexception();

    return result;
}

PyObject *callMethod(char const *method, PyObject *self, PyObject *args)
{

    if (PyObject_HasAttrString(self, const_cast<char *>(method))) {
        PyObject *callback = PyObject_GetAttrString(self, const_cast<char *>(method));
        PyObject *result = PyObject_CallObject(callback, args);
        Py_DECREF(callback);

        if (!result)
            throw pyexception();

        return result;
    }

    raiseErrorWho("Python object does not provide method '%s'", method);
    return NULL; // to make the compiler happy
}

PyObject *setCallbackFunction(PyObject *self, PyObject *args)
{
    PyObject *func;
    if (!PyArg_ParseTuple(args, "O", &func)) {
        PyErr_Format(PyExc_TypeError, "callback function for '%s' expected",
                self->ob_type->tp_name);
        Py_DECREF(self);
        return PYNULL;
    }
    else if (!PyCallable_Check(func)) {
        PyErr_Format(PyExc_TypeError, "'%s' object is not callable",
                func->ob_type->tp_name);
        Py_DECREF(self);
        return PYNULL;
    }

    PyObject_SetAttrString(self, "__callback", func);
    return self;
}

PyObject *callbackReduce(PyObject *self, TOrangeType &basetype)
{
    if (self->ob_type == (PyTypeObject *) &basetype) {
        PyObject *packed = packOrangeDictionary(self);
        PyObject *callback = PyDict_GetItemString(packed, "__callback");
        if (!callback)
            PYERROR(
                    PyExc_AttributeError,
                    "cannot pickle an invalid callback object ('__callback' attribute is missing)",
                    NULL);

        PyDict_DelItemString(packed, "__callback");
        return Py_BuildValue("O(O)N", self->ob_type, callback, packed);
    }
    else
        return Py_BuildValue("O()N", self->ob_type, packOrangeDictionary(self));
}

bool TFilter_Python::operator()(const TExample &ex)
{
    PyObject *args = Py_BuildValue("(N)", Example_FromExampleCopyRef(ex));
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    bool res = bool(PyObject_IsTrue(result)!=0);
    Py_DECREF(result);
    return res;
}

PFilter TFilter_Python::deepCopy() const
{
    PyObject *result = PyObject_CallMethod((PyObject *) myWrapper, "deepCopy",
            NULL);
    if (!result)
        raiseError("An exception has been thrown in method deepCopy!");
    if (!PyOrFilter_Check(result))
        raiseError(
                "deepCopy is expected to return an instance of a class derived from Filter");

    PFilter fil = PyOrange_AsFilter(result);
    Py_DECREF(result);
    return fil;
}

void TTransformValue_Python::transform(TValue &val)
{
    PyObject *args = Py_BuildValue("(N)", Value_FromValue(val));
    PyObject *result = callCallback((PyObject *) myWrapper, args);
    Py_DECREF(args);

    PVariable var;
    bool succ = convertFromPython(result, val, var);
    Py_DECREF(result);

    if (!succ)
        raiseError(
                "TransformValue.__call__'s result cannot be converted to a Value");
}

TMeasureAttribute_Python::TMeasureAttribute_Python() :
    TMeasureAttribute(TMeasureAttribute::DomainContingency, true, true)
{
}

float TMeasureAttribute_Python::callMeasure(PyObject *args)
{
    PyObject *res = callCallback((PyObject *) myWrapper, args);
    PyObject *resf = PyNumber_Float(res);
    Py_DECREF(res);

    if (!resf)
        raiseError("invalid result from __call__");

    float mres = (float) PyFloat_AsDouble(resf);
    Py_DECREF(resf);
    return mres;
}

float TMeasureAttribute_Python::operator ()(PDistribution d) const
{
    return (const_cast <TMeasureAttribute_Python *>(this))->callMeasure(Py_BuildValue("(N)", WrapOrange(d)));
}

float TMeasureAttribute_Python::operator ()(const TDiscDistribution &d) const
{
    PDiscDistribution nd = new TDiscDistribution(d);
    return (const_cast <TMeasureAttribute_Python *>(this))->callMeasure(Py_BuildValue("(N)", WrapOrange(nd)));
}

float TMeasureAttribute_Python::operator ()(const TContDistribution &d) const
{
    PContDistribution nd = new TContDistribution(d);
    return (const_cast <TMeasureAttribute_Python *>(this))->callMeasure(Py_BuildValue("(N)", WrapOrange(nd)));
}

float TMeasureAttribute_Python::operator()(PContingency cont, PDistribution classDistribution, PDistribution apriorClass)
{
    if (needs != Contingency_Class) {
        return TMeasureAttribute::operator()(cont, classDistribution, apriorClass);
    }

    return callMeasure(Py_BuildValue("(NNN)",
                    WrapOrange(cont),
                    WrapOrange(classDistribution),
                    WrapOrange(apriorClass)));
}

float TMeasureAttribute_Python::operator()(int attrNo, PDomainContingency dcont, PDistribution apriorClass)
{
    if (needs != DomainContingency) {
        return TMeasureAttribute::operator()(attrNo, dcont, apriorClass);
    }

    return callMeasure(Py_BuildValue("(iNN)", attrNo, WrapOrange(dcont), WrapOrange(apriorClass)));
}

float TMeasureAttribute_Python::operator()(int attrNo, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{
    if (needs != Generator) {
        return TMeasureAttribute::operator()(attrNo, gen, apriorClass, weightID);
    }

    return (const_cast <TMeasureAttribute_Python *>(this))->callMeasure(Py_BuildValue("iNNi", attrNo, WrapOrange(gen), WrapOrange(apriorClass), weightID));
}

float TMeasureAttribute_Python::operator()(PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{
    if (needs != Generator) {
        return TMeasureAttribute::operator()(var, gen, apriorClass, weightID);
    }

    return (const_cast <TMeasureAttribute_Python *>(this))->callMeasure(Py_BuildValue("NNNi", WrapOrange(var), WrapOrange(gen), WrapOrange(apriorClass), weightID));
}

void TMeasureAttribute_Python::thresholdFunction(TFloatFloatList &res,
        PVariable var, PExampleGenerator gen, PDistribution apriorClass,
        int weightID)
{
    if (!computesThresholds) {
        TMeasureAttribute::thresholdFunction(res, var, gen, apriorClass,
                weightID);
    }

    PyObject *args = Py_BuildValue("NNNi", WrapOrange(var), WrapOrange(gen),
            WrapOrange(apriorClass), weightID);
    PyObject *pyres = callMethod("thresholdFunction", (PyObject *) myWrapper,
            args);
    Py_DECREF(args);

    if (!PyList_Check(pyres)) {
        Py_DECREF(pyres);
        raiseError(
                "method 'thresholdFunction' should return a list of float tuples");
    }

    res.clear();
	const Py_ssize_t lsize = PyList_Size(pyres);
    res.reserve(lsize);
    for (Py_ssize_t i = 0; i < lsize; i++) {
        PyObject *litem = PyList_GetItem(pyres, i);
        PyObject *n1 = NULL, *n2 = NULL;
        if (!(PyTuple_Check(litem) && (PyTuple_Size(litem) == 2)
                && ((n1 = PyNumber_Float(PyTuple_GET_ITEM(litem, 0))) != NULL)
                && ((n2 = PyNumber_Float(PyTuple_GET_ITEM(litem, 1))) != NULL))) {
            Py_DECREF(pyres);
            Py_XDECREF(n1);
            Py_XDECREF(n2);
            raiseError(
                    "method 'thresholdFunction' should return a list of float tuples");
        }
        res.push_back(pair<float, float> (PyFloat_AsDouble(n1),
                PyFloat_AsDouble(n2)));
        Py_DECREF(n1);
        Py_DECREF(n2);
    }
    Py_DECREF(pyres);
}

PClassifier TLearner_Python::operator()(PExampleGenerator eg, const int &weight)
{
    if (!eg) {
        raiseError("invalid example generator");
    }

    PyObject *args = Py_BuildValue("(Ni)", WrapOrange(POrange(eg)), weight);
    PyObject *res = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrClassifier_Check(res)) {
        raiseError("__call__ is expected to return something derived from Classifier");
    }

    PClassifier clsf = PyOrange_AsClassifier(res);
    Py_DECREF(res);
    return clsf;
}

#include "vectortemplates.hpp"
#include "converts.hpp"
PAttributedFloatList TLogRegFitter_Python::operator()(PExampleGenerator eg, const int &weightID, PAttributedFloatList &beta_se, float &likelihood, int &status, PVariable &attribute)
{
    if (!eg) {
        raiseError("invalid example generator");
    }

    PyObject *args = Py_BuildValue("(Ni)", WrapOrange(POrange(eg)), weightID);
    PyObject *res = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyTuple_Check(res) || (PyTuple_Size(res)<2) || !PyInt_Check(PyTuple_GET_ITEM(res, 0))) {
        raiseError("invalid result from __call__");
    }

    status = (int)PyInt_AsLong(PyTuple_GET_ITEM(res, 0));
    if (status <= TLogRegFitter::Divergence) {
        if (PyTuple_Size(res) != 4) {
            raiseError("invalid result from __call__");
        }

        PFloatList beta = ListOfUnwrappedMethods<PAttributedFloatList, TAttributedFloatList, float>::P_FromArguments(PyTuple_GET_ITEM(res, 1));
        beta_se = ListOfUnwrappedMethods<PAttributedFloatList, TAttributedFloatList, float>::P_FromArguments(PyTuple_GET_ITEM(res, 2));
        Py_DECREF(res);
        if (!beta || !beta_se || !PyNumber_ToFloat(PyTuple_GET_ITEM(res, 3), likelihood)) {
            throw pyexception();
        }

        attribute = PVariable();
        return beta;
    }
    else {
        if (PyTuple_Size(res) != 2) {
            raiseError("invalid result from __call__");
        }

        if (!PyOrVariable_Check(PyTuple_GET_ITEM(res, 1)))
          raiseError("An instance of a class derived from Variable expected");

        attribute = PyOrange_AsVariable(PyTuple_GET_ITEM(res, 1));
        beta_se = PAttributedFloatList();
        return PAttributedFloatList();
    }
}

TValue TClassifier_Python::operator ()(const TExample &ex)
{
    PyObject *args = Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 0);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

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
{
    PyObject *args = Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 1);
    PyObject *result = callCallback((PyObject *) myWrapper, args);
    Py_DECREF(args);

    if (result == Py_None) {
        Py_DECREF(result);
        return PDistribution(classVar);
    }

    if (PyOrDistribution_Check(result)) {
        PDistribution dist = PyOrange_AsDistribution(result);
        Py_DECREF(result);
        return dist;
    }

    Py_XDECREF(result);
    raiseError("invalid result from __call__");
    return PDistribution();
}

void TClassifier_Python::predictionAndDistribution(const TExample &ex,
        TValue &val, PDistribution &dist)
{
    PyObject *args = Py_BuildValue("(Ni)", Example_FromExampleCopyRef(ex), 2);
    PyObject *result = callCallback((PyObject *) myWrapper, args);
    Py_DECREF(args);

    if (result == Py_None) {
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
    if (!PyArg_ParseTuple(result, "OO&", &obj1, cc_Distribution, &dist)
            || !convertFromPython(obj1, val, classVar)) {
        Py_XDECREF(result);
        raiseError("invalid result from __call__");
    }

    Py_DECREF(result);
}

PStringList PStringList_FromArguments(PyObject *arg);

PClassifier TTreeSplitConstructor_Python::operator()(
        PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,
        PExampleGenerator gen, const int &weightID, PDomainContingency dcont, PDistribution apriorClass, const vector<bool> &candidates, PClassifier nodeClassifier)

{
    if (!gen) {
        raiseError("invalid example generator");
    }

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

    PyObject *args = Py_BuildValue("(NiNNNN)", WrapOrange(gen), weightID,
            WrapOrange(dcont), WrapOrange(apriorClass), pycandidates, WrapOrange(
                    nodeClassifier));
    PyObject *res = callCallback((PyObject *) myWrapper, args);
    Py_DECREF(args);

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

    if (classifier && pydesc) {
        if (PyOrStringList_Check(pydesc)) {
            descriptions = PyOrange_AsStringList(pydesc);
        }
        else {
            descriptions = PStringList_FromArguments(pydesc);
            if (!descriptions) {
                Py_DECREF(res);
                throw pyexception();
            }
        }
    }
    else {
        descriptions = PStringList();
    }

    Py_DECREF(res);
    return classifier;
}

PExampleGeneratorList PExampleGeneratorList_FromArguments(PyObject *args);

PExampleGeneratorList TTreeExampleSplitter_Python::operator()(PTreeNode node, PExampleGenerator gen, const int &weightID, vector<int> &newWeights)
{   if (!gen) {
        raiseError("invalid example generator");
    }

    PyObject *args = Py_BuildValue("(NNi)", WrapOrange(node), WrapOrange(gen), weightID);
    PyObject *res = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (res == Py_None) {
        Py_DECREF(res);
        return PExampleGeneratorList();
    }

    PyObject *pygen;
    PyObject *pyweights = NULL;
    if (!PyArg_ParseTuple(res, "O|O", &pygen, &pyweights)) {
        raiseError("invalid result from __call__ (a list of list of examples and, optionally a list of weight ID's expected)");
    }

    PExampleGeneratorList eglist = PExampleGeneratorList_FromArguments(pygen);
    if (!eglist) {
        raiseError("invalid result from __call__ (a list of list of examples and, optionally a list of weight ID's expected)");
    }

    if (pyweights && (pyweights!=Py_None)) {
        if (!PyList_Check(pyweights) || (PyList_Size(pyweights)!=(Py_ssize_t)node->branches->size())) {
            raiseError("invalid result from __call__ (length of weight list should equal the number of branches)");
        }
        for (Py_ssize_t sz = 0, se = PyList_Size(pyweights); sz<se; sz++) {
            PyObject *li = PyList_GetItem(pyweights, sz);
            if (!PyInt_Check(li)) {
                raiseError("invalid weight list (int's expected).");
            }
            newWeights.push_back(int(PyInt_AsLong(li)));
        }
    }
    else {
        newWeights.clear();
    }

    Py_DECREF(res);

    return eglist;
}

bool TTreeStopCriteria_Python::operator()(PExampleGenerator gen, const int &weightID, PDomainContingency dcont)
{
    if (!gen) {
        raiseError("invalid example generator");
    }

    PyObject *args = Py_BuildValue("(NiN)", WrapOrange(gen), weightID, WrapOrange(dcont));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    bool res = bool(PyObject_IsTrue(result)!=0);
    Py_DECREF(result);
    return res;
}

int pt_DiscDistribution(PyObject *args, void *dist);

PTreeNode TTreeDescender_Python::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{
    PyObject *args = Py_BuildValue("(NN)", WrapOrange(node), Example_FromExampleCopyRef(ex));
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (result == Py_None) {
        Py_DECREF(result);
        distr->clear();
        return PTreeNode();
    }

    PTreeNode newnode;
    distr = PDiscDistribution();
    if (PyOrTreeNode_Check(result)) {
        newnode = PyOrange_AsTreeNode(result);
    }
    else if (!PyArg_ParseTuple(result, "O&|O&", cc_TreeNode, &newnode, pt_DiscDistribution, &distr)) {
        Py_DECREF(result);
        raiseError("invalid result from __call__");
    }

    Py_DECREF(result);
    return newnode;
}

bool TProgressCallback_Python::operator()(const float &f, POrange o)
{
    PyObject *args = Py_BuildValue("fN", f, WrapOrange(o));
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    bool res = PyObject_IsTrue(result) != 0;
    Py_DECREF(result);
    return res;
}

PImputer TImputerConstruct_Python::operator()(PExampleGenerator eg, const int &weight)
{
    if (!eg) {
        raiseError("invalid example generator");
    }

    PyObject *args = Py_BuildValue("(Ni)", WrapOrange(POrange(eg)), weight);
    PyObject *res = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrImputer_Check(res)) {
        raiseError("__call__ is expected to return something derived from Imputer");
    }

    PImputer imp = PyOrange_AsImputer(res);
    Py_DECREF(res);
    return imp;
}

TExample *TImputer_Python::operator()(TExample &example)
{
    PyObject *args = Py_BuildValue("(Ni)", Example_FromExampleCopyRef(example), 0);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrExample_Check(result)) {
        raiseError("__call__ is expected to return an instance of Example");
    }

    TExample *res = CLONE(TExample, PyExample_AS_Example(result));
    Py_DECREF(result);
    return res;
}

float TRuleEvaluator_Python::operator()(PRule rule, PExampleTable table, const int &weightID, const int &targetClass, PDistribution apriori)
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rule) {
        raiseError("invalid rule");
    }
    if (!apriori) {
        raiseError("invalid prior distribution");
    }

    PyObject *args = Py_BuildValue("(NNiiN)", WrapOrange(rule), WrapOrange(table), weightID, targetClass, WrapOrange(apriori));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyFloat_Check(result)) {
        raiseError("__call__ is expected to return a float value.");
    }
    float res = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
}


PEVDist TEVDistGetter_Python::operator()(const PRule rule, const int & parentLength, const int & rLength) const
{
    if (!rule) {
        raiseError("invalid rule");
    }

    PyObject *args = Py_BuildValue("(Nii)", WrapOrange(rule), parentLength, rLength);
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrEVDist_Check(result)) {
        raiseError("__call__ is expected to return an EVDist object.");
    }
    PEVDist res = PyOrange_AsEVDist(result);
    Py_DECREF(result);
    return res;
}

bool TRuleValidator_Python::operator()(PRule rule, PExampleTable table, const int &weightID, const int &targetClass, PDistribution apriori) const
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rule) {
        raiseError("invalid rule");
    }
    if (!apriori) {
        raiseError("invalid prior distribution");
    }

    PyObject *args = Py_BuildValue("(NNiiN)", WrapOrange(rule), WrapOrange(table), weightID, targetClass, WrapOrange(apriori));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyBool_Check(result)) {
        raiseError("__call__ is expected to return a Boolean value.");
    }
    bool res = bool(PyObject_IsTrue(result)!=0);
    Py_DECREF(result);
    return res;
}

PExampleTable TRuleCovererAndRemover_Python::operator()(PRule rule, PExampleTable table, const int &weightID, int &newWeightID, const int &targetClass) const
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rule) {
        raiseError("invalid rule");
    }

    PyObject *args = Py_BuildValue("(NNii)", WrapOrange(rule), WrapOrange(table), weightID, targetClass);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    PExampleGenerator gen;
    if (!PyArg_ParseTuple(result, "O&O&", pt_ExampleGenerator, &gen, pt_weightByGen(gen), &newWeightID)) {
        raiseError("__call__ is expected to return a tuple: (example table, new weight ID)");
    }
    Py_DECREF(result);
    return gen;
}

bool TRuleStoppingCriteria_Python::operator()(PRuleList ruleList, PRule rule, PExampleTable table, const int &weightID) const
{
    if (!ruleList) {
        raiseError("invalid rule list");
    }
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rule) {
        raiseError("invalid rule");
    }
    PyObject *args = Py_BuildValue("(NNNi)", WrapOrange(ruleList), WrapOrange(rule), WrapOrange(table), weightID);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyBool_Check(result)) {
        raiseError("__call__ is expected to return a Boolean value.");
    }
    bool res = bool(PyObject_IsTrue(result)!=0);
    Py_DECREF(result);
    return res;
}

bool TRuleDataStoppingCriteria_Python::operator()(PExampleTable table, const int &weightID, const int &targetClass) const
{
    if (!table) {
        raiseError("invalid example table");
    }

    PyObject *args = Py_BuildValue("(Nii)", WrapOrange(table), weightID, targetClass);
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyBool_Check(result)) {
        raiseError("__call__ is expected to return a Boolean value.");
    }
    bool res = bool(PyObject_IsTrue(result)!=0);
    Py_DECREF(result);
    return res;
}

PRule TRuleFinder_Python::operator ()(PExampleTable table, const int &weightID, const int &targetClass, PRuleList baseRules)
{
    if (!table) {
        raiseError("invalid example table");
    }

    PyObject *args = Py_BuildValue("(NiiN)", WrapOrange(table), weightID, targetClass, WrapOrange(baseRules));
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrRule_Check(result)) {
        raiseError("__call__ is expected to return a rule.");
    }
    PRule res = PyOrange_AsRule(result);
    Py_DECREF(result);
    return res;
}

PRuleList TRuleBeamRefiner_Python::operator ()(PRule rule, PExampleTable table, const int &weightID, const int &targetClass)
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rule) {
        raiseError("invalid rule");
    }

    PyObject *args = Py_BuildValue("(NNii)", WrapOrange(rule), WrapOrange(table), weightID, targetClass);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrRuleList_Check(result)) {
        raiseError("__call__ is expected to return a list of rules.");
    }
    PRuleList res = PyOrange_AsRuleList(result);
    Py_DECREF(result);
    return res;
}

PRuleList TRuleBeamInitializer_Python::operator ()(PExampleTable table, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator evaluator, PDistribution prior, PRule &bestRule)
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!evaluator) {
        raiseError("invalid evaluator function");
    }
    if (!prior) {
        raiseError("invalid prior distribution");
    }

    PyObject *args = Py_BuildValue("(NiiNNNN)", WrapOrange(table), weightID, targetClass, WrapOrange(baseRules), WrapOrange(evaluator), WrapOrange(prior), WrapOrange(bestRule));
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrRuleList_Check(result)) {
        raiseError("__call__ is expected to return a list of rules.");
    }
    PRule res = PyOrange_AsRuleList(result);
    Py_DECREF(result);
    return res;
}

PRuleList TRuleBeamCandidateSelector_Python::operator ()(PRuleList &existingRules, PExampleTable table, const int &weightID)
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!existingRules) {
        raiseError("invalid existing rules");
    }

    PyObject *args = Py_BuildValue("(NNi)", WrapOrange(existingRules), WrapOrange(table), weightID);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    PRuleList candidates;
    if (!PyArg_ParseTuple(result, "O&O&", cc_RuleList, &candidates, cc_RuleList, &existingRules)) {
        raiseError("__call__ is expected to return a tuple: (candidate rules, remaining rules)");
    }
    Py_DECREF(result);
    return candidates;
}

void TRuleBeamFilter_Python::operator ()(PRuleList &rules, PExampleTable table, const int &weightID)
{
    if (!table) {
        raiseError("invalid example table");
    }
    if (!rules) {
        raiseError("invalid existing rules");
    }

    PyObject *args = Py_BuildValue("(NNi)", WrapOrange(rules), WrapOrange(table), weightID);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (!PyOrRuleList_Check(result)) {
        raiseError("__call__ is expected to return a list of rules.");
    }
    rules = PyOrange_AsRuleList(result);
    Py_DECREF(result);
}

PRuleClassifier TRuleClassifierConstructor_Python::operator()(PRuleList rules, PExampleTable table, const int &weightID)
{
    if (!rules) {
        raiseError("invalid set of rules");
    }
    if (!table) {
        raiseError("invalid example table");
    }

    PyObject *args = Py_BuildValue("(NNi)", WrapOrange(rules), WrapOrange(table), weightID);
    PyObject *result = callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);

    if (result==Py_None) {
        Py_DECREF(result);
        return PRuleClassifier();
    }

    if (!PyOrRuleClassifier_Check(result)) {
        raiseError("__call__ is expected to return a rule classifier.");
    }
    PRuleClassifier res = PyOrange_AsRuleClassifier(result);
    Py_DECREF(result);
    return res;
}

float TKernelFunc_Python::operator ()(const TExample &e1, const TExample &e2) {
    PyObject *args=Py_BuildValue("(NN)", Example_FromExampleCopyRef(e1), Example_FromExampleCopyRef(e2));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);
    float res=PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
}

PExamplesDistance TExamplesDistanceConstructor_Python::operator ()(PExampleGenerator eg, const int &wei, PDomainDistributions dd, PDomainBasicAttrStat das) const
{
    PyObject *args=Py_BuildValue("(NiNN)", WrapOrange(eg), wei, WrapOrange(dd), WrapOrange(das));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);
    if (!PyOrExamplesDistance_Check(result))
      raiseError("ExamplesDistanceConstructor.__call__ must return an instance of ExamplesDistance");
    PExamplesDistance res = PyOrange_AsExamplesDistance(result);
    Py_DECREF(result);
    return res;
}

float TExamplesDistance_Python::operator()(const TExample &e1, const TExample &e2) const
{
    PyObject *args=Py_BuildValue("(NN)", Example_FromExampleCopyRef(e1), Example_FromExampleCopyRef(e2));
    PyObject *result=callCallback((PyObject *)myWrapper, args);
    Py_DECREF(args);
    float res=PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
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

 PyObject *args = Py_BuildValue("(NNNi)", WrapOrange(gen), boundList, freeList, weightID);
 PyObject *res = callCallback((PyObject *)myWrapper, args);
 Py_DECREF(args);

 if (!PyOrIM_Check(res))
 raiseError("invalid result from __call__");

 PIM im = PyOrange_AsIM(res);
 Py_DECREF(res);
 return im;
 }
 */
