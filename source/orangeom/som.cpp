#include "ppp/som.ppp"

#include <limits>
using namespace std;

DEFINE_TOrangeVector_classDescription(PSOMNode, "TSOMNodeList", true, ORANGEOM_API)

struct entries* examplesToEntries(PExampleTable ex){
	struct entries *data;
	data=alloc_entries();
	data->fi=NULL;
    if(!data)
        raiseError("SOM_PAK alloc_entries error");
    data->dimension=ex->domain->attributes->size();
    data->flags.totlen_known=1;
 
    struct data_entry *ent;
    PEITERATE(iter, ex){
        ent=init_entry(data, NULL);
        if(!ent)
            raiseError("SOM_PAK init_entry error");
		if(!data->dentries){
            data->dentries=ent;
            data->current=ent;
        } else {
            data->current->next=ent;
            data->current=ent;
        }
        data->num_entries++;
        for(int i=0; i<data->dimension; ++i){
            ent->points[i]=float((*iter)[i]);
            if(ent->points[i]==numeric_limits<int>::max() || ent->points[i]==numeric_limits<float>::signaling_NaN())
                ent->mask=set_mask(ent->mask, data->dimension, i);
        }
    }
	return data;
}
float TSOMNode::getDistance(const TExample &example){
    float dist=0.0f;
    PExample ex=mlnew TExample(transformedDomain, example);
    for(int i=0;i<vector->size();i++){
        if(float(ex->operator[](i))!=numeric_limits<float>::signaling_NaN())
            dist+=pow(vector->at(i)-float(ex->operator[](i)),2.0f);
    }
    return pow(dist,0.5f);   
}

const int TSOMLearner::RectangularTopology=TOPOL_RECT;
const int TSOMLearner::HexagonalTopology=TOPOL_HEXA;
const int TSOMLearner::BubbleNeighborhood=NEIGH_BUBBLE;
const int TSOMLearner::GaussianNeighborhood=NEIGH_GAUSSIAN;
const int TSOMLearner::LinearFunction=ALPHA_LINEAR;
const int TSOMLearner::InverseFunction=ALPHA_INVERSE_T;

TSOMLearner::TSOMLearner(){
    xDim=yDim=10;
    steps=2;
    topology=HexagonalTopology;
    iterations=mlnew TIntList();
    iterations->push_back(1000);
    iterations->push_back(10000); 
    neighborhood=BubbleNeighborhood;
    radius=mlnew TIntList();
    radius->push_back(10);
    radius->push_back(5);
    alphaType=LinearFunction;
    alpha=mlnew TFloatList();
    alpha->push_back(0.05f);
    alpha->push_back(0.03f);
    domainContinuizer=mlnew TDomainContinuizer();
    domainContinuizer->classTreatment=3;
    domainContinuizer->multinomialTreatment=6;
    domainContinuizer->continuousTreatment=2;
}

PClassifier TSOMLearner::operator() (PExampleGenerator examples, const int &a){
    transformedDomain=domainContinuizer->call(examples,-1,-1);
    PExampleTable ex=mlnew TExampleTable(transformedDomain, examples);
    transformedDomain=ex->domain;

	PClassifier wclassifier;
	TSOMClassifier *classifier = NULL;

    struct entries *data=NULL, *codes=NULL;
	struct data_entry *ent;

	try{
    data=examplesToEntries(ex);
    struct teach_params params;
    struct  typelist *type;
    type=get_type_by_id(alpha_list, ALPHA_LINEAR);
    params.alpha_type=type->id;
    params.alpha_func=(ALPHA_FUNC*)type->data;
    codes=randinit_codes(data, topology, neighborhood, xDim, yDim);
	codes->fi=NULL;
    set_teach_params(&params, codes, NULL, 0);
    set_som_params(&params);
    params.data=data;
    
    for(int step=0; step<steps; step++){
        params.length=iterations->at(step);
        params.alpha=alpha->at(step);
        params.radius=radius->at(step);
        codes=som_training(&params);
        if(!codes)
            raiseError("SOM_PAK som_training error");
    }
    set_teach_params(&params, codes, data, 0);
    set_som_params(&params);
    float qerror=find_qerror2(&params);
    
	if(examples->domain->classVar){
        classifier=mlnew TSOMClassifier();
        wclassifier = classifier;
		classifier->classVar=examples->domain->classVar;
	}
    else {
        classifier=mlnew TSOMMap();
        wclassifier = classifier;
    }

    PSOMNodeList nodes=mlnew TSOMNodeList(xDim*yDim);
    int i=0;
    for(ent=codes->dentries; ent!=NULL; ent=ent->next, i++){
        nodes->at(i)=mlnew TSOMNode();
        nodes->at(i)->vector=mlnew TFloatList(codes->dimension);
		nodes->at(i)->referenceExample=mlnew TExample(transformedDomain, ex->at(0));
		nodes->at(i)->referenceExample->setClass(TValue(TValue::NONE, valueDK));
		for(int j=0; j<codes->dimension; j++){
            nodes->at(i)->vector->at(j)=ent->points[j];
			nodes->at(i)->referenceExample->operator[](j).floatV=ent->points[j];
		}
        nodes->at(i)->x=i%xDim;
        nodes->at(i)->y=i/xDim;
        nodes->at(i)->transformedDomain=transformedDomain;
    }
    classifier->nodes=nodes;
    classifier->xDim=xDim;
	classifier->examples=examples;
    classifier->yDim=yDim;
    classifier->topology=topology;          
    classifier->trainingError=qerror;
	classifier->transformedDomain=transformedDomain;
    
    PEITERATE(iter, examples){
        PSOMNode node=classifier->getWinner(*iter);
        if(!node->examples)
            node->examples=mlnew TExampleTable(examples->domain);
        node->examples->addExample(*iter);
    }
   
    PLearner learner=mlnew TMajorityLearner();

    PITERATE(TSOMNodeList, nodeiter, classifier->nodes){
        if((*nodeiter)->examples && classifier->classVar)
            (*nodeiter)->classifier=learner->call((*nodeiter)->examples, a);
		else if (classifier->classVar){
            (*nodeiter)->classifier=learner->call(examples, a);
			(*nodeiter)->examples=mlnew TExampleTable(examples->domain);
		} else 
			(*nodeiter)->examples=mlnew TExampleTable(examples->domain);
    }

	classifier->som_pak_codes=codes;
	classifier->som_pak_data=data;
	classifier->params=params;
	}catch(...){
		if(codes)
			close_entries(codes);
		if(data)
			close_entries(data);
		throw;
	}
    return wclassifier; 
}
/*
TValue TSOMClassifier::operator()(const TExample &example){
    PSOMNode node=getWinner(example);
    if(node->classifier)
        return node->classifier->call(example);
    return classifier->call(example);
}
*/
PDistribution TSOMClassifier::classDistribution(const TExample &example){
    PSOMNode node=getWinner(example);
    if(node->classifier)
        return node->classifier->classDistribution(example);
    return classifier->classDistribution(example);
}
/*
void TSOMClassifier::predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist){
    PSOMNode node=getWinner(example);
    if(node->classifier)
        node->classifier->predictionAndDistribution(example, value, dist);
    else
        classifier->predictionAndDistribution(example, value, dist);
}*/

float TSOMClassifier::getError(PExampleGenerator examples){
    PExampleTable ex=mlnew TExampleTable(transformedDomain, examples);
	struct entries *data=examplesToEntries(ex);
	params.data=data;
	float error=find_qerror2(&params);
	close_entries(params.data);
	params.data=som_pak_data;
	return error;
}
    
PSOMNode TSOMClassifier::getWinner(const TExample &example){
    PSOMNode node=nodes->at(0);
    float min=numeric_limits<float>::max();
    PITERATE(TSOMNodeList, iter, nodes){
        float dist=(*iter)->getDistance(example);
        if(dist<min){
            node=*iter;
            min=dist;
        }
    }
    return node;
}

TSOMClassifier::~TSOMClassifier(){
	if(som_pak_codes)
		close_entries(som_pak_codes);
	if(som_pak_data)
		close_entries(som_pak_data);
}



/************ PYTHON INTERFACE **************/

#include "externs.px"
#include "orange_api.hpp"

C_CALL(SOMLearner, Learner, "([examples[, weight=]]) -/-> Classifier")
C_NAMED(SOMClassifier, Classifier, " ")
C_NAMED(SOMMap, Orange, " ")
C_NAMED(SOMNode, Orange, " ")

PyObject *SOMNode_getDistance(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->float")
{
    PyTRY
    PExample ex;
    if(!PyArg_ParseTuple(args, "O&:getDistance", cc_Example, &ex))
        return NULL;
    float res=SELF_AS(TSOMNode).getDistance(ex.getReference());
    return Py_BuildValue("f", res);
    PyCATCH
}

PyObject *SOMClassifier_getWinner(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->SOMNode")
{
    PyTRY
    PExample ex;
    if(!PyArg_ParseTuple(args, "O&:getWinner", cc_Example, &ex))
        return NULL;
    return WrapOrange(SELF_AS(TSOMClassifier).getWinner(ex.getReference()));
    PyCATCH
}

PyObject *SOMClassifier_getError(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(examples)->float")
{
    PyTRY
    PExampleGenerator egen;
    if(!PyArg_ParseTuple(args, "O&", cc_ExampleGenerator, &egen))
        return NULL;
    float res=SELF_AS(TSOMClassifier).getError(egen);
    return Py_BuildValue("f", res);
    PyCATCH
}

PyObject *SOMMap_getWinner(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(example)->SOMNode")
{
    PyTRY
    PExample ex;
    if(!PyArg_ParseTuple(args, "O&:getWinner", cc_Example, &ex))
        return NULL;
    return WrapOrange(SELF_AS(TSOMMap).getWinner(ex.getReference()));
    PyCATCH
}

#include "orvector.hpp"
#include "vectortemplates.hpp"

extern ORANGEOM_API TOrangeType PyOrSOMNode_Type;

PSOMNodeList PSOMNodeList_FromArguments(PyObject *arg) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::P_FromArguments(arg); }
PyObject *SOMNodeList_FromArguments(PyTypeObject *type, PyObject *arg) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_FromArguments(type, arg); }
PyObject *SOMNodeList_new(PyTypeObject *type, PyObject *arg, PyObject *kwds) BASED_ON(Orange, "(<list of SOMNode>)") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_new(type, arg, kwds); }
PyObject *SOMNodeList_getitem_sq(TPyOrange *self, int index) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_getitem(self, index); }
int       SOMNodeList_setitem_sq(TPyOrange *self, int index, PyObject *item) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_setitem(self, index, item); }
PyObject *SOMNodeList_getslice(TPyOrange *self, int start, int stop) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_getslice(self, start, stop); }
int       SOMNodeList_setslice(TPyOrange *self, int start, int stop, PyObject *item) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_setslice(self, start, stop, item); }
int       SOMNodeList_len_sq(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_len(self); }
PyObject *SOMNodeList_richcmp(TPyOrange *self, PyObject *object, int op) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_richcmp(self, object, op); }
PyObject *SOMNodeList_concat(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_concat(self, obj); }
PyObject *SOMNodeList_repeat(TPyOrange *self, int times) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_repeat(self, times); }
PyObject *SOMNodeList_str(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_str(self); }
PyObject *SOMNodeList_repr(TPyOrange *self) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_str(self); }
int       SOMNodeList_contains(TPyOrange *self, PyObject *obj) { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_contains(self, obj); }
PyObject *SOMNodeList_append(TPyOrange *self, PyObject *item) PYARGS(METH_O, "(SOMNode) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_append(self, item); }
PyObject *SOMNodeList_count(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> int") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_count(self, obj); }
PyObject *SOMNodeList_filter(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([filter-function]) -> SOMNodeList") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_filter(self, args); }
PyObject *SOMNodeList_index(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> int") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_index(self, obj); }
PyObject *SOMNodeList_insert(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(index, item) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_insert(self, args); }
PyObject *SOMNodeList_native(TPyOrange *self) PYARGS(METH_NOARGS, "() -> list") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_native(self); }
PyObject *SOMNodeList_pop(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "() -> SOMNode") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_pop(self, args); }
PyObject *SOMNodeList_remove(TPyOrange *self, PyObject *obj) PYARGS(METH_O, "(SOMNode) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_remove(self, obj); }
PyObject *SOMNodeList_reverse(TPyOrange *self) PYARGS(METH_NOARGS, "() -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_reverse(self); }
PyObject *SOMNodeList_sort(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "([cmp-func]) -> None") { return ListOfWrappedMethods<PSOMNodeList, TSOMNodeList, PSOMNode, &PyOrSOMNode_Type>::_sort(self, args); }

PYCLASSCONSTANT_INT(SOMLearner, HexagonalTopology, TSOMLearner::HexagonalTopology)
PYCLASSCONSTANT_INT(SOMLearner, RectangularTopology, TSOMLearner::RectangularTopology)
PYCLASSCONSTANT_INT(SOMLearner, BubbleNeighborhood, TSOMLearner::BubbleNeighborhood)
PYCLASSCONSTANT_INT(SOMLearner, GaussianNeighborhood, TSOMLearner::GaussianNeighborhood)
PYCLASSCONSTANT_INT(SOMLearner, LinearFunction, TSOMLearner::LinearFunction)
PYCLASSCONSTANT_INT(SOMLearner, InverseFunction, TSOMLearner::InverseFunction)


#include "som.px"