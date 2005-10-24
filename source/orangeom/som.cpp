#include "ppp/som.ppp"

#include <limits>
#include <iostream>
using namespace std;

DEFINE_TOrangeVector_classDescription(PSOMNode, "TSOMNodeList", true, ORANGEOM_API)

struct entries* examplesToEntries(PExampleTable ex){
	struct entries *data;
	data=alloc_entries();
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
    neighborhood=mlnew TIntList();
    neighborhood->push_back(BubbleNeighborhood);
    neighborhood->push_back(BubbleNeighborhood);
    radius=mlnew TIntList();
    radius->push_back(10);
    radius->push_back(5);
    alphaType=mlnew TIntList();
    alphaType->push_back(LinearFunction);
    alphaType->push_back(LinearFunction);
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
    struct entries *data, *codes;
	struct data_entry *ent;

    data=examplesToEntries(ex);
    
    struct teach_params params;
    struct  typelist *type;
    type=get_type_by_id(alpha_list, ALPHA_LINEAR);
    params.alpha_type=type->id;
    params.alpha_func=(ALPHA_FUNC*)type->data;
    codes=randinit_codes(data, topology, neighborhood, xDim, yDim);
    set_teach_params(&params, codes, NULL, 0);
    set_som_params(&params);
    params.data=data;
    
    for(int step=0; step<steps; step++){
        cout <<"step:"<<step<<endl;
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
    
    PSOMClassifier classifier;
	if(examples->domain->classVar){
        classifier=mlnew TSOMClassifier();
		classifier->classVar=examples->domain->classVar;
	}
    else
        classifier=mlnew TSOMMap();
    PSOMNodeList nodes=mlnew TSOMNodeList(xDim*yDim);
    int i=0;
    for(ent=codes->dentries; ent!=NULL; ent=ent->next, i++){
        nodes->at(i)=mlnew TSOMNode();
        nodes->at(i)->vector=mlnew TFloatList(codes->dimension);
        for(int j=0; j<codes->dimension; j++)
            nodes->at(i)->vector->at(j)=ent->points[j];
        nodes->at(i)->x=i/yDim;
        nodes->at(i)->y=i%xDim;
        nodes->at(i)->transformedDomain=transformedDomain;
    }
    classifier->nodes=nodes;
    classifier->xDim=xDim;
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
   
    PMajorityLearner learner=mlnew TMajorityLearner();

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

    return classifier; 
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
    PSOMNode node;
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

