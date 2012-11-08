/*
    This file is part of Orange.

    Copyright 1996-2011 Faculty of Computer and Information Science, University of Ljubljana
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

#include <iostream>
#include <sstream>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "liblinear_interface.ppp"

#define Malloc(type,n) (type *) malloc((n)*sizeof(type))

// Defined in linear.cpp. If a new solver is added this should be updated.

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL", NULL
};

/*
 *The folowing load save functions are used for orange pickling
 */

/*
 * Save the model to an std::ostream. This is a modified `save_model` function
 * from `linear.cpp` in LIBLINEAR package.
 */
int linear_save_model_alt(ostream &stream, struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int nr_classifier;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_classifier=1;
	else
		nr_classifier=model_->nr_class;

	stream.precision(17);

	stream << "solver_type " << solver_type_table[param.solver_type] << endl;
	stream << "nr_class " << model_->nr_class << endl;
	stream << "label";
	for(i=0; i<model_->nr_class; i++)
		stream << " " << model_->label[i];
	stream << endl;

	stream << "nr_feature " << nr_feature << endl;

	stream << "bias " << model_->bias << endl;

	stream << "w" << endl;
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			stream << model_->w[i*nr_classifier+j] << " ";
		stream << endl;
	}

	if (stream.good())
		return 0;
	else
		return -1;
}

/*
 * Save linear model into a std::string.
 */
int linear_save_model_alt(string &buffer, struct model *model_)
{
	std::ostringstream strstream;
	int ret = linear_save_model_alt(strstream, model_);
	buffer = strstream.rdbuf()->str();
	return ret;
}

/*
 * Load a linear model from std::istream. This is a modified `load_model`
 * function from `linear.cpp` in LIBLINEAR package.
 */
struct model *linear_load_model_alt(istream &stream)
{
	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	stream.width(80);
	while(stream.good())
	{
		stream >> cmd;
		if(strcmp(cmd, "solver_type")==0)
		{
			stream >> cmd;
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			stream >> nr_class;
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			stream >> nr_feature;
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			stream >> bias;
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int, nr_class);
			for(int i=0;i<nr_class;i++)
				stream >> model_->label[i];
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_->label);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int nr_classifier;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model_->w=Malloc(double, n*nr_classifier);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			stream >> model_->w[i*nr_classifier+j];
	}
	if (stream.fail())
		return NULL;
	else
		return model_;
}

/*
 * Load a linear model from a std:string.
 */
struct model *linear_load_model_alt(string &buffer)
{
	std::istringstream str_stream(buffer);
	str_stream.exceptions(ios::failbit | ios::badbit);
	return linear_load_model_alt(str_stream);
}

struct NodeSort{
	bool operator () (const feature_node &lhs, const feature_node &rhs){
		return lhs.index < rhs.index;
	}
};

int countFeatures(const TExample &ex, bool includeMeta, bool includeRegular){
	int count = 1;
	if (includeRegular)
		for (TExample::iterator i=ex.begin(); i!=ex.end(); i++)
			if ((i->varType==TValue::INTVAR || i->varType==TValue::FLOATVAR) && i->isRegular() && i!=&ex.getClass())
				count++;
	if (includeMeta)
		for (TMetaValues::const_iterator i=ex.meta.begin(); i!=ex.meta.end(); i++)
			if ((i->second.varType==TValue::INTVAR || i->second.varType==TValue::FLOATVAR) && i->second.isRegular())
				count++;
	return count;
}

feature_node *feature_nodeFromExample(const TExample &ex, double bias){
	int n_nodes = countFeatures(ex, false, true);

	if (bias >= 0.0)
	    n_nodes++;

	feature_node *nodes = new feature_node[n_nodes];
	feature_node *ptr = nodes;

	int index = 1;

    for (TExample::iterator i=ex.begin(); i!=ex.end(); i++)
        if (i!=&ex.getClass()){
            if ((i->varType==TValue::INTVAR || (i->varType==TValue::FLOATVAR && (*i==*i))) && i->isRegular()){
                if (i->varType==TValue::INTVAR)
                    ptr->value = (int) *i;
                else
                    ptr->value = (float) *i;
                ptr->index = index;
                ptr++;
            }
            index++;
        }

	if (bias >= 0.0)
	{
	    ptr->value = bias;
	    ptr->index = index;
	    ptr++;
	}

	ptr->index = -1;
	return nodes;
}

problem *problemFromExamples(PExampleGenerator examples, double bias){
	problem *prob = new problem;
	prob->l = examples->numberOfExamples();
	prob->n = examples->domain->attributes->size();

	if (bias >= 0)
	    prob->n++;

	prob->x = new feature_node* [prob->l];
	prob->y = new int [prob->l];
	prob->bias = bias;
	feature_node **ptrX = prob->x;
	int *ptrY = prob->y;
	PEITERATE(iter, examples){
		*ptrX = feature_nodeFromExample(*iter, bias);
		*ptrY = (int) (*iter).getClass();
		ptrX++;
		ptrY++;
	}
	return prob;
}

void destroy_problem(problem *prob){
	for (int i=0; i<prob->l; i++)
		delete[] prob->x[i];
	delete[] prob->x;
	delete[] prob->y;
}

static void dont_print_string(const char *s){}

TLinearLearner::TLinearLearner(){
	solver_type = L2R_LR;
	eps = 0.01f;
	C = 1;
	bias = -1.0;
	set_print_string_function(&dont_print_string);
}

PClassifier TLinearLearner::operator()(PExampleGenerator examples, const int &weight){
	parameter *param = new parameter;
	param->solver_type = solver_type;
	param->eps = eps;
	param->C = C;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;

	PVariable classVar = examples->domain->classVar;
	if (!classVar)
	    raiseError("classVar expected");
	if (classVar->varType != TValue::INTVAR)
	    raiseError("Discrete class expected");

	problem *prob = problemFromExamples(examples, bias);

	const char * error_msg = check_parameter(prob, param);
	if (error_msg){
		delete param;
		destroy_problem(prob);
		raiseError("LIBLINEAR error: %s", error_msg);
	}
	/* The solvers in liblinear use rand() function.
	 * To make the results reproducible we set the seed from the data table's
	 * crc
	 */
	PExampleTable extable(examples);
	srand(extable->checkSum(false));

	model *model = train(prob, param);
	destroy_problem(prob);

	return PClassifier(mlnew TLinearClassifier(examples->domain->classVar, examples, model));
}

TLinearClassifier::TLinearClassifier(const PVariable &var, PExampleTable _examples, struct model *_model){
	classVar = var;
	domain = _examples->domain;
	examples = _examples;
	linmodel = _model;
	bias = _model->bias;
	dbias = _model->bias;

	computesProbabilities = check_probability_model(linmodel) != 0;
	// Number of class values
	int nr_values = this->get_nr_values();

	/* Number of liblinear classifiers (if some class values are missing
	 * from the training set they are not present in the liblinear model).
	 */
	int nr_classifier = linmodel->nr_class;
	if (linmodel->nr_class == 2 && linmodel->param.solver_type != MCSVM_CS)
	{
	    nr_classifier = 1;
	}

	// Number of weight vectors exposed in orange.
	int nr_orange_weights = nr_values;
	if (nr_values == 2 && linmodel->param.solver_type != MCSVM_CS)
	{
	    nr_orange_weights = 1;
	}

	int nr_feature = linmodel->nr_feature;

	if (linmodel->bias >= 0.0)
	{
	    nr_feature++;
	}

	int* labels = new int[linmodel->nr_class];
	get_labels(linmodel, labels);

	// Initialize nr_orange_weights vectors
    weights = mlnew TFloatListList(nr_orange_weights);
    for (int i = 0; i < nr_orange_weights; i++)
    {
        weights->at(i) = mlnew TFloatList(nr_feature, 0.0f);
    }

    if (nr_classifier > 1)
    {
        for (int i = 0; i < nr_classifier; i++)
        {
            for (int j = 0; j < nr_feature; j++)
            {
                weights->at(labels[i])->at(j) = \
                        linmodel->w[j*nr_classifier + i];
            }
        }
}
    else
    {
        /* If the order of the liblinear internaly stored classes
         * is different from the order of orange's class values,
         * we reverse the weight vector.
         */
        float factor = (labels[0] == 0)? 1.0f : -1.0f;

        for (int j = 0; j < nr_feature; j++)
        {
            if (nr_orange_weights > 1)
            {
               /* There are more than 2 orange class values. This means
                * there were no instances for one or more classed in the training
                * data set.
                */
                weights->at(labels[0])->at(j) = linmodel->w[j];
                weights->at(labels[1])->at(j) = - linmodel->w[j];
            }
            else
            {
                weights->at(0)->at(j) = factor * linmodel->w[j];
            }
        }
    }
    delete[] labels;
}

TLinearClassifier::~TLinearClassifier(){
	if (linmodel)
		free_and_destroy_model(&linmodel);
}

/* Return the number of discrete class values, or raise an error
 * if the class_var is not discrete.
 */
int TLinearClassifier::get_nr_values()
{
    int nr_values = 0;
    TEnumVariable * enum_var = NULL;
    enum_var = dynamic_cast<TEnumVariable*>(classVar.getUnwrappedPtr());
    if (enum_var)
    {
        nr_values = enum_var->noOfValues();
    }
    else
    {
        raiseError("Discrete class expected.");
    }
    return nr_values;
}

PDistribution TLinearClassifier::classDistribution(const TExample &example){
    TExample new_example(domain, example);
	int numClass = get_nr_class(linmodel);
	map<int, int> indexMap;
	feature_node *x = feature_nodeFromExample(new_example, bias);

	int *labels = new int [numClass];
	get_labels(linmodel, labels);

	double *prob_est = new double [numClass];
	predict_probability(linmodel, x, prob_est);

	PDistribution dist = TDistribution::create(classVar);
	for (int i=0; i<numClass; i++)
		dist->setint(labels[i], prob_est[i]);

	delete[] x;
	delete[] labels;
	delete[] prob_est;
	return dist;
}

TValue TLinearClassifier::operator () (const TExample &example){
    TExample new_example(domain, example);
	int numClass = get_nr_class(linmodel);
	map<int, int> indexMap;
	feature_node *x = feature_nodeFromExample(new_example, bias);

	int predict_label = predict(linmodel, x);
	delete[] x;
	return TValue(predict_label);
}

