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

#include "libsvm_interface.ppp"

#include "slist.hpp"

// Defined in svm.cpp. If new svm or kernel types are added this should be updated.

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/*
 * Save load functions for use with orange pickling.
 * They are a copy of the standard libSVM save-load functions
 * except that they read/write from/to std::iostream objects.
 */

int svm_save_model_alt(std::ostream& stream, const svm_model *model){
	const svm_parameter& param = model->param;
	stream.precision(17);

	stream << "svm_type " << svm_type_table[param.svm_type] << endl;
	stream << "kernel_type " << kernel_type_table[param.kernel_type] << endl;

	if(param.kernel_type == POLY)
		stream << "degree " << param.degree << endl;

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		stream << "gamma " << param.gamma << endl;

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		stream << "coef0 " << param.coef0 << endl;

	int nr_class = model->nr_class;
	int l = model->l;
	stream << "nr_class " << nr_class << endl;
	stream << "total_sv " << l << endl;
	{
		stream << "rho";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			stream << " " << model->rho[i];
		stream << endl;
	}

	if(model->label)
	{
		stream << "label";
		for(int i=0;i<nr_class;i++)
			stream << " " << model->label[i];
		stream << endl;
	}

	if(model->probA) // regression has probA only
	{
		stream << "probA";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			stream << " " << model->probA[i];
		stream << endl;
	}
	if(model->probB)
	{
		stream << "probB";
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			stream << " " << model->probB[i];
		stream << endl;
	}

	if(model->nSV)
	{
		stream << "nr_sv";
		for(int i=0;i<nr_class;i++)
			stream << " " << model->nSV[i];
		stream << endl;
	}

	stream << "SV" << endl;
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			stream << sv_coef[j][i] << " ";

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			stream << (int)(p->value) << " ";
		else
			while(p->index != -1)
			{
				stream << (int)(p->index) << ":" << p->value << " ";
				p++;
			}
		stream << endl;
	}

	if (!stream.fail())
		return 0;
	else
		return 1;
}

int svm_save_model_alt(std::string& buffer, const svm_model *model){
	std::ostringstream strstream;
	int ret = svm_save_model_alt(strstream, model);
	buffer = strstream.rdbuf()->str();
	return ret;
}


#include <algorithm>

svm_model *svm_load_model_alt(std::istream& stream)
{
	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

	char cmd[81];
	stream.width(80);
	while (stream.good())
	{
		stream >> cmd;

		if(strcmp(cmd, "svm_type") == 0)
		{
			stream >> cmd;
			int i;
			for(i=0; svm_type_table[i]; i++)
			{
				if(strcmp(cmd, svm_type_table[i]) == 0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr, "unknown svm type.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd, "kernel_type") == 0)
		{
			stream >> cmd;
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i], cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			stream >> param.degree;
		else if(strcmp(cmd,"gamma")==0)
			stream >> param.gamma;
		else if(strcmp(cmd,"coef0")==0)
			stream >> param.coef0;
		else if(strcmp(cmd,"nr_class")==0)
			stream >> model->nr_class;
		else if(strcmp(cmd,"total_sv")==0)
			stream >> model->l;
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				stream >> model->rho[i];
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				stream >> model->label[i];
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				stream >> model->probA[i];
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				stream >> model->probB[i];
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				stream >> model->nSV[i];
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = stream.get();
				if(stream.eof() || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}
	if (stream.fail()){
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;

	}

	// read sv_coef and SV

	int elements = 0;
	long pos = stream.tellg();

	char *p,*endptr,*idx,*val;
	string str_line;
	while (!stream.eof() && !stream.fail())
	{
		getline(stream, str_line);
		elements += std::count(str_line.begin(), str_line.end(), ':');
	}

	elements += model->l;

	stream.clear();
	stream.seekg(pos, ios::beg);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	char *line;
	for(i=0;i<l;i++)
	{
		getline(stream, str_line);
		if (str_line.size() == 0)
			continue;

		line = (char *) Malloc(char, str_line.size() + 1);
		// Copy the line for strtok.
		strcpy(line, str_line.c_str());

		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
		free(line);
	}

	if (stream.fail())
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

svm_model *svm_load_model_alt(std::string& stream)
{
	std::istringstream strstream(stream);
	return svm_load_model_alt(strstream);
}

svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0){
	if(type==0){
		int index=1;
		for(TExample::iterator i=ex.begin(); i!=ex.end(); i++){
			if(i->isRegular() && i!=&ex.getClass()){
				if(i->varType==TValue::FLOATVAR)
					node->value=float(*i);
				else
					node->value=int(*i);
				node->index=index++;
				if(node->value==numeric_limits<float>::signaling_NaN() ||
					node->value==numeric_limits<int>::max())
					node--;
				node++;
			}
		}
	}
    if(type == 1){ /*one dummy attr so we can pickle the classifier and keep the SV index in the training table*/
        node->index=1;
        node->value=last;
        node++;
    }
	//cout<<(node-1)->index<<endl<<(node-2)->index<<endl;
	node->index=-1;
	node->value=last;
	node++;
	return node;
}

class SVM_NodeSort{
public:
	bool operator() (const svm_node &lhs, const svm_node &rhs){
		return lhs.index < rhs.index;
	}
};

svm_node* example_to_svm_sparse(const TExample &ex, svm_node* node, float last=0.0, bool useNonMeta=false){
	svm_node *first=node;
	int j=1;
	int index=1;
	if(useNonMeta)
		for(TExample::iterator i=ex.begin(); i!=ex.end(); i++){
			if(i->isRegular() && i!=&ex.getClass()){
				if(i->varType==TValue::FLOATVAR)
					node->value=float(*i);
				else
					node->value=int(*i);
				node->index=index;
				if(node->value==numeric_limits<float>::signaling_NaN() ||
					node->value==numeric_limits<int>::max())
					node--;
				node++;
			}
			index++;
		}
	for(TMetaValues::const_iterator i=ex.meta.begin(); i!=ex.meta.end();i++,j++){
		if(i->second.isRegular()){
			if(i->second.varType==TValue::FLOATVAR)
				node->value=float(i->second);
			else
				node->value=int(i->second);
			node->index=index-i->first;
			if(node->value==numeric_limits<float>::signaling_NaN() ||
				node->value==numeric_limits<int>::max())
				node--;
			node++;
		}
	}
	sort(first, node, SVM_NodeSort());
	//cout<<first->index<<endl<<(first+1)->index<<endl;
	node->index=-1;
	node->value=last;
	node++;
	return node;
}

/*
 * Precompute Gram matrix row for ex.
 * Used for prediction when using the PRECOMPUTED kernel.
 */
svm_node* example_to_svm_precomputed(const TExample &ex, PExampleGenerator examples, PKernelFunc kernel, svm_node* node){
	node->index = 0;
	node->value = 0.0;
	node++;
	int k = 0;
	PEITERATE(iter, examples){
		node->index = ++k;
		node->value = kernel.getReference()(*iter, ex);
		node++;
	}
	node->index = -1; // sentry
	node++;
	return node;
}

int getNumOfElements(const TExample &ex, bool meta=false, bool useNonMeta=false){
	if(!meta)
		return std::max(ex.domain->attributes->size()+1, 2);
	else{
		int count=1; //we need one to indicate the end of a sequence
		if(useNonMeta)
			count+=ex.domain->attributes->size();
		for(TMetaValues::const_iterator i=ex.meta.begin(); i!=ex.meta.end();i++)
			if(i->second.isRegular())
				count++;
		return std::max(count,2);
	}
}

int getNumOfElements(PExampleGenerator &examples, bool meta=false, bool useNonMeta=false){
	if(!meta)
		return getNumOfElements(*(examples->begin()), meta)*examples->numberOfExamples();
	else{
		int count=0;
		for(TExampleGenerator::iterator ex(examples->begin()); ex!=examples->end(); ++ex){
			count+=getNumOfElements(*ex, meta, useNonMeta);
		}
		return count;
	}
}

#include "symmatrix.hpp"
svm_node* init_precomputed_problem(svm_problem &problem, PExampleTable examples, TKernelFunc &kernel){
	int n_examples = examples->numberOfExamples();
	int i,j;
	PSymMatrix matrix = mlnew TSymMatrix(n_examples, 0.0);
	for (i = 0; i < n_examples; i++)
		for (j = 0; j <= i; j++){
			matrix->getref(i, j) = kernel(examples->at(i), examples->at(j));
//			cout << i << " " << j << " " << matrix->getitem(i, j) << endl;
		}
	svm_node *x_space = Malloc(svm_node, n_examples * (n_examples + 2));
	svm_node *node = x_space;

	problem.l = n_examples;
	problem.x = Malloc(svm_node*, n_examples);
	problem.y = Malloc(double, n_examples);

	for (i = 0; i < n_examples; i++){
		problem.x[i] = node;
		if (examples->domain->classVar->varType == TValue::FLOATVAR)
			problem.y[i] = examples->at(i).getClass().floatV;
		else
			problem.y[i] = examples->at(i).getClass().intV;

		node->index = 0;
		node->value = i + 1; // instance indices are 1 based
		node++;
		for (j = 0; j < n_examples; j++){
			node->index = j + 1;
			node->value = matrix->getitem(i, j);
			node++;
		}
		node->index = -1; // sentry
		node++;
	}
	return x_space;
}

svm_node* init_problem(svm_problem &problem, PExampleTable examples, int n_elements){
	problem.l = examples->numberOfExamples();
	problem.y = Malloc(double ,problem.l);
	problem.x = Malloc(svm_node*, problem.l);
	svm_node *x_space = Malloc(svm_node, n_elements);
	svm_node *node = x_space;

	for (int i = 0; i < problem.l; i++){
		problem.x[i] = node;
		node = example_to_svm(examples->at(i), node, i);
		if (examples->domain->classVar->varType == TValue::FLOATVAR)
			problem.y[i] = examples->at(i).getClass().floatV;
		else
			problem.y[i] = examples->at(i).getClass().intV;
	}
	return x_space;
}

static void print_string_null(const char* s) {}

TSVMLearner::TSVMLearner(){
	//sparse=false;	//if this learners supports sparse datasets (set to true in TSMVLearnerSparse subclass)
	svm_type = NU_SVC;
	kernel_type = RBF;
	degree = 3;
	gamma = 0;
	coef0 = 0;
	nu = 0.5;
	cache_size = 250;
	C = 1;
	eps = 1e-3f;
	p = 0.1f;
	shrinking = 1;
	probability = 0;
	verbose = false;
	nr_weight = 0;
	weight_label = NULL;
	weight = NULL;
};

PClassifier TSVMLearner::operator ()(PExampleGenerator examples, const int&){
	svm_parameter param;
	svm_problem prob;
	svm_model* model;
	svm_node* x_space;

//	PExampleTable table = dynamic_cast<TExampleTable *>(examples.getUnwrappedPtr());

	param.svm_type = svm_type;
	param.kernel_type = kernel_type;
	param.degree = degree;
	param.gamma = gamma;
	param.coef0 = coef0;
	param.nu = nu;
	param.C = C;
	param.eps = eps;
	param.p = p;
	param.cache_size=cache_size;
	param.shrinking=shrinking;
	param.probability=probability;
	param.nr_weight = nr_weight;
	if (nr_weight > 0) {
		param.weight_label = Malloc(int, nr_weight);
		param.weight = Malloc(double, nr_weight);
		int i;
		for (i=0; i<nr_weight; i++) {
			param.weight_label[i] = weight_label[i];
			param.weight[i] = weight[i];
		}
	} else {
		param.weight_label = NULL;
		param.weight = NULL;
	}

//	param.learner=this;
//	param.classifier=NULL;
	//cout<<param.kernel_type<<endl;

//	tempExamples=examples;
	//int exlen=examples->domain->attributes->size();
	int classVarType;
	if(examples->domain->classVar)
		classVarType=examples->domain->classVar->varType;
	else{
		classVarType=TValue::NONE;
		if(svm_type!=ONE_CLASS)
			raiseError("Domain has no class variable");
	}
	if(classVarType==TValue::FLOATVAR && !(svm_type==EPSILON_SVR || svm_type==NU_SVR ||svm_type==ONE_CLASS))
		raiseError("Domain has continuous class");

	if(kernel_type==PRECOMPUTED && !kernelFunc)
		raiseError("Custom kernel function not supplied");

	int numElements=getNumOfElements(examples);

	if(kernel_type != PRECOMPUTED)
		x_space = init_problem(prob, examples, numElements);
	else // Compute the matrix using the kernelFunc
		x_space = init_precomputed_problem(prob, examples, kernelFunc.getReference());

//	prob.l=examples->numberOfExamples();
//	prob.y=Malloc(double,prob.l);
//	prob.x=Malloc(svm_node*, prob.l);
//	x_space=Malloc(svm_node, numElements);
//	int k=0;
//	svm_node *node=x_space;
//	PEITERATE(iter, examples){
//		prob.x[k]=node;
//		node=example_to_svm(*iter, node, k, (param.kernel_type==CUSTOM)? 1:0);
//		switch(classVarType){
//			case TValue::FLOATVAR:{
//				prob.y[k]=(*iter).getClass().floatV;
//				break;
//			}
//			case TValue::INTVAR:{
//				prob.y[k]=(*iter).getClass().intV;
//				break;
//			}
//		}
//		k++;
//	}

	if(param.gamma==0)
		param.gamma=1.0f/(float(numElements)/float(prob.l)-1);

	const char* error=svm_check_parameter(&prob,&param);
	if(error){
		free(x_space);
		free(prob.y);
		free(prob.x);
		raiseError("LibSVM parameter error: %s", error);
	}
	//cout<<"training"<<endl;
//	svm_print_string = (verbose)? &print_string_stdout : &print_string_null;
	svm_set_print_string_function((verbose)? NULL : &print_string_null);
	model=svm_train(&prob,&param);

  if ((svm_type==C_SVC || svm_type==NU_SVC) && !model->nSV) {
	svm_free_and_destroy_model(&model);
    if (x_space)
      free(x_space);
      raiseError("LibSVM returned no support vectors");
  }

	//cout<<"end training"<<endl;
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
//	tempExamples=NULL;
	return PClassifier(createClassifier((param.svm_type==ONE_CLASS)?  \
		mlnew TFloatVariable("one class") : examples->domain->classVar, examples, model, x_space));
}

svm_node* TSVMLearner::example_to_svm(const TExample &ex, svm_node* node, float last, int type){
	return ::example_to_svm(ex, node, last, type);
}

int TSVMLearner::getNumOfElements(PExampleGenerator examples){
	return ::getNumOfElements(examples);
}

TSVMClassifier* TSVMLearner::createClassifier(PVariable var, PExampleTable ex, svm_model* model, svm_node* x_space){
	return mlnew TSVMClassifier(var, ex, model, x_space, kernelFunc);
}

TSVMLearner::~TSVMLearner(){
	if(weight_label)
		free(weight_label);

	if(weight)
			free(weight);
}

svm_node* TSVMLearnerSparse::example_to_svm(const TExample &ex, svm_node* node, float last, int type){
	return ::example_to_svm_sparse(ex, node, last, useNonMeta);
}

int TSVMLearnerSparse::getNumOfElements(PExampleGenerator examples){
	return ::getNumOfElements(examples, true, useNonMeta);
}

TSVMClassifier* TSVMLearnerSparse::createClassifier(PVariable var, PExampleTable ex, svm_model* model, svm_node *x_space){
	return mlnew TSVMClassifierSparse(var, ex, model, x_space, useNonMeta, kernelFunc);
}

TSVMClassifier::TSVMClassifier(const PVariable &var, PExampleTable examples, svm_model* model, svm_node* x_space, PKernelFunc kernelFunc){
	this->classVar = var;
	this->model = model;
	this->x_space = x_space;
	this->examples = examples;
	domain = examples->domain;
	svm_type = svm_get_svm_type(model);
	kernel_type = model->param.kernel_type;
	this->kernelFunc = kernelFunc;

	computesProbabilities = model && svm_check_probability_model(model) && \
			(svm_type != NU_SVR && svm_type != EPSILON_SVR); // Disable prob. estimation for regression

	int nr_class = svm_get_nr_class(model);
	int i = 0;
	supportVectors = mlnew TExampleTable(examples->domain);
	if(x_space){
		for(i = 0;i < model->l; i++){
			svm_node *node = model->SV[i];
			int sv_index = 0;
			if(model->param.kernel_type != PRECOMPUTED){
				// The value of the last node (with index == -1) holds the index of the training example.
				while(node->index != -1)
					node++;
				sv_index = int(node->value);
			}
			else
				sv_index = int(node->value) - 1; // The indices for precomputed kernels are 1 based.
			supportVectors->addExample(mlnew TExample(examples->at(sv_index)));
		}
	}
	
    if (svm_type == C_SVC || svm_type == NU_SVC){
	    nSV = mlnew TIntList(nr_class); // num of SVs for each class (sum = model->l)
	    for(i = 0;i < nr_class; i++)
		    nSV->at(i) = model->nSV[i];
    }

	coef = mlnew TFloatListList(nr_class-1);
	for(i = 0; i < nr_class - 1; i++){
		TFloatList *coefs = mlnew TFloatList(model->l);
		for(int j = 0;j < model->l; j++)
			coefs->at(j) = model->sv_coef[i][j];
		coef->at(i)=coefs;
	}
	rho = mlnew TFloatList(nr_class*(nr_class-1)/2);
	for(i = 0; i < nr_class*(nr_class-1)/2; i++)
		rho->at(i) = model->rho[i];
	if(model->probA){
		probA = mlnew TFloatList(nr_class*(nr_class-1)/2);
		if (model->param.svm_type != NU_SVR && model->param.svm_type != EPSILON_SVR && model->probB) // Regression has only probA
			probB = mlnew TFloatList(nr_class*(nr_class-1)/2);
		for(i=0; i<nr_class*(nr_class-1)/2; i++){
			probA->at(i) = model->probA[i];
			if (model->param.svm_type != NU_SVR && model->param.svm_type != EPSILON_SVR && model->probB)
				probB->at(i) = model->probB[i];
		}
	}
}

TSVMClassifier::~TSVMClassifier(){
	if (model)
		svm_free_and_destroy_model(&model);
	if(x_space)
		free(x_space);
}

PDistribution TSVMClassifier::classDistribution(const TExample & example){
	if(!model)
		raiseError("No Model");

	if(!computesProbabilities)
		return TClassifierFD::classDistribution(example);

	int n_elements;
	if (model->param.kernel_type != PRECOMPUTED)
		n_elements = getNumOfElements(example);
	else
		n_elements = examples->numberOfExamples() + 2;

	int svm_type = svm_get_svm_type(model);
	int nr_class = svm_get_nr_class(model);

	svm_node *x = Malloc(svm_node, n_elements);
	try{
		if (model->param.kernel_type != PRECOMPUTED)
			example_to_svm(example, x, -1.0);
		else
			example_to_svm_precomputed(example, examples, kernelFunc, x);
	} catch (...) {
		free(x);
		throw;
	}

	int *labels=(int *) malloc(nr_class*sizeof(int));
	svm_get_labels(model, labels);

	double *prob_estimates = (double *) malloc(nr_class*sizeof(double));;
	svm_predict_probability(model, x, prob_estimates);

	PDistribution dist = TDistribution::create(example.domain->classVar);
	for(int i=0; i<nr_class; i++)
		dist->setint(labels[i], prob_estimates[i]);
	free(x);
	free(prob_estimates);
	free(labels);
	return dist;
}

TValue TSVMClassifier::operator()(const TExample & example){
	if(!model)
		raiseError("No Model");

	int n_elements;
	if (model->param.kernel_type != PRECOMPUTED)
		n_elements = getNumOfElements(example); //example.domain->attributes->size();
	else
		n_elements = examples->numberOfExamples() + 2;

	int svm_type = svm_get_svm_type(model);
	int nr_class = svm_get_nr_class(model);

	svm_node *x = Malloc(svm_node, n_elements);
	try {
		if (model->param.kernel_type != PRECOMPUTED)
			example_to_svm(example, x);
		else
			example_to_svm_precomputed(example, examples, kernelFunc, x);
	} catch (...) {
		free(x);
		throw;
	}

	double v;

	if(svm_check_probability_model(model)){
		double *prob = (double *) malloc(nr_class*sizeof(double));
		v = svm_predict_probability(model, x, prob);
		free(prob);
	} else
		v = svm_predict(model, x);

	free(x);
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR || svm_type==ONE_CLASS)
		return TValue(v);
	else
		return TValue(int(v));
}

PFloatList TSVMClassifier::getDecisionValues(const TExample &example){
	if(!model)
		raiseError("No Model");

	int n_elements;
		if (model->param.kernel_type != PRECOMPUTED)
			n_elements = getNumOfElements(example); //example.domain->attributes->size();
		else
			n_elements = examples->numberOfExamples() + 2;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);

	svm_node *x = Malloc(svm_node, n_elements);
	try {
		if (model->param.kernel_type != PRECOMPUTED)
			example_to_svm(example, x);
		else
			example_to_svm_precomputed(example, examples, kernelFunc, x);
	} catch (...) {
		free(x);
		throw;
	}

	int nDecValues = nr_class*(nr_class-1)/2;
	double *dec = (double*) malloc(sizeof(double)*nDecValues);
	svm_predict_values(model, x, dec);
	PFloatList res = mlnew TFloatList(nDecValues);
	for(int i=0; i<nDecValues; i++){
		res->at(i) = dec[i];
	}
	free(x);
	free(dec);
	return res;
}

svm_node *TSVMClassifier::example_to_svm(const TExample &ex, svm_node *node, float last, int type){
	return ::example_to_svm(ex, node, last, type);
}

int TSVMClassifier::getNumOfElements(const TExample& example){
	return ::getNumOfElements(example);
}
svm_node *TSVMClassifierSparse::example_to_svm(const TExample &ex, svm_node *node, float last, int type){
	return ::example_to_svm_sparse(ex, node, last, useNonMeta);
}

int TSVMClassifierSparse::getNumOfElements(const TExample& example){
	return ::getNumOfElements(example, true, useNonMeta);
}


