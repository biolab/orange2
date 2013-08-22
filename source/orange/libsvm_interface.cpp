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

#include "libsvm_interface.hpp"
#include "symmatrix.hpp"

#include <algorithm>
#include <cmath>


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
			stream << "0:" << (int)(p->value) << " ";
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


svm_model *svm_load_model_alt(std::istream& stream)
{
	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

#if LIBSVM_VERSION >= 313
	// libsvm seems to ensure ordered numbers for versioning (3.0 was 300,
	// 3.1 was 310,  3.11 was 311, there was no 3.2, ...)
	model->sv_indices = NULL;
#endif

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
			string rho_str;
			for(int i=0;i<n;i++){
				// Read the number into a string and then use strtod
				// for proper handling of NaN's
				stream >> rho_str;
				model->rho[i] = strtod(rho_str.c_str(), NULL);
			}
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


std::ostream & svm_node_vector_to_stream(std::ostream & stream, const svm_node * node) {
	while (node->index != -1) {
		stream << node->index << ":" << node->value << " ";
		node++;
	}
	stream << node->index << ":" << node->value;
	return stream;
}

std::ostream & operator << (std::ostream & stream, const svm_problem & problem) {
	svm_node * node = NULL;
	for (int i = 0; i < problem.l; i++) {
		stream << problem.y[i] << " ";
		svm_node_vector_to_stream(stream, problem.x[i]);
		stream << endl;
	}
	return stream;
}

/*
 * Return a formated string representing a svm data instance (svm_node *)
 * (useful for debugging)
 */
std::string svm_node_to_string(const svm_node * node) {
	std::ostringstream strstream;
	strstream.precision(17);
	svm_node_vector_to_stream(strstream, node);
	return strstream.rdbuf()->str();
}


#ifdef _MSC_VER
	#include <float.h>
	#define isfinite _finite
#endif

/*!
 * Check if the value is valid (not a special value in 'TValue').
 */

inline bool is_valid(double value) {
	return isfinite(value) && value != numeric_limits<int>::max();
}


svm_node* example_to_svm(const TExample &ex, svm_node* node, double last=0.0) {
	int index = 1;
	double value = 0.0;
	TExample::iterator values_end;

	if (ex.domain->classVar) {
		values_end = ex.end() - 1;
	} else {
		values_end = ex.end();
	}

	for(TExample::iterator iter = ex.begin(); iter != values_end; iter++, index++) {
		if(iter->isRegular()) {
			if(iter->varType == TValue::FLOATVAR) {
				value = iter->floatV;
			} else if (iter->varType == TValue::INTVAR) {
				value = iter->intV;
			} else {
				continue;
			}

			// Only add non zero values (speedup due to sparseness)
			if (value != 0 && is_valid(value)) {
				node->index = index;
				node->value = value;
				node++;
			}
		}
	}

	// Sentinel
	node->index = -1;
	node->value = last;
	node++;
	return node;
}

class SVM_NodeSort{
public:
	bool operator() (const svm_node &lhs, const svm_node &rhs) {
		return lhs.index < rhs.index;
	}
};

svm_node* example_to_svm_sparse(const TExample &ex, svm_node* node, double last=0.0, bool include_regular=false) {
	svm_node *first = node;
	int index = 1;
	double value;

	if (include_regular) {
		node = example_to_svm(ex, node);
		// Rewind the sentinel
		node--;
		assert(node->index == -1);
		index += ex.domain->variables->size();
	}

	for (TMetaValues::const_iterator iter=ex.meta.begin(); iter!=ex.meta.end(); iter++) {
		if(iter->second.isRegular()) {
			if(iter->second.varType == TValue::FLOATVAR) {
				value = iter->second.floatV;
			} else if (iter->second.varType == TValue::INTVAR) {
				value = iter->second.intV;
			} else {
				continue;
			}

			if (value != 0 && is_valid(value)) {
				// add the (- meta_id) to index; meta_ids are negative
				node->index = index - iter->first;
				node->value = value;
				node++;
			}
		}
	}

	// sort the nodes by index (metas are not ordered)
	sort(first, node, SVM_NodeSort());

	// Sentinel
	node->index = -1;
	node->value = last;
	node++;
	return node;
}

/*
 * Precompute Gram matrix row for ex.
 * Used for prediction when using the PRECOMPUTED kernel.
 */
svm_node* example_to_svm_precomputed(const TExample &ex, PExampleGenerator examples, PKernelFunc kernel, svm_node* node) {
	// Required node with index 0
	node->index = 0;
	node->value = 0.0; // Can be any value.
	node++;
	int k = 0;
	PEITERATE(iter, examples){
		node->index = ++k;
		node->value = kernel->operator()(*iter, ex);
		node++;
	}

	// Sentinel
	node->index = -1;
	node++;
	return node;
}

int getNumOfElements(const TExample &ex, bool meta=false, bool useNonMeta=false){
	if (!meta)
		return std::max(ex.domain->attributes->size() + 1, 2);
	else {
		int count = 1; // we need one to indicate the end of a sequence
		if (useNonMeta)
			count += ex.domain->attributes->size();
		for (TMetaValues::const_iterator iter=ex.meta.begin(); iter!=ex.meta.end();iter++)
			if(iter->second.isRegular())
				count++;
		return std::max(count,2);
	}
}

int getNumOfElements(PExampleGenerator &examples, bool meta=false, bool useNonMeta=false) {
	if (!meta)
		return getNumOfElements(*(examples->begin()), meta) * examples->numberOfExamples();
	else {
		int count = 0;
		for(TExampleGenerator::iterator ex(examples->begin()); ex!=examples->end(); ++ex){
			count += getNumOfElements(*ex, meta, useNonMeta);
		}
		return count;
	}
}

svm_node* init_precomputed_problem(svm_problem &problem, PExampleTable examples, TKernelFunc &kernel){
	int n_examples = examples->numberOfExamples();
	int i,j;
	PSymMatrix matrix = mlnew TSymMatrix(n_examples, 0.0);
	for (i = 0; i < n_examples; i++)
		for (j = 0; j <= i; j++){
			matrix->getref(i, j) = kernel(examples->at(i), examples->at(j));
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

/*
 * Extract an ExampleTable corresponding to the support vectors from the
 * trained model.
 */
PExampleTable extract_support_vectors(svm_model * model, PExampleTable train_instances)
{
	PExampleTable vectors = mlnew TExampleTable(train_instances->domain);

	for (int i = 0; i < model->l; i++) {
		svm_node *node = model->SV[i];
		int sv_index = -1;
		if(model->param.kernel_type != PRECOMPUTED){
			/* The value of the last node (with index == -1) holds the
			 * index of the training example.
			 */
			while(node->index != -1) {
				node++;
			}
			sv_index = int(node->value);
		} else {
			/* The value of the first node contains the training instance
			 * index (indices 1 based).
			 */
			sv_index = int(node->value) - 1;
		}
		vectors->addExample(mlnew TExample(train_instances->at(sv_index)));
	}

	return vectors;
}


/*
 * Consolidate model->SV[1] .. SV[l] vectors into a single contiguous
 * memory block. The model will 'own' the new *(model->SV) array and
 * will be freed in destroy_svm_model (model->free_sv == 1). Note that
 * the original 'x_space' is left intact, it is the caller's
 * responsibility to free it. However the model->SV array itself is
 * reused (overwritten).
 */

void svm_model_consolidate_SV(svm_model * model) {
	int count = 0;
	svm_node * x_space = NULL;
	svm_node * ptr = NULL;
	svm_node * ptr_source = NULL;

	// Count the number of elements.
	for (int i = 0; i < model->l; i++) {
		ptr = model->SV[i];
		while (ptr->index != -1){
			count++;
			ptr++;
		}
	}
	// add the sentinel count
	count += model->l;

	x_space = Malloc(svm_node, count);
	ptr = x_space;
	for (int i = 0; i < model->l; i++) {
		ptr_source = model->SV[i];
		model->SV[i] = ptr;
		while (ptr_source->index != -1) {
			*(ptr++) = *(ptr_source++);
		}
		// copy the sentinel
		*(ptr++) = *(ptr_source++);
	}
	model->free_sv = 1; // XXX
}

static void print_string_null(const char* s) {}


TSVMLearner::TSVMLearner(){
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

	PDomain domain = examples->domain;

	int classVarType;
	if (domain->classVar)
		classVarType = domain->classVar->varType;
	else {
		classVarType = TValue::NONE;
		if(svm_type != ONE_CLASS)
			raiseError("Domain has no class variable");
	}
	if (classVarType == TValue::FLOATVAR && !(svm_type == EPSILON_SVR || svm_type == NU_SVR ||svm_type == ONE_CLASS))
		raiseError("Domain has continuous class");

	if (kernel_type == PRECOMPUTED && !kernelFunc)
		raiseError("Custom kernel function not supplied");

	PExampleTable train_data = mlnew TExampleTable(examples, /* owns= */ false);

	if (classVarType == TValue::INTVAR && svm_type != ONE_CLASS) {
		/* Sort the train data by the class columns so the order of
		 * classVar.values is preserved in libsvm's model.
		 */
		vector<int> sort_columns(domain->variables->size() - 1);
		train_data->sort(sort_columns);
	}

	// Initialize svm parameters
	param.svm_type = svm_type;
	param.kernel_type = kernel_type;
	param.degree = degree;
	param.gamma = gamma;
	param.coef0 = coef0;
	param.nu = nu;
	param.C = C;
	param.eps = eps;
	param.p = p;
	param.cache_size = cache_size;
	param.shrinking = shrinking;
	param.probability = probability;
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

	int numElements = getNumOfElements(train_data);

	prob.x = NULL;
	prob.y = NULL;

	if (kernel_type != PRECOMPUTED)
		x_space = init_problem(prob, train_data, numElements);
	else // Compute the matrix using the kernelFunc
		x_space = init_precomputed_problem(prob, train_data, kernelFunc.getReference());

	if (param.gamma == 0)
		param.gamma = 1.0f / (float(numElements) / float(prob.l) - 1);

	const char* error = svm_check_parameter(&prob, &param);
	if (error){
		free(x_space);
		free(prob.y);
		free(prob.x);
		svm_destroy_param(&param);
		raiseError("LibSVM parameter error: %s", error);
	}

	// If a probability model was requested LibSVM uses 5 fold
	// cross-validation to estimate the prediction errors. This includes a
	// random shuffle of the data. To make the results reproducible and
	// consistent with 'svm-train' (which always learns just on one dataset
	// in a process run) we reset the random seed. This could have unintended
	// consequences.
	if (param.probability)
	{
		srand(1);
	}
	svm_set_print_string_function((verbose)? NULL : &print_string_null);

	model = svm_train(&prob, &param);

	if ((svm_type==C_SVC || svm_type==NU_SVC) && !model->nSV) {
		svm_free_and_destroy_model(&model);
		free(x_space);
		free(prob.x);
		free(prob.y);
		svm_destroy_param(&param);
		raiseError("LibSVM returned no support vectors");
	}

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);

	// Consolidate the SV so x_space can be safely freed
	svm_model_consolidate_SV(model);

	free(x_space);

	PExampleTable supportVectors = extract_support_vectors(model, train_data);

	return PClassifier(createClassifier(domain, model, supportVectors, train_data));
}

svm_node* TSVMLearner::example_to_svm(const TExample &ex, svm_node* node, double last){
	return ::example_to_svm(ex, node, last);
}

svm_node* TSVMLearner::init_problem(svm_problem &problem, PExampleTable examples, int n_elements){
	problem.l = examples->numberOfExamples();
	problem.y = Malloc(double, problem.l);
	problem.x = Malloc(svm_node*, problem.l);
	svm_node *x_space = Malloc(svm_node, n_elements);
	svm_node *node = x_space;

	for (int i = 0; i < problem.l; i++){
		problem.x[i] = node;
		node = example_to_svm(examples->at(i), node, i);
		if (examples->domain->classVar)
			if (examples->domain->classVar->varType == TValue::FLOATVAR)
				problem.y[i] = examples->at(i).getClass().floatV;
			else if (examples->domain->classVar->varType == TValue::INTVAR)
				problem.y[i] = examples->at(i).getClass().intV;
	}

//	cout << problem << endl;

	return x_space;
}

int TSVMLearner::getNumOfElements(PExampleGenerator examples){
	return ::getNumOfElements(examples);
}

TSVMClassifier* TSVMLearner::createClassifier(
		PDomain domain, svm_model* model, PExampleTable supportVectors, PExampleTable examples) {
	PKernelFunc kfunc;
	if (kernel_type != PRECOMPUTED) {
		// Classifier does not need the train data and the kernelFunc.
		examples = NULL;
		kfunc = NULL;
	} else {
		kfunc = kernelFunc;
	}

	return mlnew TSVMClassifier(domain, model, supportVectors, kfunc, examples);
}

TSVMLearner::~TSVMLearner(){
	if(weight_label)
		free(weight_label);

	if(weight)
		free(weight);
}

svm_node* TSVMLearnerSparse::example_to_svm(const TExample &ex, svm_node* node, double last){
	return ::example_to_svm_sparse(ex, node, last, useNonMeta);
}

int TSVMLearnerSparse::getNumOfElements(PExampleGenerator examples){
	return ::getNumOfElements(examples, true, useNonMeta);
}

TSVMClassifier* TSVMLearnerSparse::createClassifier(
		PDomain domain, svm_model* model, PExampleTable supportVectors, PExampleTable examples) {
	PKernelFunc kfunc;
	if (kernel_type != PRECOMPUTED) {
		// Classifier does not need the train data and the kernelFunc.
		examples = NULL;
		kfunc = NULL;
	} else {
		kfunc = kernelFunc;
	}
	return mlnew TSVMClassifierSparse(domain, model, useNonMeta, supportVectors, kfunc, examples);
}


TSVMClassifier::TSVMClassifier(
		PDomain domain, svm_model * model,
		PExampleTable supportVectors,
		PKernelFunc kernelFunc,
		PExampleTable examples
		) : TClassifierFD(domain) {
	this->model = model;
	this->supportVectors = supportVectors;
	this->kernelFunc = kernelFunc;
	this->examples = examples;

	svm_type = svm_get_svm_type(model);
	kernel_type = model->param.kernel_type;

	if (svm_type == ONE_CLASS) {
		this->classVar = mlnew TFloatVariable("one class");
	}

	computesProbabilities = model && svm_check_probability_model(model) && \
				(svm_type != NU_SVR && svm_type != EPSILON_SVR); // Disable prob. estimation for regression

	int nr_class = svm_get_nr_class(model);
	int i = 0;

	/* Expose (copy) the model data (coef, rho, probA) to public
	 * class interface.
	 */
	if (svm_type == C_SVC || svm_type == NU_SVC) {
		nSV = mlnew TIntList(nr_class); // num of SVs for each class (sum(nSV) == model->l)
		for(i = 0;i < nr_class; i++) {
			nSV->at(i) = model->nSV[i];
		}
	}

	coef = mlnew TFloatListList(nr_class-1);
	for(i = 0; i < nr_class - 1; i++) {
		TFloatList *coefs = mlnew TFloatList(model->l);
		for(int j = 0;j < model->l; j++) {
			coefs->at(j) = model->sv_coef[i][j];
		}
		coef->at(i) = coefs;
	}

	// Number of binary classifiers in the model
	int nr_bin_cls = nr_class * (nr_class - 1) / 2;

	rho = mlnew TFloatList(nr_bin_cls);
	for(i = 0; i < nr_bin_cls; i++) {
		rho->at(i) = model->rho[i];
	}

	if(model->probA) {
		probA = mlnew TFloatList(nr_bin_cls);
		if (model->param.svm_type != NU_SVR && model->param.svm_type != EPSILON_SVR && model->probB) {
			// Regression only has probA
			probB = mlnew TFloatList(nr_bin_cls);
		}

		for(i=0; i<nr_bin_cls; i++) {
			probA->at(i) = model->probA[i];
			if (model->param.svm_type != NU_SVR && model->param.svm_type != EPSILON_SVR && model->probB) {
				probB->at(i) = model->probB[i];
			}
		}
	}
}


TSVMClassifier::~TSVMClassifier(){
	if (model) {
		svm_free_and_destroy_model(&model);
	}
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
		n_elements = getNumOfElements(example);
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
		n_elements = getNumOfElements(example);
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

svm_node *TSVMClassifier::example_to_svm(const TExample &ex, svm_node *node, double last){
	return ::example_to_svm(ex, node, last);
}

int TSVMClassifier::getNumOfElements(const TExample& example){
	return ::getNumOfElements(example);
}
svm_node *TSVMClassifierSparse::example_to_svm(const TExample &ex, svm_node *node, double last){
	return ::example_to_svm_sparse(ex, node, last, useNonMeta);
}

int TSVMClassifierSparse::getNumOfElements(const TExample& example){
	return ::getNumOfElements(example, true, useNonMeta);
}


#include "libsvm_interface.ppp"
