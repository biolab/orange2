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


#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
  #pragma warning (disable : 4512) // assigment operator could not be generated
#endif

#include <limits>
#include <list>
#include <math.h>
#include <assert.h>
#include "stladdon.hpp"
#include "errors.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "contingency.hpp"
#include "meta.hpp"

#include "svm.ppp"


TlibSVM::TlibSVM() {
	// create the model structure and fill in the default parameters
	model = mlnew struct svm_model;

	model->n = 0;
	model->SV = NULL;
	model->sv_coef = NULL;
	problem = NULL;
};


TSVMLearner::TSVMLearner()
{
	// copy the defaults to the Learner class
	svm = NULL; // empty

	svm_type = 0;
	kernel_type = 2;
	degree = 3;
	gamma = 0;
	coef0 = 0;
	nu = 0.5;
	cache_size = 40;
	eps = float(1e-3);
	p = 0.5;
	C = 1.0;
	max_iter = 10000;
	iter_mult = 10.0;
}


PClassifier TSVMLearner::operator()(PExampleGenerator ogen, const int &)
{ 
	if (!ogen->domain->classVar)
    raiseError("class-less domain");

  int elements, n, i, j, idx;
	struct TlibSVM::svm_node *x_space;
	
	// copy the properties back to the model

	if (svm != NULL)
		mldelete svm;
	svm = mlnew TlibSVM;
	svm->model->param.svm_type = svm_type;
	svm->model->param.kernel_type = kernel_type;
	svm->model->param.degree = degree;
	svm->model->param.gamma = gamma;
	svm->model->param.coef0 = coef0;
	svm->model->param.nu = nu;
	svm->model->param.cache_size = cache_size;
	svm->model->param.eps = eps;
	svm->model->param.p = p; 
	svm->model->param.C = C;
	
	svm->model->param.max_iter = max_iter;
	svm->model->param.iter_mult = iter_mult;

  if (!ogen->begin())
    raiseError("no examples");
	
	// copy the examples
	if (ogen->domain->classVar->varType==TValue::FLOATVAR) {
		svm->problem = mlnew TlibSVM::svm_problem;
		// allocate memory
		n = svm->problem->l = ogen->numberOfExamples();
		if (svm->model->param.gamma==0.0) {
			svm->model->param.gamma = 1.0/(ogen->domain->attributes->size());
		}
		
		elements = n*ogen->domain->variables->size();
		i = j = 0;
		
		svm->problem->y = (double*)malloc(sizeof(double)*n);
		svm->problem->x = (TlibSVM::svm_node**)malloc(sizeof(TlibSVM::svm_node*)*n);
		x_space = (TlibSVM::svm_node*)malloc(sizeof(TlibSVM::svm_node)*elements);

    TExampleIterator fei(ogen->begin());

		PEITERATE(ei, ogen) {
			// setting the class
			svm->problem->x[i] = &(x_space[j]);
      assert(i < n);
			idx = 1;
			
			// setting the values&labels
			TExample::iterator vi((*ei).begin()), evi((*ei).end());
      for(; vi!=evi; vi++) {
				if (vi!=evi-1) {
					// variable
					if (!vi->isSpecial()) {
						if (vi->varType == TValue::INTVAR) {
							x_space[j].index = idx;
							x_space[j].value = vi->intV;
							++j;
						} else if (vi->varType == TValue::FLOATVAR) {
							x_space[j].index = idx;
							x_space[j].value = vi->floatV;
							++j;
						}
					}
					++idx;
				} else {
					svm->problem->y[i] = (vi->isSpecial()) ? 0.0 : vi->floatV;
					x_space[j].index = -1;
          assert(j < elements);
					++j;
					break;
				}
				
			}
			++i;
		}
	} else
      raiseError("continuous class expected");
	
	// do the learning
	svm->svm_train();

	// create the classifier with the model now created

	if (svm->problem != NULL) {
		// mldelete the problem
		if (svm->problem->x != NULL) {
			free(*(svm->problem->x));
			free(svm->problem->x);
		}
		free(svm->problem->y);
		mldelete svm->problem;
	}

  PClassifier nc = mlnew TSVMClassifier(ogen->domain, svm);

  TlibSVM *back = mlnew TlibSVM;
  back->model->param.C = svm->model->param.C;
  back->model->param.cache_size = svm->model->param.cache_size;
  back->model->param.coef0 = svm->model->param.coef0;
  back->model->param.degree = svm->model->param.degree;
  back->model->param.eps = svm->model->param.eps;
  back->model->param.gamma = svm->model->param.gamma;
  back->model->param.iter_mult = svm->model->param.iter_mult;
  back->model->param.kernel_type = svm->model->param.kernel_type;
  back->model->param.max_iter = svm->model->param.max_iter;
  back->model->param.nu = svm->model->param.nu;
  back->model->param.p = svm->model->param.p;
  back->model->param.svm_type = svm->model->param.svm_type;
  svm->problem = NULL;
  svm = back;
	
	return nc;
}

TSVMClassifier::TSVMClassifier(PDomain dom, TlibSVM *mod) : 
TClassifierFD(dom)
{
  svm = mod;
}

TValue TSVMClassifier::operator ()(const TExample &oldexam)
{ 
  TExample exam(domain, oldexam);

	TlibSVM::svm_node x_space[5000];
	int j, idx;
	double result;
	TValue ret;

	if (exam.domain->variables->size() >= 5000)
		return classVar->DC();


	// copy the example into svm_nodes
	// setting the values&labels
	TExample::const_iterator vi(exam.begin()), evi(exam.end()-1);
	j = 0;
	idx = 1;

	for(; vi!=evi; vi++) {
			// variable
			if (!vi->isSpecial()) {
				if (vi->varType == TValue::INTVAR) {
					x_space[j].index = idx;
					x_space[j].value = vi->intV;
					++j;
				} else if (vi->varType == TValue::FLOATVAR) {
					x_space[j].index = idx;
					x_space[j].value = vi->floatV;
					++j;
				}
			}
			++idx;
	}
	x_space[j].index = -1;
	
	svm->svm_classify(x_space,&result);

	if (classVar->varType==TValue::FLOATVAR) { // if class is continuous, we shall return the median
		ret=TValue((float)result);
	};
	
	return ret;
}


//////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>

template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#define EPS_A 1e-12
#define INF DBL_MAX

enum { C_SVC, NU_SVC, ONE_CLASS, C_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID };	/* kernel_type */



//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size
//
class TlibSVM::Cache
{
public:
	Cache(int l,int size);
	~Cache();

	// return 1 if returned data is valid (cache hit)
	// return 0 if returned data needs to be filled (cache miss)
	int get_data(const int index, double **data);
private:
	int l;
	int size;
	struct head_t
	{
		head_t *prev, *next;	// a cicular list
		int index;
		double *data;
	};

	head_t* head;
	head_t* lru_head;
	head_t** index_to_head;
	void move_to_last(head_t *h);
};

TlibSVM::Cache::Cache(int i_l,int i_size):l(i_l),size(i_size)
{
	head = mlnew head_t[size];
	int i;
	for(i=0;i<size;i++)
	{
		head[i].next = &head[i+1];
		head[i].prev = &head[i-1];
		head[i].index = -1;
		head[i].data = mlnew double[l];
	}

	head[0].prev = &head[size-1];
	head[size-1].next = &head[0];
	lru_head = &head[0];

	index_to_head = mlnew head_t *[l];
	for(i=0;i<l;i++)
		index_to_head[i] = 0;
}

TlibSVM::Cache::~Cache()
{
	for(int i=0;i<size;i++)
		mldelete[] head[i].data;
	mldelete[] head;
	mldelete[] index_to_head;
}

void TlibSVM::Cache::move_to_last(head_t *h)
{
	if(lru_head == h)
		lru_head = lru_head->next;
	else
	{
		// mldelete from current location
		h->prev->next = h->next;
		h->next->prev = h->prev;

		// insert to last position
		h->next = lru_head;
		h->prev = lru_head->prev;
		h->prev->next = h;
		h->next->prev = h;
	}
}

int TlibSVM::Cache::get_data(const int index,double **data)
{
	head_t *h=index_to_head[index];
	if(h)
	{
		move_to_last(h);
		*data = h->data;
		return 1;
	}		
	else
	{
		// get one from lru_head
		h = lru_head;
		lru_head = lru_head->next;
		if(h->index!=-1)
			index_to_head[h->index] = 0;
		h->index = index;
		index_to_head[index] = h;
		*data = h->data;
		return 0;
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class TlibSVM::Kernel {
public:
	Kernel(int l, const svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual double *get_Q(int column) const = 0;

protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node * const * const x;

	// svm_parameter
	const int kernel_type;
	const double degree;
	const double gamma;
	const double coef0;

	double *x_square;
	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return pow(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
};

TlibSVM::Kernel::Kernel(int l, const svm_node * const * i_x, const svm_parameter& param)
:x(i_x), kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{

	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		default:
			fprintf(stderr,"unknown kernel function.\n");
			exit(1);
	}

	if(kernel_type == RBF)
	{
		x_square = mlnew double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

TlibSVM::Kernel::~Kernel()
{
	mldelete x_square;
}

double TlibSVM::Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double TlibSVM::Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return pow(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		default:
			fprintf(stderr,"unknown kernel function.\n");
			exit(1);
	}
}

// Generalized SMO+SVMlight algorithm
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + b^T \alpha
//
//		0 <= alpha_i <= C
//		y^T \alpha = \delta
//
// Given:
//
//	Q, b, y, C, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping criterion
//
// solution will be put in \alpha, objective value will be put in obj
//
class TlibSVM::Solver {
public:
	Solver( int i_l, const Kernel& Q, const double *b, const double *i_y,
		double *i_alpha, double i_C, double _eps,
		double& obj, double& rho, int maxiter, float mult);

	~Solver();

private:
	const int l;
	const double *y;
	double * const alpha;
	const double C;
	double eps;

	double * G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	void update_alpha_status(int i)
	{
		if(alpha[i] >= C-EPS_A)
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= EPS_A)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	int select_working_set(int &i, int &j);
};

TlibSVM::Solver::Solver( int i_l, const Kernel& Q, const double *b, const double *i_y,
		double *i_alpha, double i_C, double i_eps,
		double& obj, double& rho, int maxiter, float mult)
:l(i_l),y(i_y),alpha(i_alpha),C(i_C),eps(i_eps)
{
	// initialize alpha_status
	{
		alpha_status = mlnew char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize gradient
	{
		G = mlnew double[l];
		int i;
    for(i=0;i<l;i++)
			G[i] = b[i];
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				double *Q_i = Q.get_Q(i);
				double alpha_i = alpha[i];
				for(int j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
			}
	}

	// optimization step

	int iter = 0;

	while(1)
	{
		int i,j;
		if(select_working_set(i,j)!=0)
			break;

		++iter;
		if (iter == maxiter) {
			iter = 0;
			eps *= mult;
		}

		// update alpha[i] and alpha[j]
		
		const double *Q_i = Q.get_Q(i);
		const double *Q_j = Q.get_Q(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];
		double s = y[i]*old_alpha_i + y[j]*old_alpha_j;

		double H = s/y[j];
		double L = (s-C*y[i])/y[j];

		if(y[i]*y[j] < 0) { double t = H; H = L; L = t;}

		H = ::min(C,H);
		L = ::max(0.0,L);

		alpha[j] += y[i] * (y[j]*G[i] - y[i]*G[j]) /
			(y[i]*(y[i]*Q_j[j] - 2*Q_i[j]*y[j]) + y[j]*y[j]*Q_i[i]);

		if(alpha[j] > H) alpha[j] = H;
		else if(alpha[j] < L) alpha[j] = L;

		alpha[i] = (s - y[j]*alpha[j])/y[i];

		// update alpha_status

		update_alpha_status(i);
		update_alpha_status(j);

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<l;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}
	}
	
	// calculate rho

	{
		double r;
		int nr_free = 0;
		double ub = INF, lb = -INF, sum_free = 0;
		for(int i=0;i<l;i++)
		{
			double yG = y[i]*G[i];

			if(is_lower_bound(i))
			{
				if(y[i] > 0)
					ub = ::min(ub,yG);
				else
					lb = ::max(lb,yG);
			}
			else if(is_upper_bound(i))
			{
				if(y[i] < 0)
					ub = ::min(ub,yG);
				else
					lb = ::max(lb,yG);
			}
			else
			{
				++nr_free;
				sum_free += yG;
			}
		}

		if(nr_free>0)
			r = sum_free/nr_free;
		else
			r = (ub+lb)/2;

		rho = r;
	}

	// calculate objective value

	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i]/2 + b[i]);

		obj = v;
	}

//	printf("\noptimization finished, #iter = %d\n",iter);
}

TlibSVM::Solver::~Solver()
{
	mldelete[] alpha_status;
	mldelete[] G;
}

// return 1 if already optimal, return 0 otherwise
int TlibSVM::Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j which maximize -grad(f)^T d , under constraint
	// if alpha_i == C, d != +1
	// if alpha_i == 0, d != -1

	double Gmax1 = -INF;		// max { -grad(f)_i * d | y_i*d = +1 }
	int Gmax1_idx = -1;

	double Gmax2 = -INF;		// max { -grad(f)_i * d | y_i*d = -1 }
	int Gmax2_idx = -1;

	for(int i=0;i<l;i++)
	{
		if(y[i]>0)	// y > 0
		{
			if(!is_upper_bound(i))	// d = +1
			{
				if(-G[i] > Gmax1)
				{
					Gmax1 = -G[i];
					Gmax1_idx = i;
				}
			}
			if(!is_lower_bound(i))	// d = -1
			{
				if(G[i] > Gmax2)
				{
					Gmax2 = G[i];
					Gmax2_idx = i;
				}
			}
		}
		else		// y < 0
		{
			if(!is_upper_bound(i))	// d = +1
			{
				if(-G[i] > Gmax2)
				{
					Gmax2 = -G[i];
					Gmax2_idx = i;
				}
			}
			if(!is_lower_bound(i))	// d = -1
			{
				if(G[i] > Gmax1)
				{
					Gmax1 = G[i];
					Gmax1_idx = i;
				}
			}
		}
	}

	if(Gmax1+Gmax2 < eps)
 		return 1;

	out_i = Gmax1_idx;
	out_j = Gmax2_idx;
	return 0;
}

//
// Q matrices for different formulations
//
class TlibSVM::C_SVC_Q: public Kernel
{ 
public:
	C_SVC_Q(const svm_problem& prob, const svm_parameter& param, const double *_y)
	:Kernel(prob.l, prob.x, param), l(prob.l), y(_y)
	{
		cache = mlnew Cache(l,(int)::min((double)l,(param.cache_size*(1<<20))/(sizeof(double)*l)));
	}
	
	double *get_Q(int i) const
	{
		double *data;

		if(cache->get_data(i,&data) == 0)
		{
			for(int j=0;j<l;j++)
				data[j] = y[i]*y[j]*(this->*kernel_function)(i,j);
		}
		return data;
	}

	~C_SVC_Q()
	{
		mldelete cache;
	}
private:
	const int l;
	const double *y;
	Cache *cache;
};

class TlibSVM::NU_SVC_Q: public Kernel
{
public:
	NU_SVC_Q(const svm_problem& prob, const svm_parameter& param, const double *_y)
	:Kernel(prob.l, prob.x, param), l(prob.l), y(_y)
	{
		cache = mlnew Cache(l,(int)::min((double)l,(param.cache_size*(1<<20))/(sizeof(double)*l)));
	}
	
	double *get_Q(int i) const
	{
		double *data;

		if(cache->get_data(i,&data) == 0)
		{
			for(int j=0;j<l;j++)
				data[j] = y[i]*y[j]*(1+(this->*kernel_function)(i,j));
		}
		return data;
	}

	~NU_SVC_Q()
	{
		mldelete cache;
	}
private:
	const int l;
	const double *y;
	Cache *cache;
};

class TlibSVM::ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param), l(prob.l)
	{
		cache = mlnew Cache(l,(int)::min((double)l,(param.cache_size*(1<<20))/(sizeof(double)*l)));
	}
	
	double *get_Q(int i) const
	{
		double *data;

		if(cache->get_data(i,&data) == 0)
		{
			for(int j=0;j<l;j++)
				data[j] = (this->*kernel_function)(i,j);
		}
		return data;
	}

	~ONE_CLASS_Q()
	{
		mldelete cache;
	}
private:
	const int l;
	Cache *cache;
};

class TlibSVM::C_SVR_Q: public Kernel
{ 
public:
	C_SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param), l(prob.l)
	{
		cache = mlnew Cache(2*l,(int)::min((double)(2*l),(param.cache_size*(1<<20))/(sizeof(double)*(2*l))));
	}
	
	double *get_Q(int i) const
	{
		double *data;

		if(cache->get_data(i,&data) == 0)
		{
			if(i<l)
				for(int j=0;j<l;j++)
				{
					data[j] = (this->*kernel_function)(i,j);
					data[j+l] = -data[j];
				}
			else
				for(int j=0;j<l;j++)
				{
					data[j+l] = (this->*kernel_function)(i-l,j);
					data[j] = -data[j+l];
				}
		}
		return data;
	}

	~C_SVR_Q()
	{
		mldelete cache;
	}
private:
	const int l;
	const double *y;
	Cache *cache;
};


void TlibSVM::solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, double& obj, double& rho)
{
	int l = prob->l;
	double *minus_ones = mlnew double[l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
	}

	TlibSVM::Solver s(l, TlibSVM::C_SVC_Q(*prob,*param,prob->y), minus_ones, prob->y,
		 alpha, param->C, param->eps, obj, rho, param->max_iter, param->iter_mult);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

//	printf("nu = %f\n", sum_alpha/(param->C*prob->l));

	mldelete[] minus_ones;

	for(i=0;i<l;i++)
		alpha[i] *= prob->y[i];
}

void TlibSVM::solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, double& obj, double& rho)
{
	int l = prob->l;
	double *zeros = mlnew double[l];
	double *ones = mlnew double[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound
	if(n>=prob->l)
	{
		fprintf(stderr,"nu must be in (0,1)\n");
		exit(1);
	}
	for(i=0;i<n;i++)
		alpha[i] = 1;
	alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{	
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s(l, NU_SVC_Q(*prob,*param,prob->y), zeros, ones,
		 alpha, 1.0, param->eps, obj, rho, param->max_iter, param->iter_mult);

//	printf("C = %f\n",1/rho);

	mldelete[] zeros;
	mldelete[] ones;

	for(i=0;i<l;i++)
		alpha[i] *= prob->y[i]/rho;
}

void TlibSVM::solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, double& obj, double& rho)
{
	int l = prob->l;
	double *zeros = mlnew double[l];
	double *ones = mlnew double[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound
	if(n>=prob->l)
	{
//		fprintf(stderr,"nu must be in (0,1)\n");
		exit(1);
	}
	for(i=0;i<n;i++)
		alpha[i] = 1;
	alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		 alpha, 1.0, param->eps, obj, rho, param->max_iter, param->iter_mult);

	mldelete[] zeros;
	mldelete[] ones;
}

void TlibSVM::solve_c_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, double& obj, double& rho)
{
	int l = prob->l;
	double *alpha2 = mlnew double[2*l];
	double *linear_term = mlnew double[2*l];
	double *y = mlnew double[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s(2*l, C_SVR_Q(*prob,*param), linear_term, y,
		 alpha2, param->C, param->eps, obj, rho, param->max_iter, param->iter_mult);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	mldelete[] alpha2;
	mldelete[] linear_term;
	mldelete[] y;
}

//
// Interface functions
//
void TlibSVM::svm_train()
{
	struct svm_parameter* param = &(model->param);

	double *alpha = mlnew double[problem->l];
	double obj, rho;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(problem,param,alpha,obj,rho);
			break;
		case NU_SVC:
			solve_nu_svc(problem,param,alpha,obj,rho);
			break;
		case ONE_CLASS:
			solve_one_class(problem,param,alpha,obj,rho);
			break;
		case C_SVR:
			solve_c_svr(problem,param,alpha,obj,rho);
			break;
	}

	model->rho = rho;
	//printf("obj = %f, rho = %f\n",obj,rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
  int nElm = 0;
	for(int i=0;i<problem->l;i++)
	{
		if(fabs(alpha[i]) >= EPS_A)
		{
      int j;
      svm_node *node;

			++nSV;
      
      node = problem->x[i];
      j = 0;
      while(node->index > 0 ) {
        ++j; ++node;
      }
      nElm += j+1;

			if(fabs(alpha[i]) >= param->C - EPS_A)
				++nBSV;
		}
	}

	//printf("nSV = %d, nBSV = %d\n",nSV,nBSV);

	model->n = nSV;
	model->sv_coef = (double *)malloc(sizeof(double) * nSV);
  model->x = (svm_node*)malloc(sizeof(svm_node)*nElm);
	model->SV = (svm_node **)malloc(sizeof(svm_node*) * nSV);
  nElm = 0;

	{
		int j = 0;
		for(int i=0;i<problem->l;i++)
		{
			if(fabs(alpha[i]) >= EPS_A)
			{
        svm_node *node;

				model->sv_coef[j] = alpha[i];
				model->SV[j] = &(model->x[nElm]);
				++j;

        node = problem->x[i];
        while(node->index > 0) {
          model->x[nElm].index = node->index;
          model->x[nElm].value = node->value;
          ++nElm; ++node;
        }
        model->x[nElm++].index = -1;
			}
		}
	}

	mldelete[] alpha;
}

void TlibSVM::svm_classify(const svm_node *x, double *decision_value)
{
	const int n = model->n;
	const double *sv_coef = model->sv_coef;

	double sum = 0;
	if(model->param.svm_type == NU_SVC)
	{
		for(int i=0;i<n;i++)
			sum += sv_coef[i] * (1+Kernel::k_function(x,model->SV[i],model->param));
	}
	else
	{
		for(int i=0;i<n;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum-=(model->rho);
	}

	*decision_value = sum;
}

