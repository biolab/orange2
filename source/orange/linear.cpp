/*
Copyright (c) 2007-2008 The LIBLINEAR Project.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

    Authors: Ales Erjavec
    Contact: ales.erjavec@fri.uni-lj.si
*/

/*
Copy from linear.cpp of LIBLINEAR
Changes:
	-#include "linear.h" -> #include "linear.ppp"
	-#ifndef block around swap, min and max definition
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "linear.ppp"


typedef signed char schar;

#if _MSC_VER!=0 && _MSC_VER<1300
#ifndef swap
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#endif
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#if 0
void info(const char *fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
}
/*void info_flush()
{
	fflush(stdout);
}*/
extern void info_flush();
#else
void info(char *fmt,...) {}
//extern void info(char *fmt,...);
void info_flush() {}
//extern void info_flush();
#endif

class l2_lr_fun : public function
{
public:
	l2_lr_fun(const problem *prob, double Cp, double Cn);
	~l2_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2_lr_fun::l2_lr_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2_lr_fun::~l2_lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


double l2_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        double yz = y[i]*z[i];
		if (yz >= 0)
		        f += C[i]*log(1 + exp(-yz));
		else
		        f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2_lr_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + g[i];
}

int l2_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	feature_node **x=prob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2loss_svm_fun : public function
{
public:
	l2loss_svm_fun(const problem *prob, double Cp, double Cn);
	~l2loss_svm_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2loss_svm_fun::l2loss_svm_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];
	I = new int[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2loss_svm_fun::~l2loss_svm_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

double l2loss_svm_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        z[i] = y[i]*z[i];
		double d = z[i]-1;
		if (d < 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2loss_svm_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + 2*g[i];
}

int l2loss_svm_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2loss_svm_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2loss_svm_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2loss_svm_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2loss_svm_fun::subXTv(double *v, double *XTv)
{
	int i;
	int n=prob->n;
	feature_node **x=prob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-Svm case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static void solve_linear_c_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int n = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 20000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2LOSS_SVM_DUAL
	double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
	double upper_bound_p = INF, upper_bound_n = INF;
	if(solver_type == L1LOSS_SVM_DUAL)
	{
		diag_p = 0; diag_n = 0;
		upper_bound_p = Cp; upper_bound_n = Cn;
	}

	for(i=0; i<n; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0)
		{
			y[i] = +1;
			QD[i] = diag_p;
		}
		else
		{
			y[i] = -1;
			QD[i] = diag_n;
		}

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			QD[i] += (xi->value)*(xi->value);
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s < active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			if(yi == 1)
			{
				C = upper_bound_p;
				G += alpha[i]*diag_p;
			}
			else
			{
				C = upper_bound_n;
				G += alpha[i]*diag_n;
			}

			PG = 0;
			if (alpha[i] ==0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
			info_flush();
		}

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*"); info_flush();
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("Warning: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<n; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		if (y[i] == 1)
			v += alpha[i]*(alpha[i]*diag_p - 2);
		else
			v += alpha[i]*(alpha[i]*diag_n - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for (int i=0; i<prob->l;i++)
		if (prob->y[i]==+1)
			pos++;
	neg = prob->l - pos;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2_LR:
		{
			fun_obj=new l2_lr_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2LOSS_SVM:
		{
			fun_obj=new l2loss_svm_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2LOSS_SVM_DUAL:
			solve_linear_c_svc(prob, w, eps, Cp, Cn, L2LOSS_SVM_DUAL);
			break;
		case L1LOSS_SVM_DUAL:
			solve_linear_c_svc(prob, w, eps, Cp, Cn, L1LOSS_SVM_DUAL);
			break;
		default:
			fprintf(stderr, "Error: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	// constructing the subproblem
	feature_node **x = Malloc(feature_node *,l);
	for(i=0;i<l;i++)
		x[i] = prob->x[perm[i]];

	int k;
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	if(nr_class==2)
	{
		model_->w=Malloc(double, n);

		int e0 = start[0]+count[0];
		k=0;
		for(; k<e0; k++)
			sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++)
			sub_prob.y[k] = -1;

		train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
	}
	else
	{
		model_->w=Malloc(double, n*nr_class);
		double *w=Malloc(double, n);
		for(i=0;i<nr_class;i++)
		{
			int si = start[i];
			int ei = si+count[i];

			k=0;
			for(; k<si; k++)
				sub_prob.y[k] = -1;
			for(; k<ei; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			train_one(&sub_prob, param, w, weighted_C[i], param->C);

			for(int j=0;j<n;j++)
				model_->w[j*nr_class+i] = w[j];
		}
		free(w);
	}

	free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	free(weighted_C);
	return model_;
}

void destroy_model(struct model *model_)
{
	if(model_->w != NULL)
		free(model_->w);
	if(model_->label != NULL)
		free(model_->label);
	free(model_);
}

const char *solver_type_table[]=
{
	"L2_LR", "L2LOSS_SVM_DUAL", "L2LOSS_SVM","L1LOSS_SVM_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_classifier;
	if(model_->nr_class==2)
		nr_classifier=1;
	else
		nr_classifier=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_classifier+j]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
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
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
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
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model_->w=Malloc(double, n*nr_classifier);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_classifier+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_classifier;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_classifier;i++)
				dec_values[i] += w[(idx-1)*nr_classifier+i]*lx->value;
	}

	if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

int predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(model_->param.solver_type==L2_LR)
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_classifier;
		if(nr_class==2)
			nr_classifier = 1;
		else
			nr_classifier = nr_class;

		int label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_classifier;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2_LR
	   && param->solver_type != L2LOSS_SVM_DUAL
	   && param->solver_type != L2LOSS_SVM
	   && param->solver_type != L1LOSS_SVM_DUAL)
		return "unknown solver type";

//	if(param->solver_type == L1_LR)
//		return "sorry! sover_type = 1 (L1_LR) is not supported yet";

	return NULL;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(int,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

/*
Copy from tron.c of LIBLINEAR
Changes:
	- Comment include tron.h
	- ifdef around min, max definitions
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
//#include "tron.h"

#if _MSC_VER!=0 && _MSC_VER<1300
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif


TRON::TRON(const function *fun_obj, double eps, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double delta, snorm, one=1.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *w_new = new double[n];
	double *g = new double[n];

	for (i=0; i<n; i++)
		w[i] = 0;

        f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	delta = dnrm2_(&n, g, &inc);
	double gnorm1 = delta;
	double gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	iter = 1;

	while (iter <= max_iter && search)
	{
		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
                fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
	        actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = dnrm2_(&n, s, &inc);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));

		//printf("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double)*n);
			f = fnew;
		        fun_obj->grad(w, g);

			gnorm = dnrm2_(&n, g, &inc);
			if (gnorm <= eps*gnorm1)
				break;
		}
		if (f < -1.0e+32)
		{
			printf("warning: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0 && prered <= 0)
		{
			printf("warning: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			printf("warning: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
}

int TRON::trcg(double delta, double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		if (dnrm2_(&n, s, &inc) > delta)
		{
			//printf("cg reaches trust region boundary\n");
			alpha = -alpha;
			daxpy_(&n, &alpha, d, &inc, s, &inc);

			double std = ddot_(&n, s, &inc, d, &inc);
			double sts = ddot_(&n, s, &inc, s, &inc);
			double dtd = ddot_(&n, d, &inc, d, &inc);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			daxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}


/*
The folowing load save functions are used for orange pickling
*/

int linear_save_model_alt(string &buffer, struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	FILE *fp = tmpfile();
	if(fp==NULL) return -1;

	int nr_classifier;
	if(model_->nr_class==2)
		nr_classifier=1;
	else
		nr_classifier=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_classifier+j]);
		fprintf(fp, "\n");
	}

	fseek(fp, SEEK_SET, 0);
	char str[512];
	while(fgets(str, 512, fp)){
		buffer+=str;
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *linear_load_model_alt(string &buffer)
{
	FILE *fp = tmpfile();
	if(fp==NULL) return NULL;

	fprintf(fp, buffer.c_str());
	fseek(fp, SEEK_SET, 0);

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
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
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
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
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model_->w=Malloc(double, n*nr_classifier);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_classifier+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

#include <iostream>
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

feature_node *feature_nodeFromExample(const TExample &ex, map<int, int> &indexMap, bool includeMeta=false, bool includeRegular=true){
	//cout << "example " << endl;
	int numOfNodes = countFeatures(ex, includeMeta, includeRegular);
	/*if (includeRegular)
		numOfNodes += ex.domain->attributes->size();
	if (includeMeta)
		numOfNodes += ex.meta.size();*/
	feature_node *nodes = new feature_node[numOfNodes];
	feature_node *ptr = nodes;
	int index = 1;
	int featureIndex = 1;
	if (includeRegular){
		for (TExample::iterator i=ex.begin(); i!=ex.end(); i++){
			if ((i->varType==TValue::INTVAR || (i->varType==TValue::FLOATVAR && (*i==*i))) && i->isRegular() && i!=&ex.getClass()){
				if (i->varType==TValue::INTVAR)
					ptr->value = (int) *i;
				else
					ptr->value = (float) *i;
				ptr->index = index;
				if (indexMap.find(index)==indexMap.end()){
					ptr->index = featureIndex;
					indexMap[index] = featureIndex++;
				} else
					ptr->index = indexMap[index];
				//featureIndices.insert(index);
				//cout << ptr->value << " ";
				ptr++;
			}
			index++;
		}
	}
	if (includeMeta){
		feature_node *first = ptr;
		for (TMetaValues::const_iterator i=ex.meta.begin(); i!=ex.meta.end(); i++){
			if ((i->second.valueType==TValue::INTVAR || i->second.valueType==TValue::FLOATVAR) && i->second.isRegular()){
				ptr->value = (float) i->second;
				//ptr->index = index - i->first;
				if (indexMap.find(i->first)==indexMap.end()){
					ptr->index = featureIndex;
					indexMap[i->first] = featureIndex++;
				} else
					ptr->index = indexMap[i->first];
				//featureIndices.insert(ptr->index);
				ptr++;
			}
		}
		//cout << endl << "	sorting" << endl;
		sort(first, ptr, NodeSort());
	}
	ptr->index = -1;
	return nodes;
}

problem *problemFromExamples(PExampleGenerator examples, map<int, int> &indexMap, bool includeMeta=false, bool includeRegular=true){
	problem *prob = new problem;
	prob->l = examples->numberOfExamples();
	prob->x = new feature_node* [prob->l];
	prob->y = new int [prob->l];
	prob->bias = -1.0;
	feature_node **ptrX = prob->x;
	int *ptrY = prob->y;
	PEITERATE(iter, examples){
		*ptrX = feature_nodeFromExample(*iter, indexMap, includeMeta, includeRegular);
		*ptrY = (int) (*iter).getClass();
		ptrX++;
		ptrY++;
	}
	prob->n = indexMap.size();
	//cout << "prob->n " << prob->n <<endl;
	return prob;
}

void destroy_problem(problem *prob){
	for (int i=0; i<prob->l; i++)
		delete[] prob->x[i];
	delete[] prob->x;
	delete[] prob->y;
}

TLinearLearner::TLinearLearner(){
	solver_type = L2_LR;
	eps = 0.01f;
	C=1;
}

PClassifier TLinearLearner::operator()(PExampleGenerator examples, const int &weight){
	//cout << "initializing param" << endl;
	parameter *param = new parameter;
	param->solver_type = solver_type;
	param->eps = eps;
	param->C = C;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;
	//cout << "initializing problem" << endl;
	map<int, int> *indexMap =new map<int, int>;
	problem *prob = problemFromExamples(examples, *indexMap);
	//cout << "cheking parameters" << endl;
	const char * error_msg = check_parameter(prob, param);
	if (error_msg){
		delete param;
		destroy_problem(prob);
		raiseError("LIBLINEAR error: %s" , error_msg);
	}
	//cout << "trainig" << endl;
	model *model = train(prob, param);
	destroy_problem(prob);

	return PClassifier(mlnew TLinearClassifier(examples->domain->classVar, examples, model, indexMap));
}

TLinearClassifier::TLinearClassifier(const PVariable &var, PExampleTable _examples, struct model *_model, map<int, int> *_indexMap){
	classVar = var;
	linmodel = _model;
	examples = _examples;
	domain = examples->domain;
	indexMap = _indexMap;
	computesProbabilities = linmodel->param.solver_type == L2_LR;
	int nr_classifier = (linmodel->nr_class==2)? 1 : linmodel->nr_class;
	weights = mlnew TFloatListList(nr_classifier);
	for (int i=0; i<nr_classifier; i++){
		weights->at(i) = mlnew TFloatList(linmodel->nr_feature);
		for (int j=0; j<linmodel->nr_feature; j++)
			weights->at(i)->at(j) = linmodel->w[j*nr_classifier+i];
	}
}

TLinearClassifier::~TLinearClassifier(){
	if (linmodel)
		destroy_model(linmodel);
	if (indexMap)
		delete indexMap;
}

PDistribution TLinearClassifier::classDistribution(const TExample &example){
	int numClass = get_nr_class(linmodel);
	map<int, int> indexMap;
	feature_node *x = feature_nodeFromExample(example, indexMap, false);

	int *labels = new int [numClass];
	get_labels(linmodel, labels);

	double *prob_est = new double [numClass];
	predict_probability(linmodel, x, prob_est);

	PDistribution dist = TDistribution::create(example.domain->classVar);
	for (int i=0; i<numClass; i++)
		dist->setint(labels[i], prob_est[i]);

	delete[] x;
	delete[] labels;
	delete[] prob_est;
	return dist;
}

TValue TLinearClassifier::operator () (const TExample &example){
	int numClass = get_nr_class(linmodel);
	map<int, int> indexMap;
	feature_node *x = feature_nodeFromExample(example, indexMap, false);

	int predict_label = predict(linmodel ,x);
	delete[] x;
	return TValue(predict_label);
}

