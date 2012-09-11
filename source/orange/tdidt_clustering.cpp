/*
 This file is part of Orange.

 Copyright 1996-2012 Faculty of Computer and Information Science, University of Ljubljana
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
 along with Orange.	If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <cstring>

#include "vars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "tdidt_clustering.ppp"

#ifndef _MSC_VER
#include "err.h"
#define ASSERT(x) if (!(x)) err(1, "%s:%d", __FILE__, __LINE__)
#else
#define ASSERT(x) if(!(x)) exit(43)
#endif // _MSC_VER 
#define log2f(x) log((double) (x)) / log(2.0)
#ifndef INFINITY
#include <limits>
#define INFINITY numeric_limits<float>::infinity()
#endif // INFINITY
struct Arguments {
	int minInstances, maxDepth;
	float minMajority, minMSE, skipProb;

	int type, method, *attr_split_so_far;
	PDomain domain;
	PRandomGenerator randomGenerator;
};

struct Example {
	TExample *example;
	float weight;
};
enum {
	DiscreteNode, ContinuousNode, PredictorNode
};
enum {
	Classification, Regression
};
enum{
	Inter, Intra, Silhouette, Gini
};

/* This function uses the global variable comparator_attr.
 * Examples with unknowns are larger so that, when sorted, they appear at the bottom.
 */

int comparator_attr;
int comparator_examples(const void *ptr1, const void *ptr2) {
	struct Example *e1, *e2;

	e1 = (struct Example *) ptr1;
	e2 = (struct Example *) ptr2;
	if (e1->example->values[comparator_attr].isSpecial())
		return 1;
	if (e2->example->values[comparator_attr].isSpecial())
		return -1;
	return e1->example->values[comparator_attr].compare(e2->example->values[comparator_attr]);
}

int test_for_min_examples(float *attr_dist, int attr_vals, struct Arguments *args) {
	int i;
	if (args->minInstances == 0)
		return 1;

	for (i = 0; i < attr_vals; i++) {
		if (attr_dist[i] > 0.0 && attr_dist[i] < args->minInstances)
			return 0;
	}
	return 1;
}

float gini(float *probs, int n_vals){
	float ret = 0.0;
	for(int i = 0; i < n_vals; i++)
		ret += probs[i] * probs[i];
	return ret;
}


float distance_gini(struct Example *examples, int size, int attr, int *cls_vals, float gini_prior, struct Arguments *args)
{
	TValue *cls, *cls_end;
	struct Example *ex, *ex_end;
	int i, j, k, attr_vals, n_classes, attr_val;
	float score = 0.0, temp,  *attr_distr, ***cls_dist, **size_weight;
	
	n_classes = args->domain->classVars->size();
	attr_vals = args->domain->attributes->at(attr)->noOfValues();

	/* allocate space */
	ASSERT(attr_distr = (float *)calloc(attr_vals, sizeof(float)));
	ASSERT(cls_dist = (float ***) calloc(attr_vals, sizeof(float **)));
	ASSERT(size_weight = (float **) calloc(attr_vals, sizeof(float *)));
	for(i = 0; i < attr_vals; i++){
		ASSERT(cls_dist[i] = (float **) calloc(n_classes, sizeof(float *)));
		ASSERT(size_weight[i] = (float *) calloc(n_classes, sizeof(float)));
		for(j = 0; j < n_classes; j++){
			ASSERT(cls_dist[i][j] = (float *) calloc(cls_vals[j], sizeof(float)));
			for(k = 0; k < cls_vals[j]; k++){
				cls_dist[i][j][k]=0.0;
			}
			size_weight[i][j]=0.0;
		}
	}
	
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!ex->example->values[attr].isSpecial()) {
			attr_val = ex->example->values[attr].intV;
			attr_distr[attr_val] += ex->weight;
			for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				if (!cls->isSpecial()) {
					i = cls  + n_classes - cls_end;
					cls_dist[attr_val][i][cls->intV]  += ex->weight;
					size_weight[attr_val][i] += ex->weight;
				}
			}
		}
	}
	/* min examples in leaves */
	if (!test_for_min_examples(attr_distr, attr_vals, args)) {
		score = -INFINITY;
		goto finish;
	}

	for(i = 0; i < attr_vals; i++){
		temp = 0;
		for(j = 0; j < n_classes; j++){
			for(k = 0; k < cls_vals[j]; k++)
				cls_dist[i][j][k] /= size_weight[i][j];
			
			temp += gini(cls_dist[i][j], cls_vals[j]);
		}
		score += temp / n_classes * attr_distr[i] / size;
	}

	finish:
	free(attr_distr);
	for(i = 0; i < attr_vals; i++){
		for(j = 0; j < n_classes; j++)
			free(cls_dist[i][j]);
		free(cls_dist[i]);
		free(size_weight[i]);
	}
	free(cls_dist);
	free(size_weight);

	return score - gini_prior;
}

float** protottype_d(struct Example *examples, int size, int attr, struct Arguments *args,
		float *ptypes_size) {
	/* returns number of created prototypes (number of different attribute values) and the prototypes by reference */
	struct Example *ex, *ex_end;
	TValue *cls, *cls_end;
	int i, j, attr_vals, attr_val, n_classes;
	float *attr_dist, **ptypes, **n_ptypes;

	attr_vals = args->domain->attributes->at(attr)->noOfValues();
	n_classes = args->domain->classVars->size();

	ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof(float)));
	ASSERT(ptypes = (float **) calloc(attr_vals, sizeof(float *)));
	ASSERT(n_ptypes = (float **) calloc(attr_vals, sizeof(float *)));
	for (i = 0; i < attr_vals; i++) {
		ASSERT(ptypes[i] = (float *) calloc(n_classes, sizeof(float)));
		ASSERT(n_ptypes[i] = (float *) calloc(n_classes, sizeof(float)));
	}

	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!ex->example->values[attr].isSpecial()) {
			attr_val = ex->example->values[attr].intV;
			attr_dist[attr_val] += ex->weight;

			for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				if (!cls->isSpecial()) {
					i = cls  + n_classes - cls_end;
					ptypes[attr_val][i] += args->type == Classification ? cls->intV : cls->floatV;
					n_ptypes[attr_val][i] += ex->weight; //seperate because of missing values
				}
			}
		}
	}

	if (!test_for_min_examples(attr_dist, attr_vals, args)) {
		*ptypes_size = -1;
		for (i = 0; i < attr_vals; i++) {
			free(ptypes[i]);
		}
		free(ptypes);
		goto finish;
	}

	for (i = 0; i < attr_vals; ++i) {
		for (j = 0; j < n_classes; ++j) {
			if (n_ptypes[i][j] == 0.0)
				ptypes[i][j] = INFINITY;
			else
				ptypes[i][j] = ptypes[i][j] / n_ptypes[i][j];
		}
	}

	*ptypes_size = attr_vals;
	finish: for (i = 0; i < attr_vals; i++) {
		free(n_ptypes[i]);
	}
	free(n_ptypes);
	free(attr_dist);
	return ptypes;
}

float dist_inter(float **ptypes, int ptypes_size, struct Arguments *args){
	float dist = 0.0, d = 0.0;
	int n_dist = 0, n_classes, i, j, k;
	n_classes = args->domain->classVars->size();

	for (i = 0; i < ptypes_size - 1; i++) {
		for (j = i + 1; j < ptypes_size; j++) {
			for (k = 0; k < n_classes; k++) {
				d = ptypes[i][k] - ptypes[j][k];
				dist += d * d;
			}
			n_dist+=1;
		}
	}
	return dist / n_dist;
}


float dist_intra(float **ptypes, int ptypes_size, struct Example *examples, int size, int attr, struct Arguments *args, float split) {
	int i, n_classes, attr_val, n_dist=0;
	float dist = 0.0, d, *intra_dist, *attr_distr;
	struct Example *ex, *ex_end;
	TValue *cls, *cls_end;

	n_classes = args->domain->classVars->size();
	
	ASSERT(attr_distr = (float *)calloc(ptypes_size, sizeof(float)));
	ASSERT(intra_dist = (float *)calloc(ptypes_size, sizeof(float)));

	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!ex->example->values[attr].isSpecial()) {		
			if (split != INFINITY){
				attr_val = ex->example->values[attr].floatV;
				attr_val = attr_val >= split ? 1 : 0;
			}else
				attr_val = ex->example->values[attr].intV;

			attr_distr[attr_val] += ex->weight;
			for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				if (!cls->isSpecial()) {
					i = cls  + n_classes - cls_end;
					d = args->type == Classification ? cls->intV : cls->floatV;
					d = ptypes[attr_val][i] - d;
					intra_dist[attr_val] += d * d;
				}
			}
		}
	}
	
	for (i = 0; i < ptypes_size; i++) 
		dist+=intra_dist[i]/attr_distr[i];
	
	free(intra_dist);
	free(attr_distr);
	return dist ;
}

float dist_silhuette(float **ptypes, int ptypes_size, struct Example *examples, int size, int attr, struct Arguments *args, float split) {
	int i, j, n_classes, attr_val, n_dist=0;
	float dist = 0, temp, d, *cls_vals, *cls_vals_n, inter_dist, intra_dist;
	struct Example *ex,  *ex_end;
	TValue *cls, *cls_end;

	n_classes = args->domain->classVars->size();
	
	ASSERT(cls_vals = (float *)calloc(n_classes, sizeof(float)));
	ASSERT(cls_vals_n = (float *)calloc(n_classes, sizeof(float)));
	
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!ex->example->values[attr].isSpecial()) {
			for (i = 0; i < n_classes; i++)
				cls_vals_n[i] = 0;
			
			for (cls = ex->example->values_end,  cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				i = cls  + n_classes - cls_end;
				if (!cls->isSpecial()) {
					cls_vals[i] = args->type == Classification ? cls->intV : cls->floatV;
					cls_vals_n[i] += ex->weight;
				}else
					cls_vals[i]=-INFINITY;
			}
			
			if (split != INFINITY){
				attr_val = ex->example->values[attr].floatV;
				attr_val = attr_val >= split ? 1 : 0;
			}else
				attr_val = ex->example->values[attr].intV;
			

			for (i = 0; i < n_classes; i++){
				if(cls_vals_n[i]!=0)
					cls_vals[i] = cls_vals[i] / cls_vals_n[i];
				else
					cls_vals[i]=INFINITY;
			}

			intra_dist = 0;
			intra_dist = INFINITY;
			for (i = 0; i < ptypes_size; i++){
				temp = 0;
				for(j = 0; j < n_classes; j++){
					d = ptypes[i][j] - cls_vals[i];
					temp += d*d;
				}
				if(i == attr_val){
					inter_dist = temp;
				}else{
					if(temp < intra_dist)
						intra_dist = temp;
				}
			}
			
			temp = inter_dist - intra_dist;
			if (intra_dist > inter_dist)
				temp /= intra_dist;
			else
				temp /= inter_dist;
			dist += temp;
			n_dist++;
		}
	}
	free(cls_vals);
	free(cls_vals_n);
	dist = dist / n_dist;

	if(dist < -1)
		return -1;
	else if (dist > 1)
		return 1;
	else
		return dist;
}

float distance_d(struct Example *examples, int size, int attr, struct Arguments *args) {
	int i;
	float dist = 0, ptypes_size, **ptypes;
	
	ptypes = protottype_d(examples, size, attr, args, &ptypes_size);

	if (ptypes_size == -1){
		return -INFINITY;
	}else if(ptypes_size == 1){
		return 0.0;
	}
	ASSERT(ptypes);
	
	
	if (args->method == Intra)
		dist = dist_intra(ptypes, ptypes_size, examples, size, attr, args, INFINITY);
	else if(args->method == Silhouette)
		return dist_silhuette(ptypes, ptypes_size, examples, size, attr, args, INFINITY);
	else
		dist = dist_inter(ptypes, ptypes_size, args);

	for (i = 0; i < ptypes_size; i++) {
		free(ptypes[i]);
	}
	free(ptypes);
	return dist;
}

float distance_c(struct Example *examples, int size, int attr, struct Arguments *args,
		float *best_split) {
	struct Example *ex, *ex_end, *ex_next;
	TValue *cls, *cls_end;
	int i, j, n_classes, size_known, minInstances;
	float dist, split, best_dist = -INFINITY, d, *ptype1, *ptype2, *n_ptype1, *n_ptype2, **ptypes;

	/* minInstances should be at least 1, otherwise there is no point in splitting */
	minInstances = args->minInstances < 1 ? 1 : args->minInstances;

	n_classes = args->domain->classVars->size();
	ASSERT(ptype1 = (float *) calloc(n_classes, sizeof(float)));
	ASSERT(ptype2 = (float *) calloc(n_classes, sizeof(float)));
	ASSERT(n_ptype1 = (float *) calloc(n_classes, sizeof(float)));
	ASSERT(n_ptype2 = (float *) calloc(n_classes, sizeof(float)));
	ASSERT(ptypes = (float **) calloc(2, sizeof( float *)));
	/* sort */
	comparator_attr = attr;
	qsort(examples, size, sizeof(struct Example), comparator_examples);

	size_known = size;
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (ex->example->values[attr].isSpecial()) {
			size_known = ex - examples;
			break;
		}
		for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
				cls < cls_end; cls++) {
			if (!cls->isSpecial()) {
				i = cls  + n_classes - cls_end;
				ptype2[i] += args->type == Classification ? cls->intV : cls->floatV;
				n_ptype2[i] += ex->weight;
			}
		}
	}
	for (ex = examples, ex_end = ex + size_known - minInstances, ex_next = ex + 1, i = 0;
			ex < ex_end; ex++, ex_next++, i++) {
		for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
				cls < cls_end; cls++) {
			if (!cls->isSpecial()) {
				j = cls  + n_classes - cls_end;
				ptype1[j] += args->type == Classification ? cls->intV : cls->floatV;
				n_ptype1[j] += ex->weight;
				ptype2[j] -= args->type == Classification ? cls->intV : cls->floatV;
				n_ptype2[j] -= ex->weight;
			}
		}
		if (ex->example->values[attr] == ex_next->example->values[attr]
		|| i + 1 < minInstances)
			continue;
		
		split = (ex->example->values[attr].floatV + ex_next->example->values[attr].floatV) / 2.0;
		
		ptypes[0] = ptype1;
		ptypes[1] = ptype2;

		if(args->method == Silhouette)
			dist = dist_silhuette(ptypes, 2, examples, size, attr, args, split);
		else if (args->method == Intra)
			dist = dist_intra(ptypes, 2, examples, size, attr, args, split);
		else
			dist = dist_inter(ptypes, 2, args);

		dist = 0;
		for (j = 0; j < n_classes; j++) {
			d = ptype1[j] / n_ptype1[j] - ptype2[j] / n_ptype2[j];
			dist += d * d;
		}
		
		if (dist > best_dist) {
			best_dist = dist;
			*best_split = split;
		}
	}
	free(n_ptype1);
	free(n_ptype2);
	free(ptype1);
	free(ptype2);
	free(ptypes);
	return best_dist;
}

struct ClusteringTreeNode *
make_predictor(struct ClusteringTreeNode *node, struct Example *examples, int size,
		struct Arguments *args) {
	node->type = PredictorNode;
	node->children_size = 0;
	return node;
}

struct ClusteringTreeNode *build_tree(struct Example *examples, int size, int depth,
		struct ClusteringTreeNode *parent, struct Arguments *args) {
	int i, j, n_classes, best_attr, *cls_vals, stop_maj;
	float cls_mse, best_score, score, *size_weight, best_split, split, gini_prior;
	struct ClusteringTreeNode *node;
	struct Example *ex, *ex_end;
	TValue *cls, *cls_end;
	TVarList::const_iterator it;

	n_classes = args->domain->classVars->size();
	ASSERT(cls_vals = (int *) calloc(n_classes, sizeof(int)));
	for (i = 0; i < n_classes; i++) {
		cls_vals[i] = args->domain->classVars->at(i)->noOfValues();
	}
	
	ASSERT(node = (ClusteringTreeNode *)malloc(sizeof *node));
	node->n_classes = n_classes;
	if (args->type == Classification) {
		ASSERT(node->dist = (float **) calloc(n_classes, sizeof(float *)));
		for (i = 0; i < n_classes; i++)
			ASSERT( node->dist[i] = (float *) calloc(cls_vals[i], sizeof(float)));
		
		if (size == 0) {
			assert(parent);
			node->type = PredictorNode;
			node->children_size = 0;
			for (i = 0; i < n_classes; i++)
				memcpy(node->dist[i], parent->dist[i],
						cls_vals[i] * sizeof *node->dist[i]);
			free(cls_vals);
			return node;
		}
		
		/* class distribution */
		ASSERT(size_weight = (float *) calloc(n_classes, sizeof(float)));
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
			for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				if (!cls->isSpecial()) {
					i = cls  + n_classes - cls_end;	
					node->dist[i][cls->intV] += ex->weight;
					size_weight[i] += ex->weight;
				}
				
			}
		}

		/* stopping criterion: all class variables have to be above minMajority */
		stop_maj = 0;
		for (i = 0; i < n_classes; i++) {
			for (j = 0; j < cls_vals[i]; j++) {
				if (node->dist[i][j] / size_weight[i] >= args->minMajority)
					stop_maj++;
			}
		}
		free(size_weight);
		if (stop_maj == n_classes){
			free(cls_vals);
			return make_predictor(node, examples, size, args);
		}
		
		if(args->method == Gini){
			gini_prior = 0.0;
			for (i = 0; i < n_classes; i++){
				for (j=0; j < cls_vals[i]; j++) 
					gini_prior  = node->dist[i][j] / size_weight[i] * node->dist[i][j] / size_weight[i];
			}
			gini_prior /= n_classes;
		}

	} else {
		float *n, *sum, *sum2, cls_val;

		assert(args->type == Regression);
		if (size == 0) {
			assert(parent);
			node->type = PredictorNode;
			node->children_size = 0;
			node->n = parent->n;
			node->sum = parent->sum;
			free(cls_vals);
			return node;
		}

		ASSERT(n = (float *) calloc(n_classes, sizeof(float)));
		ASSERT(sum = (float *) calloc(n_classes, sizeof(float)));
		ASSERT(sum2 = (float *) calloc(n_classes, sizeof(float)));
		for (i = 0; i < n_classes; i++) 
			n[i] = sum[i] = sum2[i] = 0.0;
		
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
			for (cls = ex->example->values_end, cls_end = ex->example->classes_end;
					cls < cls_end; cls++) {
				if (!cls->isSpecial()) {
					cls_val = cls->floatV;
					i = cls  + n_classes - cls_end;
					n[i] += ex->weight;
					sum[i] += ex->weight * cls_val;
					sum2[i] += ex->weight * cls_val * cls_val;
				}
			}
		}

		node->n = n;
		node->sum = sum;

		stop_maj = 0;
		for (i = 0; i < n_classes; i++) {
			cls_mse = (sum2[i] - sum[i] * sum[i] / n[i]) / n[i];
			if (cls_mse < args->minMSE) {
				stop_maj += 1;
			}

		}
		free(sum2); 
		if (stop_maj == n_classes) {
			free(cls_vals);
			return make_predictor(node, examples, size, args);
		}
	}
	

	/* stopping criterion: depth exceeds limit */
	if (depth == args->maxDepth){
		free(cls_vals);
		return make_predictor(node, examples, size, args);
	}
	
	/* score attributes */
	best_score = -INFINITY;

	for (i = 0, it = args->domain->attributes->begin();
			it != args->domain->attributes->end(); it++, i++) {
		if (!args->attr_split_so_far[i]) {
			/* select random subset of attributes */
			if (args->randomGenerator->randdouble() < args->skipProb)
				continue;
			
			if ((*it)->varType == TValue::INTVAR) {
				score = args->method == Gini ? distance_gini(examples, size, i, cls_vals, gini_prior, args) :
					distance_d(examples, size, i, args);
				if (score > best_score) {
					best_score = score;
					best_attr = i;
				}
			} else if ((*it)->varType == TValue::FLOATVAR) {
				score = distance_c(examples, size, i, args, &split) ;
				if (score > best_score) {
					best_score = score;
					best_split = split;
					best_attr = i;
				}
			}
		}
	}
	
	free(cls_vals);
	if (best_score == -INFINITY){
		return make_predictor(node, examples, size, args);
	}
	//printf("* %2d %3s %3d %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_score); 

	if (args->domain->attributes->at(best_attr)->varType == TValue::INTVAR) {
		struct Example *child_examples, *child_ex;
		int attr_vals;
		float size_known, *attr_dist;

		//printf("* %2d %3s %3d %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_score); 

		attr_vals = args->domain->attributes->at(best_attr)->noOfValues();

		node->type = DiscreteNode;
		node->split_attr = best_attr;
		node->children_size = attr_vals;

		ASSERT( child_examples = (struct Example *)calloc(size, sizeof *child_examples));
		ASSERT(
				node->children = (ClusteringTreeNode **)calloc(attr_vals, sizeof *node->children));
		ASSERT( attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

		/* attribute distribution */
		size_known = 0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!ex->example->values[best_attr].isSpecial()) {
				attr_dist[ex->example->values[best_attr].intV] += ex->weight;
				size_known += ex->weight;
			}

		args->attr_split_so_far[best_attr] = 1;

		for (i = 0; i < attr_vals; i++) {
			/* create a new example table */
			for (ex = examples, ex_end = examples + size, child_ex = child_examples;
					ex < ex_end; ex++) {
				if (ex->example->values[best_attr].isSpecial()) {
					*child_ex = *ex;
					child_ex->weight *= attr_dist[i] / size_known;
					child_ex++;
				} else if (ex->example->values[best_attr].intV == i) {
					*child_ex++ = *ex;
				}
			}

			node->children[i] = build_tree(child_examples, child_ex - child_examples,
					depth + 1, node, args);
		}

		args->attr_split_so_far[best_attr] = 0;

		free(attr_dist);
		free(child_examples);
	} else {
		struct Example *examples_lt, *examples_ge, *ex_lt, *ex_ge;
		float size_lt, size_ge;

		/* printf("* %2d %3s %3d %f %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_split, best_score); */

		assert( args->domain->attributes->at(best_attr)->varType == TValue::FLOATVAR);

		ASSERT( examples_lt = (struct Example *)calloc(size, sizeof *examples));
		ASSERT( examples_ge = (struct Example *)calloc(size, sizeof *examples));

		size_lt = size_ge = 0.0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!ex->example->values[best_attr].isSpecial())
				if (ex->example->values[best_attr].floatV < best_split)
					size_lt += ex->weight;
				else
					size_ge += ex->weight;

		for (ex = examples, ex_end = examples + size, ex_lt = examples_lt, ex_ge =
				examples_ge; ex < ex_end; ex++)
			if (ex->example->values[best_attr].isSpecial()) {
				*ex_ge = *ex;
				ex_ge->weight *= size_ge / (size_lt + size_ge);
				ex_ge++;
			} else if (ex->example->values[best_attr].floatV < best_split) {
				*ex_lt++ = *ex;
			} else {
				*ex_ge++ = *ex;
			}

		node->type = ContinuousNode;
		node->split_attr = best_attr;
		node->split = best_split;
		node->children_size = 2;
		ASSERT(
				node->children = (ClusteringTreeNode **)calloc(2, sizeof *node->children));

		node->children[0] = build_tree(examples_lt, ex_lt - examples_lt, depth + 1, node,
				args);
		node->children[1] = build_tree(examples_ge, ex_ge - examples_ge, depth + 1, node,
				args);

		free(examples_lt);
		free(examples_ge);
	}

	return node;
}

TClusteringTreeLearner::TClusteringTreeLearner(const int &weight, float minMajority, float minMSE,
		int minInstances, int maxDepth, int method, float skipProb, PRandomGenerator rgen) :
		minMajority(minMajority), minMSE(minMSE), minInstances(minInstances), maxDepth(maxDepth),
			method(method), skipProb(skipProb) {
	randomGenerator = rgen ? rgen : PRandomGenerator(mlnew TRandomGenerator());
}

void _print_tree(struct ClusteringTreeNode *node, string indent, struct Arguments *args, int type, int* cls_vals) {
	int i, j, best_val;
	printf("%s",indent.c_str());
	if (node->type == DiscreteNode)
		printf("att:%s\n",args->domain->attributes->at(node->split_attr)->get_name().c_str());
	else if (node->type == ContinuousNode)
		printf("att:%s split:%f\n", args->domain->attributes->at(node->split_attr)->get_name().c_str(), node->split);
	else{
		if (type == Classification) {
			printf("[ ");
			for (i = 0; i < node->n_classes; i++) {

				best_val = 0;
				for (j = 1; j < cls_vals[i]; j++) {
					if (node->dist[i][j] > node->dist[i][best_val])
						best_val = j;
				}
				printf("%d ", best_val);
			}
			printf("]\n");
		} else {
			printf("[ ");
			assert(type == Regression);
			for (i = 0; i < node->n_classes; i++) {
				printf("%.2f ",node->sum[i] / node->n[i]);
			}
			printf("]\n");
		}
	}

	for (i = 0; i < node->children_size; i++)
		_print_tree(node->children[i], indent+"\t", args, type, cls_vals);
}


PMultiClassifier TClusteringTreeLearner::operator()(PExampleGenerator ogen,
		const int &weight) {
	struct Example *examples, *ex;
	struct ClusteringTreeNode *tree;
	struct Arguments args;
	int i, *cls_vals;

	if (ogen->domain->classVar)
		raiseError("not a multi-target domain, use standard classification trees");

	if (!ogen->numberOfExamples() > 0)
		raiseError("no examples test");

	/* create a tabel with pointers to examples */
	ASSERT(
			examples = (struct Example *)calloc(ogen->numberOfExamples(), sizeof *examples));
	ex = examples;
	PEITERATE(ei, ogen) {
		ex->example = &(*ei);
		ex->weight = 1.0;
		ex++;
	}

	ASSERT(
			args.attr_split_so_far = (int *)calloc(ogen->domain->attributes->size(), sizeof(int)));
	args.minInstances = minInstances;
	args.minMajority = minMajority;
	args.minMSE = minMSE;
	args.maxDepth = maxDepth;
	args.method = method;
	args.skipProb = skipProb;
	args.domain = ogen->domain;
	args.randomGenerator = randomGenerator;

	/* test for same type */
	args.type =
		ogen->domain->classVars->at(0)->varType == TValue::INTVAR ?
					Classification : Regression;
	for(int i = 0; i < ogen->domain->classVars->size(); i++){
		if((args.type == Classification && ogen->domain->classVars->at(i)->varType != TValue::INTVAR) ||
			(args.type == Regression && ogen->domain->classVars->at(i)->varType == TValue::INTVAR))
			raiseError("all classes must be of the same type"); //TODO: temporary?
	}

	ASSERT(cls_vals = (int *) calloc(ogen->domain->classVars->size(), sizeof(int)));
	for (i = 0; i < ogen->domain->classVars->size(); ++i)
		cls_vals[i] = ogen->domain->classVars->at(i)->noOfValues();

	tree = build_tree(examples, ogen->numberOfExamples(), 0, NULL, &args);

	//_print_tree(tree, "\t", &args, args.type, cls_vals);

	free(examples);
	free(args.attr_split_so_far);

	return new TClusteringTreeClassifier(ogen->domain->classVars, tree, args.type, cls_vals);
}

/* classifier */
TClusteringTreeClassifier::TClusteringTreeClassifier() {
}

TClusteringTreeClassifier::TClusteringTreeClassifier(const PVarList &classVars,
		struct ClusteringTreeNode *tree, int type, int *cls_vals) :
		TMultiClassifier(classVars, true), tree(tree), type(type), cls_vals(cls_vals) {
}

void destroy_tree(struct ClusteringTreeNode *node, int type) {
	int i;

	if (node->type != PredictorNode) {
		for (i = 0; i < node->children_size; i++)
			destroy_tree(node->children[i], type);
		free(node->children);
	}
	if (type == Classification) {
		for (i = 0; i < node->n_classes; i++)
			free(node->dist[i]);
		free(node->dist);
	}
	if (type == Regression) {
		free(node->sum);
		free(node->n);
	}
	free(node);
}

TClusteringTreeClassifier::~TClusteringTreeClassifier() {
	destroy_tree(tree, type);
	free(cls_vals);

}

float ** predict_classification(const TExample &ex, struct ClusteringTreeNode *node,
		int *free_dist, int *cls_vals) {
	int i, j, k;
	float **dist, **child_dist;

	while (node->type != PredictorNode)
		if (ex.values[node->split_attr].isSpecial()) {
			ASSERT(dist = (float **)calloc(node->n_classes, sizeof (float *)));
			for (i = 0; i < node->n_classes; i++)
				ASSERT( dist[i] = (float *) calloc(cls_vals[i], sizeof(float)));
			for (i = 0; i < node->children_size; i++) {
				child_dist = predict_classification(ex, node->children[i], free_dist,
						cls_vals);
				
				for (j = 0; j < node->n_classes; j++){
					for (k = 0; k < cls_vals[j]; ++k) {
						dist[j][k] += child_dist[j][k];
					}
				}
				if (*free_dist) {
					for (i = 0; i < node->n_classes; i++) 
						free(child_dist[i]);			
					free(child_dist);
				}
			}
			*free_dist = 1;
			return dist;
		} else if (node->type == DiscreteNode) {
			node = node->children[ex.values[node->split_attr].intV];
		} else {
			assert(node->type == ContinuousNode);
			node = node->children[ex.values[node->split_attr].floatV >= node->split];
		}

	*free_dist = 0;
	return node->dist;
}

void predict_regression(const TExample &ex, struct ClusteringTreeNode *node, float **sum,
		float **n) {
	int i, j;
	float *local_sum, *local_n;

	while (node->type != PredictorNode) {
		if (ex.values[node->split_attr].isSpecial()) {
			for (i = 0; i < node->n_classes; i++)
				*sum[i] = *n[i] = 0;
			for (i = 0; i < node->children_size; i++) {
				predict_regression(ex, node->children[i], &local_sum, &local_n);
				for (j = 0; j < node->n_classes; j++) {
					*sum[i] += local_sum[j];
					*n[j] += local_n[j];
				}
			}
			return;
		} else if (node->type == DiscreteNode) {
			assert( ex.values[node->split_attr].intV < node->children_size);
			node = node->children[ex.values[node->split_attr].intV];
		} else {
			assert(node->type == ContinuousNode);
			node = node->children[ex.values[node->split_attr].floatV > node->split];
		}
	}

	*sum = node->sum;
	*n = node->n;
}

void TClusteringTreeClassifier::save_tree(ostringstream &ss,
		struct ClusteringTreeNode *node) {
	int i, j;
	ss << "{ " << node->type << " " << node->children_size << " ";

	if (node->type != PredictorNode)
		ss << node->split_attr << " " << node->split << " ";

	for (i = 0; i < node->children_size; i++)
		this->save_tree(ss, node->children[i]);

	if (this->type == Classification) {

		for (i = 0; i < node->n_classes; i++) {
			for (j = 0; j < this->cls_vals[i]; ++j)
				ss << node->dist[i][j] << " ";
		}
	} else {
		assert(this->type == Regression);
		for (i = 0; i < node->n_classes; i++) {
			ss << node->n[i] << " " << node->sum[i] << " ";
		}
	}
	ss << "} ";
}


struct ClusteringTreeNode *
TClusteringTreeClassifier::load_tree(istringstream &ss, int n_classes) {
	int i, j;
	string lbracket, rbracket;
	string split_string;
	ClusteringTreeNode *node;

	ss.exceptions(istream::failbit);

	ASSERT(node = (ClusteringTreeNode *)malloc(sizeof *node));
	ss >> lbracket >> node->type >> node->children_size;
	node->n_classes=n_classes;
	if (node->type != PredictorNode) {
		ss >> node->split_attr;

		/* Read split into a string and use strtod to parse it.
		 * istream sometimes (on some platforms) seems to have problems
		 * reading formated floats.
		 */
		ss >> split_string;
		node->split = float(strtod(split_string.c_str(), NULL));
	}

	if (node->children_size) {
		ASSERT(
				node->children = (ClusteringTreeNode **)calloc(node->children_size, sizeof *node->children));
		for (i = 0; i < node->children_size; i++)
			node->children[i] = load_tree(ss, n_classes);
	}

	if (this->type == Classification) {
		ASSERT(node->dist = (float **)calloc(node->n_classes, sizeof(float *)));
		for (i = 0; i < node->n_classes; i++) {
			ASSERT( node->dist[i] = (float *)calloc(this->cls_vals[i], sizeof(float)));
			for (j = 0; j < this->cls_vals[i]; ++j) {
				ss >> node->dist[i][j];
			}
		}
	} else {
		assert(this->type == Regression);
		ASSERT(node->n = (float *)calloc(node->n_classes, sizeof(float)));
		ASSERT(node->sum = (float *)calloc(node->n_classes, sizeof(float)));
		for (i = 0; i < node->n_classes; i++) {
			ss >> node->n[i] >> node->sum[i];
		}
	}
	ss >> rbracket;

	/* Synchronization check */
	assert(lbracket == "{" && rbracket == "}");

	return node;
}

void TClusteringTreeClassifier::save_model(ostringstream &ss) {
	ss.precision(9); /* we have floats */
	ss << this->type << " " << this->tree->n_classes << " ";
	for (int i = 0; i < this->tree->n_classes; i++)
		ss << this->cls_vals[i] << " ";
	this->save_tree(ss, this->tree);
	
}

void TClusteringTreeClassifier::load_model(istringstream &ss) {
	int n_classes;
	ss >> this->type >> n_classes;
	ASSERT(this->cls_vals = (int *) calloc(n_classes, sizeof(int)));
	for (int i = 0; i < n_classes; i++)
		ss >> this->cls_vals[i];
	this->tree = load_tree(ss, n_classes);
	
}

PValueList TClusteringTreeClassifier::operator ()(const TExample &ex) {
	int i;
	PValueList classValues = new TValueList();

	if (type == Classification) {
		int j, free_dist, best_val;
		float **dist;

		dist = predict_classification(ex, tree, &free_dist, this->cls_vals);

		for (i = 0; i < tree->n_classes; i++) {
			best_val = 0;
			for (j = 1; j < this->cls_vals[i]; j++) {
				if (dist[i][j] > dist[i][best_val])
					best_val = j;
			}
			classValues->push_back(TValue(best_val));
		}
		if (free_dist) {
			for (i = 0; i < tree->n_classes; i++) {
				free(dist[i]);
			}
			free(dist);
		}
		return classValues;
	} else {
		float *sum, *n;

		assert(type == Regression);

		predict_regression(ex, tree, &sum, &n);
		for (i = 0; i < tree->n_classes; i++) {
			classValues->push_back(TValue(sum[i] / n[i]));
		}
		return classValues;
	}
}

PDistributionList TClusteringTreeClassifier::classDistribution(const TExample &ex) {
	if (type == Classification) {
		int i, j, free_dist;
		float **dist;
		PDistributionList classDists = new TDistributionList();
		PDistribution temp;
		dist = predict_classification(ex, tree, &free_dist, this->cls_vals);

		for (i = 0; i < tree->n_classes; i++) {
			temp = mlnew TDiscDistribution(this->cls_vals[i], 0.0);

			for (j = 0; j < this->cls_vals[i]; j++) {
				temp->setint(j, dist[i][j]);
			}
			temp->normalize();
			classDists->push_back(temp);
		}

		if (free_dist) {
			for (i = 0; i < tree->n_classes; i++) {
				free(dist[i]);
			}
			free(dist);
		}
		return classDists;
	} else {		
		PDistributionList classDists = new TDistributionList();
		for (int i = 0; i < tree->n_classes; i++) {
			classDists->push_back(NULL);
		}
		return classDists;
	}
}

void TClusteringTreeClassifier::predictionAndDistribution(const TExample &ex,
		PValueList &values, PDistributionList &dists) {
	values = operator()(ex);
	dists = classDistribution(ex);
}
