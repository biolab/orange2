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

#include <math.h>
#include <stdlib.h>
#include <cstring>

#include "vars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "classify.hpp"
#include "err.h"

#include "tdidt_simple.ppp"

#define ASSERT(x) if (!(x)) err(1, "%s:%d", __FILE__, __LINE__)
#define LOG2(x) log((x)) / log(2.0)

enum { DiscreteNode, ContinuousNode, PredictorNode };

struct Args {
    int minExamples, maxDepth;
    float maxMajority;

    int *attr_split_so_far;
};

int compar_attr;

/* This function uses the global variable compar_attr */
int
compar_examples(TExample **e1, TExample **e2)
{
    /*
    if ((*e1)->values[compar_attr].isSpecial())
        return 1;
    if ((*e2)->values[compar_attr].isSpecial())
        return -1;
    */
    return (*e1)->values[compar_attr].compare((*e2)->values[compar_attr]);
}

float
entropy(int *xs, int size)
{
	int *ip, *end, sum;
	float e, p;

    for (ip = xs, end = xs + size, e = 0, sum = 0; ip != end; ip++)
        if (*ip) {
            e -= *ip * LOG2(*ip);
            sum += *ip;
        }

	return e / sum + LOG2(sum);
}


float
score_attribute_c(TExample **examples, int size, int attr, float cls_entropy, int *rank, struct Args *args)
{
	PDomain domain;
    TExample **ex, **ex_end, **ex_next;
    int i, cls_vals, *attr_dist, *dist_lt, *dist_ge;
    float best_score;

    assert(size > 0);
    domain = examples[0]->domain;

    cls_vals = domain->classVar->noOfValues();

    /* allocate space */
    ASSERT(dist_lt = calloc(cls_vals, sizeof *dist_lt));
    ASSERT(dist_ge = calloc(cls_vals, sizeof *dist_ge));
    ASSERT(attr_dist = calloc(2, sizeof *attr_dist));

    /* sort */
    compar_attr = attr;
    qsort(examples, size, sizeof(TExample *), compar_examples);

    /* compute gain ratio for every split */
    for (ex = examples, ex_end = examples + size; ex != ex_end; ex++) {
        if (!(*ex)->getClass().isSpecial())
            dist_ge[(*ex)->getClass().intV]++;
        if ((*ex)->values[attr].isSpecial())
            size--;
    }

    best_score = -HUGE_VAL;
    attr_dist[1] = size;
    ex = examples + args->minExamples - 1;
    ex_end = examples + size - (args->minExamples - 1);
    for (ex_next = ex + 1, i = 0; ex_next < ex_end; ex++, ex_next++, i++) {
        int cls;
        float score;

        if (!(*ex)->getClass().isSpecial()) {
            cls = (*ex)->getClass().intV;
            dist_lt[cls]++;
            dist_ge[cls]--;
        }
        attr_dist[0]++;
        attr_dist[1]--;

        if ((*ex)->values[attr] == (*ex_next)->values[attr])
            continue;

        /* gain ratio */
        score = (attr_dist[0] * entropy(dist_lt, cls_vals) + attr_dist[1] * entropy(dist_ge, cls_vals)) / size;
        score = (cls_entropy - score) / entropy(attr_dist, 2);

        if (score > best_score) {
            best_score = score;
            *rank = i;
        }
    }

    /* cleanup */
    free(dist_lt);
    free(dist_ge);
    free(attr_dist);

    return best_score;
}

float
score_attribute_d(TExample **examples, int size, int attr, float cls_entropy, struct Args *args)
{
	PDomain domain;
	TExample **ex, **ex_end;
	int i, j, cls_vals, attr_vals, *cont, *attr_dist, *attr_dist_cls_known, min, size_attr_known, size_attr_cls_known;
	float score;

	assert(size > 0);
	domain = examples[0]->domain;

	cls_vals = domain->classVar->noOfValues();
	attr_vals = domain->attributes->at(attr)->noOfValues();

	/* allocate space */
    ASSERT(cont = calloc(cls_vals * attr_vals, sizeof *cont));
    ASSERT(attr_dist = calloc(attr_vals, sizeof *attr_dist));
    ASSERT(attr_dist_cls_known = calloc(attr_vals, sizeof *attr_dist));

	/* contingency matrix */
    size_attr_known = 0;
    size_attr_cls_known = 0;
	for (ex = examples, ex_end = examples + size; ex != ex_end; ex++) {
        if (!(*ex)->values[attr].isSpecial()) {
            int attr_val = (*ex)->values[attr].intV;

            attr_dist[attr_val]++;
            size_attr_known++;
            if (!(*ex)->getClass().isSpecial()) {
                int cls_val = (*ex)->getClass().intV;

                size_attr_cls_known++;
                attr_dist_cls_known[attr_val]++;
                cont[attr_val * cls_vals + cls_val]++;
            }
        }
    }

	/* minimum examples in leaves */
	for (i = 0; i < attr_vals; i++)
		if (attr_dist[i] < args->minExamples) {
            score = -INFINITY;
            goto finish;
        }

	/* gain ratio */
	score = 0;
	for (i = 0; i < attr_vals; i++)
		score += attr_dist_cls_known[i] * entropy(cont + i * cls_vals, cls_vals);
	score = (cls_entropy - score / size_attr_cls_known) / entropy(attr_dist, attr_vals) * ((float)size_attr_known / size);

finish:
	free(cont);
    free(attr_dist);
    free(attr_dist_cls_known);
	return score;
}

struct SimpleTreeNode *
build_tree(TExample **examples, int size, int depth, struct Args *args)
{
	PDomain domain;
	TExample **ex, **ex_top, **ex_end;
	TVarList::const_iterator it;
	struct SimpleTreeNode *node;
	float score, best_score, cls_entropy;
	int i, best_attr, best_rank, sum, best_val, finish, cls_vals;

	assert(size > 0);
	domain = examples[0]->domain;

	ASSERT(node = malloc(sizeof(*node)));

	/* class distribution */
    cls_vals = domain->classVar->noOfValues();

    ASSERT(node->dist = calloc(cls_vals, sizeof *node->dist));
	for (ex = examples, ex_end = examples + size; ex != ex_end; ex++)
        if (!(*ex)->getClass().isSpecial())
            node->dist[(*ex)->getClass().intV]++;

	/* stop splitting with majority class or depth exceeds limit */
	best_val = 0;
	for (i = 0; i < cls_vals; i++)
		if (node->dist[i] > node->dist[best_val])
			best_val = i;
	finish = depth == args->maxDepth || node->dist[best_val] >= args->maxMajority * size;

    /* score attributes */
	if (!finish) {
        cls_entropy = entropy(node->dist, cls_vals);
		best_score = -INFINITY;
		for (i = 0, it = domain->attributes->begin(); it != domain->attributes->end(); it++, i++)
			if (!args->attr_split_so_far[i]) {
                if ((*it)->varType == TValue::INTVAR) {
                    score = score_attribute_d(examples, size, i, cls_entropy, args);
                    if (score > best_score) {
                        best_score = score;
                        best_attr = i;
                    }
                } else {
                    int rank;
                    assert((*it)->varType == TValue::FLOATVAR);

                    score = score_attribute_c(examples, size, i, cls_entropy, &rank, args);
                    if (score > best_score) {
                        best_score = score;
                        best_rank = rank;
                        best_attr = i;
                    }
                }
			}
		finish = best_score == -INFINITY;
	}

    /* stop splitting - produce predictor node */
	if (finish) {
		node->type = PredictorNode;
        return node;
    }
    free(node->dist);

    /* remove examples with unknown values */
    for (ex = examples, ex_top = examples, ex_end = examples + size; ex != ex_end; ex++)
        if (!(*ex)->values[best_attr].isSpecial())
            *ex_top++ = *ex;
    size = ex_top - examples;

    if (domain->attributes->at(best_attr)->varType == TValue::INTVAR) {
		TExample **tmp;
		int *cnt, no_of_values;

		/* printf("%2d %3s %3d %f\n", depth, domain->attributes->at(best_attr)->get_name().c_str(), size, best_score); */

		node->type = DiscreteNode;
		node->split_attr = best_attr;
		
		/* counting sort */
		no_of_values = domain->attributes->at(best_attr)->noOfValues();

		ASSERT(tmp = calloc(size, sizeof *tmp));
		ASSERT(cnt = calloc(no_of_values, sizeof *cnt));
		
		for (ex = examples, ex_end = examples + size; ex != ex_end; ex++)
			cnt[(*ex)->values[best_attr].intV]++;

		for (i = 1; i < no_of_values; i++)
			cnt[i] += cnt[i - 1];

		for (ex = examples, ex_end = examples + size; ex != ex_end; ex++)
			tmp[--cnt[(*ex)->values[best_attr].intV]] = *ex;

		memcpy(examples, tmp, size * sizeof(TExample **));

		/* recursively build subtrees */
		node->children_size = no_of_values;
		ASSERT(node->children = calloc(no_of_values, sizeof *node->children));

		args->attr_split_so_far[best_attr] = 1;
		for (i = 0; i < no_of_values; i++) {
			int new_size;

			new_size = (i == no_of_values - 1) ? size - cnt[i] : cnt[i + 1] - cnt[i];
			node->children[i] = build_tree(examples + cnt[i], new_size, depth + 1, args);
		}
		args->attr_split_so_far[best_attr] = 0;

		free(tmp);
		free(cnt);
    } else if (domain->attributes->at(best_attr)->varType == TValue::FLOATVAR) {
        compar_attr = best_attr;
        qsort(examples, size, sizeof(TExample *), compar_examples);

        node->type = ContinuousNode;
        node->split_attr = best_attr;
        node->split = (examples[best_rank]->values[best_attr].floatV + examples[best_rank + 1]->values[best_attr].floatV) / 2.0;

		/* printf("%2d %3s %.4f\n", depth, domain->attributes->at(best_attr)->get_name().c_str(), node->split); */

        /* recursively build subtrees */
        node->children_size = 2;
        ASSERT(node->children = calloc(2, sizeof *node->children));
        
        node->children[0] = build_tree(examples, best_rank + 1, depth + 1, args);
        node->children[1] = build_tree(examples + best_rank + 1, size - (best_rank + 1), depth + 1, args);
    }

	return node;
}

TSimpleTreeLearner::TSimpleTreeLearner(const int &weight, float maxMajority, int minExamples, int maxDepth) :
    maxMajority(maxMajority),
    minExamples(minExamples),
    maxDepth(maxDepth)
{
}

PClassifier 
TSimpleTreeLearner::operator()(PExampleGenerator ogen, const int &weight)
{ 
	int i, *attr_split_so_far;
	TExample **examples, **ex;
	struct SimpleTreeNode *tree;
    struct Args args;

	if (!ogen->domain->classVar)
    	raiseError("class-less domain");
	
	/* create a tabel with pointers to examples */
	ASSERT(examples = calloc(ogen->numberOfExamples(), sizeof(TExample**)));
	ex = examples;
	PEITERATE(ei, ogen)
        *(ex++) = &(*ei);

	ASSERT(args.attr_split_so_far = calloc(ogen->domain->attributes->size(), sizeof(int)));
    args.minExamples = minExamples;
    args.maxMajority = maxMajority;
    args.maxDepth = maxDepth;

	tree = build_tree(examples, ogen->numberOfExamples(), 0, &args);

	free(examples);
	free(args.attr_split_so_far);

	return new TSimpleTreeClassifier(ogen->domain->classVar, tree);
}


/* classifier */
TSimpleTreeClassifier::TSimpleTreeClassifier()
{
}

TSimpleTreeClassifier::TSimpleTreeClassifier(const PVariable &classVar, struct SimpleTreeNode *t) 
	: TClassifier(classVar, true)
{
	tree = t;
}

void
destroy_tree(struct SimpleTreeNode *node)
{
    int i;

    if (node->type != PredictorNode) {
        for (i = 0; i < node->children_size; i++)
            destroy_tree(node->children[i]);
        free(node->children);
    } else {
        free(node->dist);
    }
    free(node);
}

TSimpleTreeClassifier::~TSimpleTreeClassifier()
{
    destroy_tree(tree);
}

int *
classify(const TExample &ex, struct SimpleTreeNode *node, int *free_dist)
{
	while (node->type != PredictorNode) {
        if (ex.values[node->split_attr].isSpecial()) {
            int i, j, cls_vals, *dist, *child_dist;

            cls_vals = ex.domain->classVar->noOfValues();
            ASSERT(dist = calloc(cls_vals, sizeof *dist));
            for (i = 0; i < node->children_size; i++) {
                child_dist = classify(ex, node->children[i], free_dist);
                for (j = 0; j < cls_vals; j++)
                    dist[j] += child_dist[j];
                if (*free_dist)
                    free(child_dist);
            }
            *free_dist = 1;
            return dist;
        } else if (node->type == DiscreteNode) {
            node = node->children[ex.values[node->split_attr].intV];
        } else {
            assert(node->type == ContinuousNode);
            node = node->children[ex.values[node->split_attr].floatV >= node->split];
        }
    }
    *free_dist = 0;
	return node->dist;
}

TValue 
TSimpleTreeClassifier::operator()(const TExample &ex)
{
    int i, *dist, free_dist, best_val;

    dist = classify(ex, tree, &free_dist);
    best_val = 0;
    for (i = 1; i < ex.domain->classVar->noOfValues(); i++)
        if (dist[i] > dist[best_val])
            best_val = i;

    if (free_dist)
        free(dist);
    return TValue(best_val);
}

PDistribution
TSimpleTreeClassifier::classDistribution(const TExample &ex)
{
    int i, *dist, free_dist;

    dist = classify(ex, tree, &free_dist);

	PDistribution pdist = TDistribution::create(ex.domain->classVar);
    for (i = 0; i < ex.domain->classVar->noOfValues(); i++)
        pdist->setint(i, dist[i]);
    pdist->normalize();

    if (free_dist)
        free(dist);
    return pdist;
}

void
TSimpleTreeClassifier::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{
	value = operator()(ex);
	dist = classDistribution(ex);
}
