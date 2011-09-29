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

#include "tdidt_simple.ppp"

#ifndef _MSC_VER
    #include "err.h"
    #define ASSERT(x) if (!(x)) err(1, "%s:%d", __FILE__, __LINE__)
#else
    #define ASSERT(x) if(!(x)) exit(1)
    #define log2f(x) log((double) (x)) / log(2.0)
#endif // _MSC_VER

#ifndef INFINITY
    #include <limits>
    #define INFINITY numeric_limits<float>::infinity()
#endif // INFINITY

struct Args {
    int minInstances, maxDepth;
    float maxMajority, skipProb;

    int type, *attr_split_so_far;
    PDomain domain;
};

struct Example {
    TExample *example;
    float weight;
};

enum { DiscreteNode, ContinuousNode, PredictorNode };
enum { Classification, Regression };

int compar_attr;

/* This function uses the global variable compar_attr.
 * Examples with unknowns are larger so that, when sorted, they appear at the bottom.
 */
int
compar_examples(const void *ptr1, const void *ptr2)
{
    struct Example *e1, *e2;

    e1 = (struct Example *)ptr1;
    e2 = (struct Example *)ptr2;
    if (e1->example->values[compar_attr].isSpecial())
        return 1;
    if (e2->example->values[compar_attr].isSpecial())
        return -1;
    return e1->example->values[compar_attr].compare(e2->example->values[compar_attr]);
}


float
entropy(float *xs, int size)
{
    float *ip, *end, sum, e;

    for (ip = xs, end = xs + size, e = 0.0, sum = 0.0; ip != end; ip++)
        if (*ip > 0.0) {
            e -= *ip * log2f(*ip);
            sum += *ip;
        }

    return sum == 0.0 ? 0.0 : e / sum + log2f(sum);
}

int
test_min_examples(float *attr_dist, int attr_vals, struct Args *args)
{
    int i;

    for (i = 0; i < attr_vals; i++)
        if (attr_dist[i] > 0.0 && attr_dist[i] < args->minInstances)
            return 0;
    return 1;
}

float
gain_ratio_c(struct Example *examples, int size, int attr, float cls_entropy, struct Args *args, float *best_split)
{
    struct Example *ex, *ex_end, *ex_next;
    int i, cls, cls_vals, minInstances, size_known;
    float score, *dist_lt, *dist_ge, *attr_dist, best_score, size_weight;

    cls_vals = args->domain->classVar->noOfValues();

    /* minInstances should be at least 1, otherwise there is no point in splitting */
    minInstances = args->minInstances < 1 ? 1 : args->minInstances;

    /* allocate space */
    ASSERT(dist_lt = (float *)calloc(cls_vals, sizeof *dist_lt));
    ASSERT(dist_ge = (float *)calloc(cls_vals, sizeof *dist_ge));
    ASSERT(attr_dist = (float *)calloc(2, sizeof *attr_dist));

    /* sort */
    compar_attr = attr;
    qsort(examples, size, sizeof(struct Example), compar_examples);

    /* compute gain ratio for every split */
    size_known = size;
    size_weight = 0.0;
    for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
        if (ex->example->values[attr].isSpecial()) {
            size_known = ex - examples;
            break;
        }
        if (!ex->example->getClass().isSpecial())
            dist_ge[ex->example->getClass().intV] += ex->weight;
        size_weight += ex->weight;
    }

    attr_dist[1] = size_weight;
    best_score = -INFINITY;

    for (ex = examples, ex_end = ex + size_known - minInstances, ex_next = ex + 1, i = 0; ex < ex_end; ex++, ex_next++, i++) {
        if (!ex->example->getClass().isSpecial()) {
            cls = ex->example->getClass().intV;
            dist_lt[cls] += ex->weight;
            dist_ge[cls] -= ex->weight;
        }
        attr_dist[0] += ex->weight;
        attr_dist[1] -= ex->weight;

        if (ex->example->values[attr] == ex_next->example->values[attr] || i + 1 < minInstances)
            continue;

        /* gain ratio */
        score = (attr_dist[0] * entropy(dist_lt, cls_vals) + attr_dist[1] * entropy(dist_ge, cls_vals)) / size_weight;
        score = (cls_entropy - score) / entropy(attr_dist, 2);


        if (score > best_score) {
            best_score = score;
            *best_split = (ex->example->values[attr].floatV + ex_next->example->values[attr].floatV) / 2.0;
        }
    }

    /* printf("C %s %f\n", args->domain->attributes->at(attr)->get_name().c_str(), best_score); */

    /* cleanup */
    free(dist_lt);
    free(dist_ge);
    free(attr_dist);

    return best_score;
}


float
gain_ratio_d(struct Example *examples, int size, int attr, float cls_entropy, struct Args *args)
{
    struct Example *ex, *ex_end;
    int i, cls_vals, attr_vals, attr_val, cls_val;
    float score, size_weight, size_attr_known, size_attr_cls_known, attr_entropy, *cont, *attr_dist, *attr_dist_cls_known;

    cls_vals = args->domain->classVar->noOfValues();
    attr_vals = args->domain->attributes->at(attr)->noOfValues();

    /* allocate space */
    ASSERT(cont = (float *)calloc(cls_vals * attr_vals, sizeof(float *)));
    ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof(float *)));
    ASSERT(attr_dist_cls_known = (float *)calloc(attr_vals, sizeof(float *)));

    /* contingency matrix */
    size_weight = 0.0;
    for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
        if (!ex->example->values[attr].isSpecial()) {
            attr_val = ex->example->values[attr].intV;
            attr_dist[attr_val] += ex->weight;
            if (!ex->example->getClass().isSpecial()) {
                cls_val = ex->example->getClass().intV;
                attr_dist_cls_known[attr_val] += ex->weight;
                cont[attr_val * cls_vals + cls_val] += ex->weight;
            }
        }
        size_weight += ex->weight;
    }

    /* min examples in leaves */
    if (!test_min_examples(attr_dist, attr_vals, args)) {
        score = -INFINITY;
        goto finish;
    }

    size_attr_known = size_attr_cls_known = 0.0;
    for (i = 0; i < attr_vals; i++) {
        size_attr_known += attr_dist[i];
        size_attr_cls_known += attr_dist_cls_known[i];
    }

    /* gain ratio */
    score = 0.0;
    for (i = 0; i < attr_vals; i++)
        score += attr_dist_cls_known[i] * entropy(cont + i * cls_vals, cls_vals);
    attr_entropy = entropy(attr_dist, attr_vals);

    if (size_attr_cls_known == 0.0 || attr_entropy == 0.0 || size_weight == 0.0) {
        score = -INFINITY;
        goto finish;
    }

    score = (cls_entropy - score / size_attr_cls_known) / attr_entropy * ((float)size_attr_known / size_weight);

    /* printf("D %s %f\n", args->domain->attributes->at(attr)->get_name().c_str(), score); */

finish:
    free(cont);
    free(attr_dist);
    free(attr_dist_cls_known);
    return score;
}


float
mse_c(struct Example *examples, int size, int attr, float cls_mse, struct Args *args, float *best_split)
{
    struct Example *ex, *ex_end, *ex_next;
    int i, cls_vals, minInstances, size_known;
    float size_attr_known, size_weight, cls_val, cls_score, best_score, size_attr_cls_known, score;

    struct Variance {
        float n, sum, sum2;
    } var_lt = {0.0, 0.0, 0.0}, var_ge = {0.0, 0.0, 0.0};

    cls_vals = args->domain->classVar->noOfValues();

    /* minInstances should be at least 1, otherwise there is no point in splitting */
    minInstances = args->minInstances < 1 ? 1 : args->minInstances;

    /* sort */
    compar_attr = attr;
    qsort(examples, size, sizeof(struct Example), compar_examples);

    /* compute mse for every split */
    size_known = size;
    size_attr_known = 0.0;
    for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
        if (ex->example->values[attr].isSpecial()) {
            size_known = ex - examples;
            break;
        }
        if (!ex->example->getClass().isSpecial()) {
            cls_val = ex->example->getClass().floatV;
            var_ge.n += ex->weight;
            var_ge.sum += ex->weight * cls_val;
            var_ge.sum2 += ex->weight * cls_val * cls_val;
        }
        size_attr_known += ex->weight;
    }

    /* count the remaining examples with unknown values */
    size_weight = size_attr_known;
    for (ex_end = examples + size; ex < ex_end; ex++)
        size_weight += ex->weight;

    size_attr_cls_known = var_ge.n;
    best_score = -INFINITY;

    for (ex = examples, ex_end = ex + size_known - minInstances, ex_next = ex + 1, i = 0; ex < ex_end; ex++, ex_next++, i++) {
        if (!ex->example->getClass().isSpecial()) {
            cls_val = ex->example->getClass();
            var_lt.n += ex->weight;
            var_lt.sum += ex->weight * cls_val;
            var_lt.sum2 += ex->weight * cls_val * cls_val;

            var_ge.n -= ex->weight;
            var_ge.sum -= ex->weight * cls_val;
            var_ge.sum2 -= ex->weight * cls_val * cls_val;
        }

        if (ex->example->values[attr] == ex_next->example->values[attr] || i + 1 < minInstances)
            continue;

        /* compute mse */
        score = var_lt.sum2 - var_lt.sum * var_lt.sum / var_lt.n;
        score += var_ge.sum2 - var_ge.sum * var_ge.sum / var_ge.n;
        score = (cls_mse - score / size_attr_cls_known) / cls_mse * (size_attr_known / size_weight);

        if (score > best_score) {
            best_score = score;
            *best_split = (ex->example->values[attr].floatV + ex_next->example->values[attr].floatV) / 2.0;
        }
    }

    /* printf("C %s %f\n", args->domain->attributes->at(attr)->get_name().c_str(), best_score); */
    return best_score;
}


float
mse_d(struct Example *examples, int size, int attr, float cls_mse, struct Args *args)
{
    int i, attr_vals, attr_val;
    float *attr_dist, d, score, cls_val, size_attr_cls_known, size_attr_known, size_weight;
    struct Example *ex, *ex_end;

    struct Variance {
        float n, sum, sum2;
    } *variances, *v, *v_end;

    attr_vals = args->domain->attributes->at(attr)->noOfValues();

    ASSERT(variances = (struct Variance *)calloc(attr_vals, sizeof *variances));
    ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

    size_weight = size_attr_cls_known = size_attr_known = 0.0;
    for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
        if (!ex->example->values[attr].isSpecial()) {
            attr_dist[ex->example->values[attr].intV] += ex->weight;
            size_attr_known += ex->weight;

            if (!ex->example->getClass().isSpecial()) {
                    cls_val = ex->example->getClass().floatV;
                    v = variances + ex->example->values[attr].intV;
                    v->n += ex->weight;
                    v->sum += ex->weight * cls_val;
                    v->sum2 += ex->weight * cls_val * cls_val;
                    size_attr_cls_known += ex->weight;
            }
        }
        size_weight += ex->weight;
    }

    /* minimum examples in leaves */
    if (!test_min_examples(attr_dist, attr_vals, args)) {
        score = -INFINITY;
        goto finish;
    }

    score = 0.0;
    for (v = variances, v_end = variances + attr_vals; v < v_end; v++)
        if (v->n > 0.0)
            score += v->sum2 - v->sum * v->sum / v->n;
    score = (cls_mse - score / size_attr_cls_known) / cls_mse * (size_attr_known / size_weight);

    if (size_attr_cls_known <= 0.0 || cls_mse <= 0.0 || size_weight <= 0.0)
        score = 0.0;

finish:
    free(attr_dist);
    free(variances);

    return score;
}


struct SimpleTreeNode *
make_predictor(struct SimpleTreeNode *node, struct Example *examples, int size, struct Args *args)
{
    struct Example *ex, *ex_end;

    node->type = PredictorNode;
    if (args->type == Regression) {
        node->n = node->sum = 0.0;
        for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
            if (!ex->example->getClass().isSpecial()) {
                node->sum += ex->weight * ex->example->getClass().floatV;
                node->n += ex->weight;
            }

    }

    return node;
}


struct SimpleTreeNode *
build_tree(struct Example *examples, int size, int depth, struct SimpleTreeNode *parent, struct Args *args)
{
    int i, cls_vals, best_attr;
    float cls_entropy, cls_mse, best_score, score, size_weight, best_split, split;
    struct SimpleTreeNode *node;
    struct Example *ex, *ex_end;
    TVarList::const_iterator it;

    cls_vals = args->domain->classVar->noOfValues();

    ASSERT(node = (SimpleTreeNode *)malloc(sizeof *node));

    if (args->type == Classification) {
        ASSERT(node->dist = (float *)calloc(cls_vals, sizeof(float *)));

        if (size == 0) {
            assert(parent);
            node->type = PredictorNode;
            memcpy(node->dist, parent->dist, cls_vals * sizeof *node->dist);
            return node;
        }

        /* class distribution */
        size_weight = 0.0;
        for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
            if (!ex->example->getClass().isSpecial()) {
                node->dist[ex->example->getClass().intV] += ex->weight;
                size_weight += ex->weight;
            }

        /* stopping criterion: majority class */
        for (i = 0; i < cls_vals; i++)
            if (node->dist[i] / size_weight >= args->maxMajority)
                return make_predictor(node, examples, size, args);

        cls_entropy = entropy(node->dist, cls_vals);
    } else {
        float n, sum, sum2, cls_val;

        assert(args->type == Regression);
        if (size == 0) {
            assert(parent);
            node->type = PredictorNode;
            node->n = parent->n;
            node->sum = parent->sum;
            return node;
        }

        n = sum = sum2 = 0.0;
        for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
            if (!ex->example->getClass().isSpecial()) {
                cls_val = ex->example->getClass().floatV;
                n += ex->weight;
                sum += ex->weight * cls_val;
                sum2 += ex->weight * cls_val * cls_val;
            }

        cls_mse = (sum2 - sum * sum / n) / n;
    }

    /* stopping criterion: depth exceeds limit */
    if (depth == args->maxDepth)
        return make_predictor(node, examples, size, args);

    /* score attributes */
    best_score = -INFINITY;

    for (i = 0, it = args->domain->attributes->begin(); it != args->domain->attributes->end(); it++, i++) {
        if (!args->attr_split_so_far[i]) {
            /* select random subset of attributes */
            if ((double)rand() / RAND_MAX < args->skipProb)
                continue;

            if ((*it)->varType == TValue::INTVAR) {
                score = args->type == Classification ?
                  gain_ratio_d(examples, size, i, cls_entropy, args) :
                  mse_d(examples, size, i, cls_mse, args);
                if (score > best_score) {
                    best_score = score;
                    best_attr = i;
                }
            } else if ((*it)->varType == TValue::FLOATVAR) {
                score = args->type == Classification ?
                  gain_ratio_c(examples, size, i, cls_entropy, args, &split) :
                  mse_c(examples, size, i, cls_mse, args, &split);
                if (score > best_score) {
                    best_score = score;
                    best_split = split;
                    best_attr = i;
                }
            }
        }
    }

    if (best_score == -INFINITY)
        return make_predictor(node, examples, size, args);

    if (args->domain->attributes->at(best_attr)->varType == TValue::INTVAR) {
        struct Example *child_examples, *child_ex;
        int attr_vals;
        float size_known, *attr_dist;

        /* printf("* %2d %3s %3d %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_score); */

        attr_vals = args->domain->attributes->at(best_attr)->noOfValues(); 

        node->type = DiscreteNode;
        node->split_attr = best_attr;
        node->children_size = attr_vals;

        ASSERT(child_examples = (struct Example *)calloc(size, sizeof *child_examples));
        ASSERT(node->children = (SimpleTreeNode **)calloc(attr_vals, sizeof *node->children));
        ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

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
            for (ex = examples, ex_end = examples + size, child_ex = child_examples; ex < ex_end; ex++) {
                if (ex->example->values[best_attr].isSpecial()) {
                    *child_ex = *ex;
                    child_ex->weight *= attr_dist[i] / size_known;
                    child_ex++;
                } else if (ex->example->values[best_attr].intV == i) {
                    *child_ex++ = *ex;
                }
            }

            node->children[i] = build_tree(child_examples, child_ex - child_examples, depth + 1, node, args);
        }
                    
        args->attr_split_so_far[best_attr] = 0;

        free(attr_dist);
        free(child_examples);
    } else {
        struct Example *examples_lt, *examples_ge, *ex_lt, *ex_ge;
        float size_lt, size_ge;

        /* printf("* %2d %3s %3d %f %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_split, best_score); */

        assert(args->domain->attributes->at(best_attr)->varType == TValue::FLOATVAR);

        ASSERT(examples_lt = (struct Example *)calloc(size, sizeof *examples));
        ASSERT(examples_ge = (struct Example *)calloc(size, sizeof *examples));

        size_lt = size_ge = 0.0;
        for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
            if (!ex->example->values[best_attr].isSpecial())
                if (ex->example->values[best_attr].floatV < best_split)
                    size_lt += ex->weight;
                else
                    size_ge += ex->weight;

        for (ex = examples, ex_end = examples + size, ex_lt = examples_lt, ex_ge = examples_ge; ex < ex_end; ex++)
            if (ex->example->values[best_attr].isSpecial()) {
                *ex_lt = *ex;
                *ex_ge = *ex;
                ex_lt->weight *= size_lt / (size_lt + size_ge);
                ex_ge->weight *= size_ge / (size_lt + size_ge);
                ex_lt++;
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
        ASSERT(node->children = (SimpleTreeNode **)calloc(2, sizeof *node->children));

        node->children[0] = build_tree(examples_lt, ex_lt - examples_lt, depth + 1, node, args);
        node->children[1] = build_tree(examples_ge, ex_ge - examples_ge, depth + 1, node, args);

        free(examples_lt);
        free(examples_ge);
    }

    return node;
}

TSimpleTreeLearner::TSimpleTreeLearner(const int &weight, float maxMajority, int minInstances, int maxDepth, float skipProb, unsigned int seed) :
    maxMajority(maxMajority),
    minInstances(minInstances),
    maxDepth(maxDepth),
    skipProb(skipProb),
    seed(seed)
{
}

PClassifier
TSimpleTreeLearner::operator()(PExampleGenerator ogen, const int &weight)
{
    struct Example *examples, *ex;
    struct SimpleTreeNode *tree;
    struct Args args;

    if (!ogen->domain->classVar)
        raiseError("class-less domain");

    /* create a tabel with pointers to examples */
    ASSERT(examples = (struct Example *)calloc(ogen->numberOfExamples(), sizeof *examples));
    ex = examples;
    PEITERATE(ei, ogen) {
        ex->example = &(*ei);
        ex->weight = 1.0;
        ex++;
    }

    ASSERT(args.attr_split_so_far = (int *)calloc(ogen->domain->attributes->size(), sizeof(int)));
    args.minInstances = minInstances;
    args.maxMajority = maxMajority;
    args.maxDepth = maxDepth;
    args.skipProb = skipProb;
    args.domain = ogen->domain;
    args.type = ogen->domain->classVar->varType == TValue::INTVAR ? Classification : Regression;

    srand(seed);
    tree = build_tree(examples, ogen->numberOfExamples(), 0, NULL, &args);

    free(examples);
    free(args.attr_split_so_far);

    return new TSimpleTreeClassifier(ogen->domain->classVar, tree, args.type);
}


/* classifier */
TSimpleTreeClassifier::TSimpleTreeClassifier()
{
}

TSimpleTreeClassifier::TSimpleTreeClassifier(const PVariable &classVar, struct SimpleTreeNode *tree, int type) : 
    TClassifier(classVar, true),
    tree(tree),
    type(type)
{
}


void
destroy_tree(struct SimpleTreeNode *node, int type)
{
    int i;

    if (node->type != PredictorNode) {
        for (i = 0; i < node->children_size; i++)
            destroy_tree(node->children[i], type);
        free(node->children);
    }
    if (type == Classification)
        free(node->dist);
    free(node);
}


TSimpleTreeClassifier::~TSimpleTreeClassifier()
{
    destroy_tree(tree, type);
}


float *
predict_classification(const TExample &ex, struct SimpleTreeNode *node, int *free_dist)
{
    int i, j, cls_vals;
    float *dist, *child_dist;

    while (node->type != PredictorNode)
        if (ex.values[node->split_attr].isSpecial()) {
            cls_vals = ex.domain->classVar->noOfValues();
            ASSERT(dist = (float *)calloc(cls_vals, sizeof *dist));
            for (i = 0; i < node->children_size; i++) {
                child_dist = predict_classification(ex, node->children[i], free_dist);
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

    *free_dist = 0;
    return node->dist;
}


void
predict_regression(const TExample &ex, struct SimpleTreeNode *node, float *sum, float *n)
{
    int i;
    float local_sum, local_n;

    while (node->type != PredictorNode)
        if (ex.values[node->split_attr].isSpecial()) {
            *sum = *n = 0;
            for (i = 0; i < node->children_size; i++) {
                predict_regression(ex, node->children[i], &local_sum, &local_n);
                *sum += local_sum;
                *n += local_n;
            }
            return;
        } else if (node->type == DiscreteNode) {
            node = node->children[ex.values[node->split_attr].intV];
        } else {
            assert(node->type == ContinuousNode);
            node = node->children[ex.values[node->split_attr].floatV > node->split];
        }

    *sum = node->sum;
    *n = node->n;
}


TValue
TSimpleTreeClassifier::operator()(const TExample &ex)
{
    if (type == Classification) {
        int i, free_dist, best_val;
        float *dist;

        dist = predict_classification(ex, tree, &free_dist);
        best_val = 0;
        for (i = 1; i < ex.domain->classVar->noOfValues(); i++)
            if (dist[i] > dist[best_val])
                best_val = i;

        if (free_dist)
            free(dist);
        return TValue(best_val);
    } else {
        float sum, n;

        assert(type == Regression);

        predict_regression(ex, tree, &sum, &n);
        return TValue(sum / n);
    }
}

PDistribution
TSimpleTreeClassifier::classDistribution(const TExample &ex)
{
    if (type == Classification) {
        int i, free_dist;
        float *dist;

        dist = predict_classification(ex, tree, &free_dist);

        PDistribution pdist = TDistribution::create(ex.domain->classVar);
        for (i = 0; i < ex.domain->classVar->noOfValues(); i++)
            pdist->setint(i, dist[i]);
        pdist->normalize();

        if (free_dist)
            free(dist);
        return pdist;
    } else {
        return NULL;
    }
}

void
TSimpleTreeClassifier::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{
    value = operator()(ex);
    dist = classDistribution(ex);
}
