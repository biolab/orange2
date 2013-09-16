#ifndef __TDIDT_SIMPLE_HPP
#define __TDIDT_SIMPLE_HPP

#include <limits.h>
#include "learn.hpp"

struct SimpleTreeNode {
	int type, children_size, split_attr;
	float split;
	SimpleTreeNode **children;

	float *dist;  /* classification */
	float n, sum; /* regression */
};

class ORANGE_API TSimpleTreeLearner : public TLearner {
public:
	__REGISTER_CLASS
	float maxMajority; //P
	int minInstances; //P
	int maxDepth; //P
	float skipProb; //P
	PRandomGenerator randomGenerator; //P

	TSimpleTreeLearner(const int & =0, float=1.0, int=2, int=INT_MAX, float=0.0, PRandomGenerator=NULL);
	PClassifier operator()(PExampleGenerator, const int & =0);
};

class ORANGE_API TSimpleTreeClassifier : public TClassifier {
private:
	int type, cls_vals;
	struct SimpleTreeNode *tree;

	void save_tree(ostringstream &, struct SimpleTreeNode *);
	struct SimpleTreeNode *load_tree(istringstream &);

public:
	__REGISTER_CLASS 

	TSimpleTreeClassifier();
	TSimpleTreeClassifier(const PVariable &, struct SimpleTreeNode *, int, int);
	~TSimpleTreeClassifier();

	void save_model(ostringstream &);
	void load_model(istringstream &);
	TValue operator()(const TExample &);
	PDistribution classDistribution(const TExample &);
	void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
};

WRAPPER(SimpleTreeClassifier)
#endif
