#ifndef __TDIDT_CLUSTERING_HPP
#define __TDIDT_CLUSTERING_HPP

#include <limits.h>
#include "multi_learner.hpp"

struct ClusteringTreeNode {
	int type, children_size, split_attr, n_classes;
	float split;
	ClusteringTreeNode **children;

	float **dist; /* classification */
	float *n, *sum; /* regression */
};

class ORANGE_API TClusteringTreeLearner: public TMultiLearner {
public:
	__REGISTER_CLASS
	float minMajority; //P the minimal majority each class variable must exceed to stop building
	float minMSE; //P the minimal MSE each class variable must be lower than to stop building
	int minInstances; //P the minimal number of examples for division to continue
	int maxDepth; //P the maximal depth a tree can reach
	int method; //P
	float skipProb; //P

	PRandomGenerator randomGenerator; //P

	TClusteringTreeLearner(const int & = 0, float = 0.9, float = 0.0001, int = 5, int = INT_MAX,int = 0, float =
			0.0,  PRandomGenerator = NULL);
	PMultiClassifier operator()(PExampleGenerator, const int & = 0);
};

class ORANGE_API TClusteringTreeClassifier: public TMultiClassifier {
private:
	int type, *cls_vals;
	struct ClusteringTreeNode *tree;

	void save_tree(ostringstream &, struct ClusteringTreeNode *);
	struct ClusteringTreeNode *load_tree(istringstream &, int);

public:
	__REGISTER_CLASS

	TClusteringTreeClassifier();
	TClusteringTreeClassifier(const PVarList &, struct ClusteringTreeNode *, int, int *);
	~TClusteringTreeClassifier();

	void save_model(ostringstream &);
	void load_model(istringstream &);
	PValueList operator ()(const TExample &);
	PDistributionList classDistribution(const TExample &);
	void predictionAndDistribution(const TExample &, PValueList &, PDistributionList &);
};
WRAPPER(ClusteringTreeClassifier)
#endif
