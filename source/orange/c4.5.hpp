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


#ifndef __C45_HPP
#define __C45_HPP

#include "classify.hpp"
#include "learn.hpp"


class ORANGE_API TC45Learner : public TLearner {
public:
    __REGISTER_CLASS

    bool gainRatio; //P (+g) use gain ratio (instead of information gain)
    bool subset; //P (+s) use subsetting
    bool batch; //P (+b) batch
    bool probThresh; //P (+p) probability threshold
    int minObjs; //P (+m) minimal number of objects (examples) in leaves
    int window; //P (+w) window
    int increment; //P (+i) increment
    float cf; //P (+c) cf
    int trials; //P (+t) trials

    bool prune; //P return pruned tree
    bool convertToOrange; //P return TreeClassifier instead of C45TreeClassifier
    bool storeExamples; //P stores examples when (if) converting to TreeClassifier
    bool storeContingencies; //P stores contingencies when (if) converting to TreeClassifier

    bool clearDomain();
    bool clearExamples();
    bool clearGenerator();

    bool convertDomain(PDomain);
    bool convertExamples(PExampleGenerator);
    bool convertGenerator(PExampleGenerator);

    bool convertParameters();
    bool parseCommandLine(const string &line);

    TC45Learner();

    virtual PClassifier operator()(PExampleGenerator gen, const int &weight = 0);
};


WRAPPER(C45TreeNode)

#define TC45TreeNodeList TOrangeVector<PC45TreeNode> 
VWRAPPER(C45TreeNodeList)

typedef char Boolean, *String, *Set;

typedef int ItemNo;
typedef float ItemCount;
typedef short ClassNo, DiscrValue;
typedef short Attribute;

typedef  struct _tree_record *Tree;

typedef  struct _tree_record {
  short NodeType;
  ClassNo Leaf; /* most frequent class */
  ItemCount Items, *ClassDist, Errors;
  Attribute Tested;
  short Forks;
  float Cut, Lower, Upper;
  Set *Subset;
  Tree *Branch;
} TreeRec;


WRAPPER(TreeNode);

class ORANGE_API TC45TreeNode : public TOrange {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(NodeType) enum {Leaf = 0, Branch, Cut, Subset};

  int nodeType; //P(&C45TreeNode_NodeType) 0 = leaf,  1 = branch,  2 = cut,  3 = subset
  TValue leaf; //P most frequent class at this node
  float items; //P no of items at this node
  PDiscDistribution classDist; //P class distribution of items
  // skipped: ItemCount	Errors;		/* no of errors at this node */
	PVariable	tested; //P	attribute referenced in test
	// skipped - this is len(branch)    short	Forks;		/* number of branches at this node */
  float cut; //P threshold for continuous attribute
  float lower; //P lower limit of soft threshold
  float upper; //P upper limit of soft threshold
  PIntList mapping; //P mapping for discrete value
  PC45TreeNodeList branch; //P branch[x] = (sub)tree for outcome x */

  TC45TreeNode();
  TC45TreeNode(const Tree &, PDomain);
  PDiscDistribution vote(const TExample &example, PVariable classVar);
  PDiscDistribution classDistribution(const TExample &example, PVariable classVar);

  PTreeNode asTreeNode(PExampleGenerator examples, const int &, bool storeContingencies, bool storeExamples);
};


WRAPPER(TreeClassifier);

class ORANGE_API TC45Classifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PC45TreeNode tree; //P tree

  TC45Classifier(PDomain domain = PDomain(), PC45TreeNode = PC45TreeNode());
  virtual TValue operator ()(const TExample &);
  PDistribution classDistribution(const TExample &);
  void predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist);

  PTreeClassifier asTreeClassifier(PExampleGenerator examples = PExampleGenerator(), const int &weightID = 0, bool storeContigencies = false, bool storeExamples = false);
};


#endif
