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


#ifndef _ASSOC_HPP
#define _ASSOC_HPP

#include <iostream>

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "classify.hpp"
#include "learn.hpp"

WRAPPER(Example)
VWRAPPER(IntList)
WRAPPER(ExampleTable)

class ORANGE_API TAssociationRule : public TOrange {
public:
  __REGISTER_CLASS

  PExample left; //PR left side of the rule
  PExample right; //PR right side of the rule
  float support; //P support for the rule
  float confidence; //P confidence of the rule
  float coverage; //P rule's coverage
  float strength; //P rule's strength
  float lift; //P rule's lift
  float leverage; //P rule's leverage
  float nAppliesLeft; //P number of examples covered by the rule's left side
  float nAppliesRight; //P number of examples covered by the rule's right side
  float nAppliesBoth; //P number of examples covered by the rule
  float nExamples; //P number of learning examples
  int nLeft; //PR number of items on the rule's left side
  int nRight; //PR number of items on the rule's right side

  PExampleTable examples; //PR examples which the rule was built from
  PIntList matchLeft; //PR indices of examples that match the left side of the rule
  PIntList matchBoth; //PR indices to examples that match both sides of the rule

  TAssociationRule(PExample, PExample);

  TAssociationRule(PExample al, PExample ar,
                   const float &napLeft, const float &napRight, const float &napBoth, const float &nExamples,
                   int anleft=-1, int anright=-1);

  virtual bool operator ==(const TAssociationRule &other) const
  { return (left->operator==(other.left.getReference())) && (right->operator==(other.right.getReference())); }

  static bool applies(const TExample &ex, const PExample &side);

  bool appliesLeft(const TExample &ex) const
  { return applies(ex, left); }

  bool appliesRight(const TExample &ex) const
  { return applies(ex, right); }

  bool appliesBoth(const TExample &ex) const
  { return applies(ex, left) && applies(ex, right); }

  static int countItems(PExample ex);
};

WRAPPER(AssociationRule)

#ifdef _MSC_VER
  ORANGE_EXTERN template class ORANGE_API TOrangeVector<PAssociationRule>;
#endif

#define TAssociationRules TOrangeVector<PAssociationRule>
VWRAPPER(AssociationRules)


class TItemSetNode;

/* These objects are collected in TExampleSets, lists of examples that correspond to a particular tree node.
   'example' is a unique example id (basically its index in the original dataset)
   'weight' is the example's weight. */
class TExWei {
public:
  int example;
  float weight;

  TExWei(const int &ex, const float &wei)
  : example(ex),
    weight(wei)
  {}
};

/* This is a set of examples, used to list the examples that support a particular tree node */
typedef vector<TExWei> TExampleSet;


/* A tree element that corresponds to an attribute value (ie, TItemSetNode has as many
   TlItemSetValues as there are values that appear in itemsets.
   For each value, we have the 'examples' that support it, the sum of their weights
   ('support') and the branch that contains more specialized itemsets. */
class TItemSetValue {
public:
  int value;
  TItemSetNode *branch;

  float support;
  TExampleSet examples;

  // This constructor is called when building the 1-tree
  TItemSetValue(int al);

  // This constructor is called when itemsets are intersected (makePairs ets)
  TItemSetValue(int al, const TExampleSet &ex, float asupp);

  ~TItemSetValue();
  void sumSupport();
};


/* TItemSetNode splits itemsets according to the value of attribute 'attrIndex';
   each element of 'values' corresponds to an attribute value (not necessarily to all,
   but only to those values that appear in itemsets).
   Itemsets for which the value is not defined are stored in a subtree in 'nextAttribute'.
   This can be seen in TItemSetTree::findSupport that finds a node that corresponds to the
   given itemset */
class TItemSetNode {
public:
  int attrIndex;
  TItemSetNode *nextAttribute;
  vector<TItemSetValue> values;

  // This constructor is called by 1-tree builder which initializes all values (and later reduces them)
  TItemSetNode(PVariable var, int anattri);

  // This constructor is called when extending the tree
  TItemSetNode(int anattri);

  ~TItemSetNode();
};

class TRuleTreeNode;


class ORANGE_API TAssociationRulesInducer : public TOrange {
public:
  __REGISTER_CLASS

  int maxItemSets; //P maximal number of itemsets (increase if you want)

  float confidence; //P required confidence
  float support; //P required support
  bool classificationRules; //P if true, rules will have the class and only the class attribute on the right-hand side
  bool storeExamples; //P if true, each rule is going to have tables with references to examples which match its left side or both sides

public:

  TAssociationRulesInducer(float asupp=0.1, float aconf=0.5);
  PAssociationRules operator()(PExampleGenerator, const int &weightID = 0);

  void buildTrees(PExampleGenerator, const int &weightID, TItemSetNode *&, int &depth, int &nOfExamples, TDiscDistribution &);
  int  buildTree1(PExampleGenerator, const int &weightID, TItemSetNode *&, float &suppN, int &nOfExamples, TDiscDistribution &);
  int  buildNext1(TItemSetNode *, int k, const float suppN);
  int  makePairs (TItemSetNode *, const float suppN);

  PAssociationRules generateClassificationRules(PDomain, TItemSetNode *tree, const int nOfExamples, const TDiscDistribution &);
  void generateClassificationRules1(PDomain, TItemSetNode *root, TItemSetNode *node, TExample &left, const int nLeft, const float nAppliesLeft, PAssociationRules, const int nOfExamples, const TDiscDistribution &, TExampleSet *leftSet);

  PAssociationRules generateRules(PDomain, TItemSetNode *, const int depth, const int nOfExamples);
  void generateRules1(TExample &, TItemSetNode *root, TItemSetNode *node, int k, int oldk, PAssociationRules, const int nOfExamples);
  void find1Rules(TExample &, TItemSetNode *, const float &support, int oldk, PAssociationRules, const int nOfExamples, const TExampleSet &bothSets);
  TRuleTreeNode *buildTree1FromExample(TExample &, TItemSetNode *);
  int generateNext1(TRuleTreeNode *ruleTree, TRuleTreeNode *node, TItemSetNode *itemsetsTree, TExample &right, int k, TExample &whole, const float &support, PAssociationRules, const int nOfExamples, const TExampleSet &bothSets);
  int generatePairs(TRuleTreeNode *ruleTree, TRuleTreeNode *node, TItemSetNode *itemsetsTree, TExample &right, TExample &whole, const float &support, PAssociationRules, const int nOfExamples, const TExampleSet &bothSets);
};

WRAPPER(AssociationRulesInducer)




class TSparseExample{
public:
	float weight;			// weight of thi example
	long *itemset;		// vector storing just items that have some value in original example
	int	length;

	TSparseExample(TExample *example, int weightID);
};


class TSparseExamples{
public:
	float fullWeight;					// weight of all examples
	vector<TSparseExample*> transaction;	// vector storing all sparse examples
	PDomain domain;						// domain of original example or exampleGenerator
	vector<long> intDomain;				// domain mapped longint values

	TSparseExamples(PExampleGenerator examples, int weightID);
};


class TSparseItemsetNode;
typedef map<long, TSparseItemsetNode *> TSparseISubNodes;

class TSparseItemsetNode{							//item node used in TSparseItemsetTree
public:
	float weiSupp;							//support of itemset consisting node and all of its parents
	long value;								//value of this node
	TSparseItemsetNode *parent;					//pointer to parent node
	TSparseISubNodes subNode;				//children items
	vector<int> exampleIds;

	TSparseItemsetNode(long avalue = -1);			//constructor

    TSparseItemsetNode *operator[] (long avalue);	//directly gets subnode

	TSparseItemsetNode* addNode(long avalue);		//adds new subnode
	bool hasNode(long avalue);				//returns true if has subnode with given value
};


class TSparseItemsetTree : TOrange {							//item node used in TSparseItemsetTree
public:
	TSparseItemsetTree(TSparseExamples examples);			//constructor

	int buildLevelOne(vector<long> intDomain);
	long extendNextLevel(int maxDepth, long maxCount);
	bool allowExtend(long itemset[], int iLength);
	long countLeafNodes();
	void considerItemset(long itemset[], int iLength, float weight, int aimLength);
	void considerExamples(TSparseExamples *examples, int aimLength);
  void assignExamples(TSparseItemsetNode *node, long *itemset, long *itemsetend, const int exampleId);
  void assignExamples(TSparseExamples &examples);
  void delLeafSmall(float minSupport);
	PAssociationRules genRules(int maxDepth, float minConf, float nOfExamples, bool storeExamples);
	long getItemsetRules(long itemset[], int iLength, float minConf,
						 float nAppliesBoth, float nOfExamples, PAssociationRules rules, bool storeExamples, TSparseItemsetNode *bothNode);
	PDomain domain;

//private:
	TSparseItemsetNode *root;
};


class ORANGE_API TAssociationRulesSparseInducer : public TOrange {
public:
  __REGISTER_CLASS

  int maxItemSets; //P maximal number of itemsets (increase if you want)

  float confidence; //P required confidence
  float support; //P required support

  bool storeExamples; //P stores examples corresponding to rules

  TAssociationRulesSparseInducer(float asupp=0.1, float aconf=0, int awei=0);
  TSparseItemsetTree *TAssociationRulesSparseInducer::buildTree(PExampleGenerator examples, const int &weightID, long &i, float &fullWeight);
  PAssociationRules operator()(PExampleGenerator, const int &weightID);

private:
  float nOfExamples;
};

WRAPPER(AssociationRulesSparseInducer)


WRAPPER(SparseItemsetTree)

class ORANGE_API TItemsetsSparseInducer : public TOrange {
public:
  __REGISTER_CLASS

  int maxItemSets; //P maximal number of itemsets (increase if you want)
  float support; //P required support

  bool storeExamples; //P stores examples corresponding to itemsets

  TItemsetsSparseInducer(float asupp=0.1, int awei=0);
  PSparseItemsetTree operator()(PExampleGenerator, const int &weightID);

private:
  float nOfExamples;
};


class ORANGE_API TAssociationLearner : public TLearner {
public:
  __REGISTER_CLASS

  float confidence; //P required confidence
  float support; //P required support
  int voteWeight; //P vote weight (s=support, c=confidence, p=product)
  int maxItemSets; //P maximal number of itemsets (increase if you want)

  TAssociationLearner();
  virtual PClassifier operator()(PExampleGenerator gen, const int & = 0);
};


class ORANGE_API TAssociationClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PAssociationRules rules; //P association rules
  int voteWeight; //P vote weight (s=support, c=confidence, p=product)

  TAssociationClassifier(PDomain dom=PDomain(), PAssociationRules arules=PAssociationRules(), char avote='s');
  virtual PDistribution classDistribution(const TExample &);
};

#endif
