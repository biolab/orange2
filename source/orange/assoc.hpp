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


#ifndef _ASSOC_HPP
#define _ASSOC_HPP

#include <iostream>

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "classify.hpp"
#include "learn.hpp"

class TItemSetNode;
class TExample;

class TExWei {
public:
  int example;
  float weight;

  TExWei(const int &ex, const float wei=1)
  : example(ex),
    weight(wei)
  {}

  bool operator == (const TExWei &ew2) const
  { return example==ew2.example; }

  bool operator <  (const TExWei &ew2) const
  { return example <ew2.example; }
};


class Tleb {
public:
  int label;
  TItemSetNode *branch;
  float support;
  vector<TExWei> examples;

  Tleb(int al, TItemSetNode  *ab=NULL)
  : label(al),
    branch(ab),
    support(0.0)
  {}

  Tleb(int al, const vector<TExWei> &ane, TItemSetNode  *ab=NULL)
  : label(al),
    branch(ab),
    examples(ane)
  { sumSupport(); }

  Tleb(int al, const vector<TExWei> &ane, float asupp, TItemSetNode  *ab=NULL)
  : label(al),
    branch(ab),
    support(asupp),
    examples(ane)
  {}

  ~Tleb();

  void sumSupport();
};


class TItemSetNode {
public:
  int attrIndex;
  TItemSetNode *unknown;
  vector<Tleb> values;

  TItemSetNode(PVariable var, int anattri);

  TItemSetNode(int anattri)
  : attrIndex(anattri), 
    unknown(NULL) 
  {}

  virtual ~TItemSetNode()
  { mldelete unknown; }
};



class TItemSetTree {
public:
  TItemSetNode *root;
  int largestK;

  TItemSetTree(TItemSetNode *aroot=NULL, int alK=0)
  : root(aroot), 
    largestK(alK)
  {}

  virtual ~TItemSetTree()
  { mldelete root; 
    root=NULL; 
    largestK=0; 
  }


  void addExample(const TExample &example, const vector<TExWei> &intersection, const float &support);
  void addExample(const TExample &example, const vector<TExWei> &intersection);
  void addExample(const TExample &example)
  { addExample(example, vector<TExWei>()); }

  float findSupport(const TExample &example);
};


WRAPPER(Example)

class TAssociationRule : public TOrange {
public:
  __REGISTER_CLASS

  PExample left; //P left side of the rule
  PExample right; //P right side of the rule
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
  int nleft; //P number of items on the rule's left side
  int nright; //P number of items on the rule's right side

  TAssociationRule();

  TAssociationRule(PExample al, PExample ar,
                   const float &asup, const float &acon, const float &acov,
                   const float &astr, const float &alif, const float &alev,
                   const float &napl=-1, const float &napr=-1, const float &napb=-1, const float &nexa=-1,
                   const int &anleft=-1, const int &anright=-1);

  TAssociationRule(PExample al, PExample ar,
                   const float &napLeft, const float &napRight, const float &napBoth, const float &nExamples,
                   int anleft=-1, int anright=-1);

  void correctSpecials();

  virtual bool operator ==(const TAssociationRule &other) const
  { return (left->operator==(other.left.getReference())) && (right->operator==(other.right.getReference())); }

  virtual bool appliesLeft(const TExample &ex) const
  { return left->compatible(ex); }

  virtual bool appliesRight(const TExample &ex) const
  { return right->compatible(ex); }

  virtual bool appliesBoth(const TExample &ex) const
  { return appliesLeft(ex) && appliesRight(ex); }

  static int countItems(PExample ex);
};

WRAPPER(AssociationRule)

#include "rule_conditions.hpp"


#define TAssociationRules TOrangeVector<PAssociationRule>
VWRAPPER(AssociationRules)

class TAssociationRulesInducer : public TOrange {
public:
  __REGISTER_CLASS

  int weightID; //P id of meta attribute with weight
  int maxItemSets; //P maximal number of itemsets (increase, if needed)

  float conf; //P required confidence
  float supp; //P required support
  TRuleCondDisjunctions conditions;

private:
  TItemSetTree tree;
  float nOfExamples;
public:

  TAssociationRulesInducer(float asupp=0.1, float aconf=0.5, int awei=0);
  PAssociationRules operator()(PExampleGenerator);

  int  buildTree1(PExampleGenerator gen);

  int  buildNext1(TItemSetNode *tempNode, TExample &example, int k);
  int  makePairs (TItemSetNode *tempNode, TExample &example);

  void buildTrees(PExampleGenerator gen);

  PAssociationRules generateRules(PDomain dom);
  void generateRules1(TExample &ex, TItemSetNode *root, int k, int oldk, PAssociationRules rules);
  TItemSetNode *buildTree1FromExample(TExample &ex, TItemSetNode *root1, int &itemSets);
  int generateNext1(TItemSetTree &ruleTree, TItemSetNode *tempNode,
                TExample &example, int k, TExample &wholeEx, const float &support,
                PAssociationRules rules);
  int generatePairs(TItemSetTree &ruleTree, TItemSetNode *tempNode,
                TExample &example, TExample &wholeEx, const float &support,
                PAssociationRules rules);
  void find1Rules(TExample &example, const float &support, int oldk, PAssociationRules rules);
};

WRAPPER(AssociationRulesInducer)



class TAssociationLearner : public TLearner {
public:
  __REGISTER_CLASS

  string condfile;
  float conf; //P required confidence
  float supp; //P required support
  int voteWeight; //P vote weight (s=support, c=confidence, p=product)
  int maxItemSets; //P maximal number of itemsets (increase, if needed)
 
  TAssociationLearner();
  virtual PClassifier operator()(PExampleGenerator gen, const int & = 0);
};


class TAssociationClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PAssociationRules rules; //P association rules
  int voteWeight; //P vote weight (s=support, c=confidence, p=product)
 
  TAssociationClassifier(PDomain dom=PDomain(), PAssociationRules arules=PAssociationRules(), char avote='s');
  virtual PDistribution classDistribution(const TExample &);
};

#endif
