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


#include  <iomanip>
#include <fstream>

#include "random.hpp"

#include "examplegen.hpp"
#include "spec_gen.hpp"
#include "table.hpp"

#include "assoc.ppp"

DEFINE_TOrangeVector_classDescription(PAssociationRule, "TAssociationRules", true, ORANGE_API)


TItemSetValue::TItemSetValue(int al)
: value(al),
  branch(NULL),
  support(0.0)
{}


TItemSetValue::TItemSetValue(int al, const TExampleSet &ex, float asupp)
: value(al),
  branch(NULL),
  support(asupp),
  examples(ex)
{}

TItemSetValue::~TItemSetValue()
{ mldelete branch;
}


void TItemSetValue::sumSupport()
{ 
  support = 0; 
  ITERATE(TExampleSet, wi, examples)
    support += (*wi).weight;
}


TItemSetNode::TItemSetNode(PVariable var, int anattri)
: attrIndex(anattri), 
  nextAttribute(NULL)
{ 
  for(int vi = 0, ve = var->noOfValues(); vi<ve; vi++)
    values.push_back(TItemSetValue(vi));
}


TItemSetNode::TItemSetNode(int anattri)
: attrIndex(anattri), 
  nextAttribute(NULL) 
{}


TItemSetNode::~TItemSetNode()
{ mldelete nextAttribute; }


class TRuleTreeNode {
public:
  int attrIndex;
  int value;
  float support;
  TExampleSet examples;
  TRuleTreeNode *nextAttribute;
  TRuleTreeNode *hasValue;

  TRuleTreeNode(const int ai, const int val, const float supp, const TExampleSet &ex)
  : attrIndex(ai),
    value(val),
    support(supp),
    examples(ex),
    nextAttribute(NULL),
    hasValue(NULL)
  {}

  ~TRuleTreeNode()
  { mldelete nextAttribute;
    mldelete hasValue;
  }
};




void setMatchingExamples(PAssociationRule rule, const TExampleSet &leftSet, const TExampleSet &bothSets)
{
   TIntList *matchLeft = new TIntList();
   rule->matchLeft = matchLeft;
   const_ITERATE(TExampleSet, nli, leftSet)
     matchLeft->push_back((*nli).example);

   TIntList *matchBoth = new TIntList();
   rule->matchBoth = matchBoth;
   const_ITERATE(TExampleSet, nri, bothSets)
     matchBoth->push_back((*nri).example);
}

TAssociationRule::TAssociationRule(PExample al, PExample ar)
: left(al),
  right(ar),
  support(0.0),
  confidence(0.0),
  coverage(0.0),
  strength(0.0),
  lift(0.0),
  leverage(0.0),
  nAppliesLeft(0),
  nAppliesRight(0),
  nAppliesBoth(0),
  nExamples(0),
  nLeft(countItems(al)),
  nRight(countItems(ar))
{}


TAssociationRule::TAssociationRule(PExample al, PExample ar,
                   const float &napLeft, const float &napRight, const float &napBoth, const float &nExamples,
                   int anleft, int anright)
: left(al),
  right(ar), 
  support(napBoth/nExamples),
  confidence(napBoth/napLeft),
  coverage(napLeft/nExamples),
  strength(napRight/napLeft),
  lift(nExamples * napBoth /napLeft / napRight),
  leverage((napBoth*nExamples - napLeft*napRight)/nExamples/nExamples),
  nAppliesLeft(napLeft),
  nAppliesRight(napRight),
  nAppliesBoth(napBoth),
  nExamples(nExamples),
  nLeft(anleft < 0 ? countItems(al) : anleft),
  nRight(anright < 0 ? countItems(ar) : anright)
{
  TExample::iterator ei, ee;

  for(ei = left->begin(), ee = left->end(); ei!=ee; ei++)
    if ((*ei).isSpecial())
      (*ei).setDC();

  for(ei = right->begin(), ee = right->end(); ei!=ee; ei++)
    if ((*ei).isSpecial())
      (*ei).setDK();
}


int TAssociationRule::countItems(PExample ex)
{ 
  int res = 0;
  PITERATE(TExample, ei, ex)
    if (!(*ei).isSpecial())
      res++;
  return res;
}


bool TAssociationRule::applies(const TExample &ex, const PExample &side)
{
  if (side->domain->variables->size())
    return side->compatible(ex);

  // all meta-attributes that appear in 'side' must also appear in 'ex' and be noSpecial
  const_ITERATE(TMetaValues, mi, side->meta) {
    if (!ex.hasMeta((*mi).first) || ex.getMeta((*mi).first).isSpecial())
      return false;
  }

  return true;
}


float computeIntersection(const TExampleSet &set1, const TExampleSet &set2, TExampleSet &intersection)
{
  float isupp = 0.0;
 
  TExampleSet::const_iterator se1i(set1.begin()), se1e(set1.end());
  TExampleSet::const_iterator se2i(set2.begin()), se2e(set2.end());
  while((se1i != se1e) && (se2i != se2e)) {
    if ((*se1i).example < (*se2i).example)
      se1i++;
    else if ((*se1i).example > (*se2i).example)
      se2i++;
    else {
      intersection.push_back(*se1i);
      isupp += (*se1i).weight;
      se1i++;
      se2i++;
    }
  }
  return isupp;
}


/* Searches the tree to find a node that corresponds to itemset 'ex' and returns its support */
float findSupport(const TExample &ex, TItemSetNode *node, TItemSetValue **actualNode = NULL)
{
  vector<TItemSetValue>::iterator li = node->values.begin(); // This is initialized just to avoid warnings.

  TExample::const_iterator ei(ex.begin()), eei(ex.end());
  int attrIndex = 0;
  for(; ei!=eei; ei++, attrIndex++)
    // If attribute is in the itemset
    if (!(*ei).isSpecial()) {
      // Search for the attribute in the list linked by 'nextAttribute'
      while (node && (node->attrIndex != attrIndex))
        node = node->nextAttribute;
      // this attribute does not appear in any itemset that begins with the attributes
      // that we have already encountered in the example
      if (!node)
        return 0.0;

      // Search for the value
      for(li = node->values.begin(); (li!=node->values.end()) && ((*li).value != (*ei).intV); li++);
      // this attribute value does not appear in any itemset ...
      if (li==node->values.end())
        return 0.0;

      // continue if possible
      if (!(*li).branch)
        break;
      node = (*li).branch;
    }

  // If we are not at the end of example yet, we must make sure no further values appear in the itemset
  if (ei!=ex.end())
    while((++ei!=ex.end()) && (*ei).isSpecial());

  if (ei == ex.end()) {
    if (actualNode)
      *actualNode = &*li;
    return (*li).support;
  }
  
    if (actualNode)
      *actualNode = NULL;
  return 0;
}


float findSupport(const TExample &ex, TRuleTreeNode *node, TRuleTreeNode **actualNode = NULL)
{
  TExample::const_iterator ei(ex.begin()), eei(ex.end());
  for(; (ei!=eei) && !(*ei).isSpecial(); ei++);

  for(; ei!=eei; ei++, node = node->hasValue) {
    while (node && (node->attrIndex != ei-ex.begin()))
      node = node->nextAttribute;
    if (!node || (node->value != (*ei).intV))
      raiseError("internal error in RuleTree (attribute/value not found)");

    while((++ei!=eei) && !(*ei).isSpecial());
    if (ei==eei) {
      if (actualNode)
        *actualNode = node;
      return node->support;
    }
  }

  raiseError("internal error in RuleTree (attribute/value not found)");
  return 0.0; //, to make compiler happy
}


TAssociationRulesInducer::TAssociationRulesInducer(float asupp, float aconf)
: maxItemSets(15000),
  confidence(aconf),
  support(asupp),
  classificationRules(false),
  storeExamples(false)
{}


PAssociationRules TAssociationRulesInducer::operator()(PExampleGenerator examples, const int &weightID)
{

  PITERATE(TVarList, vi, examples->domain->variables)
    if ((*vi)->varType != TValue::INTVAR)
      raiseError("cannot induce rules with non-discrete attributes (such as '%s')", (*vi)->get_name().c_str());

  TItemSetNode *tree = NULL;
  PAssociationRules rules;
  if (classificationRules && !examples->domain->classVar)
    raiseError("cannot induce classification rules on classless data");

  try {
    int depth, nOfExamples;
    TDiscDistribution classDist;
    buildTrees(examples, weightID, tree, depth, nOfExamples, classDist);
    
    rules = classificationRules ? generateClassificationRules(examples->domain, tree, nOfExamples, classDist)
                                : generateRules(examples->domain, tree, depth, nOfExamples);
                                
    if (storeExamples) {
      PExampleTable xmpls = mlnew TExampleTable(examples);
      PITERATE(TAssociationRules, ri, rules)
        (*ri)->examples = xmpls;
    }
  }
  catch (...) {
    mldelete tree; 
    throw;
  }

  mldelete tree;
  return rules;
}

    
    
void TAssociationRulesInducer::buildTrees(PExampleGenerator gen, const int &weightID, TItemSetNode *&tree, int &depth, int &nOfExamples, TDiscDistribution &classDist)
{ 
  float suppN;
  depth = 1;
  for(int totni = 0, ni = buildTree1(gen, weightID, tree, suppN, nOfExamples, classDist);
      ni;
      ni = buildNext1(tree, ++depth, suppN)) {
    totni += ni;
    if (totni>maxItemSets)
      raiseError("too many itemsets (%i); increase 'maxItemSets'", totni);
  }
  --depth;
}


// buildTree1: builds the first tree with 1-itemsets
int TAssociationRulesInducer::buildTree1(PExampleGenerator gen, const int &weightID, TItemSetNode *&tree, float &suppN, int &nOfExamples, TDiscDistribution &classDist)
{
  tree = NULL;

  if (classificationRules)
    classDist = TDiscDistribution(gen->domain->classVar);

  int index, itemSets = 0;

  TItemSetNode **toChange = &tree;

  // builds an empty tree with all possible 1-itemsets
  TVarList::const_iterator vi(gen->domain->variables->begin()), ve(gen->domain->variables->end());
  for(index = 0; vi!=ve; vi++, index++) {
    *toChange = mlnew TItemSetNode(*vi, index);
    toChange = &((*toChange)->nextAttribute);
  }

  // fills the tree with indices of examples from gen
  index = 0;
  nOfExamples = 0;
  TExampleIterator ei(gen->begin());
  for(; ei; ++ei, index++) {
    const float wex = WEIGHT(*ei);

    if (classificationRules)
      if ((*ei).getClass().isSpecial())
        continue;
      else
        classDist.add((*ei).getClass(), wex);

    nOfExamples += wex;

    TItemSetNode *ip = tree;
    TExample::const_iterator exi((*ei).begin()), exe((*ei).end());
    for(; exi!=exe; exi++, ip = ip->nextAttribute) {
      if (!(*exi).isSpecial()) {
        const int exv = (*exi).intV;
        if (exv >= (int)(ip->values.size()))
          raiseError("invalid value of attribute '%s'", gen->domain->variables->at(exi-(*ei).begin())->get_name().c_str());
        ip->values[(*exi).intV].examples.push_back(TExWei(index, wex));
      }
    }
  }

  suppN = support * nOfExamples;

  // removes all unsupported itemsets
  itemSets = 0;
  TItemSetNode **ip = &tree;
  while(*ip) {

    // computes sums; li goes through all values, and values that remain go to lue
    vector<TItemSetValue>::iterator lb((*ip)->values.begin()), li(lb), lue(lb), le((*ip)->values.end());
    for(li = lue; li!=le; li++) {
      (*li).sumSupport();
      if ((*li).support >= suppN) {
        if (li!=lue)
          *lue = *li;
        lue++;
      }
    }
    
    // no itemsets for this attribute
    if (lue == lb) {
      TItemSetNode *tip = (*ip)->nextAttribute;
      (*ip)->nextAttribute = NULL; // make sure delete doesn't remove the whole chain
      mldelete *ip; 
      *ip = tip;
    }

    // this attribute has itemset (not necessarily for all values, but 'erase' will cut them
    else {
      (*ip)->values.erase(lue, le);
      itemSets += (*ip)->values.size();
      ip = &((*ip)->nextAttribute);
    }
  }

  return itemSets;
}


/* buildNextTree: uses tree for k-1 itemset, and builds a tree for k-itemset
   by traversing to the depth where k-2 items are defined (buildNext1) and then joining
   all pairs of values descendant to the current position in the tree (makePairs).

   This function descends the tree, recursively calling itself.
   At each level of recursion, an attribute value is (implicitly) added to the itemset
   and k is reduced by one, until it reaches 2.
*/
int TAssociationRulesInducer::buildNext1(TItemSetNode *node, int k, const float suppN)
{
  if (k==2)
    return makePairs(node, suppN);

  // For each value of each attribute...
  int itemSets = 0;
  for(; node; node = node->nextAttribute)
    ITERATE(vector<TItemSetValue>, li, node->values) 
      if ((*li).branch)
        itemSets += buildNext1((*li).branch, k-1, suppN);

  return itemSets;
}


/* This function is called by buildNextTree when it reaches the depth of k-1.
   Nodes at this depth represent k-1-itemsets. Pairs of the remaining attribute
   values are now added if they are supported. 
   
   We only need to check the part of the tree at 'nextAttribute';
   past attributes have been already checked.
*/
int TAssociationRulesInducer::makePairs(TItemSetNode *node, const float suppN)
{
   int itemSets = 0;

   for(TItemSetNode *p1 = node; p1; p1 = p1->nextAttribute) {
     ITERATE(vector<TItemSetValue>, li1, p1->values) {
       TItemSetNode **li1_br = &((*li1).branch);
       for(TItemSetNode *p2 = p1->nextAttribute; p2; p2 = p2->nextAttribute) {
         ITERATE(vector<TItemSetValue>, li2, p2->values) {
           TExampleSet intersection;
           float isupp = computeIntersection((*li1).examples, (*li2).examples, intersection);
               
           // support can also be 0, so we have to check intersection size as well
           if (intersection.size() && (isupp>=suppN)) {
             if (*li1_br && ((*li1_br)->attrIndex != p2->attrIndex))
               li1_br = &((*li1_br)->nextAttribute);
             if (!*li1_br) // either we need a new attribute or no attributes have been added for p1 so far
               *li1_br = mlnew TItemSetNode(p2->attrIndex);

             (*li1_br)->values.push_back(TItemSetValue((*li2).value, intersection, isupp));
             itemSets++;
           }
         }
       }
     }
   }

   return itemSets;
}


PAssociationRules TAssociationRulesInducer::generateClassificationRules(PDomain dom, TItemSetNode *tree, const int nOfExamples, const TDiscDistribution &classDist)
{ TExample left(dom);
  PAssociationRules rules = mlnew TAssociationRules();
  generateClassificationRules1(dom, tree, tree, left, 0, nOfExamples, rules, nOfExamples, classDist, NULL);
  return rules;
}


void TAssociationRulesInducer::generateClassificationRules1(PDomain dom, TItemSetNode *root, TItemSetNode *node, TExample &left, const int nLeft, const float nAppliesLeft, PAssociationRules rules, const int nOfExamples, const TDiscDistribution &classDist, TExampleSet *leftSet)
{ 
  for(; node; node = node->nextAttribute)
    if (node->nextAttribute) {
      // this isn't the class attributes (since the class attribute is the last one)
      ITERATE(vector<TItemSetValue>, li, node->values)
        if ((*li).branch) {
          left[node->attrIndex] = TValue((*li).value);
          generateClassificationRules1(dom, root, (*li).branch, left, nLeft+1, (*li).support, rules, nOfExamples, classDist, &(*li).examples);
        }
      left[node->attrIndex].setDC();
    }
    else
      // this is the last attribute - but is it the class attribute?
      if (nLeft && (node->attrIndex == dom->attributes->size()))
        ITERATE(vector<TItemSetValue>, li, node->values) {
          const float &nAppliesBoth = (*li).support;
          const float aconf =  nAppliesBoth / nAppliesLeft;
          if (aconf >= confidence) {
            PExample right = mlnew TExample(dom);
            right->setClass(TValue((*li).value));
            PAssociationRule rule = mlnew TAssociationRule(mlnew TExample(left), right, nAppliesLeft, classDist[(*li).value], nAppliesBoth, nOfExamples, nLeft, 1);
            if (storeExamples)
              if (!leftSet) {
                set<int> allExamplesSet;
                ITERATE(vector<TItemSetValue>, ri, root->values)
                  ITERATE(TExampleSet, ei, (*ri).examples)
                    allExamplesSet.insert((*ei).example);
                TExampleSet allExamples;
                ITERATE(set<int>, ali, allExamplesSet)
                  allExamples.push_back(TExWei(*ali, 1));
                setMatchingExamples(rule, allExamples, (*li).examples);
              }
              else {
                setMatchingExamples(rule, *leftSet, (*li).examples);
              }
            rules->push_back(rule);
          }
        }
}


PAssociationRules TAssociationRulesInducer::generateRules(PDomain dom, TItemSetNode *tree, const int depth, const int nOfExamples)
{ 
  PAssociationRules rules = mlnew TAssociationRules();
  for(int k = 2; k <= depth; k++) {
    TExample example(dom);
    generateRules1(example, tree, tree, k, k, rules, nOfExamples);
  }
  return rules;
}


void TAssociationRulesInducer::generateRules1(TExample &ex, TItemSetNode *root, TItemSetNode *node, int k, const int nBoth, PAssociationRules rules, const int nOfExamples)
{ 
  /* Descends down the tree, recursively calling itself and adding a value to the
     example (ie itemset) at each call. This goes until k reaches 1. */
  if (k>1)
    for(; node; node = node->nextAttribute) {
      ITERATE(vector<TItemSetValue>, li, node->values)
        if ((*li).branch) {
          ex[node->attrIndex] = TValue((*li).value);
          generateRules1(ex, root, (*li).branch, k-1, nBoth, rules, nOfExamples);
        }

      ex[node->attrIndex].setDC();
    }
  
  else
    for(; node; node = node->nextAttribute) {
      ITERATE(vector<TItemSetValue>, li, node->values) {
        ex[node->attrIndex] = TValue((*li).value);

        /* Rule with one item on the right are treated separately.
           Incidentally, these are also the only that are suitable for classification rules */
        find1Rules(ex, root, (*li).support, nBoth, rules, nOfExamples, (*li).examples);

        if (nBoth>2) {
          TRuleTreeNode *ruleTree = buildTree1FromExample(ex, root);

          try {
            TExample example(ex.domain);
            for(int m = 2;
                (m <= nBoth-1) && generateNext1(ruleTree, ruleTree, root, example, m, ex, (*li).support, rules, nOfExamples, (*li).examples) > 2;
                m++);
          }
          catch (...) {
            mldelete ruleTree;
            throw;
          }
          mldelete ruleTree;
        }
      }

      ex[node->attrIndex].setDC();
    }
}


/* For each value in the itemset, check whether the rule with this value on the right
   and all others on the left has enough confidence, and add it if so. */
void TAssociationRulesInducer::find1Rules(TExample &example, TItemSetNode *tree, const float &nAppliesBoth, const int nBoth, PAssociationRules rules, const int nOfExamples, const TExampleSet &bothSets)
{
  TExample left(example), right(example.domain);
  for(TExample::iterator ei(example.begin()), lefti(left.begin()), righti(right.begin()); ei!=example.end(); ei++, lefti++, righti++) 
    if (!(*ei).isSpecial()) {
      (*lefti).setDC();
      *righti = *ei;
      TItemSetValue *nodeLeft;
      const float nAppliesLeft = findSupport(left, tree, &nodeLeft);
      const float tconf = nAppliesBoth/nAppliesLeft;
      if (tconf >= confidence) {
        const float nAppliesRight = findSupport(right, tree);
        PAssociationRule rule = mlnew TAssociationRule(mlnew TExample(left), mlnew TExample(right), nAppliesLeft, nAppliesRight, nAppliesBoth, nOfExamples, nBoth-1, 1);
        if (storeExamples)
          setMatchingExamples(rule, nodeLeft->examples, bothSets);
        rules->push_back(rule);
      }
      (*righti).setDC();
      *lefti = *ei;
    }
}


/* Builds a 1-tree (a list of TRuleTreeNodes linked by nextAttribute) with 1-itemsets
   that correspond to values found in example 'ex' */
TRuleTreeNode *TAssociationRulesInducer::buildTree1FromExample(TExample &ex, TItemSetNode *node)
{ 
  TRuleTreeNode *newTree = NULL;
  TRuleTreeNode **toChange = &newTree;

  ITERATE(TExample, ei, ex) 
    if  (!(*ei).isSpecial()) {
      while(node && (node->attrIndex != ei-ex.begin()))
        node = node->nextAttribute;
      _ASSERT(node);

      vector<TItemSetValue>::iterator li(node->values.begin()), le(node->values.end());
      for(; (li != le) && ((*li).value != (*ei).intV); li++);
      _ASSERT(li!=le);

      *toChange = mlnew TRuleTreeNode(node->attrIndex, (*li).value, (*li).support, (*li).examples);
      toChange = &((*toChange)->nextAttribute);
    }

  return newTree;
}


/* Extends the tree by one level, recursively calling itself until k is 2.
   (At the beginning, k-1 is the tree depth, so when we have 1-tree, this function is called with k=2.)
   At each recursive call, a value from the example 'wholeEx' is added to 'right'. */
int TAssociationRulesInducer::generateNext1(TRuleTreeNode *ruleTree, TRuleTreeNode *node, TItemSetNode *itemsets,
                                            TExample &right, int k, TExample &wholeEx, const float &nAppliesBoth,
                                            PAssociationRules rules, const int nOfExamples, const TExampleSet &bothSets)
{
  if (k==2)
    return generatePairs(ruleTree, node, itemsets, right, wholeEx, nAppliesBoth, rules, nOfExamples, bothSets);

  int itemSets = 0;
  for(; node; node = node->nextAttribute)
    if (node->hasValue) {
      right[node->attrIndex] = TValue(node->value);
      itemSets += generateNext1(ruleTree, node->hasValue, itemsets, right, k-1, wholeEx, nAppliesBoth, rules, nOfExamples, bothSets);
      right[node->attrIndex].setDC();
    }
  
  return itemSets;
}


int TAssociationRulesInducer::generatePairs(TRuleTreeNode *ruleTree, TRuleTreeNode *node, TItemSetNode *itemsets,
                                            TExample &right, TExample &wholeEx, const float &nAppliesBoth,
                                            PAssociationRules rules, const int nOfExamples, const TExampleSet &bothSets)
{
   int itemSets = 0;

   for(TRuleTreeNode *p1 = node; p1; p1 = p1->nextAttribute) {
     right[p1->attrIndex] = TValue(p1->value);
     TRuleTreeNode **li1_br = &(p1->hasValue);

     for(TRuleTreeNode *p2 = p1->nextAttribute; p2; p2 = p2->nextAttribute) {
       right[p2->attrIndex] = TValue(p2->value);

       PExample left = mlnew TExample(wholeEx.domain);
       for(TExample::iterator righti(right.begin()), wi(wholeEx.begin()), lefti(left->begin()); wi != wholeEx.end(); wi++, righti++, lefti++)
         if (!(*wi).isSpecial() && (*righti).isSpecial())
           *lefti = *wi;

       TItemSetValue *nodeLeft;
       float nAppliesLeft = findSupport(left.getReference(), itemsets, &nodeLeft);
       float aconf = nAppliesBoth/nAppliesLeft;
       if (aconf>=confidence) {

         TExampleSet intersection;
         float nAppliesRight = computeIntersection(p1->examples, p2->examples, intersection);
               
         // Add the item to the tree (confidence can also be 0, so we'd better check the intersection size)
         if (intersection.size()) {
           *li1_br = mlnew TRuleTreeNode(p2->attrIndex, p2->value, nAppliesRight, intersection);
           li1_br = &((*li1_br)->nextAttribute);
           itemSets++;
         }

         PAssociationRule rule = mlnew TAssociationRule(left, mlnew TExample(right), nAppliesLeft, nAppliesRight, nAppliesBoth, nOfExamples);
         if (storeExamples)
           setMatchingExamples(rule, nodeLeft->examples, bothSets);
         rules->push_back(rule);
       }
       right[p2->attrIndex].setDC();
     }
     right[p1->attrIndex].setDC();
   }

   return itemSets;
}


/*~***************************************************************************************
TAssociationClassifier
*****************************************************************************************/



bool notClassRule(PAssociationRule rule)
{
  if (!rule->left->getClass().isSpecial()) return true;

  TExample::const_iterator ei=rule->right->begin(), eei=rule->right->end();
  for(; (ei!=eei) && (*ei).isSpecial(); ei++);
  return (ei==eei) || (++ei!=eei);
}

TAssociationLearner::TAssociationLearner()
: confidence(0.5),
  support(0.5),
  voteWeight('s'),
  maxItemSets(15000)
{}


PClassifier TAssociationLearner::operator()(PExampleGenerator gen, const int &weight)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");

  TAssociationRulesInducer inducer(support, confidence);
  inducer.classificationRules = true;
  return mlnew TAssociationClassifier(gen->domain, inducer(gen, weight));
}


TAssociationClassifier::TAssociationClassifier(PDomain dom, PAssociationRules arules, char avote)
: TClassifierFD(dom, true),
  rules(arules),
  voteWeight(avote)
{}


PDistribution TAssociationClassifier::classDistribution(const TExample &ex)
{ PDistribution wdistval = TDistribution::create(domain->classVar);
  TDistribution &distval = wdistval.getReference();

  if ((voteWeight=='s') || (voteWeight=='c') || (voteWeight=='p'))
    PITERATE(TAssociationRules, ri, rules) {
      if (!(*ri)->right->getClass().isSpecial() && (*ri)->left->compatible(ex))
        switch (voteWeight) {
          case 's': distval.add((*ri)->right->getClass(), (*ri)->support);
          case 'c': distval.add((*ri)->right->getClass(), (*ri)->confidence);
          case 'p': distval.add((*ri)->right->getClass(), (*ri)->confidence*(*ri)->support);
        }
    }
  else
    PITERATE(TAssociationRules, ri, rules)
      if (!(*ri)->right->getClass().isSpecial() && (*ri)->left->compatible(ex)) {
        float me;
        switch (voteWeight) {
          case 'S': me=(*ri)->support;
          case 'C': me=(*ri)->confidence;
          case 'P': me=(*ri)->confidence*(*ri)->support;
        }
        if (me>distval[(*ri)->right->getClass()])
          distval.set((*ri)->right->getClass(), me);
      }
        
  distval.normalize();
  return wdistval;
}
