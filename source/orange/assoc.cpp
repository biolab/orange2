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


#include  <iomanip>
#include <fstream>

#include "random.hpp"

#include "examplegen.hpp"
#include "spec_gen.hpp"
#include "table.hpp"
#include "filter.hpp"

#include "assoc.ppp"

DEFINE_TOrangeVector_classDescription(PAssociationRule, "TAssociationRules")

/*~***************************************************************************************
TItemSetTree
*****************************************************************************************/

Tleb::~Tleb()
{ mldelete branch; }

void Tleb::sumSupport()
{ support=0; 
  ITERATE(vector<TExWei>, wi, examples) support+=(*wi).weight; }


TItemSetNode::TItemSetNode(PVariable var, int anattri)
 : attrIndex(anattri), unknown(NULL)
{ for(int vi=0; vi<var->noOfValues(); values.push_back(Tleb(vi++, vector<TExWei>(), (TItemSetNode *)NULL))); }


void TItemSetTree::addExample(const TExample &example, const vector<TExWei> &intersection)
{ float isupp=0;
  const_ITERATE(vector<TExWei>, wei, intersection) isupp+=(*wei).weight;
  addExample(example, intersection, isupp);
}

void TItemSetTree::addExample(const TExample &example, const vector<TExWei> &intersection, const float &support)
{
  TItemSetNode **toChange=&root;
  int eindex=0;

  int k=0;
  {const_ITERATE(TExample, ei, example)
    if (!(*ei).isSpecial()) k++;

  }
  if (k>largestK) largestK=k;

  const_ITERATE(TExample, ei, example) {
    if (!(*ei).isSpecial()) {
      k--;

      // find the corresponding atribute
      while((*toChange) && ((*toChange)->attrIndex!=eindex))
        toChange=&((*toChange)->unknown);

      // if attribute has not been used yet, create it
      if (!*toChange) 
        *toChange=mlnew TItemSetNode(eindex);

      // check if we have this label defined already
      vector<Tleb>::iterator li((*toChange)->values.begin());
      for(;
          (li!=(*toChange)->values.end()) && ((*li).label!=(*ei).intV);
          li++);
      if (li!=(*toChange)->values.end())
        toChange=&((*li).branch);

      else
        if (!k) {
          (*toChange)->values.push_back(Tleb((*ei).intV, intersection, support));
          return;
        }
        else {
          (*toChange)->values.push_back(Tleb((*ei).intV));
          toChange=&((*toChange)->values.back().branch);
        }
      }    
    eindex++;
  }
}



float TItemSetTree::findSupport(const TExample &ex)
{ 
  TItemSetNode *tempNode = root;
  vector<Tleb>::iterator li = tempNode->values.begin(); // This is initialized just to avoid warnings.
  TExample::const_iterator ei(ex.begin()), eei(ex.end());
  for(; ei!=eei; ei++)
    if (!(*ei).isSpecial()) {
      // search for the attribute
      while (tempNode && tempNode->attrIndex != ei-ex.begin())
        tempNode = tempNode->unknown;
      if (!tempNode)
        return 0.0;

      // search for the value
      for(li = tempNode->values.begin(); (li!=tempNode->values.end()) && ((*li).label != (*ei).intV); li++);
      if (li==tempNode->values.end())
        return 0.0;

      // continue if possible
      if (!(*li).branch)
        break;
      tempNode = (*li).branch;
    }

  // return support if that's it and 0 if more values should be find  
  if (ei!=ex.end())
    while((++ei!=ex.end()) && (*ei).isSpecial());
  return (ei==ex.end()) ? (*li).support : 0;
}



/*~***************************************************************************************
TAssociationRule
*****************************************************************************************/

TAssociationRule::TAssociationRule()
{ raiseError("invalid constructor call"); };

TAssociationRule::TAssociationRule(PExample al, PExample ar,
                 const float &asup, const float &acon, const float &acov,
                 const float &astr, const float &alif, const float &alev,
                 const float &napl, const float &napr, const float &napb, const float &nexa,
                 const int &anleft, const int &anright)
: left(al),
  right(ar), 
  support(asup),
  confidence(acon),
  coverage(acov),
  strength(astr),
  lift(alif),
  leverage(alev),
  nAppliesLeft(napl),
  nAppliesRight(napr),
  nAppliesBoth(napb),
  nExamples(nexa),
  nleft(anleft < 0 ? countItems(al) : anleft),
  nright(anright < 0 ? countItems(ar) : anright)
{ correctSpecials(); }


TAssociationRule::TAssociationRule(PExample al, PExample ar,
                   const float &napLeft, const float &napRight, const float &napBoth, const float &nExamples,
                   int anleft, int anright)
: left(al),
  right(ar), 
  support(napBoth/nExamples),
  confidence(napBoth/napRight),
  coverage(napLeft/nExamples),
  strength(napBoth/napLeft),
  lift(nExamples/napLeft),
  leverage(napBoth/nExamples - napLeft*napRight/nExamples/nExamples),
  nAppliesLeft(napLeft),
  nAppliesRight(napRight),
  nAppliesBoth(napBoth),
  nExamples(nExamples),
  nleft(anleft < 0 ? countItems(al) : anleft),
  nright(anright < 0 ? countItems(ar) : anright)
{ correctSpecials(); }


void TAssociationRule::correctSpecials()
{ TExample::iterator ei, ee;

  for(ei = left->begin(), ee = left->end(); ei!=ee; ei++)
    if ((*ei).isSpecial())
      (*ei).setDC();

  for(ei = right->begin(), ee = right->end(); ei!=ee; ei++)
    if ((*ei).isSpecial())
      (*ei).setDK();
}

int TAssociationRule::countItems(PExample ex)
{ int res = 0;
  PITERATE(TExample, ei, ex)
    if (!(*ei).isSpecial())
      res++;
  return res;
}

/*~***************************************************************************************
TAssociationRules
*****************************************************************************************/

TAssociationRulesInducer::TAssociationRulesInducer(float asupp, float aconf, int awei)
: weightID(awei),
  maxItemSets(15000),
  conf(aconf),
  supp(asupp),
  nOfExamples(0.0)
{}

PAssociationRules TAssociationRulesInducer::operator()(PExampleGenerator examples)
{ PITERATE(TVarList, vi, examples->domain->variables)
    if ((*vi)->varType != TValue::INTVAR)
      raiseError("cannot induce rules with non-discrete attributes (such as '%s')", (*vi)->name.c_str());

  buildTrees(examples);
  return generateRules(examples->domain);
}

/* Building a tree of itemsets */

void TAssociationRulesInducer::buildTrees(PExampleGenerator gen)
{ 
  int ni, totni=0;

  if ((totni=buildTree1(gen)) >0) {
    if (totni>maxItemSets)
      raiseError("too many itemsets (%i); increase 'maxItemSets'", totni);
    TExample example(gen->domain);
    while((ni=buildNext1(tree.root, example, tree.largestK+1))>0) {
      totni += ni;
      if (totni>maxItemSets)
        raiseError("too many itemsets (%i); increase 'maxItemSets'", totni);
    }
  }
}


class TNotSupported {
public:
  float supp;
  TNotSupported(float asupp) : supp(asupp) {};
  bool operator()(Tleb &leb) { float sze=leb.support; return !sze || (sze<supp); }
};


// buildTree1: builds the first tree with 1-itemset
int TAssociationRulesInducer::buildTree1(PExampleGenerator gen)
{
  TItemSetNode **toChange=&tree.root;
  int ind=0, itemSets=0;

  // builds an empty tree with all possible 1-itemsets
  PITERATE(TVarList, vi, gen->domain->variables) {
    *toChange=mlnew TItemSetNode(*vi, ind++);
    toChange=&((*toChange)->unknown);
  }

  // fills the tree with indices of examples from gen
  int index=0;
  PEITERATE(ei, gen) {
    TItemSetNode *ip=tree.root;
    float wex = WEIGHT(*ei);
    ITERATE(TExample, exi, *ei) {
      if (!(*exi).isSpecial()) {
        int exv = (*exi).intV;
        while ((int)(ip->values.size()) <= exv)
          ip->values.push_back(Tleb(ip->values.size(), vector<TExWei>(), (TItemSetNode *)NULL));

        ip->values[(*exi).intV].examples.push_back(TExWei(index, wex));
      }
      ip=ip->unknown;
    }
    index++;
    nOfExamples+=wex;
  }

  // removes all unsupported itemsets
  itemSets=0;
  TItemSetNode **ip=&tree.root;
  while(*ip) {
    ITERATE(vector<Tleb>, wi, (*ip)->values) (*wi).sumSupport();

    (*ip)->values.erase(remove_if((*ip)->values.begin(), (*ip)->values.end(), TNotSupported(supp*nOfExamples)), (*ip)->values.end());

    if (!(*ip)->values.size()) {
      TItemSetNode *tip=(*ip)->unknown;
      (*ip)->unknown=NULL; mldelete *ip; 
      *ip=tip;
    }
    else {
      itemSets+=(*ip)->values.size();
      ip=&((*ip)->unknown);
    }
  }

  if (itemSets) tree.largestK=1;
  return itemSets;
}


// buildNextTree: uses tree for k-1 itemset, and builds a tree for k-itemset
// by traversing to the depth where k-2 items are defined (buildNext1) and then joining
// all pairs of values descendant to the current position in the tree (makePairs)

int TAssociationRulesInducer::buildNext1(TItemSetNode *tempNode, TExample &example, int k)
{
  int itemSets=0;

  if (k==2)
    itemSets=makePairs(tempNode, example);
  else {
    ITERATE(vector<Tleb>, li, tempNode->values) 
      if ((*li).branch) {
        example[tempNode->attrIndex] = TValue((*li).label);
        if ((*li).branch)
          itemSets+=buildNext1((*li).branch, example, k-1);
        if (itemSets>maxItemSets)
          raiseError("too many itemsets (%i); increase 'maxItemSets'", itemSets);
      }

    example[tempNode->attrIndex].setDC();
    if (tempNode->unknown)
      itemSets+=buildNext1(tempNode->unknown, example, k);
        if (itemSets>maxItemSets)
          raiseError("too many itemsets (%i); increase 'maxItemSets'", itemSets);
  }

  return itemSets;
}



int TAssociationRulesInducer::makePairs(TItemSetNode *tempNode, TExample &example)
{
   int itemSets=0;
   
   for(TItemSetNode *p1=tempNode; p1; p1=p1->unknown) {
     for(TItemSetNode *p2=p1->unknown; p2; p2=p2->unknown) {
       ITERATE(vector<Tleb>, li1, p1->values)
         ITERATE(vector<Tleb>, li2, p2->values) {
           vector<TExWei> intersection;
           
           set_intersection((*li1).examples.begin(), (*li1).examples.end(),
                            (*li2).examples.begin(), (*li2).examples.end(),
                            inserter(intersection, intersection.begin()));
          
           float isupp=0;
           ITERATE(vector<TExWei>, wi, intersection) isupp+=(*wi).weight;

           if (intersection.size() && isupp>=supp * nOfExamples) {
             example[p1->attrIndex]= TValue((*li1).label);
             example[p2->attrIndex]= TValue((*li2).label);

             tree.addExample(example, intersection, isupp);
             itemSets++;
           }
         }
       example[p2->attrIndex].setDC();
     }
     example[p1->attrIndex].setDC();
   }

   return itemSets;
}



/* Generating rules from the tree */

PAssociationRules TAssociationRulesInducer::generateRules(PDomain dom)
{ PAssociationRules rules=mlnew TAssociationRules();
  for(int k=2; k<=tree.largestK; k++) {
    TExample example(dom);
    generateRules1(example, tree.root, k, k, rules);
  }
  return rules;
}


void TAssociationRulesInducer::generateRules1(TExample &ex, TItemSetNode *root, int k, int oldk, PAssociationRules rules)
{ if (k>1) {
    ITERATE(vector<Tleb>, li, root->values) 
      if ((*li).branch) {
        ex[root->attrIndex]=TValue((*li).label);
        generateRules1(ex, (*li).branch, k-1, oldk, rules);
      }
    ex[root->attrIndex].setDC();
    if (root->unknown) generateRules1(ex, root->unknown, k, oldk, rules);
  }
  else {
    ITERATE(vector<Tleb>, li, root->values) {
      ex[root->attrIndex]=TValue((*li).label);

      int itemSets;
      find1Rules(ex, (*li).support, oldk, rules);

      TItemSetTree ruleTree(buildTree1FromExample(ex, tree.root, itemSets), 1);

      int m=2;
      TExample example(ex.domain);
      while(   (itemSets>=2) && (oldk>=m+1)
            && generateNext1(ruleTree, ruleTree.root, example, m++, ex, (*li).support, rules));
    }
    ex[root->attrIndex].setDC();
    if (root->unknown)
      generateRules1(ex, root->unknown, k, oldk, rules);
    }
}


void TAssociationRulesInducer::find1Rules(TExample &example, const float &nAppliesBoth, int oldk, PAssociationRules rules)
{
  TExample w_e(example), e(example.domain);
  for(TExample::iterator ei(example.begin()), wei(w_e.begin()), eei(e.begin()); ei!=example.end(); ei++, wei++, eei++) 
    if (!(*ei).isSpecial()) {
      (*wei).setDC();
      *eei = *ei;
      float nAppliesLeft=tree.findSupport(w_e);
      float tconf=nAppliesBoth/nAppliesLeft;
      if (tconf>=conf) {
        float nAppliesRight=tree.findSupport(e);
        PAssociationRule rule=mlnew TAssociationRule(mlnew TExample(w_e), mlnew TExample(e), 
                                                     nAppliesLeft, nAppliesRight, nAppliesBoth, nOfExamples, 
                                                     oldk-1, 1);
        if (conditions(rule))
          rules->push_back(rule);
      }
      (*eei).setDC();
      *wei = *ei;
    }
}


TItemSetNode *TAssociationRulesInducer::buildTree1FromExample(TExample &ex, TItemSetNode *root1, int &itemSets)
{ TItemSetNode *newTree, **toChange=&newTree;
  ITERATE(TExample, ei, ex) 
    if  (!(*ei).isSpecial()) {
      while(root1 && (root1->attrIndex!=ei-ex.begin()))
        root1=root1->unknown;
      _ASSERT(root1!=NULL);

      vector<Tleb>::iterator li(root1->values.begin());
      for(; (li!=root1->values.end()) && ((*li).label!=(*ei).intV); li++);
      _ASSERT(li!=root1->values.end());

      *toChange=mlnew TItemSetNode(root1->attrIndex);
      (*toChange)->values.push_back(Tleb((*li).label, (*li).examples, (*li).support));
      toChange=&((*toChange)->unknown);
      itemSets++;
    }
  *toChange=NULL;
  return newTree;
}



int TAssociationRulesInducer::generateNext1(TItemSetTree &ruleTree, TItemSetNode *tempNode,
                TExample &example, int k, TExample &wholeEx, const float &nAppliesBoth, PAssociationRules rules)
{
  int itemSets=0;
  if (k==2)
    return generatePairs(ruleTree, tempNode, example, wholeEx, nAppliesBoth, rules);

  ITERATE(vector<Tleb>, bi, tempNode->values) {
    example[tempNode->attrIndex] = TValue((*bi).label);
    if ((*bi).branch)
      itemSets+=generateNext1(ruleTree, (*bi).branch, example, k-1, wholeEx, nAppliesBoth, rules);
  }

  example[tempNode->attrIndex].setDC();
  if (tempNode->unknown)
    itemSets+=generateNext1(ruleTree, tempNode->unknown, example, k, wholeEx, nAppliesBoth, rules);
  
  return itemSets;
}


int TAssociationRulesInducer::generatePairs(TItemSetTree &ruleTree, TItemSetNode *tempNode,
                                  TExample &example, TExample &wholeEx, const float &nAppliesBoth,
                                  PAssociationRules rules)
{
   int itemSets=0;

   for(TItemSetNode *p1=tempNode; p1; p1=p1->unknown) {
     for(TItemSetNode *p2=p1->unknown; p2; p2=p2->unknown) {
       ITERATE(vector<Tleb>, li1, p1->values) 
         ITERATE(vector<Tleb>, li2, p2->values) {
           example[p1->attrIndex]= TValue((*li1).label);
           example[p2->attrIndex]= TValue((*li2).label);

           TExample w_e(wholeEx.domain);
           for(TExample::iterator ei(example.begin()), wi(wholeEx.begin()), wei(w_e.begin()); wi!=wholeEx.end(); wi++, ei++, wei++)
             if (!(*wi).isSpecial() &&  (*ei).isSpecial())
               *wei= *wi;
           float nAppliesLeft=tree.findSupport(w_e);

           float aconf = nAppliesBoth/nAppliesLeft;
           if (aconf>=conf) {
             ruleTree.addExample(example);
             PAssociationRule rule=mlnew TAssociationRule(mlnew TExample(w_e), mlnew TExample(example), 
                                                          nAppliesLeft, tree.findSupport(example), nAppliesBoth, nOfExamples);
             rule->confidence = aconf;
             if (conditions(rule))
               rules->push_back(rule);
             itemSets++;
           }
         }
       example[p2->attrIndex].setDC();
     }
     example[p1->attrIndex].setDC();
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
: condfile(""),
  conf(0.5),
  supp(0.5),
  voteWeight('s'),
  maxItemSets(15000)
{}


PClassifier TAssociationLearner::operator()(PExampleGenerator gen, const int &weight)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");

  TAssociationRulesInducer inducer(supp, conf, weight);
  if (condfile.length()) {
    ifstream outstr(condfile.c_str());
    inducer.conditions = TRuleCondDisjunctions(gen->domain, outstr);
  }
  /* add the condition that the class value must occur on the right but not on the left side */
  PAssociationRules rules = inducer(gen);
  rules->erase(remove_if(rules->begin(), rules->end(), notClassRule), rules->end());

  return mlnew TAssociationClassifier(gen->domain, rules);
}


TAssociationClassifier::TAssociationClassifier(PDomain dom, PAssociationRules arules, char avote)
: TClassifierFD(dom, true),
  rules(arules),
  voteWeight(avote)
{}


PDistribution TAssociationClassifier::classDistribution(const TExample &ex)
{ PDistribution wdistval = TDistribution::create(domain->classVar);
  TDistribution &distval = const_cast<TDistribution &>(wdistval.getReference());

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
