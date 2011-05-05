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


#include "stladdon.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "table.hpp"

#include "contingency.hpp"
#include "distvars.hpp"

#include "measures.hpp"

#include "majority.hpp"

#include "tdidt_split.hpp"
#include "tdidt_stop.hpp"

#include "tdidt.ppp"


DEFINE_TOrangeVector_classDescription(PTreeNode, "TTreeNodeList", true, ORANGE_API)


/* Default components for split constructor -- split constructors for
   discrete and continuous attributes and classes, and the corresponding
   attribute measures.

   To avoid problems under gcc, these are initialized by an explicit call
   to tdidt_cpp_gcUnsafeInitialization made from initOrange at the end
   of module initialization.
*/

PTreeSplitConstructor defaultDiscreteTreeSplitConstructor;
PTreeSplitConstructor defaultContinuousTreeSplitConstructor;
PTreeStopCriteria defaultStop;

void tdidt_cpp_gcUnsafeInitialization()
{ PMeasureAttribute defaultDiscreteMeasure = mlnew TMeasureAttribute_gainRatio;
  PMeasureAttribute defaultContinuousMeasure = mlnew TMeasureAttribute_MSE;

  defaultDiscreteTreeSplitConstructor 
    = mlnew TTreeSplitConstructor_Combined(
         mlnew TTreeSplitConstructor_Attribute(defaultDiscreteMeasure, 0.0, 2.0),
         mlnew TTreeSplitConstructor_Threshold(defaultDiscreteMeasure, 0.0, 5.0));

  defaultContinuousTreeSplitConstructor 
    = mlnew TTreeSplitConstructor_Combined(
         mlnew TTreeSplitConstructor_Attribute(defaultContinuousMeasure, 0.0, 2.0),
         mlnew TTreeSplitConstructor_Threshold(defaultContinuousMeasure, 0.0, 5.0));

  defaultStop = mlnew TTreeStopCriteria_common();
}



int TTreeNode::treeSize() const
{ int sum = 1;
  if (branches)
    const_PITERATE(TTreeNodeList, bi, branches)
      if (*bi)
        sum += (*bi)->treeSize();
  return sum;
}


void TTreeNode::removeStoredInfo()
{ distribution = PDistribution();
  contingency = PDomainContingency();
  examples = PExampleGenerator();

  if (branches)
    const_PITERATE(TTreeNodeList, bi, branches)
      if (*bi)
        (*bi)->treeSize();
}



TTreeLearner::TTreeLearner()
: storeExamples(false),
  storeDistributions(true),
  storeContingencies(false),
  storeNodeClassifier(true),
  maxDepth(100)
{}



PClassifier TTreeLearner::operator()(PExampleGenerator ogen, const int &weight)
{ if (!ogen)
    raiseError("invalid example generator");

  PVariable &classVar = ogen->domain->classVar;

  if (!classVar)
    raiseError("class-less domain");

  bool tempSplit = !split;
  if (tempSplit)
    if (classVar->varType == TValue::INTVAR)
      split = defaultDiscreteTreeSplitConstructor;
    else if (classVar->varType == TValue::FLOATVAR)
      split = defaultContinuousTreeSplitConstructor;
    else
      raiseError("invalid class type (discrete or continuous expected)");

  bool tempStop = !stop;
  if (tempStop)
    stop = defaultStop;

  bool tempSplitter = !exampleSplitter;
  if (tempSplitter)
    exampleSplitter = mlnew TTreeExampleSplitter_UnknownsAsSelector;

  try {
    PExampleGenerator examples;
    /* If we don't intend to store them, we'll copy them if they're not in a table. 
       If we must store examples, we'll copy them in any case... */ 
    if (storeExamples)
      examples = mlnew TExampleTable(ogen);
    else
      examples = toExampleTable(ogen);

    PDistribution apriorClass = getClassDistribution (examples);
    if (apriorClass->abs == 0)
      raiseError("no examples");

    vector<bool> candidates(examples->domain->attributes->size(), true);

    PTreeNode root = call(examples, weight, apriorClass, candidates, 0);
    if (storeExamples)
      root->examples = examples;

    if (tempSplit)
      split = PTreeSplitConstructor();
    if (tempStop)
      stop = PTreeSplitConstructor();
    if (tempSplitter)
      exampleSplitter = PTreeExampleSplitter();

    return mlnew TTreeClassifier(examples->domain, root, 
                               descender ? descender : mlnew TTreeDescender_UnknownMergeAsSelector);
  }
  catch (exception) {
    if (tempSplit)
      split = PTreeSplitConstructor();
    if (tempStop)
      stop = PTreeSplitConstructor();
    if (tempSplitter)
      exampleSplitter = PTreeExampleSplitter();
    throw;
  }
}



PTreeNode TTreeLearner::operator()(PExampleGenerator examples, const int &weightID, PDistribution apriorClass, vector<bool> &candidates, const int &depth)
{
  PDomainContingency contingency;
  PDomainDistributions domainDistributions;
  PDistribution classDistribution;

  if (!examples->numberOfExamples())
    return PTreeNode();

  if (contingencyComputer)
    contingency = contingencyComputer->call(examples, weightID);

  if (storeContingencies)
    contingency = mlnew TDomainContingency(examples, weightID);

  if (contingency)
    classDistribution = contingency->classes;
  else if (storeDistributions)
    classDistribution = getClassDistribution(examples, weightID);

  if (classDistribution) {
    if (!classDistribution->abs)
      return PTreeNode();
  }
  else
    if (examples->weightOfExamples() < 1e-10)
      return PTreeNode();
  
  TTreeNode *utreeNode = mlnew TTreeNode();
  PTreeNode treeNode = utreeNode;

  utreeNode->weightID = weightID;

  bool isLeaf = ((maxDepth>=0) && (depth == maxDepth)) || stop->call(examples, weightID, contingency);

  if (isLeaf || storeNodeClassifier) {
    utreeNode->nodeClassifier = nodeLearner
                              ? nodeLearner->smartLearn(examples, weightID, contingency, domainDistributions, classDistribution)
                              : TMajorityLearner().smartLearn(examples, weightID, contingency, domainDistributions, classDistribution);

    if (isLeaf) {
      if (storeContingencies)
        utreeNode->contingency = contingency;
      if (storeDistributions)
        utreeNode->distribution = classDistribution;

      return treeNode;
    }
  }

  utreeNode->contingency = contingency;
  utreeNode->distribution = classDistribution;

  float quality;
  int spentAttribute;
  utreeNode->branchSelector = split->call(utreeNode->branchDescriptions, utreeNode->branchSizes, quality, spentAttribute,
                                          examples, weightID, contingency,
                                          apriorClass ? apriorClass : classDistribution,
                                          candidates, utreeNode->nodeClassifier);

  isLeaf = !utreeNode->branchSelector;
  bool hasNullNodes = false;

  if (!isLeaf) {
    if (spentAttribute>=0)
      if (candidates[spentAttribute])
        candidates[spentAttribute] = false;
      else
        spentAttribute = -1;
    /* BEWARE: If you add an additional 'return' in the code below,
               do not forget to restore the candidate's flag. */

    utreeNode->branches = mlnew TTreeNodeList();

    vector<int> newWeights;

    PExampleGeneratorList subsets = exampleSplitter->call(treeNode, examples, weightID, newWeights);

    if (!utreeNode->branchSizes)
      utreeNode->branchSizes = branchSizesFromSubsets(subsets, weightID, newWeights);

    if (!storeContingencies)
      utreeNode->contingency = PDomainContingency();
    if (!storeDistributions)
      utreeNode->distribution = PDistribution();

    vector<int>::iterator nwi = newWeights.begin(), nwe = newWeights.end();
    PITERATE(TExampleGeneratorList, gi, subsets) {
      if ((*gi)->numberOfExamples()) {
        utreeNode->branches->push_back(call(*gi, (nwi!=nwe) ? *nwi : weightID, apriorClass, candidates, depth+1));
        // Can store pointers to examples: the original is stored in the root
        if (storeExamples && utreeNode->branches->back())
          utreeNode->branches->back()->examples = *gi;
        else if ((nwi!=nwe) && *nwi && (*nwi != weightID))
            examples->removeMetaAttribute(*nwi);
      }
      else {
        utreeNode->branches->push_back(PTreeNode());
        hasNullNodes = true;
      }

      if (nwi!=nwe)
        nwi++;
    }

    /* If I set it to false, it must had been true before
       (otherwise, my TreeSplitConstructor wouldn't be allowed to spend it).
       Hence, I can simply set it back to true... */
    if (spentAttribute>=0)
      candidates[spentAttribute] = true;
  }

  else {  // no split constructed
    if (!utreeNode->contingency)
      utreeNode->contingency = PDomainContingency();
    if (!storeDistributions)
      utreeNode->distribution = PDistribution();
  }


  if (isLeaf || hasNullNodes) {
    if (!utreeNode->nodeClassifier)
      utreeNode->nodeClassifier = nodeLearner
                              ? nodeLearner->smartLearn(examples, weightID, contingency, domainDistributions, classDistribution)
                              : TMajorityLearner().smartLearn(examples, weightID, contingency, domainDistributions, classDistribution);
  }

  return treeNode;
}



PDiscDistribution TTreeLearner::branchSizesFromSubsets(PExampleGeneratorList subsets, const int &weightID, const vector<int> &weights) const
{
  TDiscDistribution *bs = mlnew TDiscDistribution();
  PDiscDistribution branchSizes = bs;

  vector<int>::const_iterator wi (weights.begin());
  bool hasWeights = wi != weights.end();

  PITERATE(TExampleGeneratorList, egli, subsets) {
    float totw = 0.0;
    int wID = hasWeights ? *(wi++) : weightID;

    if (!wID)
      totw = (*egli)->numberOfExamples();

    if (wID || (totw<0))
      PEITERATE(ei, *egli)
        totw += WEIGHT2(*ei, wID);

    bs->push_back(totw);
  }

  return branchSizes;
}


TTreeClassifier::TTreeClassifier(PDomain dom, PTreeNode atree, PTreeDescender adescender)
: TClassifierFD(dom),
  tree(atree), 
  descender(adescender)
{}



/* Classification is somewhat complex due to descender component.

   If descender does not require voting (i.e. always descends to a single
   node), methods simply use descender to find a node and use the
   corresponding nodeClassifier.

   If descender requires a vote, this is indicated by returning
   branch weights. The method responds by calling 'vote'. Voting is
   always done by a weighted sum class probabilities (even when
   the outcome should be single value, not distribution).

   'vote' method thus calls classDistribution for each subnode.
   classDistribution again uses descender to descend from the subnode.
   If it requires vote again, classDistribution will again call
   'vote'. We thus have a mutual (recursive) calls of 'vote'
   and 'classDistribution' - not at all levels of the tree but
   only at those where descender takes a break... */


PDistribution TTreeClassifier::findNodeDistribution(PTreeNode node, const TExample &exam)
{
  if (node->distribution)
    return node->distribution;

  if (node->contingency && node->contingency->classes)
    return node->contingency->classes;

  if (node->nodeClassifier) {
    PDistribution dist = node->nodeClassifier->classDistribution(exam);
    if (dist)
      return dist;
  }

  if (classVar->varType == TValue::INTVAR) {
    const int nval = classVar->noOfValues();
    if (nval)
      return mlnew TDiscDistribution(nval, 1.0/nval);
  }

  return PDistribution();
}


TValue TTreeClassifier::findNodeValue(PTreeNode node, const TExample &exam)
{
  PDistribution dist = findNodeDistribution(node, exam);
  if (dist)
    return dist->highestProbValue(exam);
  else
    return TValue(0.0);
}


TValue TTreeClassifier::operator()(const TExample &exam)
{
  checkProperty(descender);

  const bool convertEx = domain && (exam.domain != domain);
  TExample convex = convertEx ? TExample(domain, exam) : TExample();
  const TExample &refexam = convertEx ? convex : exam;
  PDiscDistribution branchWeights;
  PTreeNode node = descender->call(tree, refexam, branchWeights);
  if (!branchWeights) {
    if (node->nodeClassifier) 
      return node->nodeClassifier->call(refexam);
  }
  else {
    PDistribution decision = vote(node, refexam, branchWeights);
    if (decision)
      return decision->highestProbValue(exam);
  }

  // couldn't classify, so we'll return something a priori
  return findNodeValue(node, refexam);
}


PDistribution TTreeClassifier::classDistribution(const TExample &exam)
{
  checkProperty(descender);
  return classDistribution(tree, !domain || (exam.domain == domain) ? exam : TExample(domain, exam));
}


PDistribution TTreeClassifier::classDistribution(PTreeNode node, const TExample &exam)
{ PDiscDistribution branchWeights;
  node = descender->call(node, exam, branchWeights);

  if (!branchWeights) {
    if (node->nodeClassifier)
      return node->nodeClassifier->classDistribution(exam);
  }
  else 
    return vote(node, exam, branchWeights);

  return CLONE(TDistribution, findNodeDistribution(node, exam));
}


PDistribution TTreeClassifier::vote(PTreeNode node, const TExample &exam, PDiscDistribution branchWeights)
{
  PDistribution res = TDistribution::create(classVar);
  TDistribution &ures = res.getReference();
  TDiscDistribution::const_iterator bdi(branchWeights->begin()), bde(branchWeights->end());
  TTreeNodeList::const_iterator bi(node->branches->begin());
  for(; bdi!=bde; bdi++, bi++)
    if (*bdi && *bi) {
      PDistribution subDistr = classDistribution(*bi, exam);
      if (subDistr) {
        subDistr->normalize();
        subDistr->operator *= (*bdi);
        ures += subDistr;
      }
    }
  ures.normalize();
  return res;
}


void TTreeClassifier::predictionAndDistribution(const TExample &exam, TValue &val, PDistribution &distr)
{
  checkProperty(descender);
 
  TExample convex = (exam.domain != domain) ? TExample(domain, exam) : TExample();
  const TExample &refexam = (exam.domain != domain) ? convex : exam;
  PDiscDistribution splitDecision;
  PTreeNode node = descender->call(tree, refexam, splitDecision);
  if (!splitDecision) {
    if (node->nodeClassifier)
      node->nodeClassifier->predictionAndDistribution(refexam, val, distr);
    else
      distr = CLONE(TDistribution, findNodeDistribution(node, refexam));
  }
  else {
    distr = vote(node, refexam, splitDecision);
    val = distr->highestProbValue(exam);
  }
}



PTreeNode TTreeDescender_UnknownToNode::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ 
  while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    int nBranches = node->branches->size()-1;
    if (val.isSpecial() || (val.intV<0) || (val.intV>=nBranches) || !node->branches->at(val.intV))
      break;
    else
      node = node->branches->at(val.intV);
  }

  distr = PDiscDistribution();
  return node;
}


PTreeNode TTreeDescender_UnknownToBranch::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    int nBranches = node->branches->size()-1;
    if (val.isSpecial() || (val.intV<0) || (val.intV>=nBranches) || !node->branches->at(val.intV))
      node = node->branches->back();
    else
      node = node->branches->at(val.intV);
  }

  distr = PDiscDistribution();
  return node;
}


int randomNonNull(const PTreeNodeList &branches, const int &roff)
{ int nonull = 0;
  TTreeNodeList::const_iterator ni(branches->begin()), ne(branches->end());
  for (; ni!=ne; ni++)
    if (*ni)
      nonull++;
  
  if (!nonull)
    return -1;

  for(ni = branches->begin(), nonull = roff % (nonull+1); nonull; )
    if (*(ni++))
      nonull--;

  return (ni-1)-branches->begin();
}


PTreeNode TTreeDescender_UnknownToCommonBranch::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    int ind = val.isSpecial() ? -1 : val.intV;

    if ((ind<0) || (ind>=int(node->branches->size())))
      ind = node->branchSizes ? node->branchSizes->highestProbIntIndex(ex) : -1;

    if ((ind<0) || !node->branches->at(ind)) {
      ind = randomNonNull(node->branches, ex.sumValues());
      if (ind<0)
        break;
    }
    node = node->branches->at(ind);
  }

  distr = PDiscDistribution();
  return node;
}



PTreeNode TTreeDescender_UnknownToCommonSelector::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    int ind;
    if (val.isSpecial()) {
      TDiscDistribution *valdistr = val.svalV.AS(TDiscDistribution);
      ind = valdistr ? valdistr->highestProbIntIndex(ex) : -1;
    }
    else
      ind = val.intV<int(node->branches->size()) ? val.intV : -1;

    if ((ind<0) || !node->branches->at(ind)) {
      ind = randomNonNull(node->branches, ex.sumValues());
      if (ind<0)
        break;
    }

    node = node->branches->at(ind);
  }

  distr = PDiscDistribution();
  return node;
}



PTreeNode TTreeDescender_UnknownMergeAsBranchSizes::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    if (val.isSpecial() || (val.intV<0) || (val.intV>=int(node->branches->size())) || (!node->branches->at(val.intV))) {
      distr = node->branchSizes;
      return node;
    }
    else
      node = node->branches->at(val.intV);
  }

  distr = PDiscDistribution();
  return node;
}



PTreeNode TTreeDescender_UnknownMergeAsSelector::operator()(PTreeNode node, const TExample &ex, PDiscDistribution &distr)
{ while (node->branchSelector && node->branches) {
    TValue val = node->branchSelector->call(ex);
    if (val.isSpecial() || (val.intV<0) || (val.intV>=int(node->branches->size())) || (!node->branches->at(val.intV))) {
      if (val.svalV && val.svalV.is_derived_from(TDiscDistribution))
        distr = val.svalV;
      else
        distr = PDiscDistribution();
      return node;
    }
    else
      node = node->branches->at(val.intV);
  }

  distr = PDiscDistribution();
  return node;
}






PTreeNode TTreePruner_SameMajority::operator()(PTreeNode root)
{ vector<bool> tmp;
  return operator()(root, tmp);
}


/* Argument 'bestValues' gives values that are majority values for the subtree.
   While iterating through branches, an intersection is computed. Branches may
   return different sizes of 'bestValues'; the intersection is as long as the
   shortest of the reported bestValues. */

PTreeNode TTreePruner_SameMajority::operator()(PTreeNode node, vector<bool> &bestValues)
{ 
  PTreeNode newNode = CLONE(TTreeNode, node);

  if (node->branchSelector) {
    newNode->branches = mlnew TTreeNodeList();
    int notfirst = 0;
    PITERATE(TTreeNodeList, bi, node->branches)
      if (*bi) {
        if (notfirst++) {
          vector<bool> subBest;
          newNode->branches->push_back(operator()(*bi, subBest));

          if (subBest.size() < bestValues.size())
            bestValues.erase(bestValues.begin() + subBest.size(), bestValues.end());
          for(vector<bool>::iterator bvi(bestValues.begin()), bve(bestValues.end()), sbi(subBest.begin());
              bvi!=bve; bvi++, sbi++)
            *bvi = *bvi && *sbi;
        }
        else
          newNode->branches->push_back(operator()(*bi, bestValues));
      }
      else
        newNode->branches->push_back(PTreeNode());

    vector<bool>::iterator pi(bestValues.begin());
    for( ; (pi!=bestValues.end()) && !*pi; pi++);
    if (pi!=bestValues.end()) {
      newNode->branches = PTreeNodeList();
      newNode->branchDescriptions = PStringList();
      newNode->branchSelector = PClassifier();
      newNode->branchSizes = PDiscDistribution();
    }
  }

  else {
    TDefaultClassifier *maj = node->nodeClassifier.AS(TDefaultClassifier);
    if (maj) {
      TDiscDistribution *ddist = maj->defaultDistribution.AS(TDiscDistribution);
      if (ddist) {
        float bestF = -1;
        TDiscDistribution::const_iterator bi(ddist->begin()), bb=bi, be(ddist->end());
        for(; bi!=be; bi++)
          if (*bi>=bestF) {
            bb = bi;
            bestF = *bb;
          }
            
        // The loop runs to one before the last; the last is always true
        for(bi = ddist->begin(); bi!=bb; bi++)
          bestValues.push_back(*bi==bestF);
        bestValues.push_back(true);
      }
    }
  }

  return newNode;
}

  
TTreePruner_m::TTreePruner_m(const float &am)
: m(am)
{}

PTreeNode TTreePruner_m::operator()(PTreeNode root)
{ if (m<0.0)
    raiseError("'m' should be positive");
  
  PDistribution dist;
  if (root->distribution)
    dist = root->distribution;
  else if (root->contingency && root->contingency->classes)
    dist = root->contingency->classes;
  else
    raiseError("the node does not store class distribution (check your flags for TreeLearner)");


  TDiscDistribution *ddist = dist.AS(TDiscDistribution);
  if (ddist) {
    vector<float> m_by_p;
    const float mba = m/ddist->abs;
    PITERATE(TDiscDistribution, di, ddist)
      m_by_p.push_back(*di*mba);

    PTreeNode prunned;
    operator()(root, m_by_p, prunned);  
    return prunned;
  }

  TContDistribution *cdist = dist.AS(TContDistribution);
  if (cdist) {
    PTreeNode prunned;
    operator()(root, cdist->error() * m, prunned);
    return prunned;
  }

  raiseError("class distribution of unknown type (neither discrete nor continuous)");
  return PTreeNode();
}


float TTreePruner_m::estimateError(const PTreeNode &node, const vector<float> &m_by_p) const
{ 
  const TDiscDistribution *dist;
  if (node->distribution)
    dist = node->distribution.AS(TDiscDistribution);
  else if (node->contingency)
    dist = node->contingency->classes.AS(TDiscDistribution);
  else
    raiseError("the node does not store class distribution (check your flags for TreeLearner)");

  if (!dist)
    raiseError("invalid class distribution (DiscDistribution expected)");

  if ((dist->abs < 1e-10) || (dist->abs+m < 1e-10))
    return 0.0;

  float maxe = 0.0;
  vector<float>::const_iterator mi(m_by_p.begin());
  for(TDiscDistribution::const_iterator di(dist->begin()), de(dist->end()); di!=de; di++, mi++) {
    float thise = *di + *mi;
    if (thise>maxe)
      maxe = thise;
  }

  return 1.0 - maxe/(dist->abs+m);
}


float TTreePruner_m::estimateError(const PTreeNode &node, const float &m_by_se) const
{ const TContDistribution *dist;
  if (node->distribution)
    dist = node->distribution.AS(TContDistribution);
  else if (node->contingency)
    dist = node->contingency->classes.AS(TContDistribution);
  else
    raiseError("the node does not store class distribution (check your flags for TreeLearner)");
  if (!dist)
    raiseError("invalid class distribution (ContDistribution expected)");

  if ((dist->abs==0.0) || (dist->abs+m==0.0))
    return 0.0;

  return (dist->abs*dist->error() + m_by_se) / (dist->abs+m);
}

#ifdef _MSC_VER
#pragma optimize("p", off)
#endif

template<class T>
float TTreePruner_m::operator()(PTreeNode node, const T &m_by_p, PTreeNode &newNode) const
{ 
  newNode = CLONE(TTreeNode, node);

  if (node->branchSelector) {
    newNode->branches = mlnew TTreeNodeList(node->branches->size());

    float sumerr = 0, sumweights = 0;
    TDiscDistribution::const_iterator bwi (node->branchSizes->begin());
    TTreeNodeList::iterator oi (node->branches->begin()), oe (node->branches->end());
    TTreeNodeList::iterator bi (newNode->branches->begin());
    for (; oi!=oe; oi++, bi++, bwi++)
      if (*oi) {
        sumerr += *bwi * operator()(*oi, m_by_p, *bi);
        sumweights += *bwi;
      }

    const float staticError = estimateError(node, m_by_p);
    const float backupError = sumerr/sumweights;

    if (staticError<backupError) {
      newNode->branches = PTreeNodeList();
      newNode->branchDescriptions = PStringList();
      newNode->branchSelector = PClassifier();
      newNode->branchSizes = PDiscDistribution();
      return staticError;
    }
    else
      return backupError;
  }

  else
    return estimateError(node, m_by_p);
}

#ifdef _MSC_VER
#pragma optimize("p", on)
#endif
