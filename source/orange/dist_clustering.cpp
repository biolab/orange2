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

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "learn.hpp"
#include "classify.hpp"
#include "table.hpp"
#include "measures.hpp"
#include "distance.hpp"

#include "dist_clustering.ppp"

TDistClusterNode::TDistClusterNode(PDistribution adis, const PExample &example, const float &quality, TDistClusterNode *aprevNode)
: nextNode(NULL),
  prevNode(aprevNode),
  distribution(adis),
  mergeProfits(NULL),
  cluster(mlnew TExampleCluster(example)),
  distributionQuality_N(quality)
{}


TDistClusterNode::~TDistClusterNode()
{ mldelete nextNode; }



TDistProfitNode::TDistProfitNode(TDistClusterNode *c1, TDistClusterNode *c2, const float &prof, const int &qind, const long &roff)
: cluster1(c1),
  cluster2(c2),
  profit(prof),
  queueIndex(qind),
  randoff(roff)
{}


TDistProfitNode::~TDistProfitNode()
{ mldelete it1;
  mldelete it2;
}


int TDistProfitNode::compare(const TDistProfitNode &other) const
{ if (profit<other.profit)
    return -1;
  else if (profit>other.profit)
    return 1;
  else if (randoff<other.randoff)
    return -1;
  else if (randoff>other.randoff)
    return 1;
  return 0;
}



T_ExampleDist::T_ExampleDist(PExample anexample, PDistribution adist)
: example(anexample),
  distribution(adist)
{}


int TExampleDistVector::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  const_ITERATE(vector<T_ExampleDist>, pi, values) {
    PVISIT((*pi).example);
    PVISIT((*pi).distribution);
  }
  return 0;
}


int TExampleDistVector::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);
  values.clear();
  return 0;
}


PExampleDistVector TExampleDistBySorting::operator()(PExampleGenerator gen, TVarList &aboundSet, const int &weightID)
{
  // Identify bound attributes
  vector<int> bound;
  { ITERATE(TVarList, evi, aboundSet) {
      if ((*evi)->varType != TValue::INTVAR)
        raiseError("attribute '%s' is not discrete", (*evi)->name.c_str());
      bound.push_back(gen->domain->getVarNum(*evi));
    }
  }

  PDomain boundDomain = mlnew TDomain(PVariable(), aboundSet);

  TExampleTable sorted(gen, false);
  sorted.sort(bound);

  PExampleDistVector edv = mlnew TExampleDistVector();

  TDistribution *insertPoint = NULL;
  for(TExampleIterator ebegin(sorted.begin()), eend(sorted.end()), eprev(ebegin); ebegin!=eend; eprev=ebegin, ++ebegin) {
    if (insertPoint) { // we have it, but should still check for equality of bound attributes
      vector<int>::iterator bi(bound.begin()), be(bound.end());
      for( ; (bi!=be) && ((*ebegin)[*bi]==(*eprev)[*bi]); bi++);
      if (bi!=be)
        insertPoint=NULL;
    }

    if (!insertPoint) {
      edv->values.push_back(T_ExampleDist(PExample(mlnew TExample(boundDomain, *ebegin)), TDistribution::create(gen->domain->classVar)));
      insertPoint = const_cast<TDistribution *>(edv->values.back().distribution.getUnwrappedPtr());
    }
    insertPoint->add((*ebegin).getClass(), WEIGHT(*ebegin));
  }

  return edv;
}




TClustersFromDistributionsByAssessor::TClustersFromDistributionsByAssessor(float mpp, PDistributionAssessor acola)
: distributionAssessor(acola),
  minProfitProportion(mpp)
{}



void TClustersFromDistributionsByAssessor::computeQualities(TDistClusterNode *&clusters, TDistProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
// Computes errors and merge profits
{ profitQueue = TDistProfitQueue();
  baseQuality = 0.0;
  for(TDistClusterNode *cl1 = clusters; cl1; cl1=cl1->nextNode) {
    cl1->distributionQuality_N = distributionAssessor->distributionQuality(*cl1);
	  baseQuality += cl1->distributionQuality_N;
	  for(TDistClusterNode *cl2 = clusters; cl2!=cl1; cl2=cl2->nextNode) {
  	  float profit = distributionAssessor->mergeProfit(*cl1, *cl2);
      insertProfitQueueNode(cl2, cl1, profit, rgen.randsemilong(), profitQueue);
	  }
  }
}


TDistributionAssessor_Kramer defaultDistributionAssessor;

PExampleClusters TClustersFromDistributionsByAssessor::operator()(PExampleDistVector edv)
{
  bool defaultAssessorUsed = !distributionAssessor;
  if (defaultAssessorUsed)
    distributionAssessor = PDistributionAssessor(defaultDistributionAssessor);

  vector<PExampleCluster> group;

  float baseQuality;
  float N;
  TDistClusterNode *clusters = NULL;

  int nex = 0;
  ITERATE(vector<T_ExampleDist>, edvi, edv->values)
    nex += (*edvi).distribution->cases;

  TSimpleRandomGenerator rgen;

  try {
    TDistProfitQueue profitQueue;
    preparePrivateVars(edv, clusters, profitQueue, baseQuality, N, rgen);

    while(profitQueue.size() && (!stopCriterion || !stopCriterion->operator()(baseQuality, profitQueue, clusters)))
      mergeBestColumns(clusters, profitQueue, baseQuality, N, rgen);

    for(TDistClusterNode *cli = clusters; cli; cli = cli->nextNode)
      group.push_back(cli->cluster);
  }
  catch (...) {
    if (defaultAssessorUsed)
      distributionAssessor = PDistributionAssessor();
    mldelete clusters;
    clusters = NULL;
    throw;
  }

  mldelete clusters;
  clusters = NULL;
  if (defaultAssessorUsed)
    distributionAssessor = PDistributionAssessor();

  return mlnew TExampleClusters(PExampleCluster(mlnew TExampleCluster(group, numeric_limits<float>::infinity())), baseQuality);
}
  


void TClustersFromDistributionsByAssessor::preparePrivateVars(PExampleDistVector values, TDistClusterNode *&clusters, TDistProfitQueue &priorityQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{

  vector<T_ExampleDist>::iterator cli(values->values.begin()), cle(values->values.end());
  if (cli==cle)
    raiseError("empty 'ExampleDistVector'; no examples?!");

  TDistClusterNode *prevIns=clusters=mlnew TDistClusterNode((*cli).distribution, (*cli).example, 0.0, NULL);
  /* There was a bug here: the following line used to be
       PDistribution classDist=(*cli).distribution;
  */
  PDistribution classDist = CLONE(TDistribution, (*cli).distribution);
  while (++cli!=cle) {
    prevIns->nextNode=mlnew TDistClusterNode((*cli).distribution, (*cli).example, 0.0, prevIns);
    prevIns=prevIns->nextNode;
    classDist->operator += ((*cli).distribution.getReference());
  }
    
  N = classDist->abs;
  if (classDist->variable->varType==TValue::INTVAR)
    distributionAssessor->setDistribution(CAST_TO_DISCDISTRIBUTION(classDist));
  else
    distributionAssessor->setAverage(CAST_TO_CONTDISTRIBUTION(classDist).average());

  computeQualities(clusters, priorityQueue, baseQuality, N, rgen);
  baseQuality /= N;
}




TDistProfitNode *TClustersFromDistributionsByAssessor::insertProfitQueueNode(TDistClusterNode *cl1, TDistClusterNode *cl2, float profit, long roff, TDistProfitQueue &profitQueue)
{ TDistProfitNode *newNode = mlnew TDistProfitNode(cl1, cl2, profit, profitQueue.size(), roff);
  profitQueue.insert(newNode);
  newNode->it1 = mlnew TDistProfitNodeList(newNode, &cl1->mergeProfits);
  newNode->it2 = mlnew TDistProfitNodeList(newNode, &cl2->mergeProfits);
  return newNode;
}




void TClustersFromDistributionsByAssessor::mergeBestColumns(TDistClusterNode *&clusters, TDistProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{
  TDistClusterNode *cl1 = profitQueue.front()->cluster1, *cl2 = profitQueue.front()->cluster2;
  const float &profitN = profitQueue.front()->profit;

  // merge the columns and update qualities
  cl1->cluster = mlnew TExampleCluster(cl1->cluster, cl2->cluster, -profitN/N);
  cl1->distribution ->operator+= (cl2->distribution);
  cl1->distributionQuality_N += cl2->distributionQuality_N - profitN;
  baseQuality += profitN/N;

  // delete cl2 from list of clusters
  if (cl2->nextNode)
    cl2->nextNode->prevNode = cl2->prevNode;

  if (cl2->prevNode)
    cl2->prevNode->nextNode = cl2->nextNode;
  else
    clusters = cl2->nextNode;

  cl2->prevNode = cl2->nextNode = NULL;

  // remove profits from the queue; the joint is removed in cl1.
  { TDistProfitNodeList &first = cl1->mergeProfits;
    while(first.next)
      profitQueue.remove(first.next->node->queueIndex);
  }
  { TDistProfitNodeList &first = cl2->mergeProfits;
    while(first.next)
      profitQueue.remove(first.next->node->queueIndex);
  }

  // update the column error and the profits
  { for(TDistClusterNode *cn1 = clusters; cn1; cn1 = cn1->nextNode) 
      if (cn1!=cl1)
        insertProfitQueueNode(cl1, cn1, distributionAssessor->mergeProfit(*cn1, *cl1), rgen.randsemilong(), profitQueue);
  }

  delete cl2;
}



TFeatureByDistributions::TFeatureByDistributions(PClustersFromDistributions cfd, const int &comp)
: clustersFromDistributions(cfd),
  completion(comp)
{}


TExampleDistBySorting defaultEDC;
TClustersFromDistributionsByAssessor defaultCFD;

PVariable TFeatureByDistributions::operator()(PExampleGenerator egen, TVarList &boundSet, const string &name, float &quality, const int &weight)
{
  PExampleDistVector edv = defaultEDC(egen, boundSet, weight);
  if (!edv->values.size())
    return PVariable();

  PExampleClusters clusters = clustersFromDistributions ? clustersFromDistributions->call(edv) : defaultCFD(edv);
  PVariable feat = clusters->feature(float(1e30), completion);
  if (!feat)
    return PVariable();

  quality = clusters->quality;
  feat->name = name;

  return feat;
}



TDistributionAssessor::TDistributionAssessor()
{}


void TDistributionAssessor::setDistribution(const TDiscDistribution &)
{}


void TDistributionAssessor::setAverage(const float &)
{}


void TDistributionAssessor_m::setDistribution(const TDiscDistribution &classDist)
{ p_by_m = vector<float>();
  float N = classDist.abs;
  const_ITERATE(TDiscDistribution, ci, classDist)
    p_by_m.push_back(*ci/N*m);
}


TDistributionAssessor_m::TDistributionAssessor_m(const float &am)
: m(am)
{}


float TDistributionAssessor_m::m_error(const TDiscDistribution &val) const
// returns m estimate for error, multiplied by number of examples
{
  float bestok=-1, thisok;
  vector<float>::const_iterator pci(p_by_m.begin());
  const_ITERATE(TDiscDistribution, dvi, val)
	if ((thisok = (*dvi + *(pci++))) > bestok)
    bestok = thisok;
 
  return val.abs *  (1 - bestok) / (val.abs + m);
}


float TDistributionAssessor_m::m_error(const TDiscDistribution &val1, const TDiscDistribution &val2) const
// returns m estimate for error, summing the given distributions, multiplied by
//   the joint number of examples
{
  float bestok=-1, thisok;
  float N=val1.abs+val2.abs;
  vector<float>::const_iterator pci(p_by_m.begin());
  for(TDiscDistribution::const_iterator dvi1(val1.begin()), dve1(val1.end()), dvi2(val2.begin());
	  dvi1!=dve1; dvi1++, dvi2++)
	if ((thisok = (*dvi1+*dvi2+ *(pci++)) / (N + m))>bestok) bestok=thisok;
 
  return N *  (1 - bestok);
}


float TDistributionAssessor_m::distributionQuality(TDistClusterNode &node) const
{ return -m_error(CAST_TO_DISCDISTRIBUTION(node.distribution)); }


/* The profit is not divided by the total number of examples so that the function returns the
   profit multiplied by the number of examples in the merged column. */
float TDistributionAssessor_m::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{ return - m_error(CAST_TO_DISCDISTRIBUTION(clust1.distribution), CAST_TO_DISCDISTRIBUTION(clust2.distribution))
         + (clust1.distributionQuality_N + clust2.distributionQuality_N);
}



float TDistributionAssessor_Relief::distributionQuality(TDistClusterNode &node) const
{ const TDiscDistribution &dist=CAST_TO_DISCDISTRIBUTION(node.distribution);
  float sum=0.0;
  const_ITERATE(TDiscDistribution, di, dist)
    sum += *di * *di;
  return 2*sum-dist.abs*dist.abs;
}


float TDistributionAssessor_Relief::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{
  const TDiscDistribution &dist1= CAST_TO_DISCDISTRIBUTION(clust1.distribution), 
                          &dist2= CAST_TO_DISCDISTRIBUTION(clust2.distribution);
  float profit=0.0;
  for(TDiscDistribution::const_iterator v1i(dist1.begin()), v1e(dist1.end()), v2i(dist2.begin()), v2e(dist2.end());
      (v1i!=v1e) && (v2i!=v2e);
      profit += 4 * *(v1i++) * *(v2i++));
  profit -= 2 * dist1.abs * dist2.abs;
  return profit;
}


float TDistributionAssessor_mf::m_error(const float &sum, const float &sum2, const float &N) const
{ float df = sum+m*aprior;
  float N_m = N+m;
  return N/N_m * (sum2 + m*aprior*aprior - df*df/N_m);
}


TDistributionAssessor_mf::TDistributionAssessor_mf(const float &am)
: m(am)
{}


void TDistributionAssessor_mf::setAverage(const float &avg)
{ aprior = avg; }


float TDistributionAssessor_mf::distributionQuality(TDistClusterNode &node) const
{ return -m_error(node.distribution.AS(TContDistribution)->sum, 
                  node.distribution.AS(TContDistribution)->sum2,
                  node.distribution->abs); }


/* The profit is not divided by the total number of examples so that the function returns the
   profit multiplied by the number of examples in the merged column. */
float TDistributionAssessor_mf::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{
  return   clust1.distributionQuality_N + clust2.distributionQuality_N 
         - m_error(CAST_TO_CONTDISTRIBUTION(clust1.distribution).sum+CAST_TO_CONTDISTRIBUTION(clust2.distribution).sum,
                   CAST_TO_CONTDISTRIBUTION(clust1.distribution).sum2+CAST_TO_CONTDISTRIBUTION(clust2.distribution).sum2,
                   clust1.distribution->abs+clust2.distribution->abs);
}



TDistributionAssessor_Measure::TDistributionAssessor_Measure(PMeasureAttribute meas)
: measure(meas)
{}


float TDistributionAssessor_Measure::distributionQuality(TDistClusterNode &node) const
{ return measure->operator()(node.distribution); }


float TDistributionAssessor_Measure::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{ if (clust1.distribution->variable->varType==TValue::INTVAR) {
    TDiscDistribution nd (CAST_TO_DISCDISTRIBUTION(clust1.distribution));
    nd += clust2.distribution;
    return measure->operator()(nd);
  }
  else 
    raiseError("merging of continuous attributes not implemented");
  return 0.0;
}



float TDistributionAssessor_Laplace::distributionQuality(TDistClusterNode &node) const
{ const TDiscDistribution &dist=CAST_TO_DISCDISTRIBUTION(node.distribution);
  const float N = dist.abs;
  const float Nc = dist.size();
  const float error = dist.size() ? (1 - (dist.highestProb()+1)/(N+Nc)) : 0.0;
  return - N * error;
}


float TDistributionAssessor_Laplace::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{
  const TDiscDistribution &dist1=CAST_TO_DISCDISTRIBUTION(clust1.distribution),
                          &dist2=CAST_TO_DISCDISTRIBUTION(clust2.distribution);
  if (!dist1.size() && !dist2.size())
    return 0.0;

  float maxC = 0.0;
  for(TDiscDistribution::const_iterator di1(dist1.begin()), di2(dist2.begin()), de1(dist1.end()), de2(dist2.end());
      (di1!=de1) && (di2!=de2);
      di1++, di2++)
    if (*di1+*di2 > maxC)
      maxC = *di1+*di2;
  
  const float Nc = (dist1.size()>dist2.size()) ? dist1.size() : dist2.size();
  const float N = dist1.abs + dist2.abs;
  const float newError = 1 - (maxC+1)/(Nc+N);
  const float newQuality = -newError;
  return N * newQuality - (clust1.distributionQuality_N + clust2.distributionQuality_N);
}



float TDistributionAssessor_Kramer::distributionQuality(TDistClusterNode &node) const
{ const TDiscDistribution &dist=CAST_TO_DISCDISTRIBUTION(node.distribution);
  if (dist.size()>2)
    raiseError("binary class expected");
  return (dist.size()==2) ? - dist[0]*dist[1]/(dist[0]+dist[1]) : 0.0;
}


float TDistributionAssessor_Kramer::mergeProfit(const TDistClusterNode &clust1, const TDistClusterNode &clust2) const
{
  const TDiscDistribution &dist1=CAST_TO_DISCDISTRIBUTION(clust1.distribution),
                          &dist2=CAST_TO_DISCDISTRIBUTION(clust2.distribution);
  const float &p1=dist1.front();
  const float &n1=dist1.back();
  const float &p2=dist2.front();
  const float &n2=dist2.back();
  return -(p1+p2)*(n1+n2) / (p1+p2+n1+n2) - (clust1.distributionQuality_N + clust2.distributionQuality_N);
}



TStopDistributionClustering_noProfit::TStopDistributionClustering_noProfit(const float &minprof)
: minProfitProportion(minprof)
{}


bool TStopDistributionClustering_noProfit::operator()(const float &baseQuality, const TDistProfitQueue &pq, const TDistClusterNode *) const
{ return (pq.front()->profit < 0) || (pq.front()->profit<baseQuality*minProfitProportion); };


bool TStopDistributionClustering_noBigChange::operator()(const float &, const TDistProfitQueue &profitQueue, const TDistClusterNode *) const
{ int pN = profitQueue.size();
  if (pN>1) {
    float sum = 0.0;
    int i = 0;
    while(i<pN/2)
      sum += profitQueue[i++]->profit;
    if (pN%2)
      i++;
    while(i<pN)
      sum -= profitQueue[i++]->profit;
    sum /= (pN-pN%2);

    if (profitQueue.front()->profit < -sum)
      return true;
  }
  
  else if (profitQueue.front()->profit < 0)
    return true;

  return false;
}


bool TStopDistributionClustering_binary::operator()(const float &, const TDistProfitQueue &, const TDistClusterNode *clusters) const
{ return (!clusters || !clusters->nextNode || !clusters->nextNode->nextNode); }


TStopDistributionClustering_n::TStopDistributionClustering_n(const int &an)
: n(an)
{}

bool TStopDistributionClustering_n::operator()(const float &, const TDistProfitQueue &, const TDistClusterNode *clusters) const
{ TDistClusterNode const *cn = clusters;
  for (int i = n; cn && i; i--, cn = cn->nextNode);
  return !cn;
}
