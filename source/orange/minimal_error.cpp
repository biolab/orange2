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


#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "lookup.hpp"
#include "measures.hpp"
#include "distance.hpp"

#ifdef _NOISE_DEBUG
#include <iomanip>
#endif

#include "minimal_error.ppp"

#include "im_col_assess.cpp"



/*TProfitNodeList::TProfitNodeList(TProfitNode *anode, TProfitNodeList *aprev)
 : node(anode), prev(aprev), next(aprev ? aprev->next : NULL)
 { if (prev) prev->next=this;
   if (next) next->prev=this; }


TProfitNodeList::~TProfitNodeList()
{ if (prev) prev->next=next;
  if (next) next->prev=prev;
}
*/

TIMClusterNode::TIMClusterNode(TIMColumnNode *acolumn, const PExample &example, const float &quality, TIMClusterNode *aprevNode)
: nextNode(NULL),
  prevNode(aprevNode),
  mergeProfits(),
  column(acolumn),
  cluster(mlnew TExampleCluster(example)),
  columnQuality_N(quality)
{}


TIMClusterNode::~TIMClusterNode()
{ mldelete column;
  mldelete nextNode;
}



TIMClustering::TIMClustering(PIM anim)
: im(anim),
  clusters(mlnew TIntList(anim ? anim->columns.size() : 0, -1)),
  maxCluster(-1),
  quality(numeric_limits<float>::quiet_NaN())
{}




TAssessIMQuality::TAssessIMQuality(PColumnAssessor ca)
: columnAssessor(ca)
{}


float TAssessIMQuality::operator()(PIM pim)
{ checkProperty(columnAssessor);

  float abs = 0.0;
 
  if (dynamic_cast<TDIMColumnNode *>(pim->columns.front().column)) {
    TDiscDistribution classDist;
    ITERATE(vector<T_ExampleIMColumnNode>, ci, pim->columns)
      for(TIMColumnNode *colNode=(*ci).column; colNode; colNode=colNode->next) {
        TDIMColumnNode *cnode = dynamic_cast<TDIMColumnNode *>(colNode);
        classDist += TDiscDistribution(cnode->distribution, cnode->noOfValues);
      }
    columnAssessor->setDistribution(classDist);
    abs = classDist.abs;
  }
  else {
    float sum = 0.0;
    ITERATE(vector<T_ExampleIMColumnNode>, ci, pim->columns)
      for(TFIMColumnNode *colNode = dynamic_cast<TFIMColumnNode *>((*ci).column);
          colNode;
          colNode = dynamic_cast<TFIMColumnNode *>(colNode->next)) {
        sum += (*colNode).sum;
        abs += (*colNode).N;
    }
    if (!abs)
      raiseError("empty partition matrix");
    columnAssessor->setAverage(sum/abs);
  }

  float quality = 0.0;
  ITERATE(vector<T_ExampleIMColumnNode>, ci, pim->columns)
    quality += ((*ci).column->nodeQuality=columnAssessor->columnQuality((*ci).column));

  return quality/abs;
}


TClustersFromIMByAssessor::TClustersFromIMByAssessor(PColumnAssessor acola)
: columnAssessor(acola)
{}


void TClustersFromIMByAssessor::computeQualities(TIMClusterNode *clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
// Computes errors and merge profits
{ rgen.seed = int(N);

  baseQuality=0;
  for(TIMClusterNode *cl1=clusters; cl1; cl1=cl1->nextNode) {
    cl1->columnQuality_N=columnAssessor->columnQuality(cl1->column);
	  baseQuality+=cl1->columnQuality_N;
	  for(TIMClusterNode *cl2=clusters; cl2!=cl1; cl2=cl2->nextNode) {
  	  float profit=columnAssessor->mergeProfit(cl1->column, cl2->column);
      insertProfitQueueNode(cl2, cl1, profit, rgen.randsemilong(), profitQueue);
	  }
  }
}



TColumnAssessor_m defaultColumnAssessor;

PExampleClusters TClustersFromIMByAssessor::operator()(PIM pim)
{
  bool defaultAssessorUsed = !columnAssessor;
  if (defaultAssessorUsed)
    columnAssessor = PColumnAssessor(defaultColumnAssessor);

  TIMClusterNode *clusters = NULL;
  float baseQuality, N, initialQuality;

  TSimpleRandomGenerator rgen; // seed will be set later, when N is known (in computeQualities)

  try {
    TProfitQueue profitQueue;
    preparePrivateVars(pim, clusters, profitQueue, baseQuality, N, rgen);
    initialQuality = baseQuality;

    while(profitQueue.size() && (!stopCriterion || !stopCriterion->operator()(baseQuality, profitQueue, clusters)))
      mergeBestColumns(clusters, profitQueue, baseQuality, N, rgen);
  }
  catch (...) {
    if (defaultAssessorUsed)
      columnAssessor = PColumnAssessor();
    mldelete clusters;
    throw;
  }

  if (defaultAssessorUsed)
    columnAssessor = PColumnAssessor();

  vector<PExampleCluster> group;
  for(TIMClusterNode *cli=clusters; cli; cli=cli->nextNode)
    group.push_back(cli->cluster);

  mldelete clusters;

  return mlnew TExampleClusters(PExampleCluster(mlnew TExampleCluster(group, numeric_limits<float>::infinity())), baseQuality - initialQuality);
}
  

void TClustersFromIMByAssessor::preparePrivateVars(PIM pim, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{ if (pim->varType==TValue::INTVAR)
    preparePrivateVarsD(pim, clusters, profitQueue, baseQuality, N, rgen);
  else
    preparePrivateVarsF(pim, clusters, profitQueue, baseQuality, N, rgen);
}


void TClustersFromIMByAssessor::preparePrivateVarsD(PIM pim, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{
  // Random generator is not initialized yet, so you shouldn't use it in this code
  // (initialization comes in computeQualities since N is known then)

  TDiscDistribution classDist;
  clusters = NULL;

  // Creating clusters and linking them; computing class distribution.
  // Column errors are estimated later, when class distributions are known
  TIMClusterNode **clusterInsert = &clusters, *prevIns = NULL;
  ITERATE(vector<T_ExampleIMColumnNode>, cli, pim->columns) {
    prevIns = (*clusterInsert) = mlnew TIMClusterNode((*cli).column, (*cli).example, 0.0, prevIns);
    (*cli).column = NULL;
	  clusterInsert = &((*clusterInsert)->nextNode);
    for(TIMColumnNode *ci = prevIns->column; ci; ci = ci->next) {
      TDIMColumnNode *cnode = dynamic_cast<TDIMColumnNode *>(ci);
      classDist += TDiscDistribution(cnode->distribution, cnode->noOfValues);
    }
  }

  N = classDist.abs;
  columnAssessor->setDistribution(classDist);
  computeQualities(clusters, profitQueue, baseQuality, N, rgen);
  baseQuality /= N;
}


void TClustersFromIMByAssessor::preparePrivateVarsF(PIM pim, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{
  // Random generator is not initialized yet, so you shouldn't use it in this code
  // (initialization comes below, as soon as N is computed)

  float sum = 0;
  N = 0;
  clusters = NULL;

  // Creating clusters and linking them; computing class distribution.
  // Column errors are estimated later, when class distributions are known
  TIMClusterNode **clusterInsert = &clusters, *prevIns = (TIMClusterNode *)NULL;
  ITERATE(vector<T_ExampleIMColumnNode>, cli, pim->columns) {
    prevIns = (*clusterInsert) = mlnew TIMClusterNode((*cli).column, (*cli).example, 0.0, prevIns);
    (*cli).column = (TFIMColumnNode *)NULL;
	  clusterInsert = &((*clusterInsert)->nextNode);
	  for(TFIMColumnNode *ci=dynamic_cast<TFIMColumnNode *>(prevIns->column);
        ci;
        ci=dynamic_cast<TFIMColumnNode *>(ci->next)) {
      sum += (*ci).sum;
      N += (*ci).N;
    }
 	}

  columnAssessor->setAverage(sum/N);
  computeQualities(clusters, profitQueue, baseQuality, N, rgen);
  baseQuality /= N;
}



TProfitNode *TClustersFromIMByAssessor::insertProfitQueueNode(TIMClusterNode *cl1, TIMClusterNode *cl2, float profit, long randoff, TProfitQueue &profitQueue)
{ TProfitNode *newNode = mlnew TProfitNode(cl1, cl2, profit, profitQueue.size(), randoff);
  profitQueue.insert(newNode);
  newNode->it1 = mlnew TProfitNodeList(newNode, &cl1->mergeProfits);
  newNode->it2 = mlnew TProfitNodeList(newNode, &cl2->mergeProfits);
  return newNode;
}



void TClustersFromIMByAssessor::mergeBestColumns(TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &rgen)
{
  TIMClusterNode *cl1 = profitQueue.front()->column1, *cl2 = profitQueue.front()->column2;
  const float &profitN = profitQueue.front()->profit;

  cl1->cluster = mlnew TExampleCluster(cl1->cluster, cl2->cluster, -profitN/N);
  // merge the columns and update the error
  
  { TIMColumnNode **cn1 = &(cl1->column);
    for(; *cn1 && cl2->column; ) {
      TIMColumnNode **cn2 = &(cl2->column);
      for( ; *cn2 && (*cn2)->index<(*cn1)->index; cn2=&((*cn2)->next));
	  // if not empty, move the lower run [cl2->column, *cn2) to the first, before *cn1
	  if (cn2!=&(cl2->column)) {
	    TIMColumnNode *nc2=*cn1;
		*cn1 = cl2->column;
		cl2->column = *cn2;
		*cn2 = nc2;
		cn1 = cn2;
	  }
	  // join *cn1 and cl2->column, if same index
	  if (cl2->column && ((*cn1)->index==cl2->column->index)) {
      **cn1 += *cl2->column;
      (*cn1)->nodeQuality = columnAssessor->nodeQuality(**cn1);
      TIMColumnNode *n2 = cl2->column;
      cl2->column = cl2->column->next;
      n2->next = NULL;
      mldelete n2;
	  }
	  if (cl2->column)
      while(*cn1 && ((*cn1)->index<cl2->column->index))
        cn1 = &((*cn1)->next);
	  }

	  // merge the second tail, if not empty
	  if(cl2->column) {
  	  *cn1 = cl2->column;
	    cl2->column=NULL;
	  }

    cl1->columnQuality_N += cl2->columnQuality_N - profitN;
    baseQuality += profitN/N;
  }

  // delete cl2 from list of clusters
  if (cl2->nextNode)
    cl2->nextNode->prevNode = cl2->prevNode;

  if (cl2->prevNode)
    cl2->prevNode->nextNode = cl2->nextNode;
  else
    clusters = cl2->nextNode;

  cl2->prevNode = cl2->nextNode = NULL;

  // remove profits from the queue; the joint is removed in cl1.
  { TProfitNodeList &first = cl1->mergeProfits;
    while(first.next)
      profitQueue.remove(first.next->node->queueIndex);
  }
  { TProfitNodeList &first = cl2->mergeProfits;
    while(first.next)
      profitQueue.remove(first.next->node->queueIndex);
  }
  
  mldelete cl2;

  // update the column error and the profits
  { for(TIMClusterNode *cn1 = clusters; cn1; cn1 = cn1->nextNode) 
      if (cn1!=cl1) {
        float profit = columnAssessor->mergeProfit(cn1->column, cl1->column);
        insertProfitQueueNode(cl1, cn1, profit, rgen.randsemilong(), profitQueue);
	  }
  }
}



TProfitNode::TProfitNode(TIMClusterNode *c1, TIMClusterNode *c2, float prof, int qind, const long &roff)
: column1(c1),
  column2(c2),
  profit(prof),
  queueIndex(qind),
  randoff(roff)
{}


TProfitNode::~TProfitNode()
{ mldelete it1;
  mldelete it2;
}


int TProfitNode::compare(const TProfitNode &other) const
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



TFeatureByIM::TFeatureByIM(PIMConstructor cim, PClustersFromIM cfim, const int &comp)
: IMconstructor(cim),
  clustersFromIM(cfim),
  completion(comp)
{}



TIMBySorting defaultIMConstructor;
TClustersFromIMByAssessor defaultIMClusters;

PVariable TFeatureByIM::operator()(PExampleGenerator egen, TVarList &boundSet, const string &name, float &quality, const int &weight)
{
  PIM im = IMconstructor ? IMconstructor->operator()(egen, boundSet, weight) : ((TIMConstructor &)defaultIMConstructor)(egen, boundSet, weight);
  if (!im)
    return PVariable();

  PExampleClusters clusters = clustersFromIM ? clustersFromIM->call(im) : ((TClustersFromIM &)defaultIMClusters)(im);
  PVariable feat =  clusters->feature(float(1e30), completion);
  if (!feat)
    return PVariable();

  quality = clusters->quality;
  feat->name=name;

  return feat;
}



TMeasureAttribute_IM::TMeasureAttribute_IM()
: TMeasureAttribute(TMeasureAttribute::Generator, true, false)
{}


float TMeasureAttribute_IM::operator()(int attrNo, PExampleGenerator egen, PDistribution apriorClass, int weight)
{ 
  TVarList boundSet;
  boundSet.push_back(egen->domain->attributes->at(attrNo));
  PIM im = IMconstructor ? IMconstructor->operator()(egen, boundSet, weight) : ((TIMConstructor &)defaultIMConstructor)(egen, boundSet, weight);
  return TAssessIMQuality(columnAssessor)(im);
}
