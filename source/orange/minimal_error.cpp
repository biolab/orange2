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


void TColumnAssessor::setDistribution(const TDiscDistribution &)
{}


void TColumnAssessor::setAverage(const float &)
{}


void TColumnAssessor_m::setDistribution(const TDiscDistribution &classDist)
{ p_by_m = vector<float>();
  float N = classDist.abs;
  const_ITERATE(TDiscDistribution, ci, classDist)
    p_by_m.push_back(*ci/N*m);
}



TColumnAssessor_m::TColumnAssessor_m(const float &am)
: m(am)
{}



float TColumnAssessor_m::nodeQuality(TIMColumnNode &node) const
{ 
  float *val = dynamic_cast<TDIMColumnNode &>(node).distribution;
  
  float bestok=-1, thisok;
  vector<float>::const_iterator pci(p_by_m.begin());
  for(int c = p_by_m.size(); c--; ) {
	  if (  (thisok = (*(val++) + *(pci++)) )  > bestok)
      bestok = thisok;
  }
 
  float N = dynamic_cast<TDIMColumnNode &>(node).abs;
  return - N *  (1 - bestok/(N + m));
}


/* The returned error is not divided by the total number of examples so that the function
   returns the m estimate for error, multiplied by the number of examples in the column. */
float TColumnAssessor_m::columnQuality(TIMColumnNode *col) const
{
  float err=0.0;
  for(; col; col=col->next)
    err += (col->nodeQuality = nodeQuality(*col));
  return err; 
}


/* The profit is not divided by the total number of examples so that the function returns the
   profit multiplied by the number of examples in the merged column. */
float TColumnAssessor_m::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float profit=0;

  while (col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1 = col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2 = col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1 = col1->next, col2 = col2->next) {
      const TDIMColumnNode *node1 = dynamic_cast<TDIMColumnNode const *>(col1);
      const TDIMColumnNode *node2 = dynamic_cast<TDIMColumnNode const *>(col2);
      const float *val1 = node1->distribution, *val2 = node2->distribution;

      float bestok=-1, thisok;
      vector<float>::const_iterator pci(p_by_m.begin());
      for(int c= p_by_m.size(); c--; )
        if (  (thisok = (*(val1++) + *(val2++) + *(pci++)))  > bestok)
          bestok=thisok;
 
      float N = node1->abs + node2->abs;
      profit +=   -(col1->nodeQuality + col2->nodeQuality)   -   (N * (1 - bestok/(N+m)));
    }
  }

  return profit;
}


float TColumnAssessor_Laplace::nodeQuality(TIMColumnNode &node) const
{ 
  const TDIMColumnNode &dnode = dynamic_cast<TDIMColumnNode &>(node);
  if (!dnode.noOfValues)
    return 0.0;

  float *val = dnode.distribution;
  float maj = 0.0;
  for(int n = dnode.noOfValues; n--; val++)
    if (*val>maj)
      maj = *val;

  return dnode.abs * (1 - (maj+1)/(dnode.noOfValues+dnode.abs));
}


/* The returned error is not divided by the total number of examples so that the function
   returns the m estimate for error, multiplied by the number of examples in the column. */
float TColumnAssessor_Laplace::columnQuality(TIMColumnNode *col) const
{
  float err=0.0;
  for(; col; col=col->next)
    err += (col->nodeQuality = nodeQuality(*col));
  return err; 
}


/* The profit is not divided by the total number of examples so that the function returns the
   profit multiplied by the number of examples in the merged column. */
float TColumnAssessor_Laplace::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float errorDif=0;

  while (col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1 = col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2 = col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1 = col1->next, col2 = col2->next) {
      const TDIMColumnNode *node1 = dynamic_cast<TDIMColumnNode const *>(col1);
      const TDIMColumnNode *node2 = dynamic_cast<TDIMColumnNode const *>(col2);
      const float *val1 = node1->distribution, *val2 = node2->distribution;
      const float abs = node1->abs + node2->abs;

      float maj = 0.0, thisc;
      for(int c = node1->noOfValues; c--; val1++, val2++)
        if ( (thisc = (*(val1++) + *(val2++)))  > maj)
          maj = thisc;
 
      const float oldError = -(col1->nodeQuality + col2->nodeQuality);
      const float newError = abs * (1 - (maj+1)/(abs + node1->noOfValues));
      errorDif += newError -  oldError;
    }
  }

  return -errorDif;
}


float TColumnAssessor_N::nodeQuality(TIMColumnNode &node) const
{ return sqr(dynamic_cast<const TDIMColumnNode &>(node).abs); }



float TColumnAssessor_N::columnQuality(TIMColumnNode *col) const
{ float sum = 0.0;
  for(; col; col = col->next)
    sum += sqr(dynamic_cast<TDIMColumnNode *>(col)->abs);
  return sum;
}


float TColumnAssessor_N::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float profit=0;
  while(col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1=col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2=col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1=col1->next, col2=col2->next)
      profit += 2 * dynamic_cast<const TDIMColumnNode *>(col1)->abs * dynamic_cast<const TDIMColumnNode *>(col2)->abs;
  }

  return profit;
}



float TColumnAssessor_Relief::nodeQuality(TIMColumnNode &) const
{ return 0.0; }


float TColumnAssessor_Relief::columnQuality(TIMColumnNode *) const
{ return 0.0;
}


float TColumnAssessor_Relief::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float profit=0;
  while(col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1=col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2=col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1=col1->next, col2=col2->next) {
      const TDIMColumnNode *node1 = dynamic_cast<const TDIMColumnNode *>(col1);
      const TDIMColumnNode *node2 = dynamic_cast<const TDIMColumnNode *>(col2);
      const float *val1 = node1->distribution,
                  *val2 = node2->distribution;

      float tq=0.0;
      for(int c = (dynamic_cast<const TDIMColumnNode *>(col1))->noOfValues; c--; )
          tq += 4 * *(val1++) * *(val2++);

      tq -= 2 * node1->abs * node2->abs;
      profit += tq;
    }
  }

  return profit;
}


float TColumnAssessor_mf::m_error(const float &sum, const float &sum2, const float &N) const
{ float df=sum+m*aprior;
  float N_m=N+m;
  return N/N_m * (sum2 + m*aprior*aprior - df*df/N_m);
}

TColumnAssessor_mf::TColumnAssessor_mf(const float &am)
: m(am)
{}


void TColumnAssessor_mf::setAverage(const float &avg)
{ aprior=avg; }


/* The returned error is not divided by the total number of examples so that the function
   returns the m estimate for error, multiplied by the number of examples in the column. */
float TColumnAssessor_mf::columnQuality(TIMColumnNode *column) const
{
  float err = 0.0;
  for(TFIMColumnNode *col = dynamic_cast<TFIMColumnNode *>(column); col; col=dynamic_cast<TFIMColumnNode *>(col->next))
    err += (col->nodeQuality= -m_error(col->sum, col->sum2, col->N));

  return err; 
}


float TColumnAssessor_mf::nodeQuality(TIMColumnNode &node) const
{ TFIMColumnNode &nde = dynamic_cast<TFIMColumnNode &>(node);
  return -m_error(nde.sum, nde.sum2, nde.N); }


/* The profit is not divided by the total number of examples so that the function returns the
   profit multiplied by the number of examples in the merged column. */
float TColumnAssessor_mf::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float profit = 0.0;
  while(col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1=col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2=col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1=col1->next, col2=col2->next) {
      TFIMColumnNode const &c1=*dynamic_cast<TFIMColumnNode const *>(col1), &c2=*dynamic_cast<TFIMColumnNode const *>(col2);
      profit+= - m_error(c1.sum+c2.sum, c1.sum2+c2.sum2, c1.N+c2.N)
               - (c1.nodeQuality + c2.nodeQuality);
    }
  }

  return profit;
}


TColumnAssessor_Measure::TColumnAssessor_Measure(PMeasureAttribute meas)
: measure(meas)
{}


float TColumnAssessor_Measure::nodeQuality(TIMColumnNode &node) const
{ TDIMColumnNode &dnode = dynamic_cast<TDIMColumnNode &>(node);
  return dnode.abs * measure->operator()(TDiscDistribution(dnode.distribution, dnode.noOfValues));
}


float TColumnAssessor_Measure::columnQuality(TIMColumnNode *col) const
{ float sum = 0.0;
  for(; col; col=col->next) {
    TDIMColumnNode *dnode = dynamic_cast<TDIMColumnNode *>(col);
    sum += dnode->abs * measure->operator()(TDiscDistribution(dnode->distribution, dnode->noOfValues));
  }
  return sum;
}

float TColumnAssessor_Measure::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{ float profit = 0.0;
  while(col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1=col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2=col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1=col1->next, col2=col2->next) {
      TDIMColumnNode const &c1=*dynamic_cast<TDIMColumnNode const *>(col1), &c2=*dynamic_cast<TDIMColumnNode const *>(col2);
      float *val1 = c1.distribution, *val2 = c2.distribution;
      TDiscDistribution distr;
      for(int c=0, e=c1.noOfValues; c<e; distr.addint(c++, *(val1++) + *(val2++)));
      profit += (c1.abs+c2.abs) * measure->operator()(distr) - (c1.nodeQuality + c2.nodeQuality);
    }
  }

  return profit;
}



float TColumnAssessor_Kramer::nodeQuality(TIMColumnNode &node) const
{ TDIMColumnNode &col = dynamic_cast<TDIMColumnNode &>(node);
  if (col.noOfValues!=2)
    raiseError("binary class expected");
  const float *cd = col.distribution;
  // abs values cancel each other:
  return -cd[0]*cd[1];
}


float TColumnAssessor_Kramer::columnQuality(TIMColumnNode *col) const
{ float sum=0.0;
  for(; col; col=col->next) {
    TDIMColumnNode &node = dynamic_cast<TDIMColumnNode &>(*col);
    if (node.noOfValues!=2)
      raiseError("binary class expected");
    const float *cd = node.distribution;
    sum -= cd[0]*cd[1];
  }
  return sum;
}


float TColumnAssessor_Kramer::mergeProfit(TIMColumnNode *col1, TIMColumnNode *col2) const
{
  float profit = 0.0;
  while(col2 && col1) {

    while(col1 && (col1->index<col2->index))
      col1=col1->next;
    if (!col1)
      break;

    while(col2 && (col2->index<col1->index))
      col2=col2->next;

    for (; col2 && col1 && (col1->index==col2->index); col1=col1->next, col2=col2->next) {
      const TDIMColumnNode *node1 = dynamic_cast<const TDIMColumnNode *>(col1);
      const TDIMColumnNode *node2 = dynamic_cast<const TDIMColumnNode *>(col2);
      const float *cd1 = node1->distribution;
      const float *cd2 = node2->distribution;
      profit += (cd1[0]+cd2[0])*(cd1[1]+cd2[1]) - (node1->nodeQuality + node2->nodeQuality);
    }
  }

  return profit;
}



TStopIMClusteringByAssessor_noProfit::TStopIMClusteringByAssessor_noProfit(const float &minprof)
: minProfitProportion(minprof)
{}



bool TStopIMClusteringByAssessor_noProfit::operator()(const float &baseQuality, const TProfitQueue &pq, const TIMClusterNode *) const
{ return (pq.front()->profit < 0) || (pq.front()->profit<baseQuality*minProfitProportion); }


#include <math.h>

bool TStopIMClusteringByAssessor_noBigChange::operator()(const float &, const TProfitQueue &profitQueue, const TIMClusterNode *) const
{ if (profitQueue.front()->profit >= 0)
    return false;

  int pN = profitQueue.size();
  if (pN>1) {
    float sum=0.0, sum2=0.0;
    const_ITERATE(TProfitQueue, pi, profitQueue) {
      float tp = (*pi)->profit;
      sum += tp;
      sum2 += tp*tp;
    }

    int N = profitQueue.size();
    float dev = sqrt( (sum2 - sum*sum/N)/N );
    if (profitQueue.front()->profit < sum/N+1.96*dev)
      return true;
  }
  
  else if (profitQueue.front()->profit < 0)
    return true;

  return false;
}


bool TStopIMClusteringByAssessor_binary::operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const
{ return (!clusters || !clusters->nextNode || !clusters->nextNode->nextNode); }



TStopIMClusteringByAssessor_n::TStopIMClusteringByAssessor_n(const int &an)
: n(an)
{}

bool TStopIMClusteringByAssessor_n::operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const
{ TIMClusterNode const *cn = clusters;
  for (int i=n; cn && i; i--, cn=cn->nextNode);
  return !cn;
}


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
