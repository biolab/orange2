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
