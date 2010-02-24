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


#ifndef __MINIMAL_ERROR_HPP
#define __MINIMAL_ERROR_HPP

#include "measures.hpp"
#include "induce.hpp"
#include "decomposition.hpp"
#include "exampleclustering.hpp"
#include "pqueue_i.hpp"

#include "slist.hpp"

class TProfitNode;
typedef TPriorityQueue<TProfitNode> TProfitQueue;

typedef slist<TProfitNode> TProfitNodeList;

class ORANGE_API TIMClusterNode {
public:
  TIMClusterNode *nextNode, *prevNode;
  TProfitNodeList mergeProfits;
  TIMColumnNode *column;
  PExampleCluster cluster;
  float columnQuality_N;

  TIMClusterNode(TIMColumnNode *, const PExample &, const float &quality, TIMClusterNode *prevNode=NULL);
  virtual ~TIMClusterNode();
};


class ORANGE_API TColumnAssessor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual void setDistribution(const TDiscDistribution &);
  virtual void setAverage(const float &avg);

  virtual float nodeQuality(TIMColumnNode &node) const=0;
  virtual float columnQuality(TIMColumnNode *) const=0;
  virtual float mergeProfit (TIMColumnNode *, TIMColumnNode *) const=0;
};

WRAPPER(ColumnAssessor)


class ORANGE_API TColumnAssessor_m : public TColumnAssessor {
public:
  __REGISTER_CLASS

  float m; //P m for m-estimate

  TColumnAssessor_m(const float &am=2.0);

  virtual void setDistribution(const TDiscDistribution &apbym);

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;

private:
  vector<float> p_by_m;
};



class ORANGE_API TColumnAssessor_Laplace : public TColumnAssessor {
public:
  __REGISTER_CLASS
  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;
};



class ORANGE_API TColumnAssessor_N : public TColumnAssessor {
public:
  __REGISTER_CLASS

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;
};


class ORANGE_API TColumnAssessor_Relief : public TColumnAssessor {
public:
  __REGISTER_CLASS

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;
};


class ORANGE_API TColumnAssessor_Kramer: public TColumnAssessor {
public:
  __REGISTER_CLASS

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;
};


class ORANGE_API TColumnAssessor_Measure : public TColumnAssessor  {
public:
  __REGISTER_CLASS

  PMeasureAttribute measure; //P attribute quality measure

  TColumnAssessor_Measure(PMeasureAttribute =PMeasureAttribute());

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;
};


class ORANGE_API TColumnAssessor_mf : public TColumnAssessor {
public:
  __REGISTER_CLASS

  float m; //P m for m-estimate

  TColumnAssessor_mf(const float &am=2.0);

  virtual void setAverage(const float &avg);

  virtual float nodeQuality(TIMColumnNode &node) const;
  virtual float columnQuality(TIMColumnNode *) const;
  virtual float mergeProfit(TIMColumnNode *, TIMColumnNode *) const;

  float m_error(const float &sum, const float &sum2, const float &N) const;

private:
  float aprior;
};


class ORANGE_API TStopIMClusteringByAssessor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual bool operator()(const float &baseQuality, const TProfitQueue &, const TIMClusterNode *clusters) const =0;
};

WRAPPER(StopIMClusteringByAssessor);

class ORANGE_API TStopIMClusteringByAssessor_noProfit : public TStopIMClusteringByAssessor {
public:
  __REGISTER_CLASS

  float minProfitProportion; //P minimal allowable profit proportion

  TStopIMClusteringByAssessor_noProfit(const float &minprof=0.0);
  virtual bool operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const;
};

class ORANGE_API TStopIMClusteringByAssessor_noBigChange : public TStopIMClusteringByAssessor {
public:
  __REGISTER_CLASS
  virtual bool operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const;
};


class ORANGE_API TStopIMClusteringByAssessor_binary : public TStopIMClusteringByAssessor {
public:
  __REGISTER_CLASS
  virtual bool operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const;
};



class ORANGE_API TStopIMClusteringByAssessor_n : public TStopIMClusteringByAssessor {
public:
  __REGISTER_CLASS

  int n; //P number of clusters

  TStopIMClusteringByAssessor_n(const int & =2);
  virtual bool operator()(const float &, const TProfitQueue &, const TIMClusterNode *clusters) const;
};


class ORANGE_API TAssessIMQuality : public TOrange {
public:
  __REGISTER_CLASS

  PColumnAssessor columnAssessor; //P column quality assessor

  TAssessIMQuality(PColumnAssessor=PColumnAssessor());
  float operator()(PIM pim);
};

WRAPPER(AssessIMQuality);


class ORANGE_API TProfitNode {
public:
  TIMClusterNode *column1, *column2;
  float profit;

  TProfitNodeList *it1, *it2;
  int queueIndex;

  long randoff;

  TProfitNode(TIMClusterNode *c1, TIMClusterNode *c2, float prof, int qind, const long &roff);
  ~TProfitNode();

  int compare(const TProfitNode &other) const;
};




class ORANGE_API TIMClustering : public TOrange {
public:
  __REGISTER_CLASS

  PIM im; //P incompatibilty matrix
  PIntList clusters; //P cluster index for each matrix column
  int maxCluster; //P the highest cluster index
  float quality; //P cluster quality

  TIMClustering(PIM =PIM());
};

WRAPPER(IMClustering)

class ORANGE_API TClustersFromIM : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual PExampleClusters operator()(PIM im)=0;
};

WRAPPER(ClustersFromIM);


class ORANGE_API TClustersFromIMByAssessor : public TClustersFromIM {
public:
  __REGISTER_CLASS

  PColumnAssessor columnAssessor; //P column quality assessor
  PStopIMClusteringByAssessor stopCriterion; //P stop criterion

  TClustersFromIMByAssessor(PColumnAssessor = PColumnAssessor());

  virtual PExampleClusters operator()(PIM im);

protected:
  virtual void  preparePrivateVars(PIM im, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &);
  virtual void  preparePrivateVarsD(PIM im, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &);
  virtual void  preparePrivateVarsF(PIM im, TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &);

  virtual void computeQualities(TIMClusterNode *clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &);

  void  mergeBestColumns(TIMClusterNode *&clusters, TProfitQueue &profitQueue, float &baseQuality, float &N, TSimpleRandomGenerator &);

  TProfitNode *insertProfitQueueNode(TIMClusterNode *, TIMClusterNode *, float profit, long randoff, TProfitQueue &);
};

WRAPPER(ClustersFromIMByAssessor);


class ORANGE_API TFeatureByIM : public TFeatureInducer {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(Completion: NoCompletion=completion_no; CompletionByDefault=completion_default; CompletionByBayes=completion_bayes)

  PIMConstructor IMconstructor; //P incompatibility matrix constructor
  PClustersFromIM clustersFromIM; //P clustering algorithm
  int completion; //P(&FeatureByIM_Completion) decides how to determine the class for points not covered by any cluster

  TFeatureByIM(PIMConstructor = PIMConstructor(), PClustersFromIM=PClustersFromIM(), const int & =completion_bayes);
  PVariable operator()(PExampleGenerator gen, TVarList &boundSet, const string &name, float &quality, const int &weight=0);
};


class ORANGE_API TMeasureAttribute_IM : public TMeasureAttribute
{ public:
    __REGISTER_CLASS

    PIMConstructor IMconstructor; //P incompatibility matrix constructor
    PColumnAssessor columnAssessor; //P column quality assessor

    TMeasureAttribute_IM();
    virtual float operator()(int attrNo, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID=0);
};

#endif
