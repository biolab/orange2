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


#ifndef __EXAMPLECLUSTERING_HPP
#define __EXAMPLECLUSTERING_HPP

#include "root.hpp"

#include "orvector.hpp"

enum { completion_no, completion_default, completion_bayes };

PClassifier completeTable(PExampleGenerator examples, int completion, int weightID=0);


WRAPPER(ExampleCluster);
WRAPPER(Example);

class ORANGE_API TExampleCluster : public TOrange {
public:
  __REGISTER_CLASS

  PExampleCluster left;  //P 'left' cluster
  PExampleCluster right; //P 'right' cluster
  float distance;        //P distance between the two clusters (not defined for leaves)
  PExample centroid;     //P cluster's centroid (always defined for leaves)

  TExampleCluster();
  TExampleCluster(PExample);
  TExampleCluster(PExampleCluster, PExampleCluster, const float &);

  TExampleCluster(vector<PExampleCluster> &, const float &distance);
};


#define TExampleSets TOrangeVector<PExampleGenerator> 
VWRAPPER(ExampleSets)


WRAPPER(Classifier)
WRAPPER(Variable)

WRAPPER(ExampleClusters)

/* This class is a base for classes representing example clusters
   in any general format. Eg., coloring of an incompatibility graph 
   is represented as TColoredIG, which is derived from the below
   class. Example clusters are recorded in a specific format - as
   vector of integers, representing colors for graph nodes.
*/
class ORANGE_API TGeneralExampleClustering : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  /* These two functions must be defined - they return the clusters
     in two common structures - ExampleCluster and ExampleSet
  */
  virtual PExampleClusters exampleClusters() const =0;
  virtual PExampleSets exampleSets(const float &cut) const =0;

  /* Those needn't be defined. As they are, they are 'classifier'
     calls ExampleSets to obtain sets of examples in different clusters
     and 'feature' calls 'classifier' and returns the classVar of
     the resulting classifier.
     
     However, if there is a much more efficient way to derive
     classifier/feature from clustering, you may decide to overload
     the below methods. */
  virtual PClassifier classifier(const float &cut = 0.0, const int &completion = completion_bayes) const;
  virtual PVariable feature(const float &cut = 0.0, const int &completion = completion_bayes) const;
};

WRAPPER(GeneralExampleClustering)


class ORANGE_API TExampleClusters : public TGeneralExampleClustering {
public:
  __REGISTER_CLASS

  PExampleCluster root; //P root of cluster hierarchy
  float quality; //P 'quality' of clustering

  TExampleClusters();
  TExampleClusters(PExampleCluster, const float &);

  virtual PExampleClusters exampleClusters() const;
  virtual PExampleSets exampleSets(const float &cut) const;
};

#endif
