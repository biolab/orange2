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


#ifndef __HCLUST_CPP
#define __HCLUST_CPP

#include "root.hpp"
#include "orvector.hpp"
#include "symmatrix.hpp"

class TClusterW;
WRAPPER(ProgressCallback);


WRAPPER(HierarchicalCluster)

#define THierarchicalClusterList TOrangeVector<PHierarchicalCluster> 
VWRAPPER(HierarchicalClusterList)

class ORANGE_API THierarchicalCluster : public TOrange {
public:
    __REGISTER_CLASS

    PHierarchicalClusterList branches; //P subclusters
    float height;        //P height
    PIntList mapping;    //P indices to the list of all elements in the clustering
    int first;           //P the index into 'elements' to the first element of the cluster
    int last;            //P the index into 'elements' to the one after the last element of the cluster

    THierarchicalCluster();
    THierarchicalCluster(PIntList els, const int &elementIndex);
    THierarchicalCluster(PIntList els, PHierarchicalCluster left, PHierarchicalCluster right, const float &h, const int &f, const int &l);

    void swap();
    void permute(const TIntList &newOrder);

protected:
    void recursiveMove(const int &offset);
};



class ORANGE_API THierarchicalClustering : public TOrange {
public:
    __REGISTER_CLASS

    CLASSCONSTANTS(Linkage) enum {Single, Average, Complete, Ward};
    int linkage; //P(&HierarchicalClustering_Linkage) linkage
    bool overwriteMatrix; //P if true (default is false) it will use (and destroy) the distance matrix given as argument

    PProgressCallback progressCallback; //P progress callback function

    THierarchicalClustering();
    PHierarchicalCluster operator()(PSymMatrix);

    TClusterW **init(const int &dim, float *distanceMatrix);
    TClusterW *merge(TClusterW **clusters, float *callbackMilestones);
    PHierarchicalCluster restructure(TClusterW *);
    PHierarchicalCluster restructure(TClusterW *, PIntList elementIndices, TIntList::iterator &currentElement, int &currentIndex);

private:
    TClusterW *THierarchicalClustering::merge_CompleteLinkage(TClusterW **clusters, float *callbackMilestones);
    TClusterW *THierarchicalClustering::merge_SingleLinkage(TClusterW **clusters, float *callbackMilestones);
    TClusterW *THierarchicalClustering::merge_AverageLinkage(TClusterW **clusters, float *callbackMilestones);
    // Average linkage also computes Ward's linkage
};

#endif
