#ifndef __HCLUST_CPP
#define __HCLUST_CPP

#include "root.hpp"
#include "orvector.hpp"
#include "symmatrix.hpp"

class TClusterW;


WRAPPER(HierarchicalCluster)

#define THierarchicalClusterList TOrangeVector<PHierarchicalCluster> 
VWRAPPER(HierarchicalClusterList)

class THierarchicalCluster : public TOrange {
public:
    __REGISTER_CLASS

    PHierarchicalClusterList branches; //P subclusters
    float height;        //P height
    PIntList elements;   //P indices to the list of all elements in the clustering
    int first;           //P the index into 'elements' to the first element of the cluster
    int last;            //P the index into 'elements' to the one after the last element of the cluster

    THierarchicalCluster();
    THierarchicalCluster(PIntList els, const int &elementIndex);
    THierarchicalCluster(PIntList els, PHierarchicalCluster left, PHierarchicalCluster right, const float &h, const int &f, const int &l);
};



class THierarchicalClustering : public TOrange {
public:
    __REGISTER_CLASS

    enum {Complete, Single, Average};
    int linkage; //P linkage

    THierarchicalClustering();
    PHierarchicalCluster operator()(PSymMatrix);

    TClusterW **init(PSymMatrix);
    TClusterW *merge(TClusterW **clusters);
    PHierarchicalCluster restructure(TClusterW *);
    PHierarchicalCluster restructure(TClusterW *, PIntList elementIndices, TIntList::iterator &currentElement, int &currentIndex);

private:
    TClusterW *THierarchicalClustering::merge_CompleteLinkage(TClusterW **clusters);
    TClusterW *THierarchicalClustering::merge_SingleLinkage(TClusterW **clusters);
    TClusterW *THierarchicalClustering::merge_AverageLinkage(TClusterW **clusters);
};

#endif