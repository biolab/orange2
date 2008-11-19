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

#include "progress.hpp"

#include "hclust.ppp"

DEFINE_TOrangeVector_classDescription(PHierarchicalCluster, "THierarchicalClusterList", true, ORANGE_API)

class TClusterW {
public:
    TClusterW *next; // for cluster, this is next cluster, for elements next element
    TClusterW *left, *right; // subclusters, if left==NULL, this is an element
    int size;
    int elementIndex;
    float height;

    float *distances; // distances to clusters before this one (lower left matrix)
    float minDistance; // minimal distance
    int rawIndexMinDistance; // index of minimal distance
    int nDistances;

    TClusterW(const int &elIndex, float *adistances, const int &anDistances)
    : next(NULL),
      left(NULL),
      right(NULL),
      size(1),
      elementIndex(elIndex),
      height(0.0),
      distances(adistances),
      minDistance(numeric_limits<float>::max()),
      rawIndexMinDistance(-1),
      nDistances(anDistances)
    {
      if (distances)
        computeMinimalDistance();
    }

    void elevate(TClusterW **aright, const float &aheight)
    {
      left = mlnew TClusterW(*this);
      right = *aright;

      left->distances = NULL;
      size = left->size + right->size;
      elementIndex = -1;
      height = aheight;

      if (next == right)
        next = right->next;
      else
        *aright = right->next;
    }


    void computeMinimalDistance()
    {
      float *dp = distances, *minp = dp++;
      for(int i = nDistances; --i; dp++)
        if ((*dp >= 0) && (*dp < *minp))
          minp = dp;
      minDistance = *minp;
      rawIndexMinDistance = minp - distances;
    }

    TClusterW **clusterAt(int rawIndex, TClusterW **cluster)
    {
      for(float *dp = distances; rawIndex; dp++)
        if (*dp>=0) {
          cluster = &(*cluster)->next;
          rawIndex--;
        }
      return cluster;
    }

    void clearDistances()
    {
      raiseErrorWho("TClusterW", "rewrite clean to adjust indexMinDistance");

      float *dst = distances;
      int i = nDistances;
      for(; nDistances && (*dst >= 0); nDistances--, dst++);
      if (!nDistances)
        return;
      
      float *src = dst;
      for(src++, nDistances--; nDistances && (*src >= 0); src++)
        if (*src >= 0)
          *dst++ = *src;

      nDistances = dst - distances;
    }
};


THierarchicalCluster::THierarchicalCluster()
: branches(),
  height(0.0),
  mapping(),
  first(0),
  last(0)
{}


THierarchicalCluster::THierarchicalCluster(PIntList els, const int &elementIndex)
: branches(),
  height(0.0),
  mapping(els),
  first(elementIndex),
  last(elementIndex+1)
{}


THierarchicalCluster::THierarchicalCluster(PIntList els, PHierarchicalCluster left, PHierarchicalCluster right, const float &h, const int &f, const int &l)
: branches(new THierarchicalClusterList(2)),
  height(h),
  mapping(els),
  first(f),
  last(l)
{ 
  branches->at(0) = left;
  branches->at(1) = right;
}


void THierarchicalCluster::swap()
{
  if (!branches || (branches->size()<2))
    return;
  if (branches->size() > 2)
    raiseError("cannot swap multiple branches (use method 'permutation' instead)");

  const TIntList::iterator beg0 = mapping->begin() + branches->at(0)->first;
  const TIntList::iterator beg1 = mapping->begin() + branches->at(1)->first;
  const TIntList::iterator end1 = mapping->begin() + branches->at(1)->last;

  if ((branches->at(0)->first > branches->at(1)->first) || (branches->at(1)->first > branches->at(1)->last))
    raiseError("internal inconsistency in clustering structure: invalid ordering of left's and right's 'first' and 'last'");

  TIntList::iterator li0, li1;

  int *temp = new int [beg1 - beg0], *t;
  for(li0 = beg0, t = temp; li0 != beg1; *t++ = *li0++);
  for(li0 = beg0, li1 = beg1; li1 != end1; *li0++ = *li1++);
  for(t = temp; li0 != end1; *li0++ = *t++);
  delete temp;

  branches->at(0)->recursiveMove(end1 - beg1);
  branches->at(1)->recursiveMove(beg0 - beg1);

  PHierarchicalCluster tbr = branches->at(0);
  branches->at(0) = branches->at(1);
  branches->at(1) = tbr;
}


void THierarchicalCluster::permute(const TIntList &neworder)
{
  if ((!branches && neworder.size()) || (branches->size() != neworder.size()))
    raiseError("the number of clusters does not match the lenght of the permutation vector");

  int *temp = new int [last - first], *t = temp;
  TIntList::const_iterator pi = neworder.begin();
  THierarchicalClusterList::iterator bi(branches->begin()), be(branches->end());
  THierarchicalClusterList newBranches;

  for(; bi != be; bi++, pi++) {
    PHierarchicalCluster branch = branches->at(*pi);
    newBranches.push_back(branch);
    TIntList::const_iterator bei(mapping->begin() + branch->first), bee(mapping->begin() + branch->last);
    const int offset = (t - temp) - (branch->first - first);
    for(; bei != bee; *t++ = *bei++);
    if (offset)
      branch->recursiveMove(offset);
  }

  TIntList::iterator bei(mapping->begin() + first), bee(mapping->begin() + last);
  for(t = temp; bei!=bee; *bei++ = *t++);

  bi = branches->begin();
  THierarchicalClusterList::const_iterator nbi(newBranches.begin());
  for(; bi != be; *bi++ = *nbi++);
}


void THierarchicalCluster::recursiveMove(const int &offset)
{
  first += offset;
  last += offset;
  if (branches)
    PITERATE(THierarchicalClusterList, bi, branches)
      (*bi)->recursiveMove(offset);
}


THierarchicalClustering::THierarchicalClustering()
: linkage(Single),
  overwriteMatrix(false)
{}


TClusterW **THierarchicalClustering::init(const int &dim, float *distanceMatrix)
{
  for(float *ddi = distanceMatrix, *dde = ddi + ((dim+1)*(dim+2))/2; ddi!=dde; ddi++)
    if (*ddi < 0) {
      int x, y;
      TSymMatrix::index2coordinates(ddi-distanceMatrix, x, y);
      raiseError("distance matrix contains negative element at (%i, %i)", x, y);
    }

  TClusterW **clusters = mlnew TClusterW *[dim];
  TClusterW **clusteri = clusters;

  *clusters = mlnew TClusterW(0, NULL, 0);
  distanceMatrix++;
  
  for(int elementIndex = 1, e = dim; elementIndex < e; distanceMatrix += ++elementIndex) {
    TClusterW *newcluster = mlnew TClusterW(elementIndex, distanceMatrix, elementIndex);
    (*clusteri++)->next = newcluster;
    *clusteri = newcluster; 
  }

  return clusters;
}



TClusterW *THierarchicalClustering::merge_SingleLinkage(TClusterW **clusters, float *milestones)
{
  float *milestone = milestones;

  int step = 0;
  while((*clusters)->next) {
    if (milestone && (step++ ==*milestone))
      progressCallback->call(*((++milestone)++));

    TClusterW *cluster;
    TClusterW **pcluster2;

    float minDistance = numeric_limits<float>::max();
    for(TClusterW **tcluster = &((*clusters)->next); *tcluster; tcluster = &(*tcluster)->next)
      if ((*tcluster)->minDistance < minDistance) {
        minDistance = (*tcluster)->minDistance;
        pcluster2 = tcluster;
      }

    TClusterW *const cluster2 = *pcluster2;

    const int rawIndex1 = cluster2->rawIndexMinDistance;
    const int rawIndex2 = cluster2->nDistances;

    TClusterW *const cluster1 = clusters[rawIndex1];

    float *disti1 = cluster1->distances;
    float *disti2 = cluster2->distances;

    if (rawIndex1) { // not root - has no distances...
      const float *minIndex1 = cluster1->distances + cluster1->rawIndexMinDistance;
      for(int ndi = cluster1->nDistances; ndi--; disti1++, disti2++)
        if (*disti2 < *disti1) // if one is -1, they both are
          if ((*disti1 = *disti2) < *minIndex1)
            minIndex1 = disti1;
      cluster1->minDistance = *minIndex1;
      cluster1->rawIndexMinDistance = minIndex1 - cluster1->distances;
    }

    while(*disti2 < 0)
      disti2++;        // should have at least one more >=0  - the one corresponding to distance to cluster1

    for(cluster = cluster1->next; cluster != cluster2; cluster = cluster->next) {
      while(*++disti2 < 0); // should have more - the one corresponding to cluster
      if (*disti2 < cluster->distances[rawIndex1])
        if ((cluster->distances[rawIndex1] = *disti2) < cluster->minDistance) {
          cluster->minDistance = *disti2;
          cluster->rawIndexMinDistance = rawIndex1;
        }
    }

    for(cluster = cluster->next; cluster; cluster = cluster->next) {
      if (cluster->distances[rawIndex2] < cluster->distances[rawIndex1])
        cluster->distances[rawIndex1] = cluster->distances[rawIndex2];

      // don't nest this in the above if -- both distances can be equal, yet index HAS TO move
      if (rawIndex2 == cluster->rawIndexMinDistance)
        cluster->rawIndexMinDistance = rawIndex1; // the smallest element got moved

      cluster->distances[rawIndex2] = -1;
    }

    cluster1->elevate(pcluster2, minDistance);
  }

  return *clusters;
}

// Also computes Ward's linkage
TClusterW *THierarchicalClustering::merge_AverageLinkage(TClusterW **clusters, float *milestones)
{
  float *milestone = milestones;
  bool ward = linkage == Ward;

  int step = 0;
  while((*clusters)->next) {
    if (milestone && (step++ ==*milestone))
      progressCallback->call(*((++milestone)++));

    TClusterW *cluster;
    TClusterW **pcluster2;

    float minDistance = numeric_limits<float>::max();
    for(TClusterW **tcluster = &((*clusters)->next); *tcluster; tcluster = &(*tcluster)->next)
      if ((*tcluster)->minDistance < minDistance) {
        minDistance = (*tcluster)->minDistance;
        pcluster2 = tcluster;
      }

    TClusterW *const cluster2 = *pcluster2;

    const int rawIndex1 = cluster2->rawIndexMinDistance;
    const int rawIndex2 = cluster2->nDistances;

    TClusterW *const cluster1 = clusters[rawIndex1];

    float *disti1 = cluster1->distances;
    float *disti2 = cluster2->distances;

    const int size1 = cluster1->size;
    const int size2 = cluster2->size;
    const int sumsize = size1 + size2;
    cluster = (*clusters)->next;

    if (rawIndex1) { // not root - has no distances...
      const float sizeK = cluster->size;
      *disti1 = ward ? (*disti1 * (size1+sizeK) + *disti2 * (size2+sizeK) - minDistance * sizeK) / (sumsize+sizeK)
                     : (*disti1 * size1 + *disti2 * size2) / sumsize;
      const float *minIndex1 = disti1;
      int ndi = cluster1->nDistances-1;
      for(disti1++, disti2++, cluster = cluster->next; ndi--; disti1++, disti2++)
        if (*disti1 >= 0) {
          const float sizeK = cluster->size;
          cluster = cluster->next;
          *disti1 = ward ? (*disti1 * (size1+sizeK) + *disti2 * (size2+sizeK) - minDistance * sizeK) / (sumsize+sizeK)
                         : (*disti1 * size1         + *disti2 * size2) / sumsize;
          if (*disti1 < *minIndex1)
            minIndex1 = disti1;
        }
      cluster1->minDistance = *minIndex1;
      cluster1->rawIndexMinDistance = minIndex1 - cluster1->distances;
    }

    while(*disti2 < 0)
      disti2++;        // should have at least one more >=0  - the one corresponding to distance to cluster1

    for(cluster = cluster1->next; cluster != cluster2; cluster = cluster->next) {
      while(*++disti2 < 0); // should have more - the one corresponding to cluster
      float &distc = cluster->distances[rawIndex1];
      const float sizeK = cluster->size;
      distc = ward ? (distc * (size1+sizeK) + *disti2 * (size2+sizeK) - minDistance * sizeK) / (sumsize+sizeK)
                   : (distc * size1         + *disti2 * size2) / sumsize;
      if (distc < cluster->minDistance) {
        cluster->minDistance = distc;
        cluster->rawIndexMinDistance = rawIndex1;
      }
      else if ((distc > cluster->minDistance) && (cluster->rawIndexMinDistance == rawIndex1)) {
        cluster->distances[rawIndex1] = -1;
        cluster->computeMinimalDistance();
      }
    }

    for(cluster = cluster->next; cluster; cluster = cluster->next) {
      float &distc = cluster->distances[rawIndex1];
      const float sizeK = cluster->size;
      distc = ward ? (distc * (size1+sizeK) + cluster->distances[rawIndex2] * (size2+sizeK) - minDistance * sizeK) / (sumsize+sizeK)
                   : (distc * size1         + cluster->distances[rawIndex2] * size2) / sumsize;
      if (distc < cluster->minDistance) {
        cluster->minDistance = distc;
        cluster->rawIndexMinDistance = rawIndex1;
      }
      else if (   (distc > cluster->minDistance) && (cluster->rawIndexMinDistance == rawIndex1)
               || (cluster->rawIndexMinDistance == rawIndex2)) {
        cluster->distances[rawIndex1] = cluster->distances[rawIndex2] = -1;
        cluster->computeMinimalDistance();
      }
      else
        cluster->distances[rawIndex2] = -1;
    }

    cluster1->elevate(pcluster2, minDistance);
  }

  return *clusters;
}



TClusterW *THierarchicalClustering::merge_CompleteLinkage(TClusterW **clusters, float *milestones)
{
  float *milestone = milestones;

  int step = 0;
  while((*clusters)->next) {
    if (milestone && (step++ ==*milestone))
      progressCallback->call(*((++milestone)++));

    TClusterW *cluster;
    TClusterW **pcluster2;

    float minDistance = numeric_limits<float>::max();
    for(TClusterW **tcluster = &((*clusters)->next); *tcluster; tcluster = &(*tcluster)->next)
      if ((*tcluster)->minDistance < minDistance) {
        minDistance = (*tcluster)->minDistance;
        pcluster2 = tcluster;
      }

    TClusterW *const cluster2 = *pcluster2;

    const int rawIndex1 = cluster2->rawIndexMinDistance;
    const int rawIndex2 = cluster2->nDistances;

    TClusterW *const cluster1 = clusters[rawIndex1];

    float *disti1 = cluster1->distances;
    float *disti2 = cluster2->distances;

    if (rawIndex1) { // not root - has no distances...
      if (*disti2 > *disti1)
        *disti1 = *disti2;
      const float *minIndex1 = disti1;
      int ndi = cluster1->nDistances-1;
      for(disti1++, disti2++; ndi--; disti1++, disti2++) {
        if (*disti1 >= 0) {
          if (*disti2 > *disti1) // if one is -1, they both are
            *disti1 = *disti2;
          if (*disti1 < *minIndex1)
            minIndex1 = disti1;
        }
      }
      cluster1->minDistance = *minIndex1;
      cluster1->rawIndexMinDistance = minIndex1 - cluster1->distances;
    }

    while(*disti2 < 0)
      disti2++;        // should have at least one more >=0  - the one corresponding to distance to cluster1

    for(cluster = cluster1->next; cluster != cluster2; cluster = cluster->next) {
      while(*++disti2 < 0); // should have more - the one corresponding to cluster
      float &distc = cluster->distances[rawIndex1];
      if (*disti2 > distc) {
        distc = *disti2;
        if (cluster->rawIndexMinDistance == rawIndex1)
          if (distc <= cluster->minDistance)
            cluster->minDistance = distc;
          else
            cluster->computeMinimalDistance();
      }
    }

    for(cluster = cluster->next; cluster; cluster = cluster->next) {
      float &distc = cluster->distances[rawIndex1];
      if (cluster->distances[rawIndex2] > distc)
        distc = cluster->distances[rawIndex2];

      cluster->distances[rawIndex2] = -1;
      if ((cluster->rawIndexMinDistance == rawIndex1) || (cluster->rawIndexMinDistance == rawIndex2))
        cluster->computeMinimalDistance();
    }

    cluster1->elevate(pcluster2, minDistance);
  }

  return *clusters;
}



TClusterW *THierarchicalClustering::merge(TClusterW **clusters, float *milestones)
{
  switch(linkage) {
    case Complete: return merge_CompleteLinkage(clusters, milestones);
    case Single: return merge_SingleLinkage(clusters, milestones);
    case Average: 
    case Ward:
    default: return merge_AverageLinkage(clusters, milestones);
  }
}

PHierarchicalCluster THierarchicalClustering::restructure(TClusterW *root)
{
  PIntList elementIndices = new TIntList(root->size);
  TIntList::iterator currentElement(elementIndices->begin());
  int currentIndex = 0;

  return restructure(root, elementIndices, currentElement, currentIndex);
}


PHierarchicalCluster THierarchicalClustering::restructure(TClusterW *node, PIntList elementIndices, TIntList::iterator &currentElement, int &currentIndex)
{
  PHierarchicalCluster cluster;

  if (!node->left) {
    *currentElement++ = node->elementIndex;
    cluster = mlnew THierarchicalCluster(elementIndices, currentIndex++);
  }
  else {
    PHierarchicalCluster left = restructure(node->left, elementIndices, currentElement, currentIndex);
    PHierarchicalCluster right = restructure(node->right, elementIndices, currentElement, currentIndex);
    cluster = mlnew THierarchicalCluster(elementIndices, left, right, node->height, left->first, right->last);
  }

  // no need to care about 'distances' - they've been removed during clustering (in 'elevate') :)
  mldelete node;
  return cluster;
}


PHierarchicalCluster THierarchicalClustering::operator()(PSymMatrix distanceMatrix)
{
  float *distanceMatrixElements = NULL;
  TClusterW **clusters, *root;
  float *callbackMilestones = NULL;
  
  try {
    const int dim = distanceMatrix->dim;
    const int size = ((dim+1)*(dim+2))/2;
    float *distanceMatrixElements = overwriteMatrix ? distanceMatrix->elements : (float *)memcpy(new float[size], distanceMatrix->elements, size*sizeof(float));

    clusters = init(dim, distanceMatrixElements);
    callbackMilestones = (progressCallback && (distanceMatrix->dim >= 1000)) ? progressCallback->milestones(distanceMatrix->dim) : NULL;
    root = merge(clusters, callbackMilestones);
  }
  catch (...) {
    mldelete clusters;
    mldelete callbackMilestones;
    mldelete distanceMatrixElements;
    throw;
  }

  mldelete clusters;
  mldelete callbackMilestones;
  mldelete distanceMatrixElements;

  return restructure(root);
}
