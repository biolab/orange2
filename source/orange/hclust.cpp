#include "hclust.ppp"

DEFINE_TOrangeVector_classDescription(PHierarchicalCluster, "THierarchicalClusterList")

class TClusterW {
public:
    TClusterW *next; // for cluster, this is next cluster, for elements next element
    TClusterW *left, *right; // subclusters, if left==NULL, this is an element
    int size;
    int elementIndex;
    float height;

    float *distances; // distances to the clusters before this one (lower left matrix)
    float minDistance; // minimal distance
    int rawIndexMinDistance; // index of minimal distance
    int nDistances;

    TClusterW(const int &elIndex)
    : next(NULL),
      left(NULL),
      right(NULL),
      size(1),
      elementIndex(elIndex),
      height(0.0),
      distances(NULL),
      minDistance(numeric_limits<float>::max()),
      rawIndexMinDistance(-1),
      nDistances(0)
    {}

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
  elements(),
  first(0),
  last(0)
{}


THierarchicalCluster::THierarchicalCluster(PIntList els, const int &elementIndex)
: branches(),
  height(0.0),
  elements(els),
  first(elementIndex),
  last(elementIndex+1)
{}


THierarchicalCluster::THierarchicalCluster(PIntList els, PHierarchicalCluster left, PHierarchicalCluster right, const float &h, const int &f, const int &l)
: branches(new THierarchicalClusterList(2)),
  height(h),
  elements(els),
  first(f),
  last(l)
{ 
  branches->at(0) = left;
  branches->at(1) = right;
}



THierarchicalClustering::THierarchicalClustering()
: linkage(Single)
{}


TClusterW **THierarchicalClustering::init(PSymMatrix map)
{
  TClusterW **clusters = mlnew TClusterW *[map->dim];
  TClusterW **clusteri = clusters;
  *clusters = mlnew TClusterW(0);
  
  float *distances = map->elements+1;
  for(int elementIndex = 1, e = map->dim; elementIndex < e; distances += ++elementIndex) {
    TClusterW *newcluster = mlnew TClusterW(elementIndex);
    newcluster->distances = (float *)memcpy(new float[elementIndex], distances, elementIndex*sizeof(float));
    newcluster->nDistances = elementIndex;
    newcluster->computeMinimalDistance();
    (*clusteri++)->next = newcluster;
    *clusteri = newcluster; 
  }
  
  return clusters;
}



TClusterW *THierarchicalClustering::merge_SingleLinkage(TClusterW **clusters)
{
  while((*clusters)->next) {
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
      if (cluster->distances[rawIndex2] < cluster->distances[rawIndex1]) {
        cluster->distances[rawIndex1] = cluster->distances[rawIndex2];
        if (rawIndex2 == cluster->rawIndexMinDistance)
          cluster->rawIndexMinDistance = rawIndex1; // the smallest element got moved
      }
      cluster->distances[rawIndex2] = -1;
    }

    cluster1->elevate(pcluster2, minDistance);
    delete cluster2->distances;
    cluster2->distances = NULL;
  }

  return *clusters;
}


TClusterW *THierarchicalClustering::merge_AverageLinkage(TClusterW **clusters)
{
  while((*clusters)->next) {
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

    if (rawIndex1) { // not root - has no distances...
      *disti1 = (*disti1 * size1 + *disti2 * size2) / sumsize;
      const float *minIndex1 = disti1;
      int ndi = cluster1->nDistances-1;
      for(disti1++, disti2++; ndi--; disti1++, disti2++)
        if (*disti1 >= 0) {
          *disti1 = (*disti1 * size1 + *disti2 * size2) / sumsize;
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
      distc = (distc * size1 + *disti2 * size2) / sumsize;
      if (distc < cluster->minDistance) {
        cluster->minDistance = distc;
        cluster->rawIndexMinDistance = rawIndex1;
      }
      else if ((distc > cluster->minDistance) && (cluster->rawIndexMinDistance == rawIndex1))
        cluster->computeMinimalDistance();
    }

    for(cluster = cluster->next; cluster; cluster = cluster->next) {
      float &distc = cluster->distances[rawIndex1];
      distc = (distc * size1 + cluster->distances[rawIndex2] * size2) / sumsize;
      if (distc < cluster->minDistance) {
        cluster->minDistance = distc;
        cluster->rawIndexMinDistance = rawIndex1;
      }
      else if (   (distc > cluster->minDistance) && (cluster->rawIndexMinDistance == rawIndex1)
               || (cluster->rawIndexMinDistance == rawIndex2)) {
        cluster->distances[rawIndex2] = -1;
        cluster->computeMinimalDistance();
      }
      else
        cluster->distances[rawIndex2] = -1;
    }

    cluster1->elevate(pcluster2, minDistance);
    delete cluster2->distances;
    cluster2->distances = NULL;
  }

  return *clusters;
}



TClusterW *THierarchicalClustering::merge_CompleteLinkage(TClusterW **clusters)
{
  while((*clusters)->next) {
    TClusterW *cluster, *cluster1, *cluster2;
    int rawIndex1, rawIndex2;

    float minDistance = numeric_limits<float>::max();
    for(cluster = *clusters; cluster; cluster = cluster->next)
      if (cluster->minDistance < minDistance) {
        minDistance = cluster->minDistance;
        cluster2 = cluster;
      }

    rawIndex1 = cluster2->rawIndexMinDistance;
    rawIndex2 = cluster2->nDistances;
    cluster1 = *cluster2->clusterAt(rawIndex2, clusters);
    
    const int newsize = cluster1->size + cluster2->size;

    float *disti1 = cluster1->distances;
    float *disti2 = cluster2->distances;

    for(int ndi = cluster1->nDistances; ndi--; disti1++, disti2++) {
      for(; *disti1 < 0; disti1++, disti2++);
      switch (linkage) {
        case Complete:
          if (*disti1 < *disti2)
            *disti1 = *disti2;
          break;

        case Single:
          if (*disti1 > *disti2)
            *disti1 = *disti2;
          break;

        case Average:
          *disti1 = (*disti1 * cluster1->size + *disti2 * cluster2->size)  / newsize;
          break;
      }
    }

    while(*disti2 < 0)
      disti2++;        // should have at least one more >=0  - the one corresponding to distance to cluster1

    for(cluster = cluster1->next; cluster != cluster2; cluster = cluster->next) {
      while(*++disti2 < 0); // should have more - the one corresponding to cluster
      switch (linkage) {
        case Complete:
          if (cluster->distances[rawIndex1] < *disti2)
            cluster->distances[rawIndex1] = *disti2;
          break;

        case Single:
          if (cluster->distances[rawIndex1] > *disti2)
            cluster->distances[rawIndex1] = *disti2;
          break;

        case Average:
          cluster->distances[rawIndex1] = (cluster1->distances[rawIndex1] * cluster1->size + *disti2 * cluster2->size)  / newsize;
          break;
      }

    }

    rawIndex2 = cluster2->nDistances;

    if (cluster->next) {
      for(cluster = cluster->next; cluster; cluster = cluster->next) {
        switch (linkage) {
          case Complete:
            if (cluster->distances[rawIndex1] < cluster->distances[rawIndex2])
              cluster->distances[rawIndex1] = cluster->distances[rawIndex2];
            break;

          case Single:
            if (cluster->distances[rawIndex1] > cluster->distances[rawIndex2])
              cluster->distances[rawIndex1] = cluster->distances[rawIndex2];
            break;

          case Average:
            cluster->distances[rawIndex1] = (cluster->distances[rawIndex1] * cluster1->size + cluster->distances[rawIndex2] * cluster2->size)  / newsize;
            break;
        }

        cluster->distances[rawIndex2] = -1;
      }
    }

// elevate
    delete cluster2->distances;
    cluster1->distances = cluster2->distances = NULL;
  }

  return *clusters;
}



TClusterW *THierarchicalClustering::merge(TClusterW **clusters)
{
  switch(linkage) {
    case Complete: return merge_CompleteLinkage(clusters);
    case Single: return merge_SingleLinkage(clusters);
    case Average: 
    default: return merge_AverageLinkage(clusters);
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
  TClusterW **clusters = init(distanceMatrix);
  TClusterW *root = merge(clusters);
  mldelete clusters;
  return restructure(root);
}
