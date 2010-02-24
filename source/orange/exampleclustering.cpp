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


#include "values.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "lookup.hpp"
#include "table.hpp"
#include "bayes.hpp"

#include "lookup.hpp"
#include "exampleclustering.ppp"


PClassifier completeTable(PExampleGenerator examples, int completion, int weightID)
{
  if (!examples->domain->classVar)
    raiseError("completeTable: class-less domain");
  if (examples->domain->classVar!=TValue::INTVAR)
    raiseError("completeTable: discrete class expected");

  switch(completion) {

    case completion_default:
      return mlnew TDefaultClassifier(examples->domain->classVar, getClassDistribution(examples, weightID));

    case completion_bayes: {
      TBayesLearner bayes;
      return bayes(examples, weightID);
    }

    default: 
      return PClassifier();
  }
}



TExampleCluster::TExampleCluster()
: distance(0.0)
{}


TExampleCluster::TExampleCluster(PExample ac)
: distance(0.0),
  centroid(ac)
{}
  
  
TExampleCluster::TExampleCluster(PExampleCluster al, PExampleCluster ar, const float &dist)
: left(al),
  right(ar),
  distance(dist)
{}



TExampleCluster::TExampleCluster(vector<PExampleCluster> &group, const float &dist)
{
  if (!group.size())
    raiseError("invalid cluster group");

  vector<PExampleCluster> group2;
  vector<PExampleCluster>::iterator gc2(group.begin()), ge(group.end());
  while(gc2!=ge) {
    PExampleCluster &left = *(gc2++);
    if (gc2==ge) {
      group2.push_back(left);
      break;
    }
    group2.push_back(mlnew TExampleCluster(left, *(gc2++), dist));
  }

  vector<PExampleCluster>::iterator gc;
  while (group2.size()>1) {
    for(gc2=gc=group2.begin(), ge=group2.end(); gc2!=ge; gc2++, gc++) {
      PExampleCluster &left = *(gc2++);
      if (gc2==ge) {
        *(gc++) = left;
        break;
      }
      *gc = mlnew TExampleCluster(left, *gc2, dist);
    }
    for(gc2=gc; gc2!=ge; gc2++)
      *gc2 = PExampleCluster();
    group2.erase(gc, ge);
  }

  PExampleCluster &rem = group2.front();
  left = rem->left;
  right = rem->right;
  distance = rem->distance;
  centroid = rem->centroid;
}



PClassifier TGeneralExampleClustering::classifier(const float &cut, const int &completion) const
{
  PExampleSets clusters = exampleSets(cut);

  TEnumVariable *eclassVar = mlnew TEnumVariable("");
  const TVarList *attributes = NULL;
  PVariable classVar(eclassVar);

  bool isLong = false;

  PITERATE(TExampleSets, ci, clusters) {
    string value = "";

    PEITERATE(ei, *ci) {
      if (!attributes)
        attributes = (*ei).domain->attributes.getUnwrappedPtr();
      if (!value.empty())
        value+="+";

      TExample::const_iterator xi = (*ei).begin(), exi=xi;
      const_PITERATE(TVarList, vi, attributes) {
        if (exi!=xi)
          value+="-";
        string nval;
        (*vi)->val2str(*(xi++), nval);
        if ((nval.find("-")!=string::npos) || (nval.find("+")!=string::npos)) {
          value = "";
          isLong = true;
          break;
        }
        value += nval;
      }
      if (isLong)
        break;
    }
    if (isLong)
      break;
  
    if (!value.empty())
      eclassVar->addValue(value);
    else {
      char vbuf[12];
      sprintf(vbuf, "c%d", ci-clusters->begin());
      eclassVar->addValue(vbuf);
    }
  }

  if (isLong) {
    eclassVar->values->clear();
    int i = 1;
    PITERATE(TExampleSets, ci, clusters) {
      char vbuf[12];
      sprintf(vbuf, "c%d", i++);
      eclassVar->addValue(vbuf);
    }
  }

  if (!attributes)
    raiseError("no clusters");

  // *** ONE BOUND VARIABLE ***  
  if (attributes->size()==1) {
    TClassifierByLookupTable1 *cblt=mlnew TClassifierByLookupTable1(classVar, attributes->front());
    PClassifier wcblt = cblt;
    int cluster = 0;
    TDiscDistribution classDist;
    PITERATE(TExampleSets, ci, clusters) {
      PEITERATE(eci, *ci) {
        cblt->lookupTable->at((*eci).values->intV) = TValue(cluster);
        cblt->distributions->at((*eci).values->intV)->addint(cluster, 1.0);
        classDist.addint(cluster);
      }
      cluster++;
    }
    if (completion)
      cblt->replaceDKs(classDist);

    return wcblt;
  }

  // *** MORE THAN ONE BOUND VARIABLE ***  
  else {
    PExampleGenerator examples;
    PEFMDataDescription dataDes;

    PDomain domain = mlnew TDomain(classVar, *attributes);

    if (completion || (attributes->size()==3)) {
      TExampleTable *table = mlnew TExampleTable(domain);
      examples = PExampleGenerator(table);
      int cluster = 0;
      PITERATE(TExampleSets, ci, clusters) {
        PEITERATE(eci, *ci) {
          TExample ex(domain, *eci);
          ex.setClass(TValue(cluster));
          table->addExample(ex);
        }
        cluster++;
      }

      if (domain->attributes->size()<=3)
        dataDes = mlnew TEFMDataDescription(domain, mlnew TDomainDistributions(examples), 0, getMetaID());
    }

    switch (attributes->size()) {
      case 2: {
        TClassifierByLookupTable2 *cblt = mlnew TClassifierByLookupTable2(classVar, attributes->front(), attributes->back(), dataDes);
        PClassifier wcblt = cblt;
        int cluster=0;
        PITERATE(TExampleSets, ci, clusters) {
          PEITERATE(eci, *ci) {
            int index = cblt->getIndex(*eci);
            if (index>=0) {
              cblt->lookupTable->operator[](index) = TValue(cluster);
              cblt->distributions->operator[](index)->addint(cluster);
            }
          }
          cluster++;
        }

        if (completion)
          cblt->replaceDKs(examples, completion==completion_bayes);
        return wcblt;
      }

      case 3: {
        TClassifierByLookupTable3 *cblt = mlnew TClassifierByLookupTable3(classVar, (*attributes)[0], (*attributes)[1], (*attributes)[2], dataDes);
        PClassifier wcblt = cblt;
        int cluster = 0;
        PITERATE(TExampleSets, ci, clusters) {
          PEITERATE(eci, *ci) {
            int index = cblt->getIndex(*eci);
            if (index>=0) {
              cblt->lookupTable->operator[](index) = TValue(cluster);
              cblt->distributions->operator[](index)->addint(cluster);
            }
          }
          cluster++;
        }

        if (completion)
          cblt->replaceDKs(examples, completion==completion_bayes);
        return wcblt;
      }

      default: {
        static TLookupLearner lookupLearner;
        PClassifier cfgen = lookupLearner(examples);
        cfgen.AS(TClassifierByExampleTable)->classifierForUnknown = completeTable(examples, completion);
        return cfgen;
      }
    }
  }
}


PVariable TGeneralExampleClustering::feature(const float &cut, const int &completion) const
{ 
  PClassifier cl = classifier(cut, completion);
  if (!cl || !cl->classVar)
        return PVariable();
	  
  cl->classVar->getValueFrom = cl;
  return cl->classVar;
}



TExampleClusters::TExampleClusters()
: quality(0.0)
{}


TExampleClusters::TExampleClusters(PExampleCluster aroot, const float &aq)
: root(aroot),
  quality(aq)
{}


PExampleClusters TExampleClusters::exampleClusters() const
{ return mlnew TExampleClusters(root, quality); }



void mergeCluster(TExampleTable *&table, const PExampleCluster &cluster)
{
  if (cluster->centroid) {
    if (!table)
      table = mlnew TExampleTable(cluster->centroid->domain);
    table->addExample(cluster->centroid.getReference());
  }

  if (cluster->left)
    mergeCluster(table, cluster->left);
  if (cluster->right)
    mergeCluster(table, cluster->right);
}

void descend(TExampleSets *clusters, const PExampleCluster &cluster, const float &cut)
{
  if ((cluster->distance < cut)  || (!cluster->left && !cluster->right)) {
    TExampleTable *table = NULL;
    mergeCluster(table, cluster);
    if (table)
      clusters->push_back(table);
  }

  else {
    if (cluster->left)    
      descend(clusters, cluster->left, cut);
    if (cluster->right)
      descend(clusters, cluster->right, cut);
  }
}
    
PExampleSets TExampleClusters::exampleSets(const float &cut) const
{
  checkProperty(root);

  TExampleSets *sets = mlnew TExampleSets();
  PExampleSets psets = sets;
  descend(sets, root, cut);
  return psets;
}
