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


#include <queue>
#include "stladdon.hpp"

#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "measures.hpp"
#include "distance.hpp"
#include "classify.hpp"

#include "minimal_complexity.ppp"



TIGNode::TIGNode()
: example()
{}


TIGNode::TIGNode(PExample ex)
: example(ex)
{}


TIGNode::TIGNode(PExample ex, const TDiscDistribution &inc, const TDiscDistribution &comp)
: example(ex),
  incompatibility(inc),
  compatibility(comp)
{}




TIG::TIG()
: checkedForEmpty(false)
{}


int TIG::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  const_ITERATE(vector<TIGNode>, pi, nodes)
    PVISIT((*pi).example);
  return 0;
}


int TIG::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);
  nodes.clear();
  return 0;
}


void TIG::removeEmpty()
{ if (checkedForEmpty)
    return;
  checkedForEmpty = true;

  vector<bool> emptyNodes;
  int nonEmpty = 0;
  ITERATE(vector<TIGNode>, ni, nodes) {
    emptyNodes.push_back(!(*ni).incompatibility.size() && !(*ni).compatibility.size());
    if (emptyNodes.back())
      nonEmpty++;
  }
  if (nonEmpty==int(nodes.size()))
    return;

  // In place removal of non-connected nodes, among with reduction of incompatibility vectors of
  // nodes that remain.
  vector<bool>::iterator emi = emptyNodes.begin();
  vector<TIGNode>::iterator write(nodes.begin()), read(nodes.begin());
  for(; read!=nodes.end(); read++, emi++)
    if (!*emi) {
      TDiscDistribution newDis, newCom;
      newDis.reserve(nonEmpty);
      newCom.reserve(nonEmpty);
      TDiscDistribution::iterator ori((*read).incompatibility.begin()), orie((*read).incompatibility.end()),
                                  orc((*read).compatibility.begin()),   orce((*read).compatibility.end());
      ITERATE(vector<bool>, ci, emptyNodes) {
        if (!*ci) { 
          if (ori!=orie) {
            newDis.push_back(*ori);
            newDis.abs+=*ori;
          }

          if (orc!=orce) {
            newCom.push_back(*orc);
            newCom.abs+=*orc;
          }
        }

        if (ori!=orie)
          ori++;
        if (orc!=orce)
          orc++;
      }

      *(write++) = TIGNode((*read).example, newDis, newCom);
    }
      
  nodes.erase(write, nodes.end());
}


void TIG::make0or1()
{ ITERATE(vector<TIGNode>, ni, nodes) {
    int abs = 0;
    ITERATE(TDiscDistribution, di, (*ni).incompatibility)
      if (*di>0) {
        *di=1.0;
        abs++;
      }
      else
        *di=0.0;

    (*ni).incompatibility.abs = abs;
  }
}


void TIG::normalize()
{ ITERATE(vector<TIGNode>, ni, nodes) 
    (*ni).incompatibility.normalize();
}


void TIG::complete()
{ int nn = nodes.size();
  ITERATE(vector<TIGNode>, ni, nodes) {
    (*ni).incompatibility[nn-1];
    (*ni).compatibility[nn-1];
  }
}




PIG TIGByIM::operator()(PExampleGenerator , TVarList &, const int &)
{ raiseError("not implemented"); 
  return PIG();
}




PIG TIGBySorting::operator()(PExampleGenerator gen, TVarList &aboundSet, const int &)
{
  // Identify bound attributes
  vector<bool> bound = vector<bool>(gen->domain->attributes->size(), false);
  vector<bool> free = vector<bool>(gen->domain->attributes->size(), true);
  { ITERATE(TVarList, evi, aboundSet) {
      int vn = gen->domain->getVarNum(*evi);
      bound[vn] = true;
      free[vn] = false;
    }
  }

  // prepare a sorted table with examples and graph node indices
  TSortedExamples_nodeIndices sorted(gen, bound, free);

  // append the class as bound (ie. not free)
  bound.push_back(true);

  // construct the graph
  PDomain graphDomain = PDomain(mlnew TDomain(PVariable(), aboundSet));
  PIG graph = mlnew TIG();
  graph->nodes = vector<TIGNode>(sorted.maxIndex+1);
  TRandomGenerator rgen(sorted.size());
  ITERATE(vector<TIGNode>, in, graph->nodes)
    in->randint = rgen.randint();
    
  ITERATE(TSortedExamples_nodeIndices, ni, sorted) {
    TIGNode &gnode = graph->nodes[(*ni).nodeIndex];
    if (!gnode.example)
      gnode.example = mlnew TExample(graphDomain, (*ni).example.getReference());
  }
 
  TSortedExamples_nodeIndices::iterator ebegin(sorted.begin()), eend(sorted.end());
  for(TSortedExamples_nodeIndices::iterator grpbeg(ebegin), grpend(ebegin); grpbeg!=eend; ) {
    // Skip any DC values and stop if the end of the rule table is reached
    for(;(grpbeg!=eend) && (*grpbeg).example->getClass().isSpecial(); ++grpbeg);
    if (grpbeg==eend) 
      break;

    // Mark end of same free set
    TExample::iterator fbi((*grpbeg).example->begin()), fei((*grpbeg).example->end());
    for(grpend=grpbeg; (++grpend!=eend);) {
      vector<bool>::iterator bi(bound.begin());
      TExample::iterator ei1(fbi);
      for(TExample::iterator  ei2((*grpend).example->begin());
          (ei1!=fei) && (*bi || (*ei1==*ei2)); ei1++, ei2++, bi++);
      if (ei1!=fei) 
        break;
    }
    // Rules from [grpbeg, grpend) have same free attributes and are ordered by class value
      
    while (1) {
      // Find the end of the subgroup
      int oldfv = (*grpbeg).example->getClass().intV; 
      vector<TExample_nodeIndex>::iterator sgbe(grpbeg);
      while((++sgbe!=grpend) && ((*sgbe).example->getClass().intV==oldfv));

      for(TSortedExamples_nodeIndices::iterator cp1(grpbeg); cp1!=sgbe; cp1++)
        for(TSortedExamples_nodeIndices::iterator cp2(cp1); cp2!=sgbe; cp2++) {
          graph->nodes[(*cp1).nodeIndex].compatibility.addint((*cp2).nodeIndex);
          graph->nodes[(*cp2).nodeIndex].compatibility.addint((*cp1).nodeIndex);
        }

      // If that's the last subgroup, skip to the next group
      if (sgbe==grpend)
        break;      

      // Add [gbi, sbe)x[sbe, gbe) to the edges
      for(; grpbeg!=sgbe; grpbeg++)
        for(TSortedExamples_nodeIndices::iterator cp(sgbe); cp!=grpend; cp++) {
          graph->nodes[(*cp).nodeIndex].incompatibility.addint((*grpbeg).nodeIndex);
          graph->nodes[(*grpbeg).nodeIndex].incompatibility.addint((*cp).nodeIndex);
        }
    }

    grpbeg = grpend;
  }

  return graph;
}




TColoredIG::TColoredIG(PIG anig)
: ig(anig),
  colors(mlnew TIntList(anig->nodes.size(), -1))
{}



/* This is rather unnatural - graph coloring is not hierarchical
   clustering. However, we can give groups of different colors a
   distance of inf and examples in the same group have a distance
   of 0. The quality is -#colors. The order of groups and of
   examples within a group is meaningless. */
PExampleClusters TColoredIG::exampleClusters() const
{ 
  vector<PExampleCluster> groups;
  int colore = *max_element(colors->begin(), colors->end())+1;
  for(int color = 0; color!=colore; color++) {
    vector<PExampleCluster> group;

    TIntList::const_iterator ci(colors->begin()), ce(colors->end());
    vector<TIGNode>::const_iterator ni(ig->nodes.begin());
    for(; ci!=ce; ci++, ni++)
      if (*ci == color)
        group.push_back(mlnew TExampleCluster(mlnew TExample((*ni).example.getReference())));

    if (group.size())
      groups.push_back(mlnew TExampleCluster(group, 0.0));
  }

  return mlnew TExampleClusters(mlnew TExampleCluster(groups, numeric_limits<float>::infinity()), -colore);
}



PExampleSets TColoredIG::exampleSets(const float &) const
{
  PExampleSets exclusters = mlnew TExampleSets();
  if (!ig->nodes.size())
    return exclusters;

  vector<TExampleTable *> tables;
  PDomain dom = ig->nodes.front().example->domain;
  for(int i = *max_element(colors->begin(), colors->end())+1; i--; ) {
    TExampleTable *nt = mlnew TExampleTable(dom);
    exclusters->push_back(nt);
    tables.push_back(nt);
  }

  TIntList::const_iterator ci(colors->begin()), ce(colors->end());
  vector<TIGNode>::const_iterator ni(ig->nodes.begin());
  for(; ci!=ce; ci++, ni++)
    tables[*ci]->addExample((*ni).example.getReference());

  return exclusters;
}





class T__LessConnected {
public:
  PIG graph;

  T__LessConnected(PIG gr)
  : graph(gr)
  {}

  bool operator()(const int &n1, const int &n2) const
  { float ab1 = graph->nodes[n1].incompatibility.abs,
          ab2 = graph->nodes[n2].incompatibility.abs;
    return (ab1<ab2) || ((ab1==ab2) && (graph->nodes[n1].randint < graph->nodes[n2].randint)); 
  }
};


PColoredIG TColorIG_MCF::operator()(PIG graph)
{ 
  graph->removeEmpty();
  PColoredIG colored(mlnew TColoredIG(graph));

  if (graph->nodes.empty())
    return colored;

  TIntList &colors = colored->colors.getReference();
  int maxColor = -1;

  /* Construct a priority queue, with nodes priority corresponding
     to the number of its edges
  */
  typedef priority_queue<int, vector<int>, T__LessConnected> cpq;
  T__LessConnected lcg(graph);
  cpq orderedNodes = cpq(lcg);
  for(int ni = 0, nr = graph->nodes.size(); ni<nr; ni++)
    orderedNodes.push(ni);

  /* Initialize 'forbiddenColors' so that no color is forbidden
     to any node
  */
  vector<TDiscDistribution> forbiddenColors;
  int nonodes=graph->nodes.size();
  forbiddenColors.reserve(nonodes);
  while(nonodes--)
    forbiddenColors.push_back(TDiscDistribution());

  /* 'coloredBy' will be a vector of vectors; for each color, it would contain 
     the node numbers colored by it. This is used to decide for a color when
     multiple colors are allowed.
  */
  vector<vector<int> > coloredBy; 

  while (!orderedNodes.empty()) {
    int nodeNo = orderedNodes.top();
    orderedNodes.pop();

    /* Decide the color for thisNode. For each non-forbidden colors we will
       check the nodes colored by it. Compatibilities of those nodes with the
       node that is to be colored are summed and the color with the most
       compatible nodes is chosen.   
    */
    float bestComp = -1;
    int bestColor = -1;
    TDiscDistribution &compref = graph->nodes[nodeNo].compatibility;
    TDiscDistribution::iterator fi(forbiddenColors[nodeNo].begin()), fie(forbiddenColors[nodeNo].end());
    vector<vector<int> >::iterator colori(coloredBy.begin());
    for(int color = 0; color<int(coloredBy.size()); color++, colori++)
      if ((fi==fie) || (*(fi++)==0)) {
        float thisComp = 0;
        ITERATE(vector<int>, cni, *colori)
          if (int(compref.size())>*cni)
            thisComp += compref[*cni];
        if (thisComp>bestComp) {
          bestComp = thisComp;
          bestColor = colori - coloredBy.begin();
        }
      }

    /* Color the node - create a new color if needed
    */
    if (bestColor<0) {
      bestColor = ++maxColor;
      coloredBy.push_back(vector<int>(1, nodeNo));
    }
    else
      coloredBy[bestColor].push_back(nodeNo);

    colors[nodeNo] = bestColor;

    /* Forbid this color to incompatible nodes
    */
    int inode = 0;
    ITERATE(TDiscDistribution, ii, graph->nodes[nodeNo].incompatibility) {
      if (*ii>0)
        forbiddenColors[inode].addint(bestColor, *ii);
      inode++;
    }
  }

  return colored;
}



TFeatureByMinComplexity::TFeatureByMinComplexity(PColorIG cIG, const int &comp)
: colorIG(cIG),
  completion(comp)
{}


PVariable TFeatureByMinComplexity::operator()(PExampleGenerator gen, TVarList &boundSet, const string &name, float &quality, const int &weight)
{ PIG graph = TIGBySorting()(gen, boundSet, weight);
  if (!graph->nodes.size())
    raiseError("empty incompatibility graph");

  PVariable feat = (colorIG ? colorIG->call(graph) : TColorIG_MCF()(graph)) ->feature(0.0, completion);
  if (!feat)
    return PVariable();

  feat->set_name(name);

  quality = -feat->noOfValues();
  if (quality==1)
    quality = ATTRIBUTE_REJECTED;

  return feat;
}
