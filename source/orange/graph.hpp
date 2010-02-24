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


#ifndef __GRAPH_HPP
#define __GRAPH_HPP

#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <set>
#include <queue>
#include <vector>
#include "root.hpp"

using namespace std;

/* Because we sometimes interpret the edge weight as a pointer,
   the weight for no connection should be a pointer that can never occur.
   0xfff...fff, which represents double NaN is a good idea. */

extern double _disconbuf;
extern double _disconbuf;

#define DISCONNECT(d) (d) = _disconbuf

// If anybody knows a better way to do this, please tell me ;)
#define CONNECTED(d) memcmp((void *)&d, (void *)&_disconbuf, sizeof(double))



class ORANGE_API TGraph : public TOrange
{
public:
  __REGISTER_ABSTRACT_CLASS

  int nVertices; //PR the number of vertices
  int nEdgeTypes; //PR the number of edge types
  bool directed; //PR directed

  int lastAddition;
  int lastRemoval;
  int currentVersion;

  TGraph(const int &nVert, const int &nEdge, const bool dir);
  virtual double *getEdge(const int &v1, const int &v2) = 0;
  virtual double *getOrCreateEdge(const int &v1, const int &v2) = 0;
  virtual void removeEdge(const int &v1, const int &v2) = 0;

  virtual void getNeighbours(const int &v, vector<int> &) = 0;
  virtual void getNeighboursFrom(const int &v, vector<int> &) = 0;
  virtual void getNeighboursTo(const int &v, vector<int> &) = 0;

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &) = 0;
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &) = 0;
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &) = 0;

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &) = 0;
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &) = 0;

  int findPath(int &u, int &v, int level, int &maxLevel, vector<int> &path);
  vector<int> getShortestPaths(int &u, int &v);
  set<int> getConnectedComponent(int &u);
  int getDiameter();
  void getClusters();
  vector<int> getLargestFullGraphs(vector<int> nodes, vector<int> candidates);
  double getClusteringCoefficient();
};

WRAPPER(Graph)

class ORANGE_API TGraphAsMatrix : public TGraph
{
public:
  __REGISTER_CLASS

  double *edges;
  const int msize;

  TGraphAsMatrix(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsMatrix();

  virtual double *getEdge(const int &v1, const int &v2);
  virtual double *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  double *findEdge(const int &v1, const int &v2);
  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};



class ORANGE_API TGraphAsList : public TGraph
{
public:
  __REGISTER_CLASS

  class ORANGE_API TEdge {
  public:
    TEdge *next;
    int vertex;
	int nEdges;
    double weights;
/*
	TEdge(const TEdge &other) :
	  vertex(other.vertex),
	  nEdges(other.nEdges) {
		memcpy(weight, other.weights, nEdges*sizeof(double));
	}*/
  };


  TEdge **edges;

  TGraphAsList(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsList();

  virtual double *getEdge(const int &v1, const int &v2);
  virtual double *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  TEdge *createEdge(TEdge *next, const int &vertex) const;
  bool findEdgePtr(const int &v1, const int &v2, TEdge **&, int &subvert);
  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};
  

class ORANGE_API TGraphAsTree : public TGraph
{
public:
  __REGISTER_CLASS

  class ORANGE_API TEdge {
  public:
    TEdge *left, *right;
    unsigned int vertex; // 0x7ffffff for number, high bit=set -> node is red
    double weights;

    ~TEdge();
  };

  TEdge **edges;

  TGraphAsTree(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsTree();

  virtual double *getEdge(const int &v1, const int &v2);
  virtual double *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  double *getEdge(TEdge *node, const int &subvert);
  TEdge *createEdge(const int &vertex) const;
  void sortIndices(const int &v1, const int &v2, TEdge **&e, int &subvert) const;

  void getNeighbours_fromTree(TEdge *edge, vector<int> &neighbours);
  void getNeighbours_fromTree(TEdge *edge, const int &edgeTypes, vector<int> &neighbours);
  void getNeighbours_fromTree_merge(TEdge *edge, vector<int> &neighbours, const int &v, int &latest);
  void getNeighbours_fromTree_merge(TEdge *edge, const int &edgeType, vector<int> &neighbours, const int &v, int &latest);

  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};

#endif
