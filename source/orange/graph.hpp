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


#ifndef __GRAPH_HPP
#define __GRAPH_HPP

#include <vector>
#include "root.hpp"

#define GRAPH__NO_CONNECTION 1e-30f

class TGraph : public TOrange
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
  virtual float *getEdge(const int &v1, const int &v2) = 0;
  virtual float *getOrCreateEdge(const int &v1, const int &v2) = 0;
  virtual void removeEdge(const int &v1, const int &v2) = 0;

  virtual void getNeighbours(const int &v, vector<int> &) = 0;
  virtual void getNeighboursFrom(const int &v, vector<int> &) = 0;
  virtual void getNeighboursTo(const int &v, vector<int> &) = 0;

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &) = 0;
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &) = 0;
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &) = 0;

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &) = 0;
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &) = 0;
};

WRAPPER(Graph)

class TGraphAsMatrix : public TGraph
{
public:
  __REGISTER_CLASS

  float *edges;

  TGraphAsMatrix(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsMatrix();

  virtual float *getEdge(const int &v1, const int &v2);
  virtual float *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  float *findEdge(const int &v1, const int &v2);
  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};



class TGraphAsList : public TGraph
{
public:
  __REGISTER_CLASS

  class TEdge {
  public:
    TEdge *next;
    int vertex;
    float weights;
  };


  TEdge **edges;

  TGraphAsList(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsList();

  virtual float *getEdge(const int &v1, const int &v2);
  virtual float *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  TEdge *createEdge(TEdge *next, const int &vertex);
  bool findEdgePtr(const int &v1, const int &v2, TEdge **&, int &subvert);
  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};
  

class TGraphAsTree : public TGraph
{
public:
  __REGISTER_CLASS

  class TEdge {
  public:
    TEdge *left, *right;
    unsigned int vertex; // 0x7ffffff for number, high bit=set -> node is red
    float weights;

    ~TEdge();
  };

  TEdge **edges;

  TGraphAsTree(const int &nVert, const int &nEdge, const bool dir);
  ~TGraphAsTree();

  virtual float *getEdge(const int &v1, const int &v2);
  virtual float *getOrCreateEdge(const int &v1, const int &v2);
  virtual void removeEdge(const int &v1, const int &v2);

  virtual void getNeighbours(const int &v, vector<int> &);
  virtual void getNeighboursFrom(const int &v, vector<int> &);
  virtual void getNeighboursTo(const int &v, vector<int> &);

  virtual void getNeighbours(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursFrom(const int &v, const int &edgeType, vector<int> &);
  virtual void getNeighboursTo(const int &v, const int &edgeType, vector<int> &);

  virtual void getNeighboursFrom_Single(const int &v, vector<int> &);
  virtual void getNeighboursFrom_Single(const int &v, const int &edgeType, vector<int> &);

  float *getEdge(TEdge *node, const int &subvert);
  TEdge *createEdge(const int &vertex);
  void sortIndices(const int &v1, const int &v2, TEdge **&e, int &subvert) const;

  void getNeighbours_fromTree(TEdge *edge, vector<int> &neighbours);
  void getNeighbours_fromTree(TEdge *edge, const int &edgeTypes, vector<int> &neighbours);
  void getNeighbours_fromTree_merge(TEdge *edge, vector<int> &neighbours, const int &v, int &latest);
  void getNeighbours_fromTree_merge(TEdge *edge, const int &edgeType, vector<int> &neighbours, const int &v, int &latest);

  void getNeighbours_Undirected(const int &v, vector<int> &neighbours);
  void getNeighbours_Undirected(const int &v, const int &edgeType, vector<int> &neighbours);
};

#endif