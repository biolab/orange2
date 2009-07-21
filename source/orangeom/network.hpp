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
    Contact: miha.stajdohar@fri.uni-lj.si
*/


#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "Python.h"

#ifdef _MSC_VER
  /* easier to do some ifdefing here than needing to define a special
     include in every project that includes this header */
  #include "../lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#else
  #include <numpy/arrayobject.h>
#endif

#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <queue>
#include <vector>
#include <math.h>
#include <string>
#include <time.h>

#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "graph.hpp"
#include "table.hpp"
#include "symmatrix.hpp"
#include "stringvars.hpp"

#ifdef DARWIN
#include <strings.h>
#endif

using namespace std;

WRAPPER(ExampleTable)
WRAPPER(SymMatrix)

class TNetworkHierarchyNode
{
public:
	TNetworkHierarchyNode();
	~TNetworkHierarchyNode();

  int getLevel();

	//vector<int> vertices;
  int vertex;
	vector<TNetworkHierarchyNode *> childs;
	TNetworkHierarchyNode *parent;
};

class TNetworkHierarchy
{
public:
	TNetworkHierarchy();
	TNetworkHierarchy(vector<int> &topVertices);
	~TNetworkHierarchy();
	void setTop(vector<int> &vertices);
	void addToNewMeta(vector<int> &vertices);
	void expandMeta(int meta);
	void printChilds(TNetworkHierarchyNode *node);
	int getNextMetaIndex();
	int getMetaChildsCount(TNetworkHierarchyNode *node);
	int getMetasCount();

  	int meta_index;
	TNetworkHierarchyNode *top;
	TNetworkHierarchyNode *getNodeByVertex(int vertex);
	TNetworkHierarchyNode *getNodeByVertex(int vertex, TNetworkHierarchyNode &start);
};

OMWRAPPER(Network)

class ORANGEOM_API TNetwork : public TGraphAsList
{
public:
  __REGISTER_CLASS

  TNetwork(TNetwork *net);
  TNetwork(TGraphAsList *graph);
  TNetwork(const int &nVert, const int &nEdge, const bool dir);
  ~TNetwork();

  void printHierarchy();
  void hideVertices(vector<int> vertices);
  void showVertices(vector<int> vertices);
  void showAll();

  double **ptrvector(int n);
  double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
  bool *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
  void free_Carrayptrs(double **v);

  double **pos;
  PyArrayObject *coors;

  PExampleTable items; //P ExampleTable of vertices data
  PExampleTable links; //P ExampleTable of edges data
  TNetworkHierarchy hierarchy;
  set<int> optimize;
};

#endif
