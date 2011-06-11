/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Author: Miha Stajdohar, 1996--2010
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


#ifndef __GRAPH_LAYOUT_HPP
#define __GRAPH_LAYOUT_HPP

#include "Python.h"

#ifdef _MSC_VER
  /* easier to do some ifdefing here than needing to define a special
     include in every project that includes this header */
  #include "../lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#else
  #include <numpy/arrayobject.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include <queue>

#ifdef DARWIN
#include <strings.h>
#endif

#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "stringvars.hpp"
#include "table.hpp"
#include "symmatrix.hpp"

using namespace std;

class ORANGEOM_API TGraphLayout : public TOrange
{
public:
	__REGISTER_CLASS

	TGraphLayout();
	~TGraphLayout();
	int set_graph(PyObject *graph);
	void dump_coordinates();
	void dump_disp();
	void clear_disp();
	// graph layout optimization
	int random();
	void fr_repulsive_force(double kk2, int type);
	void fr_attractive_force(int type, bool weighted);
	void fr_limit_displacement();
	int fr(int steps, bool weighted);
	int fr_radial(int steps, int nCircles);
	int circular(int type);
	int circular_crossing_reduction();
	// coors
	double **ptrvector(int n);
	double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
	bool *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
	void free_Carrayptrs(double **v);
	// FR
	double k;
	double k2;
	double temperature;
	double coolFactor;
	double width;
	double height;
	// radial FR
	int *level;
	int radius;
	double *kVector;
	double *levelMin;
	double *levelMax;
	// graph
	int nLinks;
	int nVertices;
	vector<int> links[2];
	vector<double> disp[2];
	vector<double> weights;
	//set<int> vertices;
	double **pos;
	PyArrayObject *coors;	
};


class QueueVertex
{
public:
	int ndx;
	int position;
	unsigned int unplacedNeighbours;
	unsigned int placedNeighbours;
	vector<int> neighbours;

	bool hasNeighbour(int index)
	{
		vector<int>::iterator iter;

		for (iter = neighbours.begin(); iter != neighbours.end(); iter++)
			if (*iter == index)
				return true;

		return false;
	}

	friend ostream & operator<<(ostream &os, const QueueVertex &v)
	{
		os << "ndx: " << v.ndx << " unplaced: " << v.unplacedNeighbours << " placed: " << v.placedNeighbours << " neighbours: ";
		int i;
		for (i = 0; i < v.neighbours.size(); i++)
			os << v.neighbours[i] << " ";

		return (os);
	}

	QueueVertex(int index = -1, unsigned int neighbours = 0)
	{
		ndx = index;
		unplacedNeighbours = neighbours;
		placedNeighbours = 0;
	}

	bool operator () (const QueueVertex * a, const QueueVertex * b)
	{
		if (a->unplacedNeighbours < b->unplacedNeighbours)
			return false;
		else if (a->unplacedNeighbours > b->unplacedNeighbours)
			return true;
		else
		{
			return a->placedNeighbours < b->placedNeighbours;
		}
	}
};

#endif
