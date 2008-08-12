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

    Author: Miha Stajdohar, 1996--2002
*/

#ifndef __NETWORKOPTIMIZATION_HPP
#define __NETWORKOPTIMIZATION_HPP

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
#include <time.h>
#include <queue>

#ifdef DARWIN
#include <strings.h>
#endif

#include "px/orangeom_globals.hpp"
#include "root.hpp"
//#include "numeric_interface.hpp"
#include "network.hpp"
#include "stringvars.hpp"

using namespace std;

struct Edge
{
public:
	int u;
	int v;
};

class ORANGEOM_API TNetworkOptimization : public TOrange
{
public:
	__REGISTER_CLASS

	TNetworkOptimization();
	~TNetworkOptimization();

	void random();
	int fruchtermanReingold(int steps, bool weighted);
	int radialFruchtermanReingold(int steps, int nCircles);
	int smoothFruchtermanReingold(int steps, int center);
	int circular(int type);
	int circularCrossingReduction();
	//int circularRandom();
	double getTemperature() {return temperature;}
	void setTemperature(double t) {temperature = t;}
	int setNetwork(PNetwork net);
	void dumpCoordinates();

	double k;
	double k2;
	double temperature;
	double coolFactor;
	double width;
	double height;

	PNetwork network; //P Network

	int nLinks;
	int nVertices;
	vector<int> links[2];
	set<int> vertices;
	int *level;
	double *kVector;
	double *levelMin;
	double *levelMax;

	TGraphAsTree *tree;
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
