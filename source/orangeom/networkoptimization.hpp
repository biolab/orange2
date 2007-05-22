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
#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "numeric_interface.hpp"
#include "graph.hpp"
#include "stringvars.hpp"

using namespace std;

class ORANGEOM_API TNetworkOptimization : public TOrange
{
public:
	__REGISTER_CLASS

	TNetworkOptimization();
	~TNetworkOptimization();
	
	void random();
	int fruchtermanReingold(int steps);
	double getTemperature() {return temperature;}
	void setTemperature(double t) {temperature = t;}
	int setGraph(TGraphAsList *graph);
	void dumpCoordinates();

	double **ptrvector(long n);
	double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
	bool *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
	void free_Carrayptrs(double **v);
	void free_Links()
	{		
		if (links != NULL)
		{
			int i;
			for (i = 0; i < nLinks; i++)
			{
				if (links[i] != NULL)
				{
					free(links[i]);
					links[i] = NULL;
				}
			}

			free(links);
			links = NULL;
		}
	}

	float k; //PR
	float k2; //PR
	double temperature;
	int width; //P
	int height; //PR
	PyArrayObject *coors;
	PyArrayObject *filter;

	int nLinks;
	int nVertices;
	int **links;
	bool *nodes;
	double **pos;
    double attractiveForce(double x);
	double repulsiveForce(double x);
	double cool(double t);
	bool isFilter();
};

#endif