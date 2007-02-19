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

#ifndef __GRAPHOPTIMIZATION_HPP
#define __GRAPHOPTIMIZATION_HPP

#include <math.h>
#include <stdio.h>
#include <iostream>
#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "numeric_interface.hpp"

using namespace std;

class ORANGEOM_API TGraphOptimization : public TOrange
{
public:
	__REGISTER_CLASS

	TGraphOptimization(int nVertices, double **pos, int nLinks, int **links);
	~TGraphOptimization();
	
	void fruchtermanReingold(int steps);
	void setData(int nVertices, double **pos, int nLinks, int **links);
	double getTemperature() {return temperature;}
	void setTemperature(double t) {temperature = t;}

	float k; //PR
	float k2; //PR
	double temperature;
	int width; //P
	int height; //PR

	int nLinks;
	int nVertices;
	int **links;
	double **pos;

	PyArrayObject *arrayX;
	PyArrayObject *arrayY;

    double attractiveForce(double x);
	double repulsiveForce(double x);
	double cool(double t);
};

#endif