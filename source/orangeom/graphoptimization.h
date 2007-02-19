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

#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;

class GraphOptimization
{
public:
	GraphOptimization(void);
	~GraphOptimization(void);
	
	void fruchtermanReingold(int steps, int nVertices, double **pos, int nLinks, int **links);
	double getTemperature() {return temperature;}
	void setTemperature(double t) {temperature = t;}

private:
	double k;
	double k2;
	double temperature;
	int width;
	int height;
    double attractiveForce(double x);
	double repulsiveForce(double x);
	double cool(double t);
};
