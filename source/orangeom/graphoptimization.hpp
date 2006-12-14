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