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
