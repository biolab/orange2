#include "graphoptimization.h"

GraphOptimization::GraphOptimization(void)
{
	k = 1;
	k2 = 1;
	width = 1000;
	height = 1000;
	temperature = sqrt((double)(width*width + height*height)) / 10;
}

GraphOptimization::~GraphOptimization(void)
{
}

double GraphOptimization::attractiveForce(double x)
{
	return x * x / k;

}

double GraphOptimization::repulsiveForce(double x)
{
	if (x == 0)
		return k2 / 1;

	return   k2 / x; 
}

double GraphOptimization::cool(double t)
{
	return t * 0.98;
}

void GraphOptimization::fruchtermanReingold(int steps, int nVertices, double **pos, int nLinks, int **links)
{
	int i = 0;
	int count = 0;
	double kk = 1;
	double **disp = (double**)malloc(nVertices * sizeof (double));

	for (i = 0; i < nVertices; i++)
	{
		disp[i] = (double *)calloc(2, sizeof(double));

		if (disp[i] == NULL)
		{
			cerr << "Couldn't allocate memory\n";
			exit(1);
		}
	}

	int area = width * height;
	k2 = area / nVertices;
	k = sqrt(k2);
	kk = 2 * k;

	// iterations
	for (i = 0; i < steps; i++)
	{
		// reset disp
		int j = 0;
		for (j = 0; j < nVertices; j++)
		{
			disp[j][0] = 0;
			disp[j][1] = 0;
		}

		// calculate repulsive force
		int v = 0;
		for (v = 0; v < nVertices - 1; v++)
		{
			for (int u = v + 1; u < nVertices; u++)
			{
				double difX = pos[v][0] - pos[u][0];
				double difY = pos[v][1] - pos[u][1];

				double dif = sqrt(difX * difX + difY * difY);

				if (dif == 0)
					dif = 1;

				if (dif < kk)
				{
					disp[v][0] = disp[v][0] + ((difX / dif) * repulsiveForce(dif));
					disp[v][1] = disp[v][1] + ((difY / dif) * repulsiveForce(dif));

					disp[u][0] = disp[u][0] - ((difX / dif) * repulsiveForce(dif));
					disp[u][1] = disp[u][1] - ((difY / dif) * repulsiveForce(dif));
				}
			}
		}

		// calculate attractive forces
		for (j = 0; j < nLinks; j++)
		{
			int v = links[j][0];
			int u = links[j][1];

			//cout << "v: " << v << " u: " << u << endl;

			// cout << "     v: " << v << " u: " << u << " w: " << edge->weights << endl;
			
			double difX = pos[v][0] - pos[u][0];
			double difY = pos[v][1] - pos[u][1];

			double dif = sqrt(difX * difX + difY * difY);

			if (dif == 0)
				dif = 1;

			disp[v][0] = disp[v][0] - ((difX / dif) * attractiveForce(dif));
			disp[v][1] = disp[v][1] - ((difY / dif) * attractiveForce(dif));

			disp[u][0] = disp[u][0] + ((difX / dif) * attractiveForce(dif));
			disp[u][1] = disp[u][1] + ((difY / dif) * attractiveForce(dif));
		}

		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		for (v = 0; v < nVertices; v++)
		{
			double dif = sqrt(disp[v][0] * disp[v][0] + disp[v][1] * disp[v][1]);

			if (dif == 0)
				dif = 1;

			pos[v][0] = pos[v][0] + ((disp[v][0] / dif) * min(fabs(disp[v][0]), temperature));
			pos[v][1] = pos[v][1] + ((disp[v][1] / dif) * min(fabs(disp[v][1]), temperature));

			//pos[v][0] = min((double)width,  max((double)0, pos[v][0]));
			//pos[v][1] = min((double)height, max((double)0, pos[v][1]));
		}
		
		temperature = cool(temperature);
	}

	// free space
	for (i = 0; i < nVertices; i++)
	{
		free(disp[i]);
	}

	free(disp);
}