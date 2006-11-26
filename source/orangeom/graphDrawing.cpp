#include "numeric_interface.cpp"
#include "orange_api.hpp"
#include "graph.hpp"
#include "../orange/px/externs.px"
#include <stdio.h>
#include <iostream>

void dumpCoordinates(double **pos, int columns, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			cout << pos[i][j] << "  ";
		}

		cout << endl;
	}
}

double k;
double k2;

double attractiveForce(double x)
{
	return x * x / k;

}

double repulsiveForce(double x)
{
	if (x == 0)
		return k2 / 0.1;

	return k2 / x; 
}

int cool(int t)
{
	return t;
}


PyObject *graphOptimization(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(Graph, steps, coorX, coorY, width|def=100, height|def=100) -> None")
{
  PyTRY
    int steps;
	int width = 100;
	int height = 100;
	double temperature = 10;
    PyObject *pygraph;
	PyObject *pycoorX;
	PyObject *pycoorY;

	if (!PyArg_ParseTuple(args, "OiOO|ii:graphOptimization", &pygraph, &steps, &pycoorX, &pycoorY, &width, &height))
	{
		//PyErr_Clear
      return NULL;
	}

	/*
	if (pycoorY == Py_None)
		// pycoorX ima dva stolpca
	else
	    // vsak po enega
	*/

    TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

    if (graph->nVertices < 2)
      PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

	cout << "Vertices: " << graph->nVertices << endl;

	int rows;
	double *coorX;
	double *coorY;

	numericToDouble(pycoorX, coorX, rows);
	numericToDouble(pycoorY, coorY, rows);

	if (graph->nVertices != rows)
      PYERROR(PyExc_AttributeError, "graph nodes are not equal to coordinates", NULL);

	double **pos = (double**)malloc(rows * sizeof (double));
	double **disp = (double**)malloc(rows * sizeof (double));

	if ((pos == NULL) && (disp == NULL))
	{
		cerr << "Couldn't allocate memory\n";
		exit(1);
	}

	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		pos[i] = (double *)malloc(2 * sizeof(double));
		disp[i] = (double *)malloc(2 * sizeof(double));

		if ((pos[i] == NULL) && (disp[i] == NULL))
		{
			cerr << "Couldn't allocate memory\n";
			exit(1);
		}

		pos[i][0] = (double)coorX[i];
		pos[i][1] = (double)coorY[i];

		disp[i][0] = 0;
		disp[i][1] = 0;
	}
	
	int area = width * height;
	k = sqrt((double)(area / graph->nVertices));
	k2 = area / graph->nVertices;

	// iterations
	for (int i = 0; i < steps; i++)
	{
		// calculate repulsive force
		for (int v = 0; v < graph->nVertices -1 ; v++)
		{
			for (int u = v + 1; u < graph->nVertices; u++)
			{
				double difX = pos[v][0] - pos[u][0];
				double difY = pos[v][1] - pos[u][1];

				double dif = sqrt(difX*difX + difY*difY);

				if (dif == 0)
					dif = 1;

				disp[v][0] = disp[v][0] + (difX / dif * repulsiveForce(dif));
				disp[v][1] = disp[v][1] + (difY / dif * repulsiveForce(dif));

				disp[u][0] = disp[u][0] - (difX / dif * repulsiveForce(dif));
				disp[u][1] = disp[u][1] - (difY / dif * repulsiveForce(dif));
			}
		}

		// calculate attractive forces
		for (int v = 0; v < graph->nVertices; v++)
		{
			TGraphAsList::TEdge *edge = graph->edges[v];

			if (edge != NULL)
			{
				int u = edge->vertex;
				// cout << "     v: " << v << " u: " << u << " w: " << edge->weights << endl;
				
				double difX = pos[v][0] - pos[u][0];
				double difY = pos[v][1] - pos[u][1];

				double dif = sqrt(difX*difX + difY*difY);

				if (dif == 0)
					dif = 1;

				disp[v][0] = disp[v][0] - (difX / dif * attractiveForce(dif));
				disp[v][1] = disp[v][1] - (difY / dif * attractiveForce(dif));

				disp[u][0] = disp[u][0] + (difX / dif * attractiveForce(dif));
				disp[u][1] = disp[u][1] + (difY / dif * attractiveForce(dif));
	
				TGraphAsList::TEdge *next = edge->next;
				while (next != NULL)
				{
					int u = next->vertex;

					double difX = pos[v][0] - pos[u][0];
					double difY = pos[v][1] - pos[u][1];

					double dif = sqrt(difX*difX + difY*difY);

					if (dif == 0)
						dif = 1;

					disp[v][0] = disp[v][0] - (difX / dif * attractiveForce(dif));
					disp[v][1] = disp[v][1] - (difY / dif * attractiveForce(dif));

					disp[u][0] = disp[u][0] + (difX / dif * attractiveForce(dif));
					disp[u][1] = disp[u][1] + (difY / dif * attractiveForce(dif));

					//cout << "next v: " << v << " u: " << u << " w: " << next->weights << endl;
					next = next->next;
				}
			}
		}

		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		for (int v = 0; v < graph->nVertices; v++)
		{
			double dif = sqrt(disp[v][0]*disp[v][0] + disp[v][1]*disp[v][1]);

			if (dif == 0)
				dif = 1;

			pos[v][0] = pos[v][0] + disp[v][0] / dif * min(abs(disp[v][0]), temperature);
			pos[v][1] = pos[v][1] + disp[v][1] / dif * min(abs(disp[v][1]), temperature);

			pos[v][0] = min((double)width / 2, max((double)(-width) / 2, pos[v][0]));
			pos[v][1] = min((double)height / 2, max((double)(-height) / 2, pos[v][1]));
		}

		temperature = cool(temperature);
		dumpCoordinates(pos, 2, rows);
		cout << endl;


		
	}
	PyArrayObject *arrayX = (PyArrayObject *)(pycoorX);
	PyArrayObject *arrayY = (PyArrayObject *)(pycoorY);

	for (int i = 0; i < rows; i++)
	{
		*(arrayX->data + i * arrayX->strides[0]) = pos[i][0];
		*(arrayY->data + i * arrayY->strides[0]) = pos[i][1];
	}
	

	// free space
	for (int i = 0; i < rows; i++)
	{
		free(pos[i]);
		free(disp[i]);
	}

	free(pos);
	free(disp);
		
    return Py_BuildValue("ii", graph->getEdge(0, 1) ? 1 : 0, (int)(graph->edges));
  PyCATCH
}
