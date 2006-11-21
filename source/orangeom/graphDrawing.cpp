#include "numeric_interface.cpp"
#include "orange_api.hpp"
#include "graph.hpp"
#include "../orange/px/externs.px"
#include <stdio.h>
#include <iostream>

void dumpCoordinates(int **pos, int columns, int rows)
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

int k;
int k2;

int attractiveForce(int x)
{
	return x * x / k;

}

int repulsiveForce(int x)
{
	if (x == 0)
		return k2 / 1;

	return abs(k2 / x); 
}

int cool(int t)
{
	return t;
}


PyObject *graphOptimization(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(Graph, steps, coordinatesm, width|def=100, height|def=100) -> None")
{
  PyTRY
    int steps;
	int width = 100;
	int height = 100;
	int temperature = 10;
    PyObject *pygraph;
	PyObject *pycoordinates;
	
	if (!PyArg_ParseTuple(args, "OiO|ii:graphOptimization", &pygraph, &steps, &pycoordinates, &width, &height))
      return NULL;

    TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

    if (graph->nVertices < 2)
      PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

	cout << "Vertices: " << graph->nVertices << endl;

	int rows;
	int columns;
	double *matrix;

	numericToDouble(pycoordinates, matrix, columns, rows);
	
	if (graph->nVertices != rows)
      PYERROR(PyExc_AttributeError, "graph nodes are not equal to coordinates", NULL);

	int **pos = (int**)malloc(rows * sizeof (int));
	int **disp = (int**)malloc(rows * sizeof (int));

	if ((pos == NULL) && (disp == NULL))
	{
		cerr << "Couldn't allocate memory\n";
		exit(1);
	}

	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		pos[i] = (int *)malloc(columns * sizeof(int));
		disp[i] = (int *)malloc(columns * sizeof(int));

		if ((pos[i] == NULL) && (disp[i] == NULL))
		{
			cerr << "Couldn't allocate memory\n";
			exit(1);
		}

		for (int j = 0; j < columns; j++)
		{
			pos[i][j] = (int)matrix[count];
			disp[i][j] = 0;

			count++;
		}
	}
	
	int area = width * height;
	k = (int)sqrt((double)(area / graph->nVertices));
	k2 = area / graph->nVertices;

	// iterations
	for (int i = 0; i < steps; i++)
	{
		// calculate repulsive force
		for (int v = 0; v < graph->nVertices -1 ; v++)
		{
			for (int u = v + 1; u < graph->nVertices; u++)
			{
				int difX = pos[v][0] - pos[u][0];
				int difY = pos[v][1] - pos[u][1];

				int signX = 1; 
				if (difX < 0) signX = -1;

				int signY = 1;
				if (difY < 0) signY = -1;

				disp[v][0] = disp[v][0] + (signX * repulsiveForce(difX));
				disp[v][1] = disp[v][1] + (signY * repulsiveForce(difY));

				disp[u][0] = disp[u][0] - (signX * repulsiveForce(difX));
				disp[u][1] = disp[u][1] - (signY * repulsiveForce(difY));
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
				
				int difX = pos[v][0] - pos[u][0];
				int difY = pos[v][1] - pos[u][1];

				int signX = 1; 
				if (difX < 0) signX = -1;

				int signY = 1;
				if (difY < 0) signY = -1;

				disp[v][0] = disp[v][0] - (signX * attractiveForce(difX));
				disp[v][1] = disp[v][1] - (signY * attractiveForce(difY));

				disp[u][0] = disp[u][0] + (signX * attractiveForce(difX));
				disp[u][1] = disp[u][1] + (signY * attractiveForce(difY));
	
				TGraphAsList::TEdge *next = edge->next;
				while (next != NULL)
				{
					int u = next->vertex;

					int difX = pos[v][0] - pos[u][0];
					int difY = pos[v][1] - pos[u][1];

					int signX = 1; 
					if (difX < 0) signX = -1;

					int signY = 1;
					if (difY < 0) signY = -1;

					disp[v][0] = disp[v][0] - (signX * attractiveForce(difX));
					disp[v][1] = disp[v][1] - (signY * attractiveForce(difY));

					disp[u][0] = disp[u][0] + (signX * attractiveForce(difX));
					disp[u][1] = disp[u][1] + (signY * attractiveForce(difY));

					//cout << "next v: " << v << " u: " << u << " w: " << next->weights << endl;
					next = next->next;
				}
			}
		}

		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		for (int v = 0; v < graph->nVertices; v++)
		{
			int signX = 1;
			if (disp[v][0] < 0) signX = -1; else if (disp[v][0] == 0) signX = 0;

			int signY = 1;
			if (disp[v][1] < 0) signY = -1; else if (disp[v][1] == 0) signY = 0;

			pos[v][0] = pos[v][0] + signX * min(abs(disp[v][0]), temperature);
			pos[v][1] = pos[v][1] + signY * min(abs(disp[v][1]), temperature);

			pos[v][0] = min(width / 2, max(-width / 2, pos[v][0]));
			pos[v][1] = min(height / 2, max(-height / 2, pos[v][1]));
		}

		temperature = cool(temperature);
		dumpCoordinates(pos, columns, rows);
		cout << endl;
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
