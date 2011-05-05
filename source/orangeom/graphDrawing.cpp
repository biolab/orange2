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


#include "numeric_interface.cpp"
#include "orange_api.hpp"
#include "graph.hpp"
#include "../orange/px/externs.px"
#include <stdio.h>
#include <iostream>

void dumpLinks(int **link, int columns, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			cout << link[i][j] << "  ";
		}

		cout << endl;
	}
}

PyObject *graphOptimization1(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(Graph, steps, coorX, coorY, temperature, width|def=100, height|def=100) -> None")
{
  PyTRY
    int steps;
	int width = 1000;
	int height = 1000;
	double temperature = 0;
    PyObject *pygraph;
	PyObject *pycoorX;
	PyObject *pycoorY;

	//cout <<"v003" << endl;

	if (!PyArg_ParseTuple(args, "OiOOd|ii:graphOptimization1", &pygraph, &steps, &pycoorX, &pycoorY, &temperature, &width, &height))
	{
		//PyErr_Clear
      return NULL;
	}

    TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

    if (graph->nVertices < 2)
      PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

	// temperatura je 1/10 diagonale risalne povrsine
	if (temperature == 0)
		temperature = sqrt((double)(width*width + height*height)) / 10;

	int rows;
	double *coorX;
	double *coorY;
	
	numericToDouble(pycoorX, coorX, rows);
	numericToDouble(pycoorY, coorY, rows);

	if (graph->nVertices != rows)
      PYERROR(PyExc_AttributeError, "graph nodes are not equal to coordinates", NULL);

	double **pos = (double**)malloc(rows * sizeof (double));
	//double **disp = (double**)malloc(rows * sizeof (double));

	if (pos == NULL) // && (disp == NULL))
	{
		cerr << "Couldn't allocate memory\n";
		exit(1);
	}

	int count = 0;
	int i = 0;

	for (i = 0; i < rows; i++)
	{
		pos[i] = (double *)malloc(2 * sizeof(double));
		//disp[i] = (double *)malloc(2 * sizeof(double));

		if (pos[i] == NULL) //&& (disp[i] == NULL))
		{
			cerr << "Couldn't allocate memory\n";
			exit(1);
		}

		pos[i][0] = (double)coorX[i];
		pos[i][1] = (double)coorY[i];
	}

	int **links = NULL;
	int nLinks = 0;

	int v = 0;
	for (v = 0; v < graph->nVertices; v++)
	{
		TGraphAsList::TEdge *edge = graph->edges[v];

		if (edge != NULL)
		{
			int u = edge->vertex;
			links = (int**)realloc(links, (nLinks + 1) * sizeof(int));

			if (links == NULL)
			{
				cerr << "Couldn't allocate memory\n";
				exit(1);
			}

			links[nLinks] = (int *)malloc(2 * sizeof(int));

			if (links[nLinks] == NULL)
			{
				cerr << "Couldn't allocate memory\n";
				exit(1);
			}

			links[nLinks][0] = v;
			links[nLinks][1] = u;
			nLinks++;

			TGraphAsList::TEdge *next = edge->next;
			while (next != NULL)
			{
				int u = next->vertex;

				links = (int**)realloc(links, (nLinks + 1) * sizeof (int));

				if (links == NULL)
				{
					cerr << "Couldn't allocate memory\n";
					exit(1);
				}

				links[nLinks] = (int *)malloc(2 * sizeof(int));
				
				if (links[nLinks] == NULL)
				{
					cerr << "Couldn't allocate memory\n";
					exit(1);
				}

				links[nLinks][0] = v;
				links[nLinks][1] = u;
				nLinks++;

				next = next->next;
			}
		}
	}

	//dumpLinks(links, 2, nLinks);

	//dumpCoordinates(pos, 2, graph->nVertices);
	/*
	GraphOptimization optimization;
	optimization.setTemperature(temperature);
	optimization.fruchtermanReingold(steps, graph->nVertices, pos, nLinks, links);
	temperature = optimization.getTemperature();
	/**/
	//cout << endl;
	//dumpCoordinates(pos, 2, graph->nVertices);
	
	cout << "temp: " << temperature << endl;

	PyArrayObject *arrayX = (PyArrayObject *)(pycoorX);
	PyArrayObject *arrayY = (PyArrayObject *)(pycoorY);

	for (i = 0; i < rows; i++)
	{
		*(double *)(arrayX->data + i * arrayX->strides[0]) = pos[i][0];
		*(double *)(arrayY->data + i * arrayY->strides[0]) = pos[i][1];
	}

	// free space
	for (i = 0; i < rows; i++)
	{
		free(pos[i]);
		//free(disp[i]);
	}

	free(pos);
	//free(disp);
		
    //return Py_BuildValue("ii", graph->getEdge(0, 1) ? 1 : 0, (int)(graph->edges));
	return Py_BuildValue("OOd", arrayX, arrayY, temperature);
  PyCATCH
}
