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


#include "ppp/networkoptimization.ppp"
#include "graph.hpp"

TNetworkOptimization::TNetworkOptimization(int _nVertices, double **_pos, int _nLinks, int **_links)
{
	nVertices = _nVertices;
	nLinks = _nLinks;
	pos = _pos;
	links = _links;

	k = 1;
	k2 = 1;
	width = 1000;
	height = 1000;
	temperature = sqrt((double)(width*width + height*height)) / 10;
}

#ifdef _MSC_VER
#if _MSC_VER < 1300
template<class T>
inline T &min(const T&x, const T&y)
{ return x<y ? x : y; }
#endif
#endif

void TNetworkOptimization::setData(int _nVertices, double **_pos, int _nLinks, int **_links)
{
	int i;

	/*
	for (i = 0; i < nVertices; i++)
	{
		free(pos[i]);
	}
	*/

	for (i = 0; i < nLinks; i++)
	{
		free(links[i]);
	}
	
	if (pos != NULL)
	{
	cout << "set 1" << endl;
	if (pos[0] != NULL)
		free(pos[0]);
	cout << "set 2" << endl;
	if (pos[1] != NULL)
		free(pos[1]);
	cout << "set 3" << endl;
	free(pos);
	}
	free(links);

	nVertices = _nVertices;
	nLinks = _nLinks;
	pos = _pos;
	links = _links;
}

TNetworkOptimization::~TNetworkOptimization()
{
	int i;
	for (i = 0; i < nLinks; i++)
	{
		free(links[i]);
	}

	free(links);

	if (pos != NULL)
	{
		if (pos[0] != NULL)
			free(pos[0]);

		if (pos[1] != NULL)
			free(pos[1]);

		free(pos);
	}	
}

double TNetworkOptimization::attractiveForce(double x)
{
	return x * x / k;

}

double TNetworkOptimization::repulsiveForce(double x)
{
	if (x == 0)
		return k2 / 1;

	return   k2 / x; 
}

double TNetworkOptimization::cool(double t)
{
	return t * 0.98;
}
/*
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
*/
void TNetworkOptimization::fruchtermanReingold(int steps)
{
	/*
	cout << "nVertices: " << nVertices << endl << endl;
	dumpCoordinates(pos, nVertices, 2);
	/**/
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
				double difX = pos[0][v] - pos[0][u];
				double difY = pos[1][v] - pos[1][u];

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
			
			double difX = pos[0][v] - pos[0][u];
			double difY = pos[1][v] - pos[1][u];

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

			pos[0][v] = pos[0][v] + ((disp[v][0] / dif) * min(fabs(disp[v][0]), temperature));
			pos[1][v] = pos[1][v] + ((disp[v][1] / dif) * min(fabs(disp[v][1]), temperature));

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

#include "externs.px"
#include "orange_api.hpp"

int convert_(TGraphAsList *graph, PyObject *pyxcoors, PyObject *pyycoors, double **&pos, int &nLinks, int **&links)
{
	int nRows;
	double *xCoor;
	double *yCoor;
	
	numericToDouble(pyxcoors, xCoor, nRows);
	numericToDouble(pyycoors, yCoor, nRows);

	if (graph->nVertices != nRows)
      return 1;

	pos = (double**)malloc(2 * sizeof (double));

	if (pos == NULL)
	{
		cerr << "Couldn't allocate memory\n";
		exit(1);
	}

	pos[0] = (double *)malloc(graph->nVertices * sizeof(double));
	pos[1] = (double *)malloc(graph->nVertices * sizeof(double));

	if ((pos[0] == NULL) || (pos[1] == NULL))
	{
		cerr << "Couldn't allocate memory\n";
		exit(1);
	}

	//int count = 0;
	int i = 0;
	for (i = 0; i < graph->nVertices; i++)
	{
		pos[0][i] = (double)xCoor[i];
		pos[1][i] = (double)yCoor[i];
	}

	links = NULL;
	nLinks = 0;

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

	return 0;
}

PyObject *NetworkOptimization_new(PyTypeObject *type, PyObject *args, PyObject *keyw) BASED_ON (Orange, "(Graph, xCoordinates, yCoordinates) -> None") 
{
	PyObject *pygraph;
	PyObject *pyxcoors;
	PyObject *pyycoors;

	/*
	if (PyArg_ParseTuple(args, "OOO:GraphOptimization", &pygraph, &pyxcoors, &pyycoors))
	{
		TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

		if (graph->nVertices < 2)
		  PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

		int nLinks;
		int **links;
		double **pos; 

		convert(graph, pyxcoors, pyycoors, pos, nLinks, links);

		return WrapNewOrange(new TGraphOptimization(graph->nVertices, pos, nLinks, links), type);
	}
	else
	{
	/**/
		return WrapNewOrange(new TNetworkOptimization(0, NULL, 0, NULL), type);
	//}
}

PyObject *NetworkOptimization_newData(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Graph, xCoordinates, yCoordinates) -> None")
{
	PyObject *pygraph;
	PyObject *pyxcoors;
	PyObject *pyycoors;

	if (!PyArg_ParseTuple(args, "OOO:NetworkOptimization.newData", &pygraph, &pyxcoors, &pyycoors))
		return NULL;

	TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

	int nLinks;
	int **links;
	double **pos; 

	convert_(graph, pyxcoors, pyycoors, pos, nLinks, links);

	CAST_TO(TNetworkOptimization, graphOpt);
	
	graphOpt->arrayX = (PyArrayObject *)pyxcoors;
	graphOpt->arrayY = (PyArrayObject *)pyycoors;
	graphOpt->setData(graph->nVertices, pos, nLinks, links);
	
	RETURN_NONE;
}

PyObject *NetworkOptimization_fruchtermanReingold(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(steps) -> None")
{
	int steps;
	double temperature = 0;

	if (!PyArg_ParseTuple(args, "id:NetworkOptimization.fruchtermanReingold", &steps, &temperature))
		return NULL;

	CAST_TO(TNetworkOptimization, graph);
	graph->temperature = temperature;
	graph->fruchtermanReingold(steps);

	int i;
	for (i = 0; i < graph->nVertices; i++)
	{
		*(double *)(graph->arrayX->data + i * graph->arrayX->strides[0]) = graph->pos[0][i];
		*(double *)(graph->arrayY->data + i * graph->arrayY->strides[0]) = graph->pos[1][i];
	}

	return Py_BuildValue("OOd", graph->arrayX, graph->arrayY, graph->temperature);
}

#include "networkoptimization.px"