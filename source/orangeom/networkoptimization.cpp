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

TNetworkOptimization::TNetworkOptimization()
{
	import_array();
	
	nVertices = 0;
	nLinks = 0;

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

TNetworkOptimization::~TNetworkOptimization()
{
	int i;
	for (i = 0; i < nLinks; i++)
	{
		free(links[i]);
	}

	free(links);
	free_Carrayptrs(pos);
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

void TNetworkOptimization::dumpCoordinates()
{
	int rows = nVertices;
	int columns = 2;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			cout << pos[i][j] << "  ";
		}

		cout << endl;
	}
}


void TNetworkOptimization::random()
{
	srand(time(NULL));
	int i;
	for (i = 0; i < nVertices; i++)
	{
		pos[i][0] = rand() % width;
		pos[i][1] = rand() % height;
	}
}

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
	//dumpCoordinates();
}

#include "externs.px"
#include "orange_api.hpp"

PyObject *NetworkOptimization_new(PyTypeObject *type, PyObject *args, PyObject *keyw) BASED_ON (Orange, "(Graph) -> None") 
{
	PyObject *pygraph;
	
	if (PyArg_ParseTuple(args, "O:GraphOptimization", &pygraph))
	{
		TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

		if (graph->nVertices < 2)
		  PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

		//return WrapNewOrange(new TGraphOptimization(graph->nVertices, pos, nLinks, links), type);
		return WrapNewOrange(new TNetworkOptimization(), type);
	}
	else
	{
		return WrapNewOrange(new TNetworkOptimization(), type);
	}
}
/* ==== Free a double *vector (vec of pointers) ========================== */ 
void TNetworkOptimization::free_Carrayptrs(double **v)  {
	free((char*) v);
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **TNetworkOptimization::ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **TNetworkOptimization::pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i,n,m;
	
	n = arrayin->dimensions[0];
	m = arrayin->dimensions[1];
	c = ptrvector(n);
	a = (double *) arrayin->data;  /* pointer to arrayin data as double */
	
	for (i = 0; i < n; i++)
	{
		c[i] = a + i * m;
	}

	return c;
}

void TNetworkOptimization::setGraph(TGraphAsList *graph)
{
	int v, l;
	for (l = 0; l < nLinks; l++)
	{
		free(links[l]);
	}

	free(links);
	free_Carrayptrs(pos);

	nVertices = graph->nVertices;
	int dims[2];
	dims[0] = nVertices;
	dims[1] = 2;
	
	coors = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	pos = pymatrix_to_Carrayptrs(coors);
	random();

	//dumpCoordinates();

	links = NULL;
	nLinks = 0;

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
}

PyObject *NetworkOptimization_setGraph(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Graph) -> None")
{
	PyObject *pygraph;

	if (!PyArg_ParseTuple(args, "O:NetworkOptimization.setGraph", &pygraph))
		return NULL;

	TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

	CAST_TO(TNetworkOptimization, graphOpt);
	graphOpt->setGraph(graph);

	RETURN_NONE;
}

PyObject *NetworkOptimization_fruchtermanReingold(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(steps, temperature) -> temperature")
{
	int steps;
	double temperature = 0;

	if (!PyArg_ParseTuple(args, "id:NetworkOptimization.fruchtermanReingold", &steps, &temperature))
		return NULL;

	CAST_TO(TNetworkOptimization, graph);

	graph->temperature = temperature;
	graph->fruchtermanReingold(steps);
	
	return Py_BuildValue("d", graph->temperature);
}

PyObject *NetworkOptimization_getCoors(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> Coors")
{
	CAST_TO(TNetworkOptimization, graph);	
	return Py_BuildValue("O", graph->coors);
}

PyObject *NetworkOptimization_random(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
	CAST_TO(TNetworkOptimization, graph);

	graph->random();
	
	RETURN_NONE;
}

#include "networkoptimization.px"