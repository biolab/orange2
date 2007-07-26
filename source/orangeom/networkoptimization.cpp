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
#define PI 3.14159265

TNetworkOptimization::TNetworkOptimization()
{
	//cout << "constructor" << endl;
	import_array();
	
	nVertices = 0;
	nLinks = 0;

	k = 1;
	k2 = 1;
	width = 10000;
	height = 10000;
	pos = NULL;
	temperature = sqrt(width*width + height*height) / 10;
	coolFactor = 0.96;
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
	//cout << "destructor" << endl;
	free_Carrayptrs(pos);
	Py_DECREF(coors);
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
		pos[i][0] = rand() % (int)width;
		pos[i][1] = rand() % (int)height;
	}
}

int TNetworkOptimization::circularCrossingReduction()
{
	vector<QueueVertex*> vertices;
	vector<QueueVertex*> original;

	int i;
	for (i = 0; i < nVertices; i++)
	{
		vector<int> neighbours;
		graphStructure->getNeighbours(i, neighbours);

		QueueVertex *vertex = new QueueVertex();
		vertex->ndx = i;
		vertex->unplacedNeighbours = neighbours.size();
		vertex->neighbours = neighbours;

		vertices.push_back(vertex);
	}
	original.assign(vertices.begin(), vertices.end());

	deque<int> positions;
	while (vertices.size() > 0)
	{
		sort(vertices.begin(), vertices.end(), QueueVertex());
		QueueVertex *vertex = vertices.back();
		/*
		cout << "vertices" << endl;
		for (i = 0; i < vertices.size(); i++)
			cout << *vertices[i] << endl;
		cout << "ndx: " << vertex->ndx << endl;
		/**/
		// update neighbours
		for (i = 0; i < vertex->neighbours.size(); i++)
		{
			int ndx = vertex->neighbours[i];

			original[ndx]->placedNeighbours++;
			original[ndx]->unplacedNeighbours--;
		}
		// count left & right crossings
		if (vertex->placedNeighbours > 0)
		{
			int left = 0;
			vector<int> lCrossings;
			vector<int> rCrossings;
			for (i = 0; i < positions.size(); i++)
			{
				int ndx = positions[i];
				
				if (vertex->hasNeighbour(ndx))
				{
					lCrossings.push_back(left);
					left += original[ndx]->unplacedNeighbours;
					rCrossings.push_back(left);
				}
				else
					left += original[ndx]->unplacedNeighbours;
			}

			int leftCrossings = 0;
			int rightCrossings = 0;

			for (i = 0; i < lCrossings.size(); i++)
				leftCrossings += lCrossings[i];

			rCrossings.push_back(left);
			for (i = rCrossings.size() - 1; i > 0 ; i--)
				rightCrossings += rCrossings[i] - rCrossings[i - 1];
			//cout << "left: " << leftCrossings << " right: " <<rightCrossings << endl;
			if (leftCrossings < rightCrossings)
				positions.push_front(vertex->ndx);
			else
				positions.push_back(vertex->ndx);

		}
		else
			positions.push_back(vertex->ndx);

		vertices.pop_back();
	}
	/*
	cout << "original" << endl;
	for (i = 0; i < original.size(); i++)
		cout << *original[i] << endl;

	cout << "positions" << endl;
	for (i = 0; i < positions.size(); i++)
		cout << positions[i] << endl;
	/**/
	// TODO: Circular sifting
	


	int xCenter = width / 2;
	int yCenter = height / 2;
	int r = (width < height) ? width * 0.38 : height * 0.38;

	double fi = PI;
	double step = 2 * PI / nVertices;

	for (i = 0; i < nVertices; i++)
	{
		pos[positions[i]][0] = r * cos(fi) + xCenter;
		pos[positions[i]][1] = r * sin(fi) + yCenter;
	
		fi = fi - step;
	}

	for (vector<QueueVertex*>::iterator i = original.begin(); i != original.end(); ++i)
		delete *i;

	original.clear();
	vertices.clear();
	
	return 0;
}

// type
// 0 - original
// 1 - random
int TNetworkOptimization::circular(int type)
{
	int xCenter = width / 2;
	int yCenter = height / 2;
	int r = (width < height) ? width * 0.38 : height * 0.38;

	int i;
	double fi = PI;
	double step = 2 * PI / nVertices;

	srand(time(NULL));
	vector<int> vertices;
	if (type == 1)
		for (i = 0; i < nVertices; i++)
			vertices.push_back(i);

	for (i = 0; i < nVertices; i++)
	{
		if (type == 0)
		{
			pos[i][0] = r * cos(fi) + xCenter;
			pos[i][1] = r * sin(fi) + yCenter;
		}
		else if (type == 1)
		{
			int ndx = rand() % vertices.size();

			pos[vertices[ndx]][0] = r * cos(fi) + xCenter;
			pos[vertices[ndx]][1] = r * sin(fi) + yCenter;
			
			vertices.erase(vertices.begin() + ndx);
		}

		fi = fi - step;
	}

	return 0;
}
int TNetworkOptimization::fruchtermanReingold(int steps)
{ 
	/*
	cout << "nVertices: " << nVertices << endl << endl;
	dumpCoordinates();
	/**/
	double **disp = (double**)malloc(nVertices * sizeof (double));
	int i = 0;
	for (i = 0; i < nVertices; i++)
	{
		disp[i] = (double *)calloc(2, sizeof(double));

		if (disp[i] == NULL)
		{
			cerr << "Couldn't allocate memory (disp[])\n";
			return 1;
		}
	}

	int count = 0;
	double kk = 1;
	double localTemparature = 5;
	double area = width * height;

	k2 = area / nVertices;
	k = sqrt(k2);
	kk = 2 * k;
	double kk2 = kk * kk;

	// iterations
	for (i = 0; i < steps; i++)
	{
		//cout << "iteration: " << i << endl;
		// reset disp
		int j = 0;
		for (j = 0; j < nVertices; j++)
		{
			disp[j][0] = 0;
			disp[j][1] = 0;
		}

		int v = 0;
		// calculate repulsive force
		//cout << "repulsive" << endl;
		for (v = 0; v < nVertices - 1; v++)
		{
			for (int u = v + 1; u < nVertices; u++)
			{
				double difX = pos[v][0] - pos[u][0];
				double difY = pos[v][1] - pos[u][1];

				double dif2 = difX * difX + difY * difY; 

				if (dif2 < kk2)
				{
					if (dif2 == 0)
						dif2 = 1;

					double dX = difX * k2 / dif2;
					double dY = difY * k2 / dif2;

					disp[v][0] = disp[v][0] + dX;
					disp[v][1] = disp[v][1] + dY;

					disp[u][0] = disp[u][0] - dX;
					disp[u][1] = disp[u][1] - dY;
				}
			}
		}
		// calculate attractive forces
		//cout << "attractive" << endl;
		for (j = 0; j < nLinks; j++)
		{
			//int v = links[j][0];
			//int u = links[j][1];
			int v = links[0][j];
			int u = links[1][j];

			double difX = pos[v][0] - pos[u][0];
			double difY = pos[v][1] - pos[u][1];

			double dif = sqrt(difX * difX + difY * difY);

			double dX = difX * dif / k;
			double dY = difY * dif / k;

			disp[v][0] = disp[v][0] - dX;
			disp[v][1] = disp[v][1] - dY;

			disp[u][0] = disp[u][0] + dX;
			disp[u][1] = disp[u][1] + dY;
		}
		//cout << "limit" << endl;
		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		for (v = 0; v < nVertices; v++)
		{
			double dif = sqrt(disp[v][0] * disp[v][0] + disp[v][1] * disp[v][1]);

			if (dif == 0)
				dif = 1;

			pos[v][0] = pos[v][0] + (disp[v][0] * min(fabs(disp[v][0]), temperature) / dif);
			pos[v][1] = pos[v][1] + (disp[v][1] * min(fabs(disp[v][1]), temperature) / dif);

			//pos[v][0] = min((double)width,  max((double)0, pos[v][0]));
			//pos[v][1] = min((double)height, max((double)0, pos[v][1]));
		}
		//cout << temperature << ", ";
		temperature = temperature * coolFactor;
	}

	//cout << "end coors: " << endl;
	//dumpCoordinates();

	// free space
	for (i = 0; i < nVertices; i++)
	{
		free(disp[i]);
		disp[i] = NULL;
	}
	//cout << endl;
	free(disp);
	disp = NULL;
	
	return 0;
}


int TNetworkOptimization::radialFruchtermanReingold(int steps, int nCircles)
{ 
	/*
	cout << "nVertices: " << nVertices << endl << endl;
	dumpCoordinates();
	/**/
	double **disp = (double**)malloc(nVertices * sizeof (double));
	int i = 0;

	for (i = 0; i < nVertices; i++)
	{
		disp[i] = (double *)calloc(2, sizeof(double));

		if (disp[i] == NULL)
		{
			cerr << "Couldn't allocate memory (disp[])\n";
			return 1;
		}
	}

	int radius = width / nCircles / 2;
	//cout << "radius: " << radius << endl;
	int count = 0;
	double kk = 1;
	double localTemparature = 5;
	double area = width * height;

	k2 = area / nVertices;
	k = sqrt(k2);
	kk = 2 * k;
	double kk2 = kk * kk;
	// iterations
	for (i = 0; i < steps; i++)
	{
		//cout << "iteration: " << i << endl;
		// reset disp
		int j = 0;
		for (j = 0; j < nVertices; j++)
		{
			disp[j][0] = 0;
			disp[j][1] = 0;
		}

		int v = 0;
		// calculate repulsive force
		for (v = 0; v < nVertices - 1; v++)
		{
			for (int u = v + 1; u < nVertices; u++)
			{
				// only for vertices on the same level
				//if (level[v] != level[u])
				//	continue;
		
				double difX = pos[v][0] - pos[u][0];
				double difY = pos[v][1] - pos[u][1];

				double dif2 = difX * difX + difY * difY; 

				if (dif2 < kk2)
				{
					if (dif2 == 0)
						dif2 = 1;

					double dX = difX * k2 / dif2;
					double dY = difY * k2 / dif2;

					disp[v][0] = disp[v][0] + dX;
					disp[v][1] = disp[v][1] + dY;

					disp[u][0] = disp[u][0] - dX;
					disp[u][1] = disp[u][1] - dY;
				}
			}
		}
		// calculate attractive forces
		for (j = 0; j < nLinks; j++)
		{
			int v = links[0][j];
			int u = links[1][j];

			// only for vertices on the same level
			//if (level[v] != level[u])
			//	continue;

			double difX = pos[v][0] - pos[u][0];
			double difY = pos[v][1] - pos[u][1];

			double dif = sqrt(difX * difX + difY * difY);

			double dX = difX * dif / k;
			double dY = difY * dif / k;

			disp[v][0] = disp[v][0] - dX;
			disp[v][1] = disp[v][1] - dY;

			disp[u][0] = disp[u][0] + dX;
			disp[u][1] = disp[u][1] + dY;
		}
		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		for (v = 0; v < nVertices; v++)
		{
			double dif = sqrt(disp[v][0] * disp[v][0] + disp[v][1] * disp[v][1]);

			if (dif == 0)
				dif = 1;

			pos[v][0] = pos[v][0] + (disp[v][0] * min(fabs(disp[v][0]), temperature) / dif);
			pos[v][1] = pos[v][1] + (disp[v][1] * min(fabs(disp[v][1]), temperature) / dif);

			double distance = (pos[v][0] - (width/2)) * (pos[v][0] - (width/2)) + (pos[v][1] - (height/2)) * (pos[v][1] - (height/2));
			//cout << "x: " << pos[v][0] << " y: " << pos[v][1] << " width: " << width << " height: " << height << endl;
			//cout << "distance: " << distance << " radius: " << (level[v] * radius) * (level[v] * radius) << endl;
			if (level[v] == 0)
			{
				// move to center
				pos[v][0] = width / 2;
				pos[v][1] = height / 2;

				//cout << "center, x: " << pos[v][0] << " y: " << pos[v][1] << endl;
			}
			//*
			else if (distance > ((level[v] * radius) * (level[v] * radius)))
			{
				// move to outer ring
				double fi = atan((pos[v][1] - (height / 2)) / (pos[v][0] - (width / 2)));

				pos[v][0] = level[v] * radius * cos(fi) + (width / 2);
				pos[v][1] = level[v] * radius * sin(fi) + (height / 2);

				//cout << "outer, x: " << pos[v][0] << " y: " << pos[v][1] << " radius: " << radius << " fi: " << fi << " level: " << level[v] << " v: " << v << endl;
			}
			else if (distance < (((level[v] - 1) * radius) * ((level[v] - 1) * radius)))
			{
				// move to inner ring
				double fi = atan((pos[v][1] - (height / 2)) / (pos[v][0] - (width / 2)));

				pos[v][0] = (level[v] - 1) * radius * cos(fi) + (width / 2);
				pos[v][1] = (level[v] - 1) * radius * sin(fi) + (height / 2);

				//cout << "inner, x: " << pos[v][0] << " y: " << pos[v][1] << endl;
			}
			/**/
		}
		//cout << temperature << ", ";
		temperature = temperature * coolFactor;
	}
	/*
	for (i = 0; i < nVertices; i++)
		cout << "level " << i << ": " << level[i] << endl;
	/**/
	//cout << "end coors: " << endl;
	//dumpCoordinates();

	// free space
	for (i = 0; i < nVertices; i++)
	{
		free(disp[i]);
		disp[i] = NULL;
	}
	//cout << endl;
	free(disp);
	disp = NULL;
	
	return 0;
}



/* ==== Free a double *vector (vec of pointers) ========================== */ 
void TNetworkOptimization::free_Carrayptrs(double **v)  {

	free((char*) v);
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **TNetworkOptimization::ptrvector(double n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));

	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  
	}
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
	
	for (i = 0; i < n; i++) {
		c[i] = a + i * m;
	}

	return c;
}

/* ==== Create 1D Carray from PyArray ======================
 129     Assumes PyArray is contiguous in memory.             */
bool *TNetworkOptimization::pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	int n;

	n = arrayin->dimensions[0];
	return (bool *) arrayin->data;  /* pointer to arrayin data as double */
}

int TNetworkOptimization::setGraph(TGraphAsList *graph)
{
	//cout << "-1" << endl;
	links[0].clear();
	links[1].clear();
	free_Carrayptrs(pos);

	nVertices = graph->nVertices;
	int dims[2];
	dims[0] = nVertices;
	dims[1] = 2;
	//cout << "0" << endl;
	coors = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	pos = pymatrix_to_Carrayptrs(coors);

	random();
 
	//dumpCoordinates();
	nLinks = 0;
	int v;
	for (v = 0; v < graph->nVertices; v++)
	{
		TGraphAsList::TEdge *edge = graph->edges[v];

		if (edge != NULL)
		{
			int u = edge->vertex;
			
			links[0].push_back(v);
			links[1].push_back(u);
			nLinks++;

			TGraphAsList::TEdge *next = edge->next;
			while (next != NULL)
			{
				int u = next->vertex;
				
				links[0].push_back(v);
				links[1].push_back(u);
				nLinks++;

				next = next->next;
			}
		}
	}
	//cout << "5" << endl;
	return 0;
}


#include "externs.px"
#include "orange_api.hpp"

PyObject *NetworkOptimization_new(PyTypeObject *type, PyObject *args, PyObject *keyw) BASED_ON (Orange, "(Graph) -> None") 
{
  PyTRY
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
  PyCATCH
}

int getWords(string const& s, vector<string> &container)
{
    int n = 0;
	bool quotation = false;
    string::const_iterator it = s.begin(), end = s.end(), first;
    for (first = it; it != end; ++it)
    {
        // Examine each character and if it matches the delimiter
        if (((!quotation) && ((' ' == *it) || ('\t' == *it) || ('\r' == *it) || ('\f' == *it) || ('\v' == *it))) || ('\n' == *it))
        {
            if (first != it)
            {
                // extract the current field from the string and
                // append the current field to the given container
                container.push_back(string(first, it));
                ++n;
                
                // skip the delimiter
                first = it + 1;
            }
            else
            {
                ++first;
            }
        }
		else if (('\"' == *it) || ('\'' == *it))
		{
			if (quotation)
			{
				quotation = false;

				// extract the current field from the string and
                // append the current field to the given container
                container.push_back(string(first, it));
                ++n;
                
                // skip the delimiter
                first = it + 1;
			}
			else
			{
				quotation = true;

				// skip the quotation
				first = it + 1;
			}
		}
    }
    if (first != it)
    {
        // extract the last field from the string and
        // append the last field to the given container
        container.push_back(string(first, it));
        ++n;
    }
    return n;
}



PyObject *NetworkOptimization_setGraph(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Graph) -> None")
{
  PyTRY
	PyObject *pygraph;

	if (!PyArg_ParseTuple(args, "O:NetworkOptimization.setGraph", &pygraph))
		return NULL;

	TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

	CAST_TO(TNetworkOptimization, graphOpt);
	//cout << "networkoptimization.cpp/setGraph: setting graph..." << endl;
	if (graphOpt->setGraph(graph) > 0)
		PYERROR(PyExc_SystemError, "setGraph failed", NULL);
	
	graphOpt->graphStructure = graph;

	//cout << "done." << endl;
	RETURN_NONE;
  PyCATCH
}

bool hasVertex(int vertex, vector<int> list)
{
	int i;
	for (i = 0; i < list.size(); i++)
	{
		//cout << list[i] << " " << vertex << endl;
		if (list[i] == vertex)
			return true;
	}

	return false;
}

PyObject *NetworkOptimization_radialFruchtermanReingold(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(center, steps, temperature) -> temperature")
{
  PyTRY
	int steps, center;
	double temperature = 0;

	if (!PyArg_ParseTuple(args, "iid:NetworkOptimization.radialFruchtermanReingold", &center, &steps, &temperature))
		return NULL;

	CAST_TO(TNetworkOptimization, graph);

	graph->pos[center][0] = graph->width / 2;
	graph->pos[center][1] = graph->height / 2;

	int nCircles = 6;
	int r = graph->width / nCircles / 2;

	graph->level = new int[graph->nVertices];
	int i;
	for (i = 0; i < graph->nVertices; i++)
		graph->level[i] = nCircles;

	vector<int> removedLinks[2];
	vector<int> vertices;
	vector<int> allVertices;
	vertices.push_back(center);
	graph->level[center] = 0;

	for (i = 0; i < nCircles; i++)
	{
		// position vertices
		double fi = 360 / vertices.size();
		int v;
		for (v = 0; v < vertices.size(); v++)
		{
			double x = i * r * cos(v * fi * PI / 180) + (graph->width / 2);
			double y = i * r * sin(v * fi * PI / 180) + (graph->height / 2);

			graph->pos[vertices[v]][0] = x;
			graph->pos[vertices[v]][1] = y;

			//cout << "v: " << vertices[v] << " X: " << x << " Y: " << y << " level: " << graph->level[vertices[v]] << endl;
		}
		//cout << endl;
		vector<int> newVertices;
		for (v = 0; v < vertices.size(); v++)
		{
			int j;
			int node = vertices[v];

			for (j = graph->links[0].size() - 1; j >= 0; j--)
			{
				if (graph->links[0][j] == node)
				{
					//cout << "j: " << j << " u: " << graph->links1[0][j] << " v: " << graph->links1[1][j] << endl;
					removedLinks[0].push_back(graph->links[0][j]);
					removedLinks[1].push_back(graph->links[1][j]);

					if (!hasVertex(graph->links[1][j], allVertices))
					{
						newVertices.push_back(graph->links[1][j]);
						allVertices.push_back(graph->links[1][j]);
						graph->level[graph->links[1][j]] = i + 1;
					}
					graph->links[0].erase(graph->links[0].begin() + j);
					graph->links[1].erase(graph->links[1].begin() + j);
				}
				else if (graph->links[1][j] == node)
				{
					//cout << "j: " << j << " u: " << graph->links1[0][j] << " v: " << graph->links1[1][j] << endl;
					removedLinks[0].push_back(graph->links[0][j]);
					removedLinks[1].push_back(graph->links[1][j]);

					if (!hasVertex(graph->links[0][j], allVertices))
					{
						//cout << "adding: " << 
						newVertices.push_back(graph->links[0][j]);
						allVertices.push_back(graph->links[0][j]);
						graph->level[graph->links[0][j]] = i + 1;
					}

					graph->links[0].erase(graph->links[0].begin() + j);
					graph->links[1].erase(graph->links[1].begin() + j);
				}
			}
		}

		vertices.clear();

		if (newVertices.size() == 0)
			break;

		for (v = 0; v < newVertices.size(); v++)
		{
			vertices.push_back(newVertices[v]);
		}
	}
	// adds back removed links
	for (i = 0; i < removedLinks[0].size(); i++)
	{
		graph->links[0].push_back(removedLinks[0][i]);
		graph->links[1].push_back(removedLinks[1][i]);
	}

	graph->temperature = temperature;
	graph->coolFactor = exp(log(10.0/10000.0) / steps);
	/*
	for (i = 0; i < graph->nVertices; i++)
		cout << "level " << i << ": " << graph->level[i] << endl;
	/**/
	if (graph->radialFruchtermanReingold(steps, nCircles) > 0)
	{
		delete[] graph->level;
		PYERROR(PyExc_SystemError, "radialFruchtermanReingold failed", NULL);
	}

	delete[] graph->level;

	return Py_BuildValue("d", graph->temperature);
  PyCATCH
}


PyObject *NetworkOptimization_circularOriginal(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TNetworkOptimization, graph);
	graph->circular(0);
	RETURN_NONE;
  PyCATCH
}

PyObject *NetworkOptimization_circularRandom(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TNetworkOptimization, graph);
	graph->circular(1);
	RETURN_NONE;
  PyCATCH
}


PyObject *NetworkOptimization_circularCrossingReduction(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TNetworkOptimization, graph);
	graph->circularCrossingReduction();
	RETURN_NONE;
  PyCATCH
}
PyObject *NetworkOptimization_fruchtermanReingold(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(steps, temperature, hiddenNodes) -> temperature")
{
  PyTRY
	int steps;
	double temperature = 0;
	PyObject* hiddenNodes;

	if (!PyArg_ParseTuple(args, "id|O:NetworkOptimization.fruchtermanReingold", &steps, &temperature, &hiddenNodes))
		return NULL;

	int size = PyList_Size(hiddenNodes);

	CAST_TO(TNetworkOptimization, graph);

	// remove links for hidden nodes
	vector<int> removedLinks[2];
	int i, j;
	for (i = 0; i < size; i++)
	{
		int node = PyInt_AsLong(PyList_GetItem(hiddenNodes, i));
		
		//cout <<"size: " << graph->links1->size() << endl;
		for (j = graph->links[0].size() - 1; j >= 0; j--)
		{
			if (graph->links[0][j] == node || graph->links[1][j] == node)
			{
				//cout << "j: " << j << " u: " << graph->links1[0][j] << " v: " << graph->links1[1][j] << endl;
				removedLinks[0].push_back(graph->links[0][j]);
				removedLinks[1].push_back(graph->links[1][j]);

				graph->links[0].erase(graph->links[0].begin() + j);
				graph->links[1].erase(graph->links[1].begin() + j);
			}
		}
	}
	graph->nLinks = graph->links[0].size();

	graph->temperature = temperature;
	graph->coolFactor = exp(log(10.0/10000.0) / steps);
	
	if (graph->fruchtermanReingold(steps) > 0)
	{
		PYERROR(PyExc_SystemError, "fruchtermanReingold failed", NULL);
	}

	// adds back removed links
	for (i = 0; i < removedLinks[0].size(); i++)
	{
		graph->links[0].push_back(removedLinks[0][i]);
		graph->links[1].push_back(removedLinks[1][i]);
	}

	graph->nLinks = graph->links[0].size();
	
	return Py_BuildValue("d", graph->temperature);
  PyCATCH
}

PyObject *NetworkOptimization_get_coors(PyObject *self, PyObject *args) /*P Y A RGS(METH_VARARGS, "() -> Coors")*/
{
  PyTRY
	CAST_TO(TNetworkOptimization, graph);	
	Py_INCREF(graph->coors);
	return (PyObject *)graph->coors;  
  PyCATCH
}

int *getVertexPowers(TNetworkOptimization *graph)
{
	int *vertexPower = new int[graph->nVertices];

	int i;
	for (i=0; i < graph->nVertices; i++)
	{
		vertexPower[i] = 0;
	}

	for (i=0; i < graph->nLinks; i++)
	{
		vertexPower[graph->links[0][i]]++;
		vertexPower[graph->links[1][i]]++;
	}
	
  return vertexPower;
}

PyObject *NetworkOptimization_getVertexPowers(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "() -> list")
{
  PyTRY
    CAST_TO(TNetworkOptimization, graph);
    int *vertexPower = getVertexPowers(graph);
    PyObject *pypowers = PyList_New(graph->nVertices);
    for(int i =0; i < graph->nVertices; i++)
      PyList_SetItem(pypowers, i, PyInt_FromLong(vertexPower[i]));
    delete [] vertexPower;
    return pypowers;
  PyCATCH;
}

PyObject *NetworkOptimization_getHubs(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(n) -> HubList")
{
  PyTRY
	int n;

	if (!PyArg_ParseTuple(args, "n:NetworkOptimization.getHubs", &n))
		return NULL;

	CAST_TO(TNetworkOptimization, graph);

  int i;
  int *vertexPower = getVertexPowers(graph);
  
	PyObject* hubList = PyList_New(n);
	
	for (i=0; i < n; i++)
	{
		int j;
		int ndx_max = -1;
		int max = 0;
		for (j=0; j < graph->nVertices; j++)
		{
			if (vertexPower[j] > max)
			{
				ndx_max = j;
				max = vertexPower[j];
			}
		}
		//cout << "pow: " << vertexPower[ndx_max] << " ndx: " << ndx_max << endl;

		vertexPower[ndx_max] = -2;
		PyList_SetItem(hubList, i, PyInt_FromLong(ndx_max));
	}

	delete [] vertexPower;
	return hubList;
  PyCATCH
}

PyObject *NetworkOptimization_closestVertex(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x, y) -> Ndx")
{
  PyTRY
	double x;
	double y;

	if (!PyArg_ParseTuple(args, "dd:NetworkOptimization.closestVertex", &x, &y))
		return NULL;

	CAST_TO(TNetworkOptimization, graph);

	int i;
	double min = 100000000;
	int ndx = -1;
	for (i=0; i < graph->nVertices; i++)
	{
		double dX = graph->pos[i][0] - x;
		double dY = graph->pos[i][1] - y;
		double d = dX*dX + dY*dY;

		if (d < min)
		{
			min = d;
			ndx = i;
		}
	}

	return Py_BuildValue("id", ndx, sqrt(min));
  PyCATCH
}

PyObject *NetworkOptimization_random(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TNetworkOptimization, graph);

	graph->random();
	
	RETURN_NONE;
  PyCATCH
}

WRAPPER(ExampleTable)

PyObject *readNetwork(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(fn) -> Graph")
{
  PyTRY
	TGraph *graph;
	PGraph wgraph;
	TDomain *domain = new TDomain();
	PDomain wdomain = domain;
	TExampleTable *table;
	PExampleTable wtable;

	//cout << "readNetwork" << endl;
	char *fn;

	if (!PyArg_ParseTuple(args, "s:orangeom.readNetwork", &fn))
		return NULL;

	//cout << "File: " << fn << endl;

	string line;
	ifstream file(fn);
	string graphName = "";
	int nVertices = 0;

	if (file.is_open())
	{
		// read head
		while (!file.eof())
		{
			getline (file, line);
			vector<string> words;
			int n = getWords(line, words);
			//cout << line << "  -  " << n << endl;
			if (n > 0)
			{
				if (stricmp(words[0].c_str(), "*network") == 0)
				{
					//cout << "Network" << endl;
					if (n > 1)
					{
						graphName = words[1];
						//cout << "Graph name: " << graphName << endl;
					}
					else
					{
						file.close();
						PYERROR(PyExc_SystemError, "invalid file format", NULL);
					}
				}
				else if (stricmp(words[0].c_str(), "*vertices") == 0)
				{
					//cout << "Vertices" << endl;
					if (n > 1)
					{
						istringstream strVertices(words[1]);
						strVertices >> nVertices;
						if (nVertices == 0)
						{
							file.close();
							PYERROR(PyExc_SystemError, "invalid file format", NULL);
						}

						//cout << "nVertices: " << nVertices << endl;
					}
					else
					{
						file.close();
						PYERROR(PyExc_SystemError, "invalid file format", NULL);
					}

					break;
				}
			}
		}
		graph = new TGraphAsList(nVertices, 0, false);
		wgraph = graph;

		domain->addVariable(new TIntVariable("index"));
		domain->addVariable(new TStringVariable("label"));
		domain->addVariable(new TFloatVariable("x"));
		domain->addVariable(new TFloatVariable("y"));
		domain->addVariable(new TFloatVariable("z"));
		domain->addVariable(new TStringVariable("ic"));
		domain->addVariable(new TStringVariable("bc"));
		domain->addVariable(new TStringVariable("bw"));
		table = new TExampleTable(domain);
  	wtable = table;

		// read vertex descriptions
		while (!file.eof())
		{
			getline(file, line);
			vector<string> words;
			int n = getWords(line, words);
			//cout << line << "  -  " << n << endl;
			if (n > 0)
			{
				TExample *example = new TExample(domain);

				if ((stricmp(words[0].c_str(), "*arcs") == 0) || (stricmp(words[0].c_str(), "*edges") == 0))
					break;

				int index = -1;
				istringstream strIndex(words[0]);
				strIndex >> index;
				if ((index <= 0) || (index > nVertices))
				{
					file.close();
					PYERROR(PyExc_SystemError, "invalid file format", NULL);
				}

				//cout << "index: " << index << " n: " << n << endl;
				(*example)[0] = TValue(index);
				
				if (n > 1)
				{
					string label = words[1];
					//cout << "label: " << label << endl;
					(*example)[1] = TValue(new TStringValue(label), STRINGVAR);
					
					int i = 2;
					char *xyz = "  xyz";
					// read coordinates
					while ((i <= 4) && (i < n))
					{
						float coor = -1;	
						istringstream strCoor(words[i]);
						strCoor >> coor;

						if ((coor < 0) || (coor > 1))
							break;
						
						//cout << xyz[i] << ": " << coor * 1000 << endl;
						(*example)[i] = TValue(coor);
						i++;
					}
					// read attributes
					while (i < n)
					{
						if (stricmp(words[i].c_str(), "ic") == 0)
						{
							if (i + 1 < n) 
								i++; 
							else 
							{
								file.close();
								PYERROR(PyExc_SystemError, "invalid file format", NULL);
							}

							//cout << "ic: " << words[i] << endl;
							(*example)[5] = TValue(new TStringValue(words[i]), STRINGVAR);
						}
						else if (stricmp(words[i].c_str(), "bc") == 0)
						{
							if (i + 1 < n) 
								i++; 
							else 
							{
								file.close();
								PYERROR(PyExc_SystemError, "invalid file format", NULL);
							}

							//cout << "bc: " << words[i] << endl;
							(*example)[6] = TValue(new TStringValue(words[i]), STRINGVAR);
						}
						else if (stricmp(words[i].c_str(), "bw") == 0)
						{
							if (i + 1 < n) 
								i++; 
							else
							{
								file.close();
								PYERROR(PyExc_SystemError, "invalid file format", NULL);
							}

							//cout << "bw: " << words[i] << endl;
							(*example)[7] = TValue(new TStringValue(words[i]), STRINGVAR);
						}
						i++;
					}
					
				}
				table->push_back(example);
				//cout << "push back" <<endl;
			}
		}
		// read arcs
		vector<string> words;
		int n = getWords(line, words);
		if (n > 0)
		{
			if (stricmp(words[0].c_str(), "*arcs") == 0)
			{
				while (!file.eof())
				{
					getline (file, line);
					vector<string> words;
					int n = getWords(line, words);
					//cout << line << "  -  " << n << endl;
					if (n > 0)
					{
						if (stricmp(words[0].c_str(), "*edges") == 0)
							break;

						if (n > 1)
						{
							int i1 = -1;
							int i2 = -1;
							istringstream strI1(words[0]);
							istringstream strI2(words[1]);

							strI1 >> i1;
							strI2 >> i2;

							if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices))
							{
								file.close();
								PYERROR(PyExc_SystemError, "invalid file format", NULL);
							}

							if (i1 == i2)
								continue;
							
							//cout << "i1: " << i1 << " i2: " << i2 << endl;
							*graph->getOrCreateEdge(i1 - 1, i2 - 1) = 1;
						}
					}
				}
			}
		}
		// read edges
		n = getWords(line, words);
		if (n > 0)
		{
			if (stricmp(words[0].c_str(), "*edges") == 0)
			{
				while (!file.eof())
				{
					getline (file, line);
					vector<string> words;
					int n = getWords(line, words);
					//cout << line << "  -  " << n << endl;
					if (n > 1)
					{
						int i1 = -1;
						int i2 = -1;
						istringstream strI1(words[0]);
						istringstream strI2(words[1]);

						strI1 >> i1;
						strI2 >> i2;

						if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices))
						{
							file.close();
							PYERROR(PyExc_SystemError, "invalid file format", NULL);
						}

						if (i1 == i2)
							continue;

						*graph->getOrCreateEdge(i1 - 1, i2 - 1) = 1;
						*graph->getOrCreateEdge(i2 - 1, i1 - 1) = 1;
					}
				}
			}
		}

		file.close();
	}
	else
	{
	  PyErr_Format(PyExc_SystemError, "unable to open file '%s'", fn);
	  return NULL;
	}
	
	//graph->setProperty("items", wtable);

	return Py_BuildValue("NN", WrapOrange(wgraph), WrapOrange(wtable));
  PyCATCH
}

#include "networkoptimization.px"