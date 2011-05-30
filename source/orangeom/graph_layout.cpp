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


#include "ppp/graph_layout.ppp"
#define PI 3.14159265

TGraphLayout::TGraphLayout()
{

	//cout << "TGraphLayout::constructor" << endl;
	import_array();

	nVertices = 0;
	nLinks = 0;

	k = 1;
	k2 = 1;
	radius = 1;
	width = 10000;
	height = 10000;
	temperature = sqrt(width*width + height*height) / 10;
	coolFactor = 0.96;
	coors = NULL;
	pos = NULL;
	//networkx_graph = NULL;
	//cout << "constructor end" << endl;
}

#ifdef _MSC_VER
#if _MSC_VER < 1300
template<class T>
inline T &min(const T&x, const T&y)
{ return x<y ? x : y; }
#endif
#endif

TGraphLayout::~TGraphLayout()
{
	free_Carrayptrs(pos);
	/*Py_XDECREF(coors);*/
}

void TGraphLayout::dump_coordinates()
{
	for (int i = 0; i < nVertices; i++)
	{
		cout << pos[0][i] << "  " << pos[1][i] << endl;
	}
}

void TGraphLayout::dump_disp()
{
	for (int i = 0; i < nVertices; i++)
	{
		cout << disp[0][i] << "  " << disp[1][i] << endl;
	}
}


int TGraphLayout::random()
{
	srand(time(NULL));
	int i;
	for (i = 0; i < nVertices; i++)
	{
		pos[0][i] = rand() % (int)width;
		pos[1][i] = rand() % (int)height;
	}
	return 0;
}

int TGraphLayout::circular_crossing_reduction()
{
	vector<QueueVertex*> vertices;
	vector<QueueVertex*> original;

	int i;
	for (i = 0; i < nVertices; i++)
	{
		vector<int> neighbours;
		/* network->getNeighbours(i, neighbours); TODO: FIX! */

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

	// Circular sifting
	for (i = 0; i < positions.size(); i++)
		original[positions[i]]->position = i;

	int step;
	for (step = 0; step < 5; step++)
	{
		for (i = 0; i < nVertices; i++)
		{
			bool stop = false;
			int switchNdx = -1;
			QueueVertex *u = original[positions[i]];
			int vNdx = (i + 1) % nVertices;

			while (!stop)
			{
				QueueVertex *v = original[positions[vNdx]];

				int midCrossings = u->neighbours.size() * v->neighbours.size() / 2;
				int crossings = 0;
				int j,k;
				for (j = 0; j < u->neighbours.size(); j++)
					for (k = 0; k < v->neighbours.size(); k++)
						if ((original[u->neighbours[j]]->position == v->position) || (original[v->neighbours[k]]->position == u->position))
							midCrossings = (u->neighbours.size() - 1) * (v->neighbours.size() - 1) / 2;
						else if ((original[u->neighbours[j]]->position + nVertices - u->position) % nVertices < (original[v->neighbours[k]]->position + nVertices - u->position) % nVertices)
							crossings++;

				//cout << "v: " <<  v->ndx << " crossings: " << crossings << " u.n.size: " << u->neighbours.size() << " v.n.size: " << v->neighbours.size() << " mid: " << midCrossings << endl;
				if (crossings > midCrossings)
					switchNdx = vNdx;
				else
					stop = true;

				vNdx = (vNdx + 1) % nVertices;
			}
			int j;
			if (switchNdx > -1)
			{
				//cout << "u: " << u->ndx << " switch: " << original[switchNdx]->ndx << endl << endl;
				positions.erase(positions.begin() + i);
				positions.insert(positions.begin() + switchNdx, u->ndx);

				for (j = i; j <= switchNdx; j++)
					original[positions[j]]->position = j;
			}
			//else
			//	cout << "u: " << u->ndx << " switch: " << switchNdx << endl;
		}
	}

	int xCenter = width / 2;
	int yCenter = height / 2;
	int r = (width < height) ? width * 0.38 : height * 0.38;

	double fi = PI;
	double fiStep = 2 * PI / nVertices;

	for (i = 0; i < nVertices; i++)
	{
		pos[0][positions[i]] = r * cos(fi) + xCenter;
		pos[1][positions[i]] = r * sin(fi) + yCenter;

		fi = fi - fiStep;
	}

	for (vector<QueueVertex*>::iterator i = original.begin(); i != original.end(); ++i)
		delete *i;

	original.clear();
	vertices.clear();

	return 0;
}

int TGraphLayout::circular(int type)
{
	// type
	// 0 - original
	// 1 - random
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
			pos[0][i] = r * cos(fi) + xCenter;
			pos[1][i] = r * sin(fi) + yCenter;
		}
		else if (type == 1)
		{
			int ndx = rand() % vertices.size();

			pos[0][vertices[ndx]] = r * cos(fi) + xCenter;
			pos[1][vertices[ndx]] = r * sin(fi) + yCenter;

			vertices.erase(vertices.begin() + ndx);
		}

		fi = fi - step;
	}

	return 0;
}

void TGraphLayout::clear_disp()
{
	int j;
	for (j = 0; j < nVertices; j++) {
			disp[0][j] = 0;
			disp[1][j] = 0;
	}
}

void TGraphLayout::fr_repulsive_force(double kk2, int type)
{
	// type = 0 - classic fr
	// type = 1 - radial fr
	// type = 2 - smooth fr
	int u, v;
	for (v = 0; v < nVertices; v++) {
		for (u = 0; u < v; u++) {
			if (type == 1) {
				if (level[u] == level[v]) {
					k = kVector[level[u]];
				} else {
					k = radius;
				}
				//kk = 2 * k;
				k2 = k*k;
			} else if (type == 2) {
				if (level[u] == level[v]) {
					if (level[u] == 0) {
						k = kVector[0];
					} else {
						k = kVector[1];
					}
				} else {
					k = kVector[2];
				}
				k2 = k * k;
				kk2 = 4 * k2;
			}

			double difX = pos[0][v] - pos[0][u];
			double difY = pos[1][v] - pos[1][u];

			double dif2 = difX * difX + difY * difY;

			if (dif2 < kk2) {
				if (dif2 == 0) {
					dif2 = 1;
				}
				double dX = difX * k2 / dif2;
				double dY = difY * k2 / dif2;

				disp[0][v] += dX;
				disp[1][v] += dY;

				disp[0][u] -= dX;
				disp[1][u] -= dY;
			}
		}
	}
}

void TGraphLayout::fr_attractive_force(int type, bool weighted)
{
	// type = 0 - classic fr
	// type = 1 - radial fr
	// type = 2 - smooth fr
	int j, u, v;
	for (j = 0; j < nLinks; j++) {
		v = links[0][j];
		u = links[1][j];
		
		if (type == 1) {
			if (level[u] == level[v]) {
				k = kVector[level[u]];
			} else {
				k = radius;
			}
		} else if (type == 2) {
			if (level[u] == level[v]) {
				if (level[u] == 0) {
					k = kVector[0];
				} else {
					k = kVector[1];
				}
			} else {
				k = kVector[2];
			}
		}

		double difX = pos[0][v] - pos[0][u];
		double difY = pos[1][v] - pos[1][u];

		double dif = sqrt(difX * difX + difY * difY);
		
		double dX, dY;

		if (weighted) {
			dX = difX * dif / k * weights[j];
			dY = difY * dif / k * weights[j];
		} else {
			dX = difX * dif / k;
			dY = difY * dif / k;
		}

		disp[0][v] -= dX;
		disp[1][v] -= dY;

		disp[0][u] += dX;
		disp[1][u] += dY;
	}
}

void TGraphLayout::fr_limit_displacement()
{
	int v;
	// limit the maximum displacement to the temperature t
	// and then prevent from being displaced outside frame
	for (v = 0; v < nVertices; v++) {
		double dif = sqrt(pow(disp[0][v], 2) + pow(disp[1][v], 2));

		if (dif == 0) {
			dif = 1;
		}
		pos[0][v] += (disp[0][v] * min(fabs(disp[0][v]), temperature) / dif);
		pos[1][v] += (disp[1][v] * min(fabs(disp[1][v]), temperature) / dif);

		//pos[v][0] = min((double)width,  max((double)0, pos[v][0]));
		//pos[v][1] = min((double)height, max((double)0, pos[v][1]));
	}
}

int TGraphLayout::fr(int steps, bool weighted)
{
	int i;
	int count = 0;
	double kk = 1;
	double localTemparature = 5;
	double area = width * height;

	k2 = area / nVertices;
	k = sqrt(k2);
	kk = 2 * k;
	double kk2 = kk * kk;
	
	// iterations
	for (i = 0; i < steps; i++) {
		clear_disp();
		fr_repulsive_force(kk2, 0);
		fr_attractive_force(0, weighted);
		fr_limit_displacement();
		temperature = temperature * coolFactor;
	}

	return 0;
}

int TGraphLayout::fr_radial(int steps, int nCircles)
{
	int i, v;
	radius = width / nCircles / 2;
	int count = 0;
	double kk = 1;
	double localTemparature = 5;
	double area = width * height;

	k2 = area / nVertices;
	k = sqrt(k2);
	kk = 2 * k;
	double kk2 = kk * kk;
	// iterations
	for (i = 0; i < steps; i++) {
		clear_disp();
		fr_repulsive_force(kk2, 1);
		fr_attractive_force(1, false);
		fr_limit_displacement();
		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame

		for (v = 0; v < nCircles; v++) {
			levelMin[v] = INT_MAX;
			levelMax[v] = 0;
		}

		for (v = 0; v < nVertices; v++) {
			double distance = (pos[0][v] - (width/2)) * (pos[0][v] - (width/2)) + (pos[1][v] - (height/2)) * (pos[1][v] - (height/2));

			if (distance < levelMin[level[v]])
				levelMin[level[v]] = distance;

			if (distance > levelMax[level[v]])
				levelMax[level[v]] = distance;
		}

		for (v = 1; v < nCircles; v++) {
			levelMin[v] = (v - 1) * radius / sqrt(levelMin[v]);
			levelMax[v] =  v      * radius / sqrt(levelMax[v]);
		}

		for (v = 0; v < nVertices; v++) {
			double distance = sqrt((pos[0][v] - (width/2)) * (pos[0][v] - (width/2)) + (pos[1][v] - (height/2)) * (pos[1][v] - (height/2)));

			if (level[v] == 0) {
				// move to center
				pos[0][v] = width / 2;
				pos[1][v] = height / 2;

				//cout << "center, x: " << pos[v][0] << " y: " << pos[v][1] << endl;
			} else if (distance > level[v] * radius - radius / 2) {
				// move to outer ring
				if (levelMax[level[v]] < 1) {
					double fi = 0;
					double x = pos[0][v] - (width / 2);
					double y = pos[1][v] - (height / 2);

					if (x < 0)
						fi = atan(y / x) + PI;
					else if ((x > 0) && (y >= 0))
						fi = atan(y / x);
					else if ((x > 0) && (y < 0))
						fi = atan(y / x) + 2 * PI;
					else if ((x == 0) && (y > 0))
						fi = PI / 2;
					else if ((x == 0) && (y < 0))
						fi = 3 * PI / 2;

					pos[0][v] = levelMax[level[v]] * distance * cos(fi) + (width / 2);
					pos[1][v] = levelMax[level[v]] * distance * sin(fi) + (height / 2);

					//cout << "outer, x: " << pos[v][0] << " y: " << pos[v][1] << " radius: " << radius << " fi: " << fi << " level: " << level[v] << " v: " << v << endl;
				}
			} else if (distance < (level[v] - 1) * radius + radius / 2) {
				// move to inner ring
				if (levelMin[level[v]] > 1) {
					double fi = 0;
					double x = pos[0][v] - (width / 2);
					double y = pos[1][v] - (height / 2);

					if (x < 0)
						fi = atan(y / x) + PI;
					else if ((x > 0) && (y >= 0))
						fi = atan(y / x);
					else if ((x > 0) && (y < 0))
						fi = atan(y / x) + 2 * PI;
					else if ((x == 0) && (y > 0))
						fi = PI / 2;
					else if ((x == 0) && (y < 0))
						fi = 3 * PI / 2;

					pos[0][v] = levelMin[level[v]] * distance * cos(fi) + (width / 2);
					pos[1][v] = levelMin[level[v]] * distance * sin(fi) + (height / 2);

					//cout << "inner, x: " << pos[v][0] << " y: " << pos[v][1] << endl;
				}
			}
		}
		temperature = temperature * coolFactor;
	}
	return 0;
}

/* ==== Free a double *vector (vec of pointers) ========================== */
void TGraphLayout::free_Carrayptrs(double **v)  {

	free((char*) v);
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **TGraphLayout::ptrvector(int n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double *)));

	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);
	}
	return v;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **TGraphLayout::pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
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
bool *TGraphLayout::pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	int n;

	n = arrayin->dimensions[0];
	return (bool *) arrayin->data;  /* pointer to arrayin data as double */
}

int TGraphLayout::set_graph(PyObject *graph)
{
	if (graph == NULL) {
		nVertices = 0;
		nLinks = 0;
		npy_intp dims[2];
		dims[0] = 2;
		dims[1] = nVertices;
		free_Carrayptrs(pos);
		coors = NULL;
		pos = NULL;
		links[0].clear();
		links[1].clear();
		weights.clear();
		disp[0].clear();
		disp[1].clear();
		return 0;
	}
	PyObject* nodes = PyObject_GetAttrString(graph, "node");
	PyObject* adj = PyObject_GetAttrString(graph, "adj");

	nVertices = PyDict_Size(nodes);
	nLinks = 0;

	npy_intp dims[2];
	dims[0] = 2;
	dims[1] = nVertices;

	free_Carrayptrs(pos);
	/*Py_XDECREF(coors);*/

	coors = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	pos = pymatrix_to_Carrayptrs(coors);
	/*Py_INCREF(coors);*/
	srand(time(NULL));
	int i;
	for (i = 0; i < nVertices; i++) {
		pos[0][i] = rand() % 10000;
		pos[1][i] = rand() % 10000;
	}

	links[0].clear();
	links[1].clear();
	weights.clear();
	disp[0].resize(nVertices, 0);
	disp[1].resize(nVertices, 0);

	PyObject *key_u, *value_u, *key_v, *value_v;
	Py_ssize_t pos_u = 0;

	while (PyDict_Next(adj, &pos_u, &key_u, &value_u)) {
		int u = PyInt_AS_LONG(key_u);

		Py_ssize_t pos_v = 0;
		while (PyDict_Next(value_u, &pos_v, &key_v, &value_v)) {
			int v = PyInt_AS_LONG(key_v);
			links[0].push_back(u);
			links[1].push_back(v);
			weights.push_back(1); // TODO: compute weight
			nLinks++;
		}
	}

	return 0;
}

#include "externs.px"
#include "orange_api.hpp"

PyObject *GraphLayout_new(PyTypeObject *type, PyObject *args, PyObject *keyw) BASED_ON (Orange, "() -> None")
{
  PyTRY
	/*
	  PyObject *pygraph;

	if (PyArg_ParseTuple(args, "O:GraphLayout", &pygraph))
	{
		TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

		if (graph->nVertices < 2)
		  PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);

		//return WrapNewOrange(new TGraphOptimization(graph->nVertices, pos, nLinks, links), type);
		return WrapNewOrange(new TGraphLayout(), type);
	}
	else
	{
		return WrapNewOrange(new TGraphLayout(), type);
	}
	*/
	return WrapNewOrange(new TGraphLayout(), type);
  PyCATCH
}

PyObject *GraphLayout_set_graph(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(Orange.network.Graph) -> None")
{
  PyTRY
	PyObject* graph = Py_None;
	PyObject* positions = Py_None;

	if (!PyArg_ParseTuple(args, "|OO:GraphLayout.set_graph", &graph, &positions)) {
		  return PYNULL;
	}

	CAST_TO(TGraphLayout, graph_layout);

	if (graph == Py_None) {
		graph_layout->set_graph(NULL);
	} else {
		graph_layout->set_graph(graph);

		if (positions != Py_None) {
			int i;
			for (i = 0; i < graph_layout->nVertices; i++) {
				double x,y;
				PyArg_ParseTuple(PyList_GetItem(positions, i), "dd", &x, &y);
				graph_layout->pos[0][i] = x;
				graph_layout->pos[1][i] = y;
			}
		}
	}
	
	RETURN_NONE;
  PyCATCH
}

PyObject *GraphLayout_get_coors(PyObject *self, PyObject *args) /*P Y A RGS(METH_VARARGS, "() -> Coors")*/
{
  PyTRY
	CAST_TO(TGraphLayout, graph_layout);
	Py_INCREF(graph_layout->coors);
	return (PyObject *)graph_layout->coors;
  PyCATCH
}

bool has_vertex(int vertex, vector<int> list)
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

PyObject *GraphLayout_random(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TGraphLayout, graph_layout);
	graph_layout->random();
	RETURN_NONE;
  PyCATCH
}

PyObject *GraphLayout_fr(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(steps, temperature, coolFactor, weighted) -> temperature")
{
  PyTRY
	int steps;
	double temperature = 0;
	double coolFactor = 0;
	bool weighted = false;

	if (!PyArg_ParseTuple(args, "id|db:GraphLayout.fr", &steps, &temperature, &coolFactor, &weighted)) {
		return NULL;
	}

	CAST_TO(TGraphLayout, graph_layout);

	graph_layout->temperature = temperature;

	if (coolFactor == 0) {
		graph_layout->coolFactor = exp(log(10.0/10000.0) / steps);
	} else { 
		graph_layout->coolFactor = coolFactor;
	}

	if (graph_layout->fr(steps, weighted) > 0) {
		PYERROR(PyExc_SystemError, "fr failed", PYNULL);
	}

	return Py_BuildValue("d", graph_layout->temperature);
  PyCATCH
}

PyObject *GraphLayout_fr_radial(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(center, steps, temperature) -> temperature")
{
  PyTRY

	int steps, center;
	double temperature = 0;

	if (!PyArg_ParseTuple(args, "iid:GraphLayout.fr_radial", &center, &steps, &temperature))
		return NULL;

	CAST_TO(TGraphLayout, graph_layout);

	graph_layout->pos[0][center] = graph_layout->width / 2;
	graph_layout->pos[1][center] = graph_layout->height / 2;

	int nCircles = 6;
	int r = graph_layout->width / nCircles / 2;

	graph_layout->level = new int[graph_layout->nVertices];
	graph_layout->kVector = new double[nCircles];
	graph_layout->levelMin = new double[nCircles];
	graph_layout->levelMax = new double[nCircles];
	int i;
	for (i = 0; i < graph_layout->nVertices; i++)
		graph_layout->level[i] = nCircles;

	for (i = 0; i < nCircles; i++)
	{
		graph_layout->kVector[i] = 0;
		graph_layout->levelMin[i] = INT_MAX;
		graph_layout->levelMax[i] = 0;
	}
	vector<int> removedLinks[2];
	vector<int> vertices;
	vector<int> allVertices;
	vertices.push_back(center);
	graph_layout->level[center] = 0;

	for (i = 0; i < nCircles; i++)
	{
		// position vertices
		double fi = 360 / vertices.size();
		int v;
		for (v = 0; v < vertices.size(); v++)
		{
			double x = i * r * cos(v * fi * PI / 180) + (graph_layout->width / 2);
			double y = i * r * sin(v * fi * PI / 180) + (graph_layout->height / 2);

			graph_layout->pos[0][vertices[v]] = x;
			graph_layout->pos[1][vertices[v]] = y;

			//cout << "v: " << vertices[v] << " X: " << x << " Y: " << y << " level: " << graph_layout->level[vertices[v]] << endl;
		}
		//cout << endl;
		vector<int> newVertices;
		for (v = 0; v < vertices.size(); v++)
		{
			int j;
			int node = vertices[v];

			for (j = graph_layout->links[0].size() - 1; j >= 0; j--)
			{
				if (graph_layout->links[0][j] == node)
				{
					//cout << "j: " << j << " u: " << graph_layout->links1[0][j] << " v: " << graph_layout->links1[1][j] << endl;
					removedLinks[0].push_back(graph_layout->links[0][j]);
					removedLinks[1].push_back(graph_layout->links[1][j]);

					if (!has_vertex(graph_layout->links[1][j], allVertices))
					{
						newVertices.push_back(graph_layout->links[1][j]);
						allVertices.push_back(graph_layout->links[1][j]);
						graph_layout->level[graph_layout->links[1][j]] = i + 1;
					}
					graph_layout->links[0].erase(graph_layout->links[0].begin() + j);
					graph_layout->links[1].erase(graph_layout->links[1].begin() + j);
				}
				else if (graph_layout->links[1][j] == node)
				{
					//cout << "j: " << j << " u: " << graph_layout->links1[0][j] << " v: " << graph_layout->links1[1][j] << endl;
					removedLinks[0].push_back(graph_layout->links[0][j]);
					removedLinks[1].push_back(graph_layout->links[1][j]);

					if (!has_vertex(graph_layout->links[0][j], allVertices))
					{
						//cout << "adding: " <<
						newVertices.push_back(graph_layout->links[0][j]);
						allVertices.push_back(graph_layout->links[0][j]);
						graph_layout->level[graph_layout->links[0][j]] = i + 1;
					}

					graph_layout->links[0].erase(graph_layout->links[0].begin() + j);
					graph_layout->links[1].erase(graph_layout->links[1].begin() + j);
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
		graph_layout->links[0].push_back(removedLinks[0][i]);
		graph_layout->links[1].push_back(removedLinks[1][i]);
	}


	for (i = 0; i < graph_layout->nVertices; i++)
	{
		if (graph_layout->level[i] >= nCircles)
			graph_layout->level[i] = nCircles - 1;

		graph_layout->kVector[graph_layout->level[i]]++;
	}

	double radius = graph_layout->width / nCircles / 2;
	for (i = 0; i < nCircles; i++)
	{
		//cout << "n: " << graph_layout->kVector[i] << endl;
		//cout << "r: " << radius * i;
		if (graph_layout->kVector[i] > 0)
			graph_layout->kVector[i] = 2 * i * radius * sin(PI / graph_layout->kVector[i]);
		else
			graph_layout->kVector[i] = -1;

		//cout << "kvec: " << graph_layout->kVector[i] << endl;
	}

	graph_layout->temperature = temperature;
	graph_layout->coolFactor = exp(log(10.0/10000.0) / steps);
	/*
	for (i = 0; i < graph_layout->nVertices; i++)
		cout << "level " << i << ": " << graph_layout->level[i] << endl;
	/**/
	if (graph_layout->fr_radial(steps, nCircles) > 0)
	{
		delete[] graph_layout->level;
		delete[] graph_layout->kVector;
		delete[] graph_layout->levelMin;
		delete[] graph_layout->levelMax;
		PYERROR(PyExc_SystemError, "fr_radial failed", NULL);
	}

	delete[] graph_layout->level;
	delete[] graph_layout->kVector;
	delete[] graph_layout->levelMin;
	delete[] graph_layout->levelMax;
	return Py_BuildValue("d", graph_layout->temperature);
  PyCATCH
}

PyObject *GraphLayout_circular_original(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TGraphLayout, graph_layout);
	graph_layout->circular(0);
	RETURN_NONE;
  PyCATCH
}

PyObject *GraphLayout_circular_random(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TGraphLayout, graph_layout);
	graph_layout->circular(1);
	RETURN_NONE;
  PyCATCH
}

PyObject *GraphLayout_circular_crossing_reduction(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
{
  PyTRY
	CAST_TO(TGraphLayout, graph_layout);
	graph_layout->circular_crossing_reduction();
	RETURN_NONE;
  PyCATCH
}

int *get_vertex_powers(TGraphLayout *graph)
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

PyObject *GraphLayout_get_vertex_powers(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "() -> list")
{
  PyTRY
    CAST_TO(TGraphLayout, graph);
    int *vertexPower = get_vertex_powers(graph);
    PyObject *pypowers = PyList_New(graph->nVertices);
    for(int i =0; i < graph->nVertices; i++)
      PyList_SetItem(pypowers, i, PyInt_FromLong(vertexPower[i]));
    delete [] vertexPower;
    return pypowers;
  PyCATCH;
}

PyObject *GraphLayout_closest_vertex(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x, y) -> vertex id")
{
  PyTRY
	double x;
	double y;

	if (!PyArg_ParseTuple(args, "dd:GraphLayout.closest_vertex", &x, &y))
		return NULL;

	CAST_TO(TGraphLayout, graph);

	int i;
	double min = 100000000;
	int ndx = -1;
	for (i=0; i < graph->nVertices; i++)
	{
		double dX = graph->pos[0][i] - x;
		double dY = graph->pos[1][i] - y;
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

PyObject *GraphLayout_vertex_distances(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x, y) -> List of (distance, vertex)")
{
  PyTRY
	double x;
	double y;

	if (!PyArg_ParseTuple(args, "dd:GraphLayout.vertex_distances", &x, &y))
		return NULL;

	CAST_TO(TGraphLayout, graph);

	int i;
	PyObject* distancies = PyList_New(0);
	for (i=0; i < graph->nVertices; i++)
	{
		double dX = graph->pos[0][i] - x;
		double dY = graph->pos[1][i] - y;
		double d = dX*dX + dY*dY;

		PyObject *nel = Py_BuildValue("di", d, i);
		PyList_Append(distancies, nel);
		Py_DECREF(nel);
	}

	return distancies;
  PyCATCH
}

PyObject *GraphLayout_get_vertices_in_rect(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(x1, y1, x2, y2) -> list of vertices")
{
  PyTRY
	double x1, y1, x2, y2;

	if (!PyArg_ParseTuple(args, "dddd:GraphLayout.get_vertices_in_rect", &x1, &y1, &x2, &y2))
		return NULL;

	if (x1 > x2) {
		double tmp = x2;
		x2 = x1;
		x1 = tmp;
	}

	if (y1 > y2) {
		double tmp = y2;
		y2 = y1;
		y1 = tmp;
	}

	CAST_TO(TGraphLayout, graph);
	PyObject* vertices = PyList_New(0);
	int i;
	for (i = 0; i < graph->nVertices; i++) {
		double vX = graph->pos[0][i];
		double vY = graph->pos[1][i];

		if ((x1 <= vX) && (x2 >= vX) && (y1 <= vY) && (y2 >= vY)) {
			PyObject *nel = Py_BuildValue("i", i);
			PyList_Append(vertices, nel);
			Py_DECREF(nel);
		}
	}

	return vertices;
  PyCATCH
}

PyObject *GraphLayout_edges_from_distance_matrix(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(matrix, lower, upper, kNN) -> list of edges")
{
  PyTRY
	PyObject *pyMatrix;
	double lower;
	double upper;
	int kNN = 0;
	int andor = 0;

	if (!PyArg_ParseTuple(args, "Odd|ii:GraphLayout.edges_from_distance_matrix", &pyMatrix, &lower, &upper, &kNN))
		return PYNULL;

	TSymMatrix *matrix = &dynamic_cast<TSymMatrix &>(PyOrange_AsOrange(pyMatrix).getReference());
	//cout << "kNN: " << kNN << endl;
	//cout << "andor: " << andor << endl;

	int i,j;
	//vector<coord_t> edges_interval;
	PyObject* edges = PyList_New(0);

	if (matrix->matrixType == 0) {
		// lower
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				//cout << "i " << i << " j " << j;
				double value = matrix->getitem(j,i);
				//cout << " value " << value << endl;
				if (lower <=  value && value <= upper) {
					connected = true;
					//edges_interval.push_back(coord_t(j, i));
					PyObject *nel = Py_BuildValue("iid", j, i, value);
					PyList_Append(edges, nel);
					Py_DECREF(nel);
				}
			}
		}
	} else {
		// upper
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				double value = matrix->getitem(i,j);
				if (lower <=  value && value <= upper) {
					connected = true;
					//edges_interval.push_back(coord_t(i, j));
					PyObject *nel = Py_BuildValue("iid", i, j, value);
					PyList_Append(edges, nel);
					Py_DECREF(nel);
				}
			}
		}
	}
	
	//PyObject* edges_knn = PyList_New(0);
	if (kNN > 0) {
		for (i = 0; i < matrix->dim; i++) {
			vector<int> closest;
			matrix->getknn(i, kNN, closest);

			for (j = 0; j < closest.size(); j++) {
				//edges_knn.push_back(coord_t(i, closest[j]));
				double value = matrix->getitem(i, closest[j]);
				PyObject *nel = Py_BuildValue("iid", i, closest[j], value);
				PyList_Append(edges, nel);
				Py_DECREF(nel);
			}
		}
	}

	if (andor == 0) {
		
	}

	return edges;
  PyCATCH;
}


WRAPPER(ExampleTable)

int getWords1(string const& s, vector<string> &container)
{
    int n = 0;
	bool quotation = false;
	container.clear();
    string::const_iterator it = s.begin(), end = s.end(), first;
    for (first = it; it != end; ++it) {
        // Examine each character and if it matches the delimiter
        if ((!quotation && (' ' == *it || '\t' == *it || '\r' == *it || '\f' == *it || '\v' == *it || ',' == *it)) || ('\n' == *it)) {
            if (first != it) {
                // extract the current field from the string and
                // append the current field to the given container
                container.push_back(string(first, it));
                ++n;

                // skip the delimiter
                first = it + 1;
            } else {
                ++first;
            }
        }
		else if (('\"' == *it) || ('\'' == *it) || ('(' == *it) || (')' == *it)) {
			if (quotation) {
				quotation = false;

				// extract the current field from the string and
                // append the current field to the given container
                container.push_back(string(first, it));
                ++n;

                // skip the delimiter
                first = it + 1;
			} else {
				quotation = true;

				// skip the quotation
				first = it + 1;
			}
		}
    }
    if (first != it) {
        // extract the last field from the string and
        // append the last field to the given container
        container.push_back(string(first, it));
        ++n;
    }
    return n;
}

PyObject *GraphLayout_readPajek(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(fn) -> Edge List")
{
  PyTRY
	TDomain *domain = new TDomain();
	PDomain wdomain = domain;
	TExampleTable *table;
	PExampleTable wtable;
	PyObject *edgeList = PyList_New(0);
	PyObject *arcList = PyList_New(0);
	char *fn;

	if (!PyArg_ParseTuple(args, "s:GraphLayout.readPajek", &fn))
		PYERROR(PyExc_TypeError, "invalid arguments (string expected)", PYNULL);

	string line;
	
	istream *stream;
	ifstream file(fn);
	istringstream text(fn, istringstream::in);

	if (file.is_open()) {
		stream = &file;
	} else {
		if (text.good()) {
			stream = &text;	
		} else {
			PyErr_Format(PyExc_SystemError, "Unable to read network.", fn);
			return PYNULL;
		}
	}
	
	string graphName = "";
	string description = "";
	int nVertices = 0;
	int nEdges = 0;

	TFloatVariable *indexVar = new TFloatVariable("index");
	indexVar->numberOfDecimals = 0;
	domain->addVariable(indexVar);
	domain->addVariable(new TStringVariable("label"));
	domain->addVariable(new TFloatVariable("x"));
	domain->addVariable(new TFloatVariable("y"));
	domain->addVariable(new TFloatVariable("z"));
	domain->addVariable(new TStringVariable("shape"));
	domain->addVariable(new TFloatVariable("size"));
	domain->addVariable(new TStringVariable("vertex color"));
	domain->addVariable(new TStringVariable("boundary color"));
	domain->addVariable(new TFloatVariable("boundary width"));
	table = new TExampleTable(domain);
	wtable = table;
	vector<string> words;

	// read head
	while (!stream->eof()) {
		getline(*stream, line);
		int n = getWords1(line, words);
		if (n > 0) 	{
			std::transform(words[0].begin(), words[0].end(), words[0].begin(), ::tolower);
			if (words[0].compare("*network") == 0) {
				if (n > 1) {
					graphName = words[1];
				}
			} else if (words[0].compare("*description") == 0) {
				if (n > 1) {
					description = words[1];
				}
			} else if (words[0].compare("*vertices") == 0) {
				if (n > 1) {
					nVertices = atoi(words[1].c_str());
					if (nVertices < 1) {
						file.close();
						PYERROR(PyExc_SystemError, "Invalid number of vertices.", PYNULL);
					}
				} else {
					file.close();
					PYERROR(PyExc_SystemError, "Invalid file format. Number of vertices not given.", PYNULL);
				}
				if (n > 2) {
					nEdges = atoi(words[2].c_str());
				}

				// read vertices
				int row = 0;
				while (!stream->eof()) {
					getline(*stream, line);
					int n = getWords1(line, words);
					if (n > 0) {
						std::transform(words[0].begin(), words[0].end(), words[0].begin(), ::tolower);
						if (words[0].compare("*arcs") == 0 || words[0].compare("*edges") == 0)
							break;

						TExample *example = new TExample(domain);
						float index = -1; istringstream strIndex(words[0]); strIndex >> index;
							
						if ((index <= 0) || (index > nVertices)) {
							file.close();
							PYERROR(PyExc_SystemError, "Invalid file format. Invalid vertex index.", PYNULL);
						}

						(*example)[0] = TValue(index);
						if (n > 1) {
							string label = words[1];
							(*example)[1] = TValue(new TStringValue(label), STRINGVAR);

							int i = 2;
							char *xyz = "  xyz";
							// read coordinates
							while ((i <= 4) && (i < n)) {
								double coor = -1; istringstream strCoor(words[i]); strCoor >> coor;
								(*example)[i] = TValue((float)coor);

								// if only 2 coordinates are given, shape follows
								if ((i == 4) && (coor == -1)) {
									(*example)[i+1] = TValue(new TStringValue(words[i]), STRINGVAR);
								} else if ((i == 4) && (n > i + 1)) {
									(*example)[i+1] = TValue(new TStringValue(words[i+1]), STRINGVAR);
								}
								i++;
							}
							// read attributes
							while (i < n) {
								if (stricmp(words[i].c_str(), "s_size") == 0) {
									if (i + 1 < n) {
										i++;
									} else {
										file.close();
										PYERROR(PyExc_SystemError, "invalid file format. Error reading vertex size.", PYNULL);
									}
									float size = -1; istringstream strSize(words[i]); strSize >> size;
									(*example)[6] = TValue(size);

								} else if (stricmp(words[i].c_str(), "ic") == 0) {
									if (i + 1 < n) {
										i++;
									} else {
										file.close();
										PYERROR(PyExc_SystemError, "invalid file format. Error reading vertex color.", PYNULL);
									}
									(*example)[7] = TValue(new TStringValue(words[i]), STRINGVAR);

								} else if (stricmp(words[i].c_str(), "bc") == 0) {
									if (i + 1 < n) {
										i++;
									} else {
										file.close();
										PYERROR(PyExc_SystemError, "invalid file format. Error reading boundary color.", PYNULL);
									}
									(*example)[8] = TValue(new TStringValue(words[i]), STRINGVAR);
								} else if (stricmp(words[i].c_str(), "bw") == 0) {
									if (i + 1 < n) {
										i++;
									} else {
										file.close();
										PYERROR(PyExc_SystemError, "invalid file format. Error reading boundary width.", PYNULL);
									}
									float bwidth = -1; istringstream strBWidth(words[i]); strBWidth >> bwidth;
									(*example)[9] = TValue(bwidth);
								}
								i++;
							}
						}
						example->id = getExampleId();
						table->push_back(example);
					}
					row++;
				}
			}
			if (words[0].compare("*arcs") == 0) {
				// read arcs
				while (!stream->eof()) {
					getline(*stream, line);
					//vector<string> words;
					int n = getWords1(line, words);
					if (n > 0) {
						std::transform(words[0].begin(), words[0].end(), words[0].begin(), ::tolower);
						if (words[0].compare("*edges") == 0) {
							break;
						}
						if (n > 1) {
							int i1 = -1; istringstream strI1(words[0]); strI1 >> i1;
							int i2 = -1; istringstream strI2(words[1]); strI2 >> i2;
								
							if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices)) {
								file.close();
								PYERROR(PyExc_SystemError, "Invalid file format. Adding arcs: vertex index out of range.", PYNULL);
							}

							if (i1 == i2) continue;
							if (n > 2) {
								vector<string> weights;
								int m = getWords1(words[2], weights);
								int i;
								for (i=0; i < m; i++) {
									double i3 = 0; istringstream strI3(weights[i]); strI3 >> i3;
									PyObject *nel = Py_BuildValue("iid", i1 - 1, i2 - 1, i3);
									PyList_Append(arcList, nel);
									Py_DECREF(nel);
								}
							}
						}
					}
				}
			}
			if (words[0].compare("*edges") == 0) {
				// read edges
				while (!stream->eof()) {
					getline(*stream, line);
					int n = getWords1(line, words);
					if (n > 1) {
						int i1 = -1; istringstream strI1(words[0]); strI1 >> i1;
						int i2 = -1; istringstream strI2(words[1]); strI2 >> i2;
							
						if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices)) {
							file.close();
							PYERROR(PyExc_SystemError, "Invalid file format. Adding edges: vertex index out of range.", PYNULL);
						}

						if (i1 == i2) continue;
						if (n > 2) {
							vector<string> weights;
							int m = getWords1(words[2], weights);
							int i;
							for (i=0; i < m; i++) {
								double i3 = 0; istringstream strI3(weights[i]); strI3 >> i3;
								PyObject *nel = Py_BuildValue("iid", i1 - 1, i2 - 1, i3);
								PyList_Append(edgeList, nel);
								Py_DECREF(nel);
							}
						}
					}
				}
			}
		}
	}
	file.close();
	return Py_BuildValue("NNN", edgeList, arcList, WrapOrange(wtable));
  PyCATCH
}

#include "graph_layout.px"
