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

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: miha.stajdohar@fri.uni-lj.si
*/

#include "ppp/network.ppp"

TNetwork::TNetwork(TGraphAsList *graph)
: TGraphAsList(graph->nVertices, graph->nEdgeTypes, graph->directed)
{
	//cout << "TNetwork::constructor 1" << endl;
	import_array();
	optimize.clear();
	vector<int> vertices;
	vector<int> neighbours;

	for(int v1 = 0; v1 < graph->nVertices; v1++) {
		graph->getNeighboursFrom_Single(v1, neighbours);

		ITERATE(vector<int>, ni, neighbours) {
			double *w = getOrCreateEdge(v1, *ni);
			*w = *graph->getEdge(v1, *ni);
		}

		vertices.push_back(v1);
		optimize.insert(v1);
	}

	hierarchy.setTop(vertices);

	int dims[2];
	dims[0] = 2;
	dims[1] = graph->nVertices;
	coors = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	pos = pymatrix_to_Carrayptrs(coors);

	srand(time(NULL));
	int i;
	for (i = 0; i < graph->nVertices; i++)
	{
		pos[0][i] = rand() % 10000;
		pos[1][i] = rand() % 10000;
	}
}

TNetwork::TNetwork(const int &nVert, const int &nEdge, const bool dir)
: TGraphAsList(nVert, nEdge, dir)
{
	//cout << "TNetwork::constructor 2" << endl;
	import_array();
	optimize.clear();
	vector<int> vertices;
	int i;
	for (i = 0; i < nVert; i++)
	{
		vertices.push_back(i);
		optimize.insert(i);
	}

	hierarchy.setTop(vertices);

	//cout << "nVert: " << nVert << endl;

	int dims[2];
	dims[0] = 2;
	dims[1] = nVert;
	coors = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);
	pos = pymatrix_to_Carrayptrs(coors);

	srand(time(NULL));
	for (i = 0; i < nVert; i++)
	{
		pos[0][i] = rand() % 10000;
		pos[1][i] = rand() % 10000;
	}
}

TNetwork::~TNetwork()
{
	free_Carrayptrs(pos);
	Py_DECREF(coors);
}

void TNetwork::hideVertices(vector<int> vertices)
{
  for (vector<int>::iterator it = vertices.begin(); it != vertices.end(); ++it)
	{
    optimize.erase(*it);
  }
}

void TNetwork::showVertices(vector<int> vertices)
{
  for (vector<int>::iterator it = vertices.begin(); it != vertices.end(); ++it)
	{
    optimize.insert(*it);
  }
}

void TNetwork::showAll()
{
  optimize.clear();
  int i;
  for (i = 0; i < nVertices; i++)
  {
    optimize.insert(i);
  }
}

void TNetwork::printHierarchy()
{
  hierarchy.printChilds(hierarchy.top);
  cout << endl;
}


/* ==== Free a double *vector (vec of pointers) ========================== */
void TNetwork::free_Carrayptrs(double **v)  {

	free((char*) v);
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **TNetwork::ptrvector(int n)  {
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
double **TNetwork::pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
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
bool *TNetwork::pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	int n;

	n = arrayin->dimensions[0];
	return (bool *) arrayin->data;  /* pointer to arrayin data as double */
}

TNetworkHierarchyNode::TNetworkHierarchyNode()
{
	parent = NULL;
  vertex = INT_MIN;
}

TNetworkHierarchyNode::~TNetworkHierarchyNode()
{
  int i;
  for (i = 0; i < childs.size(); i++)
  {
    if (childs[i])
    {
      delete childs[i];
    }
  }
}

int TNetworkHierarchyNode::getLevel()
{
  int level = 0;
  TNetworkHierarchyNode *next_parent = parent;

  while (next_parent != NULL)
  {
    if (next_parent->parent == NULL)
      next_parent = NULL;
    else
      next_parent = next_parent->parent;
    level++;
  }

  return level;
}

TNetworkHierarchy::TNetworkHierarchy()
{
	top = new TNetworkHierarchyNode();
  meta_index = 0;
  top->vertex = getNextMetaIndex();
  top->parent = NULL;
}

TNetworkHierarchy::TNetworkHierarchy(vector<int> &topVertices)
{
	top = new TNetworkHierarchyNode();
  meta_index = 0;
  top->vertex = getNextMetaIndex();
  top->parent = NULL;
	setTop(topVertices);
}

TNetworkHierarchy::~TNetworkHierarchy()
{
	if (top)
	{
		delete top;
	}
}

int TNetworkHierarchy::getNextMetaIndex()
{
  meta_index--;
  return meta_index;
}

int TNetworkHierarchy::getMetaChildsCount(TNetworkHierarchyNode *node)
{
  int rv = 0;
  int i;

  for (i = 0; i < node->childs.size(); i++)
  {
    if (node->childs[i]->vertex < 0)
      rv++;

    rv += getMetaChildsCount(node->childs[i]);
  }

  return rv;
}

int TNetworkHierarchy::getMetasCount()
{
  return getMetaChildsCount(top);
}

void TNetworkHierarchy::printChilds(TNetworkHierarchyNode *node)
{
  if (node->childs.size() > 0)
  {
    cout << node->vertex << " | ";
    int i;
    for (i = 0; i < node->childs.size(); i++)
    {
      cout << node->childs[i]->vertex << " ";
    }

    cout << endl;

    for (i = 0; i < node->childs.size(); i++)
    {
      printChilds(node->childs[i]);
    }
  }
}

void TNetworkHierarchy::setTop(vector<int> &vertices)
{
	top->childs.clear();
  top->parent = NULL;

	for (vector<int>::iterator it = vertices.begin(); it != vertices.end(); ++it)
	{
    TNetworkHierarchyNode *child = new TNetworkHierarchyNode();

    child->vertex = *it;
    child->parent = top;

    top->childs.push_back(child);
	}
}

void TNetworkHierarchy::addToNewMeta(vector<int> &vertices)
{
  vector<TNetworkHierarchyNode *> nodes;
  int i;
  TNetworkHierarchyNode *highest_parent = NULL;
  for (i = 0; i < vertices.size(); i++)
  {
    TNetworkHierarchyNode *node = getNodeByVertex(vertices[i]);
    nodes.push_back(node);
    if (highest_parent)
    {
      if (node->parent && highest_parent->getLevel() > node->parent->getLevel())
      {
        highest_parent = node->parent;
      }
    }
    else
    {
      highest_parent = node->parent;
    }
  }

  TNetworkHierarchyNode *meta = new TNetworkHierarchyNode();
  meta->parent = highest_parent;
  meta->vertex = getNextMetaIndex();
  highest_parent->childs.push_back(meta);

  for (i = 0; i < nodes.size(); i++)
  {
    for (vector<TNetworkHierarchyNode *>::iterator it = nodes[i]->parent->childs.begin(); it != nodes[i]->parent->childs.end(); ++it)
	  {
      if ((*it)->vertex == nodes[i]->vertex)
      {
        nodes[i]->parent->childs.erase(it);

        // TODO: erase meta-nodes with 1 or 0 childs
      }
    }

    nodes[i]->parent = meta;
    meta->childs.push_back(nodes[i]);
  }
}

void TNetworkHierarchy::expandMeta(int meta)
{
  TNetworkHierarchyNode *metaNode = getNodeByVertex(meta);

  int i;
  for (i = 0; i < metaNode->childs.size(); i++)
  {
    TNetworkHierarchyNode *node = node->childs[i];

    node->parent = metaNode->parent;
    metaNode->parent->childs.push_back(node);
  }

  // erase meta from parent
  for (vector<TNetworkHierarchyNode *>::iterator it = metaNode->parent->childs.begin(); it != metaNode->parent->childs.end(); ++it)
  {
    if ((*it)->vertex == metaNode->vertex)
    {
      metaNode->parent->childs.erase(it);
      break;
    }
  }

  metaNode->childs.clear();
  metaNode->parent = NULL;
}

TNetworkHierarchyNode *TNetworkHierarchy::getNodeByVertex(int vertex, TNetworkHierarchyNode &start)
{
  int i;
  for (i = 0; i < start.childs.size(); i++)
  {
    if (start.childs[i]->vertex == vertex)
    {
      return start.childs[i];
    }
    else
    {
      TNetworkHierarchyNode *child = getNodeByVertex(vertex, *start.childs[i]);

      if (child)
      {
        return child;
      }
    }
  }

  return NULL;
}

TNetworkHierarchyNode *TNetworkHierarchy::getNodeByVertex(int vertex)
{
  return getNodeByVertex(vertex, *top);
}

#include "externs.px"
#include "orange_api.hpp"
WRAPPER(GraphAsList);
PyObject *Network_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON (GraphAsList, "(nVertices, directed[, nEdgeTypes])")
{
	PyTRY
		int nVertices, directed, nEdgeTypes = 1;
    PyObject *pygraph;

    if (PyArg_ParseTuple(args, "O:Network", &pygraph))
    {
      if (!PyOrGraphAsList_Check(pygraph))
      {
        PyErr_Format(PyExc_TypeError, "Network.__new__: an instance of GraphAsList expected got '%s'", pygraph->ob_type->tp_name);
        return PYNULL;
      }

      TGraphAsList *graph = PyOrange_AsGraphAsList(pygraph).getUnwrappedPtr();

      TNetwork *network = mlnew TNetwork(graph);

      // set graphs attribut items of type ExampleTable to subgraph
      PyObject *strItems = PyString_FromString("items");

		  if (PyObject_HasAttr(pygraph, strItems) == 1)
		  {
			  PyObject* items = PyObject_GetAttr(pygraph, strItems);
        network->items = &dynamic_cast<TExampleTable &>(PyOrange_AsOrange(items).getReference());
      }

	  Py_DECREF(strItems);

      return WrapNewOrange(network, type);
    }

    PyErr_Clear();

    if (PyArg_ParseTuple(args, "ii|i:Network", &nVertices, &directed, &nEdgeTypes))
    {
		  return WrapNewOrange(mlnew TNetwork(nVertices, nEdgeTypes, directed != 0), type);
    }

    PYERROR(PyExc_TypeError, "Network.__new__: number of vertices directedness and optionaly, number of edge types expected", PYNULL);

	PyCATCH
}


PyObject *Network_fromSymMatrix(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(matrix, lower, upper) -> noConnectedNodes")
{
	PyTRY
	CAST_TO(TNetwork, network);

	PyObject *pyMatrix;
	double lower;
	double upper;

	if (!PyArg_ParseTuple(args, "Odd:Network.fromDistanceMatrix", &pyMatrix, &lower, &upper))
		return PYNULL;

	TSymMatrix *matrix = &dynamic_cast<TSymMatrix &>(PyOrange_AsOrange(pyMatrix).getReference());

	if (matrix->dim != network->nVertices)
		PYERROR(PyExc_TypeError, "DistanceMatrix dimension should equal number of vertices.", PYNULL);

	int i,j;
	int nConnected = 0;

	if (matrix->matrixType == 0) {
		// lower
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				//cout << "i " << i << " j " << j;
				double value = matrix->getitem(j,i);
				//cout << " value " << value << endl;
				if (lower <=  value && value <= upper) {
					//cout << "value: " << value << endl;
					double* w = network->getOrCreateEdge(j, i);
					*w = value;

					connected = true;
				}
			}

			if (connected)
				nConnected++;
		}

		vector<int> neighbours;
		network->getNeighbours(0, neighbours);
		if (neighbours.size() > 0)
			nConnected++;
	}
	else {
		// upper
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				double value = matrix->getitem(i,j);
				if (lower <=  value && value <= upper) {
					double* w = network->getOrCreateEdge(i, j);
					*w = value;
					connected = true;
				}
			}

			if (connected)
				nConnected++;

			vector<int> neighbours;
			network->getNeighbours(matrix->dim - 1, neighbours);
			if (neighbours.size() > 0)
				nConnected++;
		}
	}

	return Py_BuildValue("i", nConnected);
	PyCATCH;
}

PyObject *Network_fromDistanceMatrix(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(matrix, lower, upper) -> noConnectedNodes")
{
	PyTRY
	CAST_TO(TNetwork, network);

	PyObject *pyMatrix;
	double lower;
	double upper;

	if (!PyArg_ParseTuple(args, "Odd:Network.fromDistanceMatrix", &pyMatrix, &lower, &upper))
		return PYNULL;

	TSymMatrix *matrix = &dynamic_cast<TSymMatrix &>(PyOrange_AsOrange(pyMatrix).getReference());

	if (matrix->dim != network->nVertices)
		PYERROR(PyExc_TypeError, "DistanceMatrix dimension should equal number of vertices.", PYNULL);

	int i,j;
	int nConnected = 0;

	if (matrix->matrixType == 0) {
		// lower
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				//cout << "i " << i << " j " << j;
				double value = matrix->getitem(j,i);
				//cout << " value " << value << endl;
				if (lower <=  value && value <= upper) {
					//cout << "value: " << value << endl;
					double* w = network->getOrCreateEdge(j, i);
					*w = value;

					connected = true;
				}
			}

			if (connected)
				nConnected++;
		}

		vector<int> neighbours;
		network->getNeighbours(0, neighbours);
		if (neighbours.size() > 0)
			nConnected++;
	}
	else {
		// upper
		for (i = 0; i < matrix->dim; i++) {
			bool connected = false;
			for (j = i+1; j < matrix->dim; j++) {
				double value = matrix->getitem(i,j);
				if (lower <=  value && value <= upper) {
					double* w = network->getOrCreateEdge(i, j);
					*w = value;
					connected = true;
				}
			}

			if (connected)
				nConnected++;

			vector<int> neighbours;
			network->getNeighbours(matrix->dim - 1, neighbours);
			if (neighbours.size() > 0)
				nConnected++;
		}
	}

	return Py_BuildValue("i", nConnected);
	PyCATCH;
}

PyObject *Network_printHierarchy(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "None -> None")
{
  PyTRY
    CAST_TO(TNetwork, network);
    network->printHierarchy();
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_printNodeByVertex(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(vertex) -> None")
{
  PyTRY
    int vertexNdx;

    if (!PyArg_ParseTuple(args, "i:Network.printNodeByVertex", &vertexNdx))
		  return PYNULL;

    CAST_TO(TNetwork, network);
    TNetworkHierarchyNode* vertex = network->hierarchy.getNodeByVertex(vertexNdx);
    cout << "vertex: " << vertex->vertex << endl;
    cout << "n of childs: " << vertex->childs.size() << endl;
    cout << "level: " << vertex->getLevel() << endl;
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_groupVerticesInHierarchy(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(List of vertices) -> None")
{
  PyTRY
    PyObject *pyVertices;

    if (!PyArg_ParseTuple(args, "O:Network.groupVerticesInHierarchy", &pyVertices))
		  return PYNULL;

    int size = PyList_Size(pyVertices);
    int i;
		vector<int> vertices;
		for (i = 0; i < size; i++)
		{
      int vertex = PyInt_AsLong(PyList_GetItem(pyVertices, i));
      vertices.push_back(vertex);
    }

    CAST_TO(TNetwork, network);
    network->hierarchy.addToNewMeta(vertices);
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_expandMeta(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(index) -> None")
{
  PyTRY
    int meta;

    if (!PyArg_ParseTuple(args, "i:Network.groupVerticesInHierarchy", &meta))
		  return PYNULL;

    CAST_TO(TNetwork, network);
    network->hierarchy.expandMeta(meta);
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_hideVertices(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(List of vertices) -> None")
{
  PyTRY
    PyObject *pyVertices;

    if (!PyArg_ParseTuple(args, "O:Network.hideVertices", &pyVertices))
		  return PYNULL;

    int size = PyList_Size(pyVertices);
    int i;
		vector<int> vertices;
		for (i = 0; i < size; i++)
		{
      int vertex = PyInt_AsLong(PyList_GetItem(pyVertices, i));
      vertices.push_back(vertex);
    }

    CAST_TO(TNetwork, network);
    network->hideVertices(vertices);
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_showVertices(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "(List of vertices) -> None")
{
  PyTRY
    PyObject *pyVertices;

    if (!PyArg_ParseTuple(args, "O:Network.showVertices", &pyVertices))
		  return PYNULL;

    int size = PyList_Size(pyVertices);
    int i;
		vector<int> vertices;
		for (i = 0; i < size; i++)
		{
      int vertex = PyInt_AsLong(PyList_GetItem(pyVertices, i));
      vertices.push_back(vertex);
    }

    CAST_TO(TNetwork, network);
    network->showVertices(vertices);
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_showAll(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "None -> None")
{
  PyTRY
    CAST_TO(TNetwork, network);
    network->showAll();
    RETURN_NONE
  PyCATCH;
}

PyObject *Network_getVisible(PyObject *self, PyObject *) PYARGS(METH_NOARGS, "None -> None")
{
  PyTRY
    CAST_TO(TNetwork, network);

    PyObject *pyVisible = PyList_New(0);

    for (set<int>::iterator it = network->optimize.begin(); it != network->optimize.end(); ++it)
	  {
      PyObject *nel = Py_BuildValue("i", *it);
			PyList_Append(pyVisible, nel);
			Py_DECREF(nel);
    }

	  return pyVisible;
  PyCATCH;
}

PyObject *Network_get_coors(PyObject *self, PyObject *args) /*P Y A RGS(METH_VARARGS, "() -> Coors")*/
{
  PyTRY
	CAST_TO(TNetwork, graph);
	Py_INCREF(graph->coors);
	return (PyObject *)graph->coors;
  PyCATCH
}

#include "network.px"
