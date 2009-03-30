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
		int nVertices = 1, directed = 0, nEdgeTypes = 1;
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

    if (PyArg_ParseTuple(args, "|iii:Network", &nVertices, &directed, &nEdgeTypes))
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

int getWords(string const& s, vector<string> &container)
{
    int n = 0;
	bool quotation = false;
	container.clear();
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

WRAPPER(ExampleTable)
PyObject *Network_readNetwork(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(fn) -> Network")
{
  PyTRY

	TNetwork *graph;
	PNetwork wgraph;
	TDomain *domain = new TDomain();
	PDomain wdomain = domain;
	TExampleTable *table;
	PExampleTable wtable;
	int directed = 0;
	//cout << "readNetwork" << endl;
	char *fn;

	if (!PyArg_ParseTuple(args, "s|i:Network.readNetwork", &fn, &directed))
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
				}
				else if (stricmp(words[0].c_str(), "*arcs") == 0)
				{
					directed = 1;
					break;
				}
			}
		}
		file.close();
	}

	ifstream file1(fn);
	if (file1.is_open())
	{
		// read head
		while (!file1.eof())
		{
			getline (file1, line);
			vector<string> words;
			int n = getWords(line, words);
			//cout << line << "  -  " << n << endl;
			if (n > 0)
			{
				if (stricmp(words[0].c_str(), "*vertices") == 0)
				{
					//cout << "Vertices" << endl;
					if (n > 1)
					{
						istringstream strVertices(words[1]);
						strVertices >> nVertices;
						if (nVertices == 0)
						{
							file1.close();
							PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
						}

						//cout << "nVertices: " << nVertices << endl;
					}
					else
					{
						file1.close();
						PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
					}

					break;
				}
			}
		}

		if (nVertices <= 1) {
			file1.close();
			PYERROR(PyExc_SystemError, "invalid file1 format; invalid number of vertices (less than 1)", NULL);
		}

		graph = new TNetwork(nVertices, 0, directed == 1);
		wgraph = graph;

		TFloatVariable *indexVar = new TFloatVariable("index");
		indexVar->numberOfDecimals = 0;
		domain->addVariable(indexVar);
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
		int row = 0;
		while (!file1.eof())
		{
			getline(file1, line);
			vector<string> words;
			int n = getWords(line, words);
			//cout << line << "  -  " << n << endl;
			if (n > 0)
			{
				TExample *example = new TExample(domain);

				if ((stricmp(words[0].c_str(), "*arcs") == 0) || (stricmp(words[0].c_str(), "*edges") == 0))
					break;

				float index = -1;
				istringstream strIndex(words[0]);
				strIndex >> index;
				if ((index <= 0) || (index > nVertices))
				{
					file1.close();
					PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
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
						double coor = -1;
						istringstream strCoor(words[i]);
						strCoor >> coor;

						//if ((coor < 0) || (coor > 1))
						//	break;

						//cout << xyz[i] << ": " << coor << endl;
						(*example)[i] = TValue((float)coor);

						if (i == 2)
							graph->pos[0][row] = coor;

						if (i == 3)
							graph->pos[1][row] = coor;

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
								file1.close();
								PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
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
								file1.close();
								PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
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
								file1.close();
								PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
							}

							//cout << "bw: " << words[i] << endl;
							(*example)[7] = TValue(new TStringValue(words[i]), STRINGVAR);
						}
						i++;
					}

				}
				example->id = getExampleId();
				table->push_back(example);
				//cout << "push back" <<endl;
			}

			row++;
		}
		// read arcs
		vector<string> words;
		int n = getWords(line, words);
		if (n > 0)
		{
			if (stricmp(words[0].c_str(), "*arcs") == 0)
			{
				while (!file1.eof())
				{
					getline (file1, line);
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
							int i3 = -1;
							istringstream strI1(words[0]);
							istringstream strI2(words[1]);
							istringstream strI3(words[2]);
							strI1 >> i1;
							strI2 >> i2;
							strI3 >> i3;

							if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices))
							{
								file1.close();
								PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
							}

							if (i1 == i2)
								continue;

							//cout << "i1: " << i1 << " i2: " << i2 << endl;
							*graph->getOrCreateEdge(i1 - 1, i2 - 1) = i3;
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
				while (!file1.eof())
				{
					getline (file1, line);
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

						int i3 = 1;
            if (n > 2) {
  						istringstream strI3(words[2]);
	  					strI3 >> i3;
	  			  }

						if ((i1 <= 0) || (i1 > nVertices) || (i2 <= 0) || (i2 > nVertices))
						{
							file1.close();
							PYERROR(PyExc_SystemError, "invalid file1 format", NULL);
						}

						if (i1 == i2)
							continue;

						*graph->getOrCreateEdge(i1 - 1, i2 - 1) = i3;

						if (directed == 1) {
							*graph->getOrCreateEdge(i2 - 1, i1 - 1) = i3;
						}
					}
				}
			}
		}

		file1.close();
	}
	else
	{
	  PyErr_Format(PyExc_SystemError, "unable to open file1 '%s'", fn);
	  return NULL;
	}

  graph->items = wtable;
	return WrapOrange(wgraph);

  PyCATCH
}
#include "network.px"
