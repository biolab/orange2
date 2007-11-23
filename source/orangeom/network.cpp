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
  vector<int> neighbours;
	for(int v1 = 0; v1 < graph->nVertices; v1++) {
		graph->getNeighboursFrom_Single(v1, neighbours);

		ITERATE(vector<int>, ni, neighbours) {
      double *w = getOrCreateEdge(v1, *ni);
			*w = *graph->getOrCreateEdge(v1, *ni);
		}
	}
}

TNetwork::TNetwork(const int &nVert, const int &nEdge, const bool dir)
: TGraphAsList(nVert, nEdge, dir)
{
  
}

TNetwork::~TNetwork()
{

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
      TGraphAsList *graph = &dynamic_cast<TGraphAsList &>(PyOrange_AsOrange(pygraph).getReference());

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
    else if (PyArg_ParseTuple(args, "ii|i:Network", &nVertices, &directed, &nEdgeTypes))
    {
		  return WrapNewOrange(mlnew TNetwork(nVertices, nEdgeTypes, directed != 0), type);
    }
    else
    {
      PYERROR(PyExc_TypeError, "Network.__new__: number of vertices directedness and optionaly, number of edge types expected", PYNULL);
    }
	  
	PyCATCH
}

#include "network.px"