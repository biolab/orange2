#include "orange_api.hpp"
#include "graph.hpp"
#include "../orange/px/externs.px"

PyObject *graphOptimization(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(Graph, steps, coordinates) -> None")
{
  PyTRY
    PyObject *pygraph;
    int steps;
    PyObject *pycoordinates;
    if (!PyArg_ParseTuple(args, "OiO:graphOptimization", &pygraph, &steps, &pycoordinates))
      return NULL;

    TGraphAsMatrix *graph = &dynamic_cast<TGraphAsMatrix &>(PyOrange_AsOrange(pygraph).getReference());

    if (graph->nVertices < 2)
      PYERROR(PyExc_AttributeError, "graph has less than two nodes", NULL);



    return Py_BuildValue("ii", graph->getEdge(0, 1) ? 1 : 0, (int)(graph->edges));
  PyCATCH
}
