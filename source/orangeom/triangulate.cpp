#include "wml/WmlVector2.h"
#include "wml/WmlDelaunay2a.h"
using namespace Wml;

#include "orange_api.hpp"
#include "graph.hpp"
#include "examplegen.hpp"

PyObject *triangulate(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(examples[, attr1, attr2, nEdgeTypes]) -> Graph")
{
  PyTRY
    PExampleGenerator egen;
    int nEdgeTypes = 1;
    PyObject *pyvar1 = NULL, *pyvar2 = NULL;
    int attr1 = 0, attr2 = 1;
    if (   !PyArg_ParseTuple(args, "O&|OOi:triangulate", pt_ExampleGenerator, &egen, &pyvar1, &pyvar2, &nEdgeTypes)
        || pyvar1 && !varNumFromVarDom(pyvar1, egen->domain, attr1)
        || pyvar2 && !varNumFromVarDom(pyvar2, egen->domain, attr2))
      return NULL;

    if (  (egen->domain->attributes->at(attr1)->varType != TValue::FLOATVAR)
	     || (egen->domain->attributes->at(attr2)->varType != TValue::FLOATVAR))
      PYERROR(PyExc_TypeError, "triangulate expects continuous attributes", NULL);

    const int nofex = egen->numberOfExamples();

    Vector2<float> *points = new Vector2<float>[nofex];
    Vector2<float> *pi = points;

    PEITERATE(ei, egen) {
      if ((*ei)[attr1].isSpecial() || (*ei)[attr2].isSpecial())
        PYERROR(PyExc_AttributeError, "triangulate cannod handle unknown values", NULL);

      *(pi++) = Vector2<float>((*ei)[attr1].floatV, (*ei)[attr2].floatV);
    }

    int nTriangles;
    int *triangles, *adjacent;
    Delaunay2a<float> delaunay(nofex, points, nTriangles, triangles, adjacent);
    delete adjacent;

    TGraph *graph = new TGraphAsList(nofex, nEdgeTypes, 0);
    PGraph wgraph = graph;
    try {
      for(int *ti = triangles, *te = ti+nTriangles*3; ti!=te; ti+=3) {
        for(int se = 3; se--; ) {
          double *gedge = graph->getOrCreateEdge(ti[se], ti[(se+1) % 3]);
          for(int et = nEdgeTypes; et--; *gedge++ = 1.0);
        }
      }
    }
    catch (...) {
      delete triangles;
      throw;
    }
    delete triangles;

    PyObject *res = WrapOrange(wgraph);

    PyObject *objs = PyString_FromString("objects");
    Orange_setattrLow((TPyOrange *)res, objs, PyTuple_GET_ITEM(args, 0), false);
    Py_DECREF(objs);
    return res;
  PyCATCH
}

#include "px/triangulate.px"