#include "wml/WmlVector2.h"
#include "wml/WmlDelaunay2a.h"
using namespace Wml;

//#include "module.hpp"
#include "lib_kernel.hpp"
#include "cls_orange.hpp"

#include "graph.hpp"
#include "examplegen.hpp"

PGraph triangulate(PExampleGenerator egen, const int &nEdgeTypes)
{
  if ((egen->domain->attributes->size() != 2)
       || (egen->domain->attributes->at(0)->varType != TValue::FLOATVAR)
	   || (egen->domain->attributes->at(1)->varType != TValue::FLOATVAR)
	   || !egen->domain->classVar
	   || (egen->domain->classVar->varType != TValue::INTVAR))
    raiseError("traingulate expects two continuous attributes and a discrete class");

  const int nofex = egen->numberOfExamples();

  Vector2<float> *points = new Vector2<float>[nofex];
  Vector2<float> *pi = points;

  for(TExampleIterator ei(egen->begin()); ei; ++ei, pi++) {
    if ((*ei)[0].isSpecial() || (*ei)[1].isSpecial())
      raiseError("triangulate cannot handle unknown values");

    *pi = Vector2<float>((*ei)[0].floatV, (*ei)[1].floatV);
  }

  int nTriangles;
  int *triangles, *adjacent;
  Delaunay2a<float> delaunay(nofex, points, nTriangles, triangles, adjacent);
  delete adjacent;

  TGraph *graph = new TGraphAsList(nofex, nEdgeTypes, 0);
  PGraph wgraph = graph;

  for(int *ti = triangles, *te = ti+nTriangles*3; ti!=te; ti+=3) {
    for(int se = 3; se--; ) {
      float *gedge = graph->getOrCreateEdge(ti[se], ti[(se+1) % 3]);
      for(int et = nEdgeTypes; et--; *gedge++ = 1.0);
    }
  }

  delete triangles;

  return wgraph;
}


PyObject *py_triangulate(PyObject *, PyObject *args, PyObject *)
{
  PyTRY
    PExampleGenerator egen;
    int nEdgeTypes = 1;
    if (!PyArg_ParseTuple(args, "O&|i:triangulate", pt_ExampleGenerator, &egen, &nEdgeTypes))
      return NULL;

    PyObject *res = WrapOrange(triangulate(egen, nEdgeTypes));

    PyObject *objs = PyString_FromString("objects");
    Orange_setattrLow((TPyOrange *)res, objs, PyTuple_GET_ITEM(args, 0), false);
    Py_DECREF(objs);
    return res;
  PyCATCH
}

