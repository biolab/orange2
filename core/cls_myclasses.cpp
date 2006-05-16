#include "coreclasses.hpp"


C_CALL(RandomForestLearner, Learner, "() -/-> RandomForest")

BASED_ON(RandomForest, ClassifierFD)
/*
PyObject *RandomForest_new(PyTypeObject *type, PyObject *args, PyObject *keyws) BASED_ON(Classifier, "(classVar | examples, [randomGenerator | int])")
{
  PyTRY {
    PVariable classVar;
    PRandomGenerator rg;

    int seed;
    if (PyArg_ParseTuple(args, "O&|i", cc_Variable, &classVar, &seed)) {
      rg = new TRandomGenerator(seed);
    }

    else {
      PyErr_Clear();
      if (!PyArg_ParseTuple(args, "O&", cc_Variable, &classVar, cc_RandomGenerator, &rg))
        PYERROR(PyExc_AttributeError, "RandomClassifier.__new__ expects a class variable and, optionally, a random generator or a seed", NULL);
    }

    TClassifier *rclass = new TMyClassifier(classVar, rg);
    return WrapNewOrange(rclass, type);
  }
  PyCATCH;
}
*/


#include "px/cls_coreclasses.px"