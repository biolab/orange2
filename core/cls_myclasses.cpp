/* When the files are so small, it makes sense to merge this file and myclasses.cpp */


#include "myclasses.hpp"


/* We shall just use the default constructor.
   If the user wants to set the seed in constructor, he can call
    l = mymodule.MyLearner(randomSeed=12)
*/

C_CALL(MyLearner, Learner, "([randomSeed=]) -/-> MyClassifier")

// Nothing else to be done for the learner. The interface for the call operator is inherited



/* Here we shall define a constructor.
   The first argument will have to be a variable (class attribute),
   while the second can be either integer or an instance of RandomGenerator.
   In each case, we get or construct a random generator, then we construct
   an instance of MyClassifier and pass it the random generator and the
   class attribute. We wrap and return the constructed classifier. */

PyObject *MyClassifier_new(PyTypeObject *type, PyObject *args, PyObject *keyws) BASED_ON(Classifier, "(classVar | examples, [randomGenerator | int])")
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


/* For the filter, we shall use the default constructor and the inherited
   call operator */
C_NAMED(MyFilter, Filter, "([randomGenerator=]) -> MyFilter")


#include "px/cls_myclasses.px"