"""

.. index:: feature; classifier

****************************************
Classifier for variable (``classifier``)
****************************************


Classifiers from variable predict the class based on the value of a single variable.
While they can be used for making predictions, they actually play a different,
yet important role in Orange. They are used not to predict class values
but to compute variable's values. For instance, when a continuous variable
is discretized and replaced by a discrete variable, an instance of a classifier
from variable takes care of automatic value computation when needed.


Similarly, a classifier from variable usually decides the branch when a data instance
is classified in decision trees.

There are two classifiers from variable; the simpler `obj:ClassifierFromVarFD`
supposes that example is from some fixed domain and the safer `obj:ClassifierFromVar` does not.
You should primarily use the latter, moreover since it uses a caching schema which helps the
class to be practically as fast as the former.

Both classifiers can be given a transformer that can modify the value.
In discretization, for instance, the transformer is responsible to compute
a discrete interval for a continuous value of the original variable.

.. literalinclude:: code/classifier-from-var-example.py
    :lines: 1-5


ClassifierFromVar
=================

.. index::
   single: feature; classifier

.. py:class:: ClassifierFromVar(whichVar, transformer) 
    
    lala
    
    .. attribute:: variable1[, variable2[, variable3]](read only)
        

    .. attribute:: whichVar

        The descriptor of the attribute whose value is to be returned.

    .. attribute:: transformer        

        The transformer for the value. It should be a class derived from TransformValue, but you can also use a callback function.

    .. distributionForUnknown::

        The distribution that is returned when the whichVar's value is undefined.    




distributionForUnknown

When given an example, ClassifierFromVar will return transformer(example[whichVar]). whichVar can be either an ordinary attribute, a meta attribute or an attribute which is not defined for the example but has getValueFrom that can be used to compute the value. If none goes through or if the value found is unknown, a Value of subtype Distribution containing distributionForUnknown is returned.

The class stores the domain version for the last example and its position in the domain. If consecutive examples come from the same domain (which is usually the case), ClassifierFromVar is just two simple ifs slower than ClassifierFromVarFD.

As you might have guessed, the crucial component here is the transformer. Let us, for sake of demonstration, load a Monk 1 dataset and construct an attribute e1 that will have value "1", when e is "1", and "not 1" when e is different than 1. There are many ways to do it, and that same problem is covered in different places in Orange documentation. Although the way presented here is not the simplest, it will serve to demonstrate how ClassifierFromVar works.




ClassifierFromVarFD
===================

"""

from Orange.core import ClassifierFromVar
from Orange.core import ClassifierFromVarFD


