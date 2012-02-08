.. index: classification from variable
.. index:
   single: classification; classification from variable

************************
Classifier from variable 
************************

Classifiers from variable are used not to predict class values
but to compute variable's values from another variables.
For instance, when a continuous variable is discretized and replaced by
a discrete variable, an instance of a classifier from variable takes
care of automatic value computation when needed.

There are two classifiers from variable; the simpler :obj:`~Orange.classification.ClassifierFromVarFD`
supposes that example is from some fixed domain and the safer
:obj:`~Orange.classification.ClassifierFromVar` does not.

Both classifiers can be given a transformer that can modify the value.
In discretization, for instance, the transformer is responsible to compute
a discrete interval for a continuous value of the original variable.


ClassifierFromVar
=================

.. index::
   single: feature; classifier

.. class:: ClassifierFromVar(which_var, transformer) 
    
    Compute variable's values from variable which_var using
    transformation defined by transformer.        

    .. attribute:: which_var

        The descriptor of the attribute whose value is to be returned.

    .. attribute:: transformer        

        The transformer for the value. It should be a class derived from
        TransformValue, but you can also use a callback function.

    .. distribution_for_unknown::

        The distribution that is returned when the which_var's value is undefined.    


When given an instance, :obj:`~Orange.classification.ClassifierFromVar` will return
transformer(instance[which_var]). Attribute which_var can be either an ordinary variable,
a meta variable or a variable which is not defined for the instance but has getValueFrom
that can be used to compute the value. If none goes through or if the value found is unknown,
a Value of subtype Distribution containing distributionForUnknown is returned.

The class stores the domain version for the last example and its position in the domain.
If consecutive examples come from the same domain (which is usually the case),
:obj:`~Orange.classification.ClassifierFromVar` is just two simple ifs slower than 
:obj:`~Orange.classification.ClassifierFromVarFD`.

As you might have guessed, the crucial component here is the transformer.
Let us, for sake of demonstration, load a Monk 1 dataset and construct an attribute
e1 that will have value "1", when e is "1", and "not 1" when e is different than 1.
There are many ways to do it, and that same problem is covered in different places
in Orange documentation. Although the way presented here is not the simplest,
it will serve to demonstrate how ClassifierFromVar works.


.. literalinclude:: code/classifier-from-var-example.py
    :lines: 1-19

ClassifierFromVarFD
===================

.. class:: ClassifierFromVarFD

    :obj:`~Orange.classification.ClassifierFromVarFD` is very similar to :obj:`~Orange.classification.ClassifierFromVar` except that the variable
    is not given as a descriptor (like which_var) but as an index. The index can be
    either a position of the variable in the domain or a meta-id. Given that :obj:`~Orange.classification.ClassifierFromVarFD`
    is practically no faster than :obj:`~Orange.classification.ClassifierFromVar` (and can in future even be merged with the latter),
    you should seldom need to use the class.

    .. attribute:: domain (inherited from ClassifierFromVarFD)
    
        The domain on which the classifier operates.

    .. attribute:: position

        The position of the attribute in the domain or its meta-id.

    .. attriubte:: transformer

        The transformer for the value.

    .. attribute:: distribution_for_unknown

        The distribution that is returned when the which_var's value is undefined.

When an example is passed to :obj:`~Orange.classification.ClassifierFromVarFD`,
it is first checked whether it is
from the correct domain; an exception is raised if not. If the domain is OK,
the corresponding attribute value is retrieved, transformed and returned.

:obj:`~Orange.classification.ClassifierFromVarFD`'s twin brother, :obj:`~Orange.classification.ClassifierFromVar`, can also handle variables that
are not in the instances' domain or meta-variables, but can be computed therefrom by using
their getValueFrom. Since :obj:`~Orange.classification.ClassifierFromVarFD` doesn't store attribute descriptor but
only an index, such functionality is obviously impossible.

To rewrite the above script to use :obj:`~Orange.classification.ClassifierFromVarFD`,
we need to set the domain and the e's index to position
(equivalent to setting which_var in :obj:`~Orange.classification.ClassifierFromVar`).
The initialization of :obj:`~Orange.classification.ClassifierFromVarFD` thus goes like this:

.. literalinclude:: code/classifier-from-var-example.py
    :lines: 21-25