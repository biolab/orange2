"""
.. index: knn

Module :mod:`orange.classification.knn` includes classes for classification based on nearest neighbours
algorithm and also classes for finding instances near the given one. 

====================
k Nearest Neighbours
====================

kNN is one of the simplest learning techniques - the learner only needs to 
store the instances, while the classifier does its work by observing the most 
similar instances of the instance to be classified.

.. class:: kNNLearner(k, distanceConstructor, weightID)

    :param instances: table of instances
    :type instances: Orange.data.Table
    
    :param k: number of nearest neighbours used in classification
    :type k: int
    
    :param weightID: id of meta attribute with instance weights
    :type weightID: int
    
    :rtype: :class:`kNNLearner`
    
    .. method:: __call__(instances)
        
        Return learned kNNClassifier
        
        :param instances: table of instances
        :type instances: Orange.data.Table
        
        :rtype: :class:`kNNClassifier`


    .. attribute:: k
    
    Number of neighbours. If set to 0 (which is also the default value), the 
    square root of the number of instances is used.
    
    .. attribute:: rankWeight
    
    Enables weighting by ranks (default: :obj:`true`)
    
    .. attribute:: distanceConstructor
    
    A component that constructs the object for measuring distances between 
    instances.

kNNLearner first constructs an object for measuring distances between 
instances. distanceConstructor is used if given; otherwise, Euclidean 
metrics will be used. :class:`kNNLearner` then constructs an instance of 
:class:`FindNearest_BruteForce`. Together with ID of meta feature with 
weights of instances, :obj:`k` and :obj:`rankWeight`, it is passed to a :class:`kNNClassifier`.

.. class:: kNNClassifier(domain, weightID, k, FindNearest, rankWeight, nExamples)

    .. method:: __call__(instance)
    
        :param instance: given instance to be classified
        :type instance: Orange.data.Instance
        
        :param return_type: return value and probabilities, only value or only probabilities
        :type return_type: Orange.classifier.getBoth, Orange.classifier.getValue, Orange.classifier.getProbilities
        
        :rtype: :class:`Orange.data.Value`, :class:`Orange.statistics.distribution`, or a tuple with both
        
    .. method:: findNearest(instance)
    
    A component that finds nearest neighbours of a given instance.
        
    :param instance: given instance
    :type instance: Orange.data.instance
        
    :rtype: :class:`Orange.data.instance`
    
    
    .. attribute:: k
    
    Number of neighbours. If set to 0 (which is also the default value), 
    the square root of the number of examples is used.
    
    .. attribute:: rankWeight
    
    Enables weighting by ranks (default: :obj:`true`).
    
    .. attribute:: weightID
    
    ID of meta attribute with weights of examples
    
    .. attribute:: nExamples
    
    The number of learning instances. It is used to compute the number of 
    neighbours if :obj:`k` is zero.

When called to classify an instance, the classifier first calls 
:meth:`kNNClassifier.findNearest` 
to retrieve a list with :obj:`k` nearest neighbours. The component 
:meth:`kNNClassifier.findNearest` has 
a stored table of instances (those that have been passed to the learner) 
together with their weights. If instances are weighted (non-zero 
:obj:`weightID`), weights are considered when counting the neighbours.

If :meth:`kNNClassifier.findNearest` returns only one neighbour 
(this is the case if k=1), :class:`kNNClassifier` returns the neighbour's
class.

Otherwise, the retrieved neighbours vote about the class prediction
(or probability of classes). Voting has double weights. As first, if
instances are weighted, their weights are respected. Secondly, nearer
neighbours have greater impact on the prediction; weight of instance
is computed as exp(-t:sup:`2`/s:sup:`2`), where the meaning of t depends
on the setting of :obj:`rankWeight`.

* if :obj:`rankWeight` is false, t is a distance from the instance being
  classified
* if :obj:`rankWeight` is true, neighbours are ordered and t is the position
  of the neighbour on the list (a rank)


In both cases, s is chosen so that the impact of the farthest instance is
0.001.

Weighting gives the classifier certain insensitivity to the number of
neighbours used, making it possible to use large k's.

The classifier can treat continuous and discrete features, and can even
distinguish between ordinal and nominal features. See information on
distance measuring for details.

Examples
========

We will test the learner on 'iris' dataset. We shall split it onto train
(80%) and test (20%) sets, learn on training instances and test on five
randomly selected test instances.

part of (`knnlearner.py`_, uses `iris.tab`_)

.. literalinclude:: code/knnExample1.py

The output of this code is:: 
    
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The secret of kNN's success is that the instances in iris dataset appear in
three well separated clusters. The classifier's accuraccy will remain
excellent even with very large or small number of neighbours.

As many experiments have shown, a selection of instances distance measure
does not have a greater and predictable effect on the performance of kNN
classifiers. So there is not much point in changing the default. If you
decide to do so, you need to set the distanceConstructor to an instance
of one of the classes for distance measuring.

.. literalinclude:: code/knnExample2.py

The output of this code is::

    Iris-virginica Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The result is still perfect.

.. _iris.tab: code/iris.tab
.. _knnlearner.py: code/knnlearner.py

.. index: fnn

==========================
Finding Nearest Neighbours
==========================

Orange provides classes for finding the nearest neighbours of the given
reference instance. While we might add some smarter classes in future, we
now have only two - abstract classes that defines the general behaviour of
neighbour searching classes, and classes that implement brute force search.

As usually in Orange, there is a pair of classes: a class that does the work
(:class:`FindNearest`) and a class that constructs it ("learning" - getting the
instances and arranging them in an appropriate data structure that allows for
searching) (:class:`FindNearestConstructor`).

.. class:: FindNearest

    A class for brute force search for nearest neighbours. It stores a table of
    instances (it's its own copy of instances, not only Orange.data.Table with
    references to another Orange.data.Table). When asked for neighbours, it
    measures distances to all instances, stores them in a heap and returns the
    first k as an Orange.data.Table with references to instances stored in
    FindNearest's field instances).
    
    .. attribute:: distance
    
    a component that measures distance between examples
    
    .. attribute:: examples
    
    a stored list of instances
    
    .. attribute:: weightID
    
    ID of meta attribute with weight
    
    .. method:: __call__(instance, n)
    
    :param instance: given instance
    :type instance: Orange.data.instance
    
    :param n: number of neighbours
    :type n: int
    
    :rtype: list(Orange.data.instance)
    
.. class:: FindNearestConstructor()

    A class that constructs FindNearest. It calls the inherited 
    distanceConstructor and then passes the constructed distance measure,
    among with instances, weightIDand distanceID, to the just constructed
    instance of FindNearest_BruteForce.
    
    If there are more instances with the same distance fighting for the last
    places, the tie is resolved by randomly picking the appropriate number of
    instances. A local random generator is constructed and initiated by a constant
    computed from the reference instance. The effect of this is that same random
    neighbours will be chosen for the instance each time FindNearest_BruteForce
    is called.
    
    .. attribute:: distanceConstructor
    
    A component of class ExamplesDistanceConstructor that "learns" to measure
    distances between instances. Learning can be, for instances, storing the
    ranges of continuous features or the number of value of a discrete feature
    (see the page about measuring distances for more information). The result of
    learning is an instance of ExamplesDistance that should be used for measuring
    distances between instances.
    
    .. attribute:: includeSame
    
    Tells whether to include the examples that are same as the reference; default is true.
    
    .. method:: __call__(table, weightID, distanceID)
    
        Constructs an instance of FindNearest that would return neighbours of a
        given instance, obeying weightID when counting them (also, some measures
        of distance might consider weights as well) and store the distances in a
        meta attribute with ID distanceID.
    
        :param table: table of instances
        :type table: Orange.data.Table
        
        :param weightID: id of meta attribute with weights of instances
        :type weightID: int
        
        :param distanceID: id of meta attribute that will save distances
        :type distanceID: int
        
        :rtype: :class:`FindNearest`

Example
=======

The following script shows how to find the five nearest neighbours of the
first instance in the lenses dataset.

(`knnInstanceDistance.py`_, uses `lenses.tab`_)

.. literalinclude:: code/knnInstanceDistance.py

.. _lenses.tab: code/lenses.tab
.. _knnInstanceDistance.py: code/knnInstanceDistance.py

"""

from Orange.core import \
            kNNLearner, \
            FindNearest_BruteForce as FindNearest, \
            FindNearestConstructor_BruteForce as FindNearestConstructor, \
            kNNClassifier, \
            P2NN
            #FindNearest
            #FindNearestConstructor
