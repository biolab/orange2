"""
.. index: k-nearest neighbors (kNN)
.. index:
   single: classification; k-nearest neighbors (kNN)
   
*******************
k-nearest neighbors
*******************

The module includes implementation of `nearest neighbors 
algorithm <http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm>`_ and classes
for finding nearest instances according to chosen distance metrics.

k-nearest neighbor algorithm
============================

Nearest neighbors algorithm is one of most basic, 
`lazy <http://en.wikipedia.org/wiki/Lazy_learning>`_ machine learning algorithms.
The learner only needs to store the training data instances, while the classifier
does all the work by searching this list for the most similar instances for 
the data instance being classified:

.. literalinclude:: code/knnExample0.py

.. class:: kNNLearner(k, distanceConstructor, weightID)

    :param instances: table of instances
    :type instances: Orange.data.Table
    
    :param k: number of nearest neighbours used in classification
    :type k: int
    
    :param weightID: id of meta attribute with instance weights
    :type weightID: int
    
    :rtype: :class:`kNNLearner`
    
    .. method:: __call__(instances)
        
        Return instance of :class:`kNNClassifier` that learns from the
        :obj:`instances`.
        
        :param instances: table of instances
        :type instances: Orange.data.Table
        
        :rtype: :class:`kNNClassifier`


    .. attribute:: k
    
        Number of neighbors. If set to 0 (which is also the default value), 
        the square root of the number of instances is used.
    
    .. attribute:: rankWeight
    
        Enables weighting by ranks (default: :obj:`true`)
    
    .. attribute:: distanceConstructor
    
        component that constructs the object for measuring distances between 
        instances.

kNNLearner first constructs an object for measuring distances between 
instances. distanceConstructor is used if given; otherwise, Euclidean 
metrics will be used. :class:`kNNLearner` then constructs an instance of 
:class:`FindNearest_BruteForce`. Together with ID of meta feature with 
weights of instances, :attr:`kNNLearner.k` and :attr:`kNNLearner.rankWeight`,
it is passed to a :class:`kNNClassifier`.

.. class:: kNNClassifier(domain, weightID, k, FindNearest, rankWeight, \
nExamples)

    .. method:: __call__(instance)
    
        :param instance: given instance to be classified
        :type instance: Orange.data.Instance
        
        :param return_type: return value and probabilities, only value or only
                            probabilities
        :type return_type: Orange.classifier.getBoth, 
                           Orange.classifier.getValue,
                           Orange.classifier.getProbilities
        
        :rtype: :class:`Orange.data.Value`,
                :class:`Orange.statistics.distribution`, or a tuple with both
        
    .. method:: findNearest(instance)
    
    A component that finds nearest neighbors of a given instance.
        
    :param instance: given instance
    :type instance: Orange.data.Instance
        
    :rtype: :class:`Orange.data.Instance`
    
    
    .. attribute:: k
    
        Number of neighbors. If set to 0 (which is also the default value), 
        the square root of the number of examples is used.
    
    .. attribute:: rankWeight
    
        Enables weighting by ranks (default: :obj:`true`).
    
    .. attribute:: weightID
    
        ID of meta attribute with weights of examples
    
    .. attribute:: nExamples
    
        The number of learning instances. It is used to compute the number of 
        neighbours if :attr:`kNNClassifier.k` is zero.

When called to classify an instance, the classifier first calls 
:meth:`kNNClassifier.findNearest` 
to retrieve a list with :attr:`kNNClassifier.k` nearest neighbors. The
component :meth:`kNNClassifier.findNearest` has 
a stored table of instances (those that have been passed to the learner) 
together with their weights. If instances are weighted (non-zero 
:obj:`weightID`), weights are considered when counting the neighbors.

If :meth:`kNNClassifier.findNearest` returns only one neighbor 
(this is the case if :obj:`k=1`), :class:`kNNClassifier` returns the
neighbour's class.

Otherwise, the retrieved neighbours vote about the class prediction
(or probability of classes). Voting has double weights. As first, if
instances are weighted, their weights are respected. Secondly, nearer
neighbours have greater impact on the prediction; weight of instance
is computed as exp(-t:sup:`2`/s:sup:`2`), where the meaning of t depends
on the setting of :obj:`rankWeight`.

* if :obj:`rankWeight` is :obj:`false`, :obj:`t` is a distance from the
  instance being classified
* if :obj:`rankWeight` is :obj:`true`, neighbors are ordered and :obj:`t`
  is the position of the neighbor on the list (a rank)


In both cases, :obj:`s` is chosen so that the impact of the farthest instance
is 0.001.

Weighting gives the classifier certain insensitivity to the number of
neighbors used, making it possible to use large :obj:`k`'s.

The classifier can treat continuous and discrete features, and can even
distinguish between ordinal and nominal features. See information on
distance measuring for details.

Examples
--------

We will test the learner on 'iris' data set. We shall split it onto train
(80%) and test (20%) sets, learn on training instances and test on five
randomly selected test instances, in part of 
(`knnlearner.py`_, uses `iris.tab`_):

.. literalinclude:: code/knnExample1.py

The output of this code is:: 
    
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The secret of kNN's success is that the instances in iris data set appear in
three well separated clusters. The classifier's accuracy will remain
excellent even with very large or small number of neighbors.

As many experiments have shown, a selection of instances distance measure
does not have a greater and predictable effect on the performance of kNN
classifiers. So there is not much point in changing the default. If you
decide to do so, you need to set the distanceConstructor to an instance
of one of the classes for distance measuring. This can be seen in the following
part of (`knnlearner.py`_, uses `iris.tab`_):

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


Finding nearest neighbors
=========================

Orange provides classes for finding the nearest neighbors of the given
reference instance. While we might add some smarter classes in future, we
now have only two - abstract classes that defines the general behavior of
neighbor searching classes, and classes that implement brute force search.

As usually in Orange, there is a pair of classes: a class that does the work
(:class:`FindNearest`) and a class that constructs it ("learning" - getting the
instances and arranging them in an appropriate data structure that allows for
searching) (:class:`FindNearestConstructor`).

.. class:: FindNearest

    A class for brute force search for nearest neighbours. It stores a table 
    of instances (it's its own copy of instances, not only Orange.data.Table
    with references to another Orange.data.Table). When asked for neighbours,
    it measures distances to all instances, stores them in a heap and returns 
    the first k as an Orange.data.Table with references to instances stored in
    FindNearest's field instances).
    
    .. attribute:: distance
    
        a component that measures distance between examples
    
    .. attribute:: examples
    
        a stored list of instances
    
    .. attribute:: weightID
    
        ID of meta attribute with weight
    
    .. method:: __call__(instance, n)
    
    :param instance: given instance
    :type instance: Orange.data.Instance
    
    :param n: number of neighbours
    :type n: int
    
    :rtype: list of :obj:`Orange.data.Instance`
    
.. class:: FindNearestConstructor()

    A class that constructs FindNearest. It calls the inherited 
    distanceConstructor and then passes the constructed distance measure,
    among with instances, weightIDand distanceID, to the just constructed
    instance of FindNearest_BruteForce.
    
    If there are more instances with the same distance fighting for the last
    places, the tie is resolved by randomly picking the appropriate number of
    instances. A local random generator is constructed and initiated by a
    constant computed from the reference instance. The effect of this is that
    same random neighbours will be chosen for the instance each time
    FindNearest_BruteForce
    is called.
    
    .. attribute:: distanceConstructor
    
        A component of class ExamplesDistanceConstructor that "learns" to
        measure distances between instances. Learning can be, for instances,
        storing the ranges of continuous features or the number of value of
        a discrete feature (see the page about measuring distances for more
        information). The result of learning is an instance of 
        ExamplesDistance that should be used for measuring distances
        between instances.
    
    .. attribute:: includeSame
    
        Tells whether to include the examples that are same as the reference;
        default is true.
    
    .. method:: __call__(table, weightID, distanceID)
    
        Constructs an instance of FindNearest that would return neighbours of
        a given instance, obeying weightID when counting them (also, some 
        measures of distance might consider weights as well) and store the 
        distances in a meta attribute with ID distanceID.
    
        :param table: table of instances
        :type table: Orange.data.Table
        
        :param weightID: id of meta attribute with weights of instances
        :type weightID: int
        
        :param distanceID: id of meta attribute that will save distances
        :type distanceID: int
        
        :rtype: :class:`FindNearest`

Examples
--------

The following script (`knnInstanceDistance.py`_, uses `lenses.tab`_) 
shows how to find the five nearest neighbours of the first instance
in the lenses dataset.

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
