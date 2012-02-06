"""
.. index: k-nearest neighbors (kNN)
.. index:
   single: classification; k-nearest neighbors (kNN)
   
*****************************
k-nearest neighbors (``knn``)
*****************************

The `nearest neighbors
algorithm <http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm>`_ is one of the most basic,
`lazy <http://en.wikipedia.org/wiki/Lazy_learning>`_ machine learning algorithms.
The learner only needs to store the instances of training data, while the classifier
does all the work by searching this list for the instances most similar to
the data instance being classified:

.. literalinclude:: code/knnExample0.py

.. class:: kNNLearner(k, distance_constructor, weight_id)

    Lazy classifier that stores instances from the training set. Constructor
    parameters set the corresponding attributes.

    .. attribute:: k

        number of nearest neighbors used in classification. If set to 0
        (default), the square root of the numbers of instances is used.

    .. attribute:: distance_constructor

        component that constructs the object for measuring distances between
        instances. Defaults to :class:`~Orange.distance.instances.EuclideanConstructor`.

    .. attribute:: weight_id
    
        id of meta attribute with instance weights

    .. attribute:: rank_weight

        Enables weighting by ranks (default: :obj:`true`)

    .. method:: __call__(instances)

        Return a learned :class:`~kNNClassifier`. Learning consists of
        constructing a distance measure and passing it to the classifier
        along with :obj:`instances` and all attributes.

        :param instances: training instances
        :type instances: :class:`~Orange.data.Table`


.. class:: kNNClassifier(domain, weight_id, k, find_nearest, rank_weight, n_examples)

    .. method:: __call__(instance, return_type)

        :param instance: given instance to be classified
        :type instance: Orange.data.Instance
        
        :param return_type: return value and probabilities, only value or only
                            probabilities
        :type return_type: :obj:`~Orange.classification.Classifier.GetBoth`,
                           :obj:`~Orange.classification.Classifier.GetValue`,
                           :obj:`~Orange.classification.Classifier.GetProbabilities`
        
        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both
        
    .. method:: find_nearest(instance)
    
    A component which finds the nearest neighbors of a given instance.
        
    :param instance: given instance
    :type instance: :class:`~Orange.data.Instance`
        
    :rtype: :class:`Orange.data.Instance`
    
    
    .. attribute:: k
    
        Number of neighbors. If set to 0 (which is also the default value), 
        the square root of the number of examples is used.
    
    .. attribute:: rank_weight
    
        Enables weighting by rank (default: :obj:`true`).
    
    .. attribute:: weight_id
    
        ID of meta attribute with weights of examples
    
    .. attribute:: n_examples
    
        The number of learning instances. It is used to compute the number of 
        neighbors if the value of :attr:`kNNClassifier.k` is zero.

When called to classify an instance, the classifier first calls 
:meth:`kNNClassifier.find_nearest` 
to retrieve a list with :attr:`kNNClassifier.k` nearest neighbors. The
component :meth:`kNNClassifier.find_nearest` has 
a stored table of instances (those that have been passed to the learner) 
together with their weights. If instances are weighted (non-zero 
:obj:`weight_ID`), weights are considered when counting the neighbors.

If :meth:`kNNClassifier.find_nearest` returns only one neighbor 
(this is the case if :obj:`k=1`), :class:`kNNClassifier` returns the
neighbor's class.

Otherwise, the retrieved neighbors vote about the class prediction
(or probability of classes). Voting has double weights. As first, if
instances are weighted, their weights are respected. Secondly, nearer
neighbors have a greater impact on the prediction; the weight of instance
is computed as exp(-t:sup:`2`/s:sup:`2`), where the meaning of t depends
on the setting of :obj:`rank_weight`.

* if :obj:`rank_weight` is :obj:`false`, :obj:`t` is the distance from the
  instance being classified
* if :obj:`rank_weight` is :obj:`true`, neighbors are ordered and :obj:`t`
  is the position of the neighbor on the list (a rank)


In both cases, :obj:`s` is chosen so that the impact of the farthest instance
is 0.001.

Weighting gives the classifier a certain insensitivity to the number of
neighbors used, making it possible to use large :obj:`k`'s.

The classifier can treat continuous and discrete features, and can even
distinguish between ordinal and nominal features. See information on
distance measuring for details.

Examples
--------

The learner will be tested on an 'iris' data set. The data will be split 
into training (80%) and testing (20%) instances. We will use the former 
for "training" the classifier and test it on five testing instances 
randomly selected from a part of (:download:`knnlearner.py <code/knnlearner.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/knnExample1.py

The output of this code is:: 
    
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The secret to kNN's success is that the instances in the iris data set appear in
three well separated clusters. The classifier's accuracy will remain
excellent even with a very large or very small number of neighbors.

As many experiments have shown, a selection of instances of distance measures
has neither a greater nor more predictable effect on the performance of kNN
classifiers. Therefore there is not much point in changing the default. If you
decide to do so, the distance_constructor must be set to an instance
of one of the classes for distance measuring. This can be seen in the following
part of (:download:`knnlearner.py <code/knnlearner.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/knnExample2.py

The output of this code is::

    Iris-virginica Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The result is still perfect.

.. index: fnn


Finding nearest neighbors
-------------------------

Orange provides classes for finding the nearest neighbors of a given
reference instance. While we might add some smarter classes in the future, we
now have only two - abstract classes that define the general behavior of
neighbor searching classes, and classes that implement brute force search.

As is the norm in Orange, there are a pair of classes: a class that does the work
(:class:`FindNearest`) and a class that constructs it ("learning" - getting the
instances and arranging them in an appropriate data structure that allows for
searching) (:class:`FindNearestConstructor`).

.. class:: FindNearest

    A class for a brute force search for nearest neighbors. It stores a table 
    of instances (it's its own copy of instances, not only Orange.data.Table
    with references to another Orange.data.Table). When asked for neighbors,
    it measures distances to all instances, stores them in a heap and returns 
    the first k as an Orange.data.Table with references to instances stored in
    FindNearest's field instances).
    
    .. attribute:: distance
    
        a component that measures the distance between examples
    
    .. attribute:: examples
    
        a stored list of instances
    
    .. attribute:: weight_ID
    
        ID of meta attribute with weight
    
    .. method:: __call__(instance, n)
    
    :param instance: given instance
    :type instance: Orange.data.Instance
    
    :param n: number of neighbors
    :type n: int
    
    :rtype: list of :obj:`Orange.data.Instance`
    
.. class:: FindNearestConstructor()

    
    A class that constructs FindNearest. It calls the inherited
    distance_constructor, which constructs a distance measure.
    The distance measure, along with the instances weight_ID and
    distance_ID, is then passed to the just constructed instance
    of FindNearest_BruteForce.

    If there are more instances with the same distance fighting for the last
    places, the tie is resolved by randomly picking the appropriate number of
    instances. A local random generator is constructed and initiated by a
    constant computed from the reference instance. The effect of this is that
    the same random neighbors will be chosen for the instance each time
    FindNearest_BruteForce
    is called.
    
    .. attribute:: distance_constructor
    
        A component of class ExamplesDistanceConstructor that "learns" to
        measure distances between instances. Learning can mean, for instances,
        storing the ranges of continuous features or the number of values of
        a discrete feature (see the page about measuring distances for more
        information). The result of learning is an instance of 
        ExamplesDistance that should be used for measuring distances
        between instances.
    
    .. attribute:: include_same
    
        Tells whether or not to include the examples that are same as the reference;
        the default is true.
    
    .. method:: __call__(table, weightID, distanceID)
    
        Constructs an instance of FindNearest that would return neighbors of
        a given instance, obeying weight_ID when counting them (also, some 
        measures of distance might consider weights as well) and storing the 
        distances in a meta attribute with ID distance_ID.
    
        :param table: table of instances
        :type table: Orange.data.Table
        
        :param weight_ID: id of meta attribute with weights of instances
        :type weight_ID: int
        
        :param distance_ID: id of meta attribute that will save distances
        :type distance_ID: int
        
        :rtype: :class:`FindNearest`

Examples
--------

The following script (:download:`knnInstanceDistance.py <code/knnInstanceDistance.py>`, uses :download:`lenses.tab <code/lenses.tab>`)
shows how to find the five nearest neighbors of the first instance
in the lenses dataset.

.. literalinclude:: code/knnInstanceDistance.py

"""

from Orange.core import \
            kNNLearner, \
            FindNearest_BruteForce as FindNearest, \
            FindNearestConstructor_BruteForce as FindNearestConstructor, \
            kNNClassifier, \
            P2NN
            #FindNearest
            #FindNearestConstructor
