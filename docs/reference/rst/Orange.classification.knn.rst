.. py:currentmodule:: Orange.classification.knn

.. index: k-nearest neighbors (kNN)
.. index:
   single: classification; k-nearest neighbors (kNN)
   
*****************************
k-nearest neighbors (``knn``)
*****************************

The `nearest neighbors algorithm
<http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm>`_ is one
of the most basic, `lazy
<http://en.wikipedia.org/wiki/Lazy_learning>`_ machine learning
algorithms.  The learner only stores the training data, and the
classifier makes predictions based on the instances most similar to
the data instance being classified:

.. literalinclude:: code/knnExample0.py

.. class:: kNNLearner(k, distance_constructor, weight_id)

    Lazy classifier that stores instances from the training set. Constructor
    parameters set the corresponding attributes.

    .. attribute:: k

        Number of nearest neighbors used in classification. If 0
        (default), the square root of the numbers of instances is
        used.

    .. attribute:: distance_constructor

        Component that constructs the object for measuring distances between
        instances. Defaults to :class:`~Orange.distance.Euclidean`.

    .. attribute:: weight_id
    
        Id of meta attribute with instance weights.

    .. attribute:: rank_weight

        If ``True`` (default), neighbours are weighted according to
        their order and not their (normalized) distances to the
        instance that is being classified.

    .. method:: __call__(data)

        Return a :class:`~kNNClassifier`. Learning consists of
        constructing a distance measure and passing it to the
        classifier along with :obj:`instances` and attributes (:obj:`k`,
        :obj:`rank_weight` and :obj:`weight_id`).

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
        
    .. attribute:: find_nearest
    
        A callable component that finds the nearest :obj:`k` neighbors
        of the given instance.
        
        :param instance: given instance
        :type instance: :class:`~Orange.data.Instance`
        :rtype: :class:`Orange.data.Instance`
    
    .. attribute:: k
    
        Number of neighbors. If set to 0 (which is also the default value), 
        the square root of the number of examples is used.
    
    .. attribute:: weight_id
    
        Id of meta attribute with instance weights.

    .. attribute:: rank_weight

        If ``True`` (default), neighbours are weighted according to
        their order and not their (normalized) distances to the
        instance that is being classified.

    .. attribute:: n_examples
    
        The number of learning instances, used to compute the number of 
        neighbors if the value of :attr:`kNNClassifier.k` is zero.

When called to classify instance ``inst``, the classifier first calls
:obj:`kNNClassifier.find_nearest(inst)` to retrieve a list with
:attr:`kNNClassifier.k` nearest neighbors. The component
:meth:`kNNClassifier.find_nearest` has a stored table of training
instances together with their weights. If instances are weighted
(non-zero :obj:`weight_id`), weights are considered when counting the
neighbors.

If :meth:`kNNClassifier.find_nearest` returns only one neighbor (this
is the case if :obj:`k=1`), :class:`kNNClassifier` returns the
neighbor's class.

Otherwise, the retrieved neighbors vote for the class prediction or
probability of classes. Voting can be a product of two weights:
weights of training instances, if they are given, and weights that
reflect the distance from ``inst``. Nearer neighbors have a greater
impact on the prediction: the weight is computed as
exp(-t:sup:`2`/s:sup:`2`), where the meaning of `t` depends on the
setting of :obj:`rank_weight`.

* if :obj:`rank_weight` is :obj:`False`, :obj:`t` is the distance from the
  instance being classified
* if :obj:`rank_weight` is :obj:`True`, neighbors are ordered and :obj:`t`
  is the position of the neighbor on the list (a rank)

In both cases, :obj:`s` is chosen so that the weight of the farthest
instance is 0.001.

Weighting gives the classifier a certain insensitivity to the number of
neighbors used, making it possible to use large :obj:`k`'s.

The classifier can use continuous and discrete features, and can even
distinguish between ordinal and nominal features. See information on
distance measuring for details.

Examples
--------

The learner will be tested on an 'iris' data set. The data will be split 
into training (80%) and testing (20%) instances. We will use the former 
for "training" the classifier and test it on five testing instances 
randomly selected from a part of (:download:`knnlearner.py <code/knnlearner.py>`):

.. literalinclude:: code/knnExample1.py

The output of this code is:: 
    
    Iris-setosa Iris-setosa
    Iris-versicolor Iris-versicolor
    Iris-versicolor Iris-versicolor
    Iris-setosa Iris-setosa
    Iris-setosa Iris-setosa

The choice of metric usually has not greater impact on the performance
of kNN classifiers, so default should work fine. To change it,
distance_constructor must be set to an instance of one of the classes
for distance measuring.

.. literalinclude:: code/knnExample2.py
    :lines: 4-7

.. index: fnn


Finding nearest neighbors
-------------------------

Orange provides classes for finding the nearest neighbors of a given
reference instance.

As usual in Orange, there are two classes: one that does the work
(:class:`FindNearest`) and another that constructs the former from
data (:class:`FindNearestConstructor`).

.. class:: FindNearest

    Brute force search for nearest neighbors in the stored data table.
    
    .. attribute:: distance
    
        An instance of :obj:`Orange.distance.Distance` used for
        computing distances between data instances.
    
    .. attribute:: instances
    
        Stored data table
    
    .. attribute:: weight_ID
    
        ID of meta attribute with weight. If present (non-null) the
        class does not return ``k`` instances but a set of instances
        with a total weight of ``k``.

    .. attribute:: distance_ID

        The id of meta attribute that will be added to the found
        neighbours and to store the distances between the returned
        data instances and the reference. If zero, the distances is
        not stored.
    
    .. method:: __call__(instance, k)
    
        Return a data table with ``k`` nearest neighbours of
	``instance``.  Any ties for the last place(s) are resolved by
	randomly picking the appropriate number of instances. A local
	random generator is constructed and seeded by a constant
	computed from :obj:`instance`, so the same random neighbors
	are always returned for the same instance.

	:param instance: given instance
	:type instance: Orange.data.Instance

	:param k: number of neighbors
	:type k: int

	:rtype: :obj:`Orange.data.Table`
    
.. class:: FindNearestConstructor()

    A class that constructs :obj:`FindNearest` and initializes it with a
    distance metric, constructed by :obj:`distance_constructor`.
    
    .. attribute:: distance_constructor
    
        An instance of :obj:`Orange.distance.DistanceConstructor` that
        "learns" to measure distances between instances. Learning can
        mean, for example, storing the ranges of continuous features
        or the number of values of a discrete feature. The result of
        learning is an instance of :obj:`Orange.distance.Distance` that is
        used for measuring distances between instances.
    
    .. attribute:: include_same
    
        Indicates whether to include the instances that are same as
        the reference; default is ``true``.
    
    .. method:: __call__(data, weight_ID, distance_ID)
    
        Constructs an instance of :obj:`FindNearest` for the given
        data. Arguments :obj:`weight_ID` and :obj:`distance_ID` are copied to the new object.

        :param table: table of instances
        :type table: Orange.data.Table
        
        :param weight_ID: id of meta attribute with weights of instances
        :type weight_ID: int
        
        :param distance_ID: id of meta attribute that will store distances
        :type distance_ID: int
        
        :rtype: :obj:`FindNearest`

Examples
--------

The following script (:download:`knnInstanceDistance.py <code/knnInstanceDistance.py>`)
shows how to find the five nearest neighbors of the first instance
in the lenses dataset.

.. literalinclude:: code/knnInstanceDistance.py


.. automodule:: Orange.classification.knn
