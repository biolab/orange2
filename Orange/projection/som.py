"""
******************************
Self-organizing maps (``som``)
******************************

.. index:: self-organizing map (SOM)

.. index:: 
   single: projection; self-organizing map (SOM)

An implementation of `self-organizing map <http://en.wikipedia.org/wiki/Self-organizing_map>`_ algorithm (SOM). 
SOM is an unsupervised learning 
algorithm that infers low, typically two-dimensional discretized representation of the input space,
called a map. The map preserves topological properties of the input space, such that
the cells that are close in the map include data instances that are similar to each other.

=================================
Inference of Self-Organizing Maps
=================================

The main class for inference of self-organizing maps is :obj:`SOMLearner`. The class initializes
the topology of the map and returns an inference objects which, given the data, performs the 
optimization of the map:: 

   import Orange
   som = Orange.projection.som.SOMLearner(map_shape=(8, 8), 
            initialize=Orange.projection.som.InitializeRandom)
   data = Orange.data.table("iris.tab")
   map = som(data)

.. autodata:: NeighbourhoodGaussian

.. autodata:: HexagonalTopology

.. autodata:: RectangularTopology

.. autodata:: InitializeLinear

.. autodata:: InitializeRandom

.. autodata:: NeighbourhoodGaussian 

.. autodata:: NeighbourhoodBubble

.. autodata:: NeighbourhoodEpanechicov 

.. autoclass:: SOMLearner
   :members:

.. autoclass:: Solver
   :members:
   
.. autoclass:: SOMMap
   :members:

=============================================
Supervised Learning with Self-Organizing Maps
=============================================

Supervised learning requires class-labeled data. For training,
class information is first added to data instances as a regular feature 
by extending the feature vectors accordingly. Next, the map is trained, and the
training data projected to nodes. Each node then classifies to the majority class.
For classification, the data instance is projected the cell, returning the associated class.
An example of the code that trains and then classifies on the same data set is::

    import Orange
    import random
    learner = Orange.projection.som.SOMSupervisedLearner(map_shape=(4, 4))
    data = Orange.data.Table("iris.tab")
    classifier = learner(data)
    random.seed(50)
    for d in random.sample(data, 5):
        print "%-15s originally %-15s" % (classifier(d), d.getclass())

.. autoclass:: SOMSupervisedLearner
   :members:
   
==================
Supporting Classes
==================

Class :obj:`Map` stores the self-organizing map composed of :obj:`Node` objects. The code below
(:download:`som-node.py <code/som-node.py>`, uses :download:`iris.tab <code/iris.tab>`) shows an example how to access the information stored in the 
node of the map:

.. literalinclude:: code/som-node.py
    :lines: 7-

.. autoclass:: Map
   :members:
   
.. autoclass:: Node
   :members:
 
========
Examples
========

The following code  (:download:`som-mapping.py <code/som-mapping.py>`, uses :download:`iris.tab <code/iris.tab>`) infers self-organizing map from Iris data set. The map is rather small, and consists 
of only 9 cells. We optimize the network, and then report how many data instances were mapped
into each cell. The second part of the code reports on data instances from one of the corner cells:

.. literalinclude:: code/som-mapping.py
    :lines: 7-

The output of this code is::

    Node    Instances
    (0, 0)  21
    (0, 1)  1
    (0, 2)  23
    (1, 0)  22
    (1, 1)  7
    (1, 2)  6
    (2, 0)  32
    (2, 1)  16
    (2, 2)  22
    
    Data instances in cell (1, 2):
    [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor']
    [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor']
    [5.6, 2.9, 3.6, 1.3, 'Iris-versicolor']
    [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor']
    [5.5, 2.4, 3.7, 1.0, 'Iris-versicolor']
    [5.0, 2.3, 3.3, 1.0, 'Iris-versicolor']
"""

import sys, os

import numpy
import numpy.ma as ma
import orange
import random

random.seed(42)

HexagonalTopology = 0
"""Hexagonal topology, cells are hexagon-shaped."""
RectangularTopology = 1
"""Rectangular topology, cells are square-shaped"""

InitializeLinear = 0
"""Data instances are initially assigned to cells according to their two-dimensional PCA projection."""
InitializeRandom = 1
"""Data instances are initially randomly assigned to cells."""

NeighbourhoodGaussian = 0 
"""Gaussian (smoothed) neighborhood."""
NeighbourhoodBubble = 1
"""Bubble (crisp) neighborhood."""
NeighbourhoodEpanechicov = 2
"""Epanechicov (cut and smoothed) neighborhood."""

##########################################################################
# Inference of Self-Organizing Maps 

class Solver(object):
    """ SOM Solver class used to train the map. Supports batch and sequential training.
    Based on ideas from `SOM Toolkit for Matlab <http://www.cis.hut.fi/somtoolbox>`_.

    :param neighbourhood: neighborhood function id
    :type neighbourhood: :obj:`NeighbourhoodGaussian`, :obj:`NeighbourhoodBubble`, or :obj:`NeighbourhoodEpanechicov`
    :param radius_ini: initial radius
    :type radius_ini: int
    :param raduis_fin: final radius
    :type raduis_fin: int
    :param epoch: number of training interactions
    :type epoch: int
    :param batch_train: if True run the batch training algorithm (default), else use the sequential one
    :type batch_train: bool
    :param learning_rate: learning rate for the sequential training algorithm
    :type learning_rate: float
    """
    
    def __init__(self, **kwargs):
        self.neighbourhood = NeighbourhoodGaussian
        self.learning_rate = 0.05
        self.radius_ini = 2
        self.radius_fin = 1
        self.epochs = 100
        self.random_order = False
        self.batch_train = True
        self.eps = 1e-5
        self.qerror = []
        self.__dict__.update(kwargs)

    def radius(self, epoch):
        return self.radius_ini - (float(self.radius_ini) - self.radius_fin)*(float(epoch) / self.epochs)

    def alpha(self, epoch):
        """Compute the learning rate from epoch, starting with learning_rate to 0 at the end of training. 
        """
        return (1 - epoch/self.epochs)*self.learning_rate
            
    def __call__(self, data, map, progressCallback=None):
        """ Train the map from data. Pass progressCallback function to report on the progress.
        """
        self.data = data
        self.map = map

        self.qerror = []
        self.bmu_cache = {}
        if self.batch_train:
            self.train_batch(progressCallback)
        else:
            self.train_sequential(progressCallback)
        return self.map

    def train_sequential(self, progressCallback):
        """Sequential training algorithm. 
        """
        self.vectors = self.map.vectors()
        self.unit_distances = self.map.unit_distances()
        
#        from pylab import plot, show, draw, ion
#        ion()
#        plot(self.data[:, 0], self.data[:, 1], "ro")
#        vec_plot = plot(self.vectors[:, 0], self.vectors[:, 1], "bo")[0]
        
        for epoch in range(self.epochs):
            self.distances = []
            ind = range(len(self.data))
            if self.random_order:
                random.shuffle(ind)
            self.train_step_sequential(epoch, ind)
            if progressCallback:
                progressCallback(100.0*epoch/self.epochs)
            self.qerror.append(numpy.mean(numpy.sqrt(self.distances)))
#            print epoch, "q error:", numpy.mean(numpy.sqrt(self.distances)), self.radius(epoch)
            if epoch > 5 and numpy.mean(numpy.abs(numpy.array(self.qerror[-5:-1]) - self.qerror[-1])) <= self.eps:
                break
            
#            vec_plot.set_xdata(self.vectors[:, 0])
#            vec_plot.set_ydata(self.vectors[:, 1])
#            draw()
#        show()

    def train_step_sequential(self, epoch, indices=None):
        """A single step of sequential training algorithm.
        """
        indices = range(len(self.data)) if indices == None else indices
        for ind in indices:
            x = self.data[ind]
            Dx = self.vectors - self.data[ind]
            Dist = ma.sum(Dx**2, 1)
            min_dist = ma.min(Dist)
            bmu = ma.argmin(Dist)
            self.distances.append(min_dist)

            if self.neighbourhood == Map.NeighbourhoodGaussian:
                h = numpy.exp(-self.unit_distances[:, bmu]/(2*self.radius(epoch))) * (self.unit_distances[:, bmu] <= self.radius(epoch))
            elif self.neighbourhood == Map.NeighbourhoodEpanechicov:
                h = 1.0 - (self.unit_distances[:bmu]/self.radius(epoch))**2
                h = h * (h >= 0.0)
            else:
                h = 1.0*(self.unit_distances[:, bmu] <= self.radius(epoch))
            h = h * self.alpha(epoch)

            nonzero = ma.nonzero(h)
            h = h[nonzero]

            self.vectors[nonzero] = self.vectors[nonzero] - Dx[nonzero] * numpy.reshape(h, (len(h), 1))

    def train_batch(self, progressCallback=None):
        """Batch training algorithm.
        """
        
        self.unit_distances = self.map.unit_distances()
        self.constant_matrix = 2 * ma.dot(numpy.eye(self.data.shape[1]), numpy.transpose(self.data))
        self.dist_cons = numpy.transpose(ma.dot(self.data**2, numpy.ones(self.data.shape[1])))
        self.weight_matrix = numpy.ones((self.data.shape[1], self.data.shape[0]))
        self.vectors = self.map.vectors()
        
##        from pylab import plot, show, draw, ion
##        ion()
##        plot(self.data[:, 0], self.data[:, 1], "ro")
##        vec_plot = plot(self.vectors[:, 0], self.vectors[:, 1], "bo")[0]
        
        for epoch in range(self.epochs):
            self.train_step_batch(epoch)
            if progressCallback:
                progressCallback(100.0*epoch/self.epochs)
            if False and epoch > 5 and numpy.mean(numpy.abs(numpy.array(self.qerror[-5:-1]) - self.qerror[-1])) <= self.eps:
                break
##            vec_plot.set_xdata(self.vectors[:, 0])
##            vec_plot.set_ydata(self.vectors[:, 1])
##            draw()
##        show()
        
        for node, vector in zip(self.map, self.vectors):
            node.vector = vector

    def train_step_batch(self, epoch):
        """A single step of batch training algorithm.
        """
        D1 = ma.dot(self.vectors**2, self.weight_matrix)
        D2 = ma.dot(self.vectors, self.constant_matrix)
        Dist = D1 - D2

        best_nodes = ma.argmin(Dist, 0)
        distances = ma.min(Dist, 0)
##        print "q error:", ma.mean(ma.sqrt(distances + self.dist_cons)), self.radius(epoch)
        self.qerror.append(ma.mean(ma.sqrt(distances + self.dist_cons)))

        if self.neighbourhood == Map.NeighbourhoodGaussian:        
            H = numpy.exp(-self.unit_distances/(2*self.radius(epoch))) * (self.unit_distances <= self.radius(epoch))
        elif self.neighbourhood == Map.NeighbourhoodEpanechicov:
            H = 1.0 - (self.unit_distances/self.radius(epoch))**2
            H = H * (H >= 0.0)
        else:
            H = 1.0*(self.unit_distances <= self.radius(epoch))

        P =  numpy.zeros((self.vectors.shape[0], self.data.shape[0]))
        
        P[(best_nodes, range(len(best_nodes)))] = numpy.ones(len(best_nodes))
        
        S = ma.dot(H, ma.dot(P, self.data))
        
        A = ma.dot(H, ma.dot(P, ~self.data._mask))

##        nonzero = (range(epoch%2, len(self.vectors), 2), )
        nonzero = (numpy.array(sorted(set(ma.nonzero(A)[0]))), )
        
        self.vectors[nonzero] = S[nonzero] / A[nonzero]


class SOMLearner(orange.Learner):
    """An implementation of self-organizing map. Considers an input data set, projects the data 
    instances onto a map, and returns a result in the form of a classifier holding projection
    information together with an algorithm to project new data instances. Uses :obj:`Map` for
    representation of projection space, :obj:`Solver` for training, and returns a trained 
    map with information on projection of the training data as crafted by :obj:`SOMMap`.
    
    :param map_shape: dimension of the map
    :type map_shape: tuple
    :param initialize: initialization type id; linear 
      initialization assigns the data to the cells according to its position in two-dimensional
      principal component projection
    :type initialize: :obj:`InitializeRandom` or :obj:`InitializeLinear`
    :param topology: topology type id
    :type topology: :obj:`HexagonalTopology` or :obj:`RectangularTopology`
    :param neighbourhood: cell neighborhood type id
    :type neighbourhood: :obj:`NeighbourhoodGaussian`, obj:`NeighbourhoodBubble`, or obj:`NeighbourhoodEpanechicov`
    :param batch_train: perform batch training?
    :type batch_train: bool
    :param learning_rate: learning rate
    :type learning_rate: float
    :param radius_ini: initial radius
    :type radius_ini: int
    :param radius_fin: final radius
    :type radius_fin: int
    :param epochs: number of epochs (iterations of a training steps)
    :type epochs: int
    :param solver: a class with the optimization algorithm
    """
    
    def __new__(cls, examples=None, weightId=0, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if examples is not None:
            self.__init__(**kwargs)
            return self.__call__(examples, weightId)
        else:
            return self
        
    def __init__(self, map_shape=(5, 10), initialize=InitializeLinear, topology=HexagonalTopology, neighbourhood=NeighbourhoodGaussian,
                 batch_train=True, learning_rate=0.05, radius_ini=3.0, radius_fin=1.0, epochs=1000, solver=Solver, **kwargs):

        self.map_shape = map_shape
        self.initialize = initialize
        self.topology = topology
        self.neighbourhood = neighbourhood
        self.batch_train = batch_train
        self.learning_rate = learning_rate
        self.radius_ini = radius_ini
        self.radius_fin = radius_fin
        self.epochs = epochs
        self.solver = solver
        self.eps = 1e-4
        
        orange.Learner.__init__(self, **kwargs)
        
    def __call__(self, data, weightID=0, progressCallback=None):
        numdata, classes, w = data.toNumpyMA()
        map = Map(self.map_shape, topology=self.topology)
        if self.initialize == Map.InitializeLinear:
            map.initialize_map_linear(numdata)
        else:
            map.initialize_map_random(numdata)
        map = self.solver(batch_train=self.batch_train, eps=self.eps, neighbourhood=self.neighbourhood,
                     radius_ini=self.radius_ini, radius_fin=self.radius_fin, learning_rate=self.learning_rate,
                     epochs=self.epochs)(numdata, map, progressCallback=progressCallback)
        return SOMMap(map, data)

class SOMMap(orange.Classifier):
    """Project the data onto the inferred self-organizing map.
    
    :param map: a trained self-organizing map
    :type map: :obj:`SOMMap`
    :param data: the data to be mapped on the map
    :type data: :obj:`Orange.data.Table`
    """
    
    def __init__(self, map=[], data=[]):
        self.map = map
        self.examples = data
        for node in map:
            node.referenceExample = orange.Example(orange.Domain(self.examples.domain.attributes, False),
                                                 [(var(value) if var.varType == orange.VarTypes.Continuous else var(int(value))) \
                                                  for var, value in zip(self.examples.domain.attributes, node.vector)])
            node.examples = orange.ExampleTable(self.examples.domain)

        for ex in self.examples:
            node = self.getBestMatchingNode(ex)
            node.examples.append(ex)

        if self.examples and self.examples.domain.classVar:
            for node in self.map:
                node.classifier = orange.MajorityLearner(node.examples if node.examples else self.examples)
                     
            self.classVar = self.examples.domain.classVar
        else:
            self.classVar = None

    def getBestMatchingNode(self, example):
        """Return the best matching node for a given data instance
        """
        example, c, w = orange.ExampleTable([example]).toNumpyMA()
        vectors = self.map.vectors()
        Dist = vectors - example
        bmu = ma.argmin(ma.sum(Dist**2, 1))
        return list(self.map)[bmu]
        
    def __call__(self, example, what=orange.GetValue):
        bmu = self.getBestMatchingNode(example)
        return bmu.classifier(example, what)

    def __getattr__(self, name):
        try:
            return getattr(self.__dict__["map"], name)
        except (KeyError, AttributeError):
            raise AttributeError(name)

    def __iter__(self):
        """ Iterate over all nodes in the map
        """
        return iter(self.map)

    def __getitem__(self, val):
        """ Return the node at position x, y
        """
        return self.map.__getitem__(val)

##########################################################################
# Supervised learning

class SOMSupervisedLearner(SOMLearner):
    """SOMSupervisedLearner is a class used to learn SOM from orange.ExampleTable, by using the
    class information in the learning process. This is achieved by adding a value for each class
    to the training instances, where 1.0 signals class membership and all other values are 0.0.
    After the training, the new values are discarded from the node vectors.
    
    :param data: class-labeled data set
    :type data: :obj:`Orange.data.Table`
    :param progressCallback: a one argument function to report on inference progress (in %)
    """
    def __call__(self, examples, weightID=0, progressCallback=None):
        data, classes, w = examples.toNumpyMA()
        nval = len(examples.domain.classVar.values)
        ext = ma.zeros((len(data), nval))
        ext[([i for i, m in enumerate(classes.mask) if m], [int(c) for c, m in zip(classes, classes.mask) if m])] = 1.0
        data = ma.hstack((data, ext))
        map = Map(self.map_shape, topology=self.topology)
        if self.initialize == Map.InitializeLinear:
            map.initialize_map_linear(data)
        else:
            map.initialize_map_random(data)
        map = Solver(batch_train=self.batch_train, eps=self.eps, neighbourhood=self.neighbourhood,
                     radius_ini=self.radius_ini, radius_fin=self.radius_fin, learning_rate=self.learning_rate,
                     epoch=self.epochs)(data, map, progressCallback=progressCallback)
        for node in map:
            node.vector = node.vector[:-nval]
        return SOMMap(map, examples)

##########################################################################
# Supporting Classes 

class Node(object):
    """An object holding the information about the node in the map.

    .. attribute:: pos

        Node position.

    .. attribute:: referenceExample

        Reference data instance (a prototype).
        
    .. attribute:: examples
    
        Data set with instances training instances that were mapped to the node. 
    """
    def __init__(self, pos, map=None, vector=None):
        self.pos = pos
        self.map = map
        self.vector = vector

class Map(object):
    """Self organizing map (the structure). Includes methods for data initialization.
    
    .. attribute:: map

        Self orginzing map. A list of lists of :obj:`Node`.
        
    .. attribute:: examples
    
        Data set that was considered when optimizing the map.
    """
    
    HexagonalTopology = HexagonalTopology
    RectangularTopology = RectangularTopology
    InitializeLinear = InitializeLinear
    InitializeRandom = InitializeRandom
    NeighbourhoodGaussian = NeighbourhoodGaussian
    NeighbourhoodBubble = NeighbourhoodBubble
    NeighbourhoodEpanechicov = NeighbourhoodEpanechicov
        
    def __init__(self, map_shape=(20, 40), topology=HexagonalTopology):
        self.map_shape = map_shape
        self.topology = topology
        self.map = [[Node((i, j), self) for j in range(map_shape[1])] for i in range(map_shape[0])]
        
    def __getitem__(self, pos):
        """ Return the node at position x, y.
        """
        x, y = pos
        return self.map[x][y]

    def __iter__(self):
        """ Iterate over all nodes in the map.
        """
        for row in self.map:
            for node in row:
                yield node

    def vectors(self):
        """Return all vectors of the map as rows in an numpy.array.
        """
        return numpy.array([node.vector for node in self])

    def unit_distances(self):
        """Return a NxN numpy.array of internode distances (based on
        node position in the map, not vector space) where N is the number of
        nodes.
        """
        nodes = list(self)
        dist = numpy.zeros((len(nodes), len(nodes)))

        coords = self.unit_coords()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                dist[i, j] = numpy.sqrt(numpy.dot(coords[i] - coords[j], coords[i] - coords[j]))
        return numpy.array(dist)

    def unit_coords(self):
        """ Return the unit coordinates of all nodes in the map as an numpy.array.
        """
        nodes = list(self)
        coords = numpy.zeros((len(nodes), len(self.map_shape)))
        coords[:, 0] = numpy.floor(numpy.arange(len(nodes)) / self.map_shape[0])
        coords[:, 1] = numpy.mod(numpy.arange(len(nodes)), self.map_shape[1])
        
        ## in hexagonal topology we move every odd map row by 0.5 and multiply all by sqrt(0.75)
        if self.topology == Map.HexagonalTopology:
            ind = numpy.nonzero(1 - numpy.mod(coords[:, 0], 2))
            coords[ind] = coords[ind] + 0.5
            coords = coords * numpy.sqrt(0.75)
        return coords


    def initialize_map_random(self, data=None, dimension=5):
        """Initialize the map nodes vectors randomly, by supplying
        either training data or dimension of the data.
        """
        if data is not None:
            min, max = ma.min(data, 0), ma.max(data, 0);
            dimension = data.shape[1]
        else:
            min, max = numpy.zeros(dimension), numpy.ones(dimension)
        for node in self:
#            node.vector = min + numpy.random.rand(dimension) * (max - min)
            node.vector = min + random.randint(0, dimension) * (max - min)

    def initialize_map_linear(self, data, map_shape=(10, 20)):
        """ Initialize the map node vectors linearly over the subspace
        of the two most significant eigenvectors.
        """
        data = data.copy() #ma.array(data)
        dim = data.shape[1]
        mdim = len(map_shape)
        munits = len(list(self))
        me = ma.mean(data, 0)
        A = numpy.zeros((dim ,dim))

        for i in range(dim):
            data[:, i] = data[:, i] - me[i]
        
        for i in range(dim):
            for j in range(dim):
                c = data[:, i] * data[:, j]
                A[i, j] = ma.sum(c) / len(c)
                A[j, i] = A[i, j]

        eigval, eigvec = numpy.linalg.eig(A)
        ind = list(reversed(numpy.argsort(eigval)))
        eigval = eigval[ind[:mdim]]
        eigvec = eigvec[:, ind[:mdim]]

        for i in range(mdim):
            eigvec[:, i] = eigvec[:, i] / numpy.sqrt(numpy.dot(eigvec[:, i], eigvec[:, i])) * numpy.sqrt(eigval[i])

        unit_coords = self.unit_coords()
        for d in range(mdim):
            max, min = numpy.max(unit_coords[:, d]), numpy.min(unit_coords[:, d])
            unit_coords[:, d] = (unit_coords[:, d] - min)/(max - min)
        unit_coords = (unit_coords - 0.5) * 2

        vectors = numpy.array([me for i in range(munits)])
        for i in range(munits):
            for d in range(mdim):
                vectors[i] = vectors[i] +  unit_coords[i][d] * numpy.transpose(eigvec[:, d])

        for i, node in enumerate(self):
            node.vector = vectors[i]

    def getUMat(self):
        return getUMat(self)
        
##########################################################################
# Supporting functions 

def getUMat(som):
    dim1=som.map_shape[0]*2-1
    dim2=som.map_shape[1]*2-1

    a=numpy.zeros((dim1, dim2))
    if som.topology == HexagonalTopology:
        return __fillHex(a, som)
    else:
        return __fillRect(a, som)

def __fillHex(array, som):
    xDim, yDim = som.map_shape
##    for n in som.nodes:
##        d[tuple(n.pos)]=n
    d = dict([((i, j), som[i, j]) for i in range(xDim) for j in range(yDim)])
    check=lambda x,y:x>=0 and x<(xDim*2-1) and y>=0 and y<(yDim*2-1)
    dx=[1,0,-1]
    dy=[0,1, 1]
    for i in range(0, xDim*2,2):
        for j in range(0, yDim*2,2):
            for ddx, ddy in zip(dx, dy):
                if check(i+ddx, j+ddy):
##                    array[i+ddx][j+ddy]=d[(i/2, j/2)].getDistance(d[(i/2+ddx, j/2+ddy)].referenceExample)
                    array[i+ddx][j+ddy] = numpy.sqrt(ma.sum((d[(i/2, j/2)].vector - d[(i/2+ddx, j/2+ddy)].vector)**2))
    dx=[1,-1,0,-1, 0, 1]
    dy=[0, 0,1, 1,-1,-1]
    for i in range(0, xDim*2, 2):
        for j in range(0, yDim*2, 2):
            l=[array[i+ddx, j+ddy] for ddx, ddy in zip(dx, dy) if check(i+ddx, j+ddy)]
            array[i][j]=sum(l)/len(l)
    return array

def __fillRect(array, som):
    xDim, yDim = som.map_shape
    d = dict([((i, j), som[i, j]) for i in range(xDim) for j in range(yDim)])
    check=lambda x,y:x>=0 and x<xDim*2-1 and y>=0 and y<yDim*2-1
    dx=[1, 0, 1]
    dy=[0, 1, 1]
    for i in range(0, xDim*2, 2):
        for j in range(0, yDim*2, 2):
            for ddx, ddy in zip(dx, dy):
                if check(i+ddx, j+ddy):
##                    array[i+ddx][j+ddy]=d[(i/2, j/2)].getDistance(d[(i/2+ddx, j/2+ddy)].referenceExample)
                    array[i+ddx][j+ddy] = numpy.sqrt(ma.sum((d[(i/2, j/2)].vector - d[(i/2+ddx, j/2+ddy)].vector)**2))
    dx=[1,-1, 0,0,1,-1,-1, 1]
    dy=[0, 0,-1,1,1,-1, 1,-1]
    for i in range(0, xDim*2, 2):
        for j in range(0, yDim*2, 2):
            l=[array[i+ddx,j+ddy] for ddx,ddy in zip(dx,dy) if check(i+ddx, j+ddy)]
            array[i][j]=sum(l)/len(l)
    return array

##########################################################################
# Testing (deprecated, use regression tests instead  

if __name__ == "__main__":
    data = orange.ExampleTable("iris.tab")
    learner = SOMLearner()
    learner = SOMLearner(batch_train=True, initialize=InitializeLinear, radius_ini=3, radius_fin=1, neighbourhood=Map.NeighbourhoodGaussian, epochs=1000)
    map = learner(data)
    for e in data:
        print map(e), e.getclass()
