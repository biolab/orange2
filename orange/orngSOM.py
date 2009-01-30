import sys, os

import numpy
import numpy.ma as ma
import orange
##import math

##from numpy import dot, abs, sqrt, random, array

HexagonalTopology = 0
RectangularTopology = 1
InitializeLinear = 0
InitializeRandom = 1
NeighbourhoodGaussian = 0
NeighbourhoodBubble = 1
NeighbourhoodEpanechicov = 2

class Node(object):
    def __init__(self, pos, map=None, vector=None):
        self.pos = pos
        self.map = map
        self.vector = vector

class Map(object):
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
        """ Return the node at position x, y
        """
        x, y = pos
        return self.map[x][y]

    def __iter__(self):
        """ Iterate over all nodes in the map
        """
        for row in self.map:
            for node in row:
                yield node

    def vectors(self):
        """ Return all vectors of the map as rows in an numpy.array
        """
        return numpy.array([node.vector for node in self])

    def unit_distances(self):
        """ Return a NxN numpy.array of internode distances (based on
        node position in the map, not vector space) where N is the number of
        nodes
        """
        nodes = list(self)
        dist = numpy.zeros((len(nodes), len(nodes)))

        coords = self.unit_coords()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                dist[i, j] = numpy.sqrt(numpy.dot(coords[i] - coords[j], coords[i] - coords[j]))
        return numpy.array(dist)

    def unit_coords(self):
        """ Return the unit coordinates of all nodes in the map as an numpy.array
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
        """ Initialize the map nodes vectors randomly, by supplying
        either training data or dimension of the data
        """
        if data != None:
            min, max = ma.min(data, 0), ma.max(data, 0);
            dimension = data.shape[1]
        else:
            min, max = numpy.zeros(dimension), numpy.ones(dimension)
        for node in self:
            node.vector = min + numpy.random.rand(dimension) * (max - min)

    def initialize_map_linear(self, data, map_shape=(10, 20)):
        """ Initialize the map node vectors lineary over the subspace
        of the two most significant eigenvectors
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
##        print eigvec, eigval

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
        
class Solver(object):
    """ SOM Solver class used to train the map.
    Arguments:
        * neighbourhood - Neighbourhood function (NeighbourhoodGaussian, or NeighbourhoodBubble)
        * radius_ini    - Inttial radius
        * raduis_fin    - Final radius
        * epoch         - Number of training iterations
        * batch_train   - If True run the batch training algorithem (default), else use the sequential one
        * learning_rate - If learning rate for the sequential training algorithem

    Both the batch ans sequential algorithems are based on SOM Toolkit for Matlab
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
        return self.radius_ini + (float(self.radius_ini) - self.radius_fin)*(float(epoch) / self.epochs)

    def alpha(self, epoch):
        return (1 - epoch/self.epochs)*self.learning_rate
            
    def __call__(self, data, map, progressCallback=None):
        """ Train the map on data. Use progressCallback to Report on the progress.
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
        self.vectors = self.map.vectors()
        self.unit_distances = self.map.unit_distances()
        
##        from pylab import plot, show, draw, ion
##        ion()
##        plot(self.data[:, 0], self.data[:, 1], "ro")
##        vec_plot = plot(self.vectors[:, 0], self.vectors[:, 1], "bo")[0]
        
        for epoch in range(self.epochs):
            self.distances = []
            ind = range(len(self.data))
            if self.random_order:
                random.shuffle(ind)
            self.train_step_sequential(epoch, ind)
            if progressCallback:
                progressCallback(100.0*epoch/self.epochs)
            self.qerror.append(numpy.mean(numpy.sqrt(self.distances)))
##            print epoch, "q error:", numpy.mean(numpy.sqrt(self.distances)), self.radius(epoch)
            if epoch > 5 and numpy.mean(numpy.abs(numpy.array(self.qerror[-5:-1]) - self.qerror[-1])) <= self.eps:
                break
            
##            vec_plot.set_xdata(self.vectors[:, 0])
##            vec_plot.set_ydata(self.vectors[:, 1])
##            draw()
##        show()

    def train_step_sequential(self, epoch, indices=None):
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
            if epoch > 5 and numpy.mean(numpy.abs(numpy.array(self.qerror[-5:-1]) - self.qerror[-1])) <= self.eps:
                break
##            vec_plot.set_xdata(self.vectors[:, 0])
##            vec_plot.set_ydata(self.vectors[:, 1])
##            draw()
##        show()
        
        for node, vector in zip(self.map, self.vectors):
            node.vector = vector

    def train_step_batch(self, epoch):
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
    """ SOMLearner is a class used to learn SOM from orange.ExampleTable

    Example:
        >>> som = orngSOM.SOMLearner(map_shape=()
    """
    def __init__(self, map_shape=(5, 10), initialize=InitializeLinear, neighbourhood=NeighbourhoodGaussian,
                 batch_train=True, learning_rate=0.05, radius_ini=3, radius_fin=1, epochs=1000, **kwargs):
        self.map_shape = (5, 10)
        self.initialize = initialize
        self.topology = topology
        self.neighbourhood = neighbourhood
        self.batch_train = batch_train
        self.learning_rate = learning_rate
        self.radius_ini = radius_ini
        self.radius_fin = radius_fin
        self.epochs = epochs
        self.eps = 1e-4
        
        orange.Learner.__init__(self, **kwargs)
        
    def __call__(self, examples, weightID=0, progressCallback=None):
        data, classes, w = examples.toNumpyMA()
        map = Map(self.map_shape, topology=self.topology)
        if self.initialize == Map.InitializeLinear:
            map.initialize_map_linear(data)
        else:
            map.initialize_map_random(data)
        map = Solver(batch_train=self.batch_train, eps=self.eps, neighbourhood=self.neighbourhood,
                     radius_ini=self.radius_ini, radius_fin=self.radius_fin, learning_rate=self.learning_rate,
                     epoch=self.epochs)(data, map, progressCallback=progressCallback)
        return SOMMap(map, examples)

class SOMSupervisedLearner(SOMLearner):
    """ SOMSupervisedLearner is a class used to learn SOM from orange.ExampleTable, by using the
    class information in the learning process. This is achieved by adding a value for each class to the training
    instances, where 1.0 signals class membership and all other values are 0.0. After the training,
    the new values are discarded from the node vectors.
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

class SOMMap(orange.Classifier):
    def __init__(self, map, examples):
        self.map = map
        self.examples = examples
        for node in map:
            node.referenceExample = orange.Example(orange.Domain(examples.domain.attributes, False),
                                                 [(var(value) if var.varType == orange.VarTypes.Continuous else var(int(value))) \
                                                  for var, value in zip(examples.domain.attributes, node.vector)])
            node.examples = orange.ExampleTable(examples.domain)

        for ex in examples:
            node = self.getBestMatchingNode(ex)
            node.examples.append(ex)

        if examples.domain.classVar:
            for node in self.map:
                node.classifier = orange.MajorityLearner(node.examples)

    def getBestMatchingNode(self, example):
        """ Return the best matching node
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
            return getattr(self.map, name)
        except AttributeError:
            raise AttributeError(name)

    def __iter__(self):
        """ Iterate over all nodes in the map
        """
        return iter(self.map)

    def __getitem__(self, val):
        """ Return the node at position x, y
        """
        return self.map.__getitem__(val)

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

if __name__ == "__main__":
    data = orange.ExampleTable("doc//datasets//iris.tab")
    learner = SOMLearner()
    learner = SOMLearner(batch_train=True, initialize=InitializeRandom)
    map = learner(data)
    for e in data:
        print map(e), e.getclass()
    