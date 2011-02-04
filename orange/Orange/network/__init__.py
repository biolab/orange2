""" 
.. index:: network

Network analysis and layout optimization.

=======
Network
=======

.. autoclass:: Orange.network.Network
   :members:
   
Examples
========

Reading and saving a network
----------------------------

This example demonstrates reading a network. Network class can read or write 
Pajek (.net) or GML file format.

`network-read.py`_ (uses: `K5.net`_):

.. literalinclude:: code/network-read.py
    :lines: 4-5

Visualize a network in NetExplorer widget
-----------------------------------------

This example demonstrates how to display a network in NetExplorer.

part of `network-widget.py`_

.. literalinclude:: code/network-widget.py
    :lines: 10-16
    
.. image:: files/network-explorer.png
    :width: 100%
   
===========================
Network Layout Optimization
===========================

    .. autoclass:: Orange.network.NetworkOptimization
       :members:
       :exclude-members: collapse 
       
Examples
========

Network constructor and random layout
-------------------------------------

In our first example we create a Network object with a simple full graph (K5). 
Vertices are initially placed randomly. Graph is visualized using pylabs 
matplotlib. NetworkOptimization class is not needed because we do not apply any 
layout optimization method in this example.

`network-constructor.py`_

.. literalinclude:: code/network-constructor.py

Executing the above script pops-up a pylab window with the following graph 
drawing:

.. image:: files/network-K5-random.png

Network layout optimization
---------------------------

This example demonstrates how to optimize network layout using one of included 
algorithms.

part of `network-optimization.py`_

.. literalinclude:: code/network-optimization.py
    :lines: 12-16
    
The following optimization algorithms are supported:

* .random()
* .fruchtermanReingold(steps, temperature, coolingFactor=Default, hiddenNodes=[], weighted=False)
* .radialFruchtermanReingold(center, steps, temperature)
* .circularOriginal()
* .circularRandom()
* .circularCrossingReduction()

Spring forces layout optimization is the result of the above script:

.. image:: files/network-K5-fr.png
   
======
Graphs
======

Orange offers a data structure for representing directed and undirected graphs 
with various types of weighted connections.

Basic graphs have only one type of edges. Each edge can have an associated 
number, representing a strength of the edge - with whatever underlying 
physical interpretation. Orange's graphs are more general and two vertices 
can be connected by edges of various types. One use for this would be in 
genetics, where one gene can excite or inhibit another - or both 
simultaneously, which is why we can't simply assign negative numbers to the 
edges. The number of edge types is unlimited, but needs to be set in advance.

Before constructing a graph, you will also need to decide for the underlying
data structure. The differences for smaller graphs (e.g. with less than 100
nodes) should be insignificant, while for the larger, the decision should be
based upon the expected number of edges ("density" of the graph) and the
operations you plan to execute. Graphs with large number of edges (eg.>n2/4,
where n is the number of vertices) should be represented with adjacency
matrices (class :obj:`Orange.network.GraphAsMatrix`), graphs with small number
of edges with adjacency lists (:obj:`Orange.network.GraphAsList`) and those in
between with adjacency trees (:obj:`Orange.network.GraphAsTree`). Regarding
the speed, matrices are generally the fastest, while some operations, such as
finding all edges leading from certain node, will sometimes be faster with
lists or trees.

One thing that is not supported (at the moment?) are multiple edges of the 
same type between two vertices.

Construction
============

When constructing a graph, you will need to decide about the data structure for
representation of edges, and call the corresponding constructor. All
constructors take the same arguments: the number of vertices (needs to be given
in advance, you cannot add additional vertices later), a flag telling whether
the graph is directed or not, and the number of edge types. The default number
of edge types is 1 (a normal graph), while the other two arguments are
mandatory.

You can choose between three constructors, all derived from a single ancestor
:obj:`Orange.network.Graph`:

.. class:: GraphAsMatrix(nVertices, directed[, nEdgeTypes])

    Bases: :obj:`Orange.network.Graph`

    Edges are stored in a matrix with either n^2 or n(n+1)/2 elements,
    depending upon whether the graph is directed or not. (In C++, it is stored
    as float * pointing to an array of length n*n*nEdgeTypes or
    (n*(n+1))/2*nEdgeTypes elements, where nEdgeTypes is the number of edge
    types.) This representation is suitable for smaller graphs and for dense
    large graphs. For graph with only one edge type, this representation is
    more economical than representation with lists or trees when the number of
    edges is larger than n2/4. Inserting, deleting and checking the edges is
    fast; listing the neighbours of a certain node is fast unless the graph is
    sparse, in which case a graph represented with a list or a tree would be
    faster.
    
.. class:: GraphAsList(nVertices, directed[, nEdgeTypes]) 
    
    Bases: :obj:`Orange.network.Graph`
    
    Edges are stored in an ordered lists of neighbours, one list for each node.
    In C++, for each neighbour, the connection is stored in a structure with
    the vertex number (int), a pointer to the next structure, and an array of
    floats, one for each integer. With 16-byte alignment, this would take 16
    bytes for graphs with one or two edge types on the usual 32-bit platforms.
    For undirected graphs, each edge is stored only once, in the list of the
    edge with the smaller index. This makes the structure smaller and insertion
    and lookup faster; it slows down finding the neighbours of a given node.
    This structure is convenient for graphs with a very small number of edges.
    For them, inserting and removing edges is relatively fast; getting all
    edges leading from a vertex is fast, while getting edges leading to a
    vertex or getting all neighbours (in directed or undirected graph) is slow.
    
.. class:: GraphAsTree(nVertices, directed[, nEdgeTypes]) 
    
    Bases: :obj:`Orange.network.Graph`
    
    This structure is similar to GraphAsTree except that the edges are stored
    in trees instead of lists. This should be a structure of choice for all
    graph between really sparse and those having one quarter of possible edges.
    As expected, queries are fast, while insertion and removal of edges is
    somewhat slower (though still faster than for GraphAsList unless the number
    of edges is really small). Internally, nodes of the tree contain the vertex
    number, two pointers and a list of floats. With one edge type, this should
    be 16 bytes on 32-bit platforms.

Examples
--------

An ordinary undirected graph with 10 vertices stored in a matrix would thus be
constructed by::

    >>> graph = Orange.network.GraphAsMatrix(10, 0)

A directed graph with 1000 vertices and edges of three types, stored with
adjacency trees would be constructed by::

    >>> graph = Orange.network.GraphAsTree(1000, 1, 3)

Usage
=====

All three graph types are used in the same way, independent of the underlying
structure. All methods are defined in basic class :obj:`Orange.network.Graph`.

.. class:: Graph

    .. attribute:: nVertices
    
        The number of vertices (read-only, set at construction)
    
    .. attribute:: nEdgeTypes
    
        The number of different edge types (read-only, set at construction)
    
    .. attribute:: directed
    
        Tells whether the graph is directed (read-only, set at construction)
    
    .. attribute:: objects
    
        A dictionary, list or other sequence of objects that correspond to
        graph nodes. The use of this object is described in section on
        indexing.

    .. attribute:: forceMapping

        Determines whether to map integer indices through 'objects'. Details are
        described below.

    .. attribute:: returnIndices

        If set, the methods that return list of neighbours will return lists of
        integers even when objects are given.

    **Indexing**
    
    Vertices are referred to by either integer indices or Python objects of any
    type. In the latter case, a mapping should be provided by assigning the
    'objects' attribute. For instance, if you set graph.objects to ["Age",
    "Gender", "Height", "Weight"] then graph["Age", "Height"] would be equivalent
    to graph[0, 2] and graph.getNeighbours("Weight") to graph.getNeighbours(3).
    Vertex identifier doesn't need to be a string, it can be any Python
    object.
    
    If objects contains a dictionary, its keys are vertex identifiers and the
    values in the dictionary should be integers, eg.
       
    part of `network-graph.py`_

    .. literalinclude:: code/network-graph.py
        :lines: 20-23
 
    If not a dictionary, objects can be any kind of sequence. Usually, you will
    give it a list of the same length as the number of vertices in the graph,
    so each element would identify a vertex. When indexing, the index is sought
    for in the list. objects can also be a list of attributes, a domain, an
    example table or even a single string; Orange will run a code equivalent to
    "for o in graph.objects", so everything for which such a loop works, goes.
    
    Searching through the list is, of course, rather slow, so it is recommended
    to use integer indices for larger graphs. So, if you request
    graph.getNeighbours(0), the method will return the neighbours of the first
    vertex, even if objects is given. But - what if you want to map all
    indices, even integers, through objects? In this case, you need to set
    graph.forceMapping to 1. If graph.forceMapping is set and graph.objects is
    given, even getNeighbours(0) will search the graph.objects for 0 and return
    the neighbours of the corresponding (not necessarily the first) node.

    **Getting and Setting Edges**
    
    .. method:: graph[v1, v2]
    
        For graphs with a single edge type, graph[v1, v2] returns the weight of
        the edge between v1 and v2, or None if there is no edge (edge's weight
        can also be 0).
        
        For graphs with multiple edge types, graph[v1, v2] returns a list of
        weights for various edge types. Some (or all, if there is no edge)
        elements of the list can be None. If the edge does not exist, graph[v1,
        v2] returns a list of Nones, not a None.
        
        Edges can also be set by assigning them a weight, e.g.graph[2, 5]=1.5.
        As described above, if objects is a set, we can use other objects, such
        as names, as v1 and v2 (the same goes for all other functions described
        below).
        
        You can assign a list to graph[v1, v2]; in graph with three edge
        types you can set graph[2, 5] = [1.5, None, -2.0]. After that, there
        are two edges between vertices 2 and 5, one of the first type with
        weight 1.5, and one of the third with weight -2.0. To remove an edge,
        you can assign it a list of Nones or a single None, e.g. graph[2,
        5]=None; this removes edges of all types between the two nodes. 
        
        The list returned for graphs with multiple edge types is actually a
        reference to the edge, therefore you can set e = graph[v1, v2] and then
        manipulate e, for instance e[1]=10 or e[0]=None. Edge will behave just
        as an ordinary list (well, almost - no slicing ets). However, don't try
        to assign a list to e, eg e=[1, None, 4]. This assigns a list to e, not
        to the corresponding edge... 
        
    .. method:: graph[v1, v2, type] 
        
        This is defined only for graph with multiple edge types; it returns the
        weight for the edge of type type between v1 and v2, or None if there is
        no such edge. You can also establish an edge by assigning a weight
        (e.g. graph[2, 5, 2] = -2.0) or remove it by assigning it a None
        (graph[2, 5, 2] = None).
        
    .. method:: edgeExists(v1, v2[, type]) 
        
        Returns true if the edge between v1 and v2 exists. For multiple edge
        type graphs you can also specify the type of the edge you check for. If
        the third argument is omitted, the method returns true if there is any
        kind of edge between the two vertices. It is recommended to use this
        method when you want to check for a node. In single edge type graphs,
        if graph[v1, v2]: will fail when there is an edge but it has a weight
        of zero. With multiple edge types, if graph[v1, v2]: will always
        success since graph[v1, v2] returns a non- empty list; if there is no
        edges, this will be a list of Nones, but Python still treats it as
        "true". 
        
    .. method:: addCluster(list_of_vertices) 
        
        Creates a cluster - adds edges between all listed vertices.
    
    **Queries**
        
    Graph provides a set of functions that return nodes connected to a certain
    node.
    
    .. method:: getNeighbours(v1[, type])
    
        Returns all the nodes that are connected to v1. In directed graphs,
        this includes vertices with edges toward or from v1. In graphs with
        multiple edge types you can also specify the edge type you are
        interested in: getNeighbours will the return only the vertices that are
        connected to v1 by edges of that type.
    
    .. method:: getEdgesFrom(v1[, type])
    
        Return all the vertices which are connected to v1 by the edges leading
        from v1. In edges with multiple edge types, you can specify the edge
        type. In undirected graph, this function is equivalent to
        getNeighbours.
    
    .. method:: getEdgesTo(v1[, type])
    
        Returns all the vertices with edges leading to v1. Again, you can
        decide for a single edge type to observe, and, again again, in
        undirected graphs this function is equivalent to getNeighbours.
        
    If objects is set, functions return a list of objects (names of
    vertices or whatever objects you stored in objects). Otherwise, a list
    of integer indices is returned. If you want to force Graph to return
    integer indices even if objects is set, set graph.returnIndices to
    True.
    
    Of the three operations, the expensive one is to look for the vertices with
    edges pointing to the given edge. There is no problem when graph is
    represented as a matrix (:obj:`Orange.network.GraphAsMatrix`); these are
    always fast. On directed graph, getEdgeFrom is always fast as well.
    
    In undirected graphs represented as lists or trees, the edge between
    vertices with indices v1 and v2 is stored at the list/tree in the
    smaller of the two indices. Therefore to list all neighbours of v1,
    edges with v2<v1 are copied form v1's list, while for edges with v2>v1
    the function needs to look for v1 in each v2's list/tree. Lookup in
    trees is fast, while in representation with adjacency list, the
    function is slower for v1 closer to nVertices/2. If v1 is small there
    is a great number of v2>v1 whose lists are to be checked, but since the
    lists are ordered, v1 is more to the beginning of these lists (and when
    a vertex with index higher than v1 is encountered, we know that v1 is
    not on the list). If v2 is great, there it is more toward the end of
    the list, but there is smaller number of lists to be checked.
    Generally, the average number of traversed list elements for
    getNeighbours/getEdgesFrom/getEdgesTo on undirected graphs with
    p*nVertices2 edges is p(nVertices-v1)v1.
    
    To sum up, if you have a large undirected graph and intend to query for
    neighbours (or, equivalently, edges to or from a node) a lot, don't use
    :obj:`Orange.network.GraphAsList`. If the graph is small or you won't use
    these functions, it doesn't matter.
    
    For directed graphs, getEdgesFrom is trivial. The other two functions are
    even slower than for undirected graphs, since to find the edges leading
    from any vertex to a given one, all lists/trees need to be searched. So, if
    your algorithm will extensively use getEdgesTo or getNeighbours and your
    graph is large but the number of edges is less than nEdges2/2, you should
    use :obj:`Orange.network.GraphAsTree` or, to be faster but consume more
    memory store the graph as :obj:`Orange.network.GraphAsMatrix`. If the
    number of edges is greater, :obj:`Orange.network.GraphAsMatrix` is more
    economic anyway. This calculation is for graph with only one edge type;
    see the description of graph types for details.
    
    However, this is all programmed in C++, so whatever you do, the bottleneck
    will probably still be in your Python code and not in C++. You probably
    cannot miss by using :obj:`Orange.Network.GraphAsTree` disregarding the
    size of the graph and the operations you perform on it.

    **Graph analysis**
    
    .. method:: getSubGraph(vertices)
    
        Return a new graph of type :obj:`Orange.network.Graph` that is a
        subgraph of the original graph and consists of given vertices.
    
    .. method:: getClusteringCoefficient()
    
        Return the graph average local clustering coefficient, described in
        Watts DJ, Strogatz SH: Collective dynamics of 'small-world' networks.
        Nature 1998, 393(6684):440-442.
    
    .. method:: getConnectedComponents()
    
        Return a list of all connected components sorted descending by
        component size.
    
    .. method:: getDegreeDistribution()
    
        Return degree distribution as dictionary of type
        {degree:number_of_vertices}.
    
    .. method:: getDegrees()
    
        Return a list of degrees. List size matches number of vertices. Index
        of given degree matches index of corresponding vertex.
    
    .. method:: getHubs(n)
    
        Return a list of n largest hubs.
    
    .. method:: getShortestPaths(u, v)
    
        Return a list of vertices in the shortest path between u and v.
    
    .. method:: getDistance(u, v)
    
        Return a distance between vertices u and v.
    
    .. method:: getDiameter()
    
        Return a diameter of the graph.
        
Examples
--------

How to use graphs, part of `network-graph.py`_

.. literalinclude:: code/network-graph.py
    :lines: 9-56

Results::

    [(1, 0), (2, 0), (2, 1)]
    0.3
    0.3
    0.1
    0.3
    ['Gender', 'Height']
    [1, 2]
    (None, None, None)
    12.0
    (None, 12, None)
    1
    0
    1
    0
    0
    1
    (None, None, 3)
    (None, None, None)

How to use graphs with objects on edges, part of `network-graph-obj.py`_

.. literalinclude:: code/network-graph-obj.py
    :lines: 9-59
    
Results::

    [(1, 0), (2, 1)]
    [1, 2, 3]
    [1, 2, 3]
    a string
    None
    a string
    [1, 2, 3]
    ['Gender']
    [1]
    (None, None, None)
    12.0
    (None, 12, None)
    1
    0
    1
    0
    0
    1
    (None, None, 3)
    (None, None, None)

An example of network analysis, part of `network-graph-analysis.py`_ (uses:
`combination.net`_):

.. literalinclude:: code/network-graph-analysis.py
    :lines: 12-49
    
Results::

    Connected components
    [[0, 1, 2, 3, 4, 5, 6, 7, 8], [13, 14, 15, 16, 17, 18], [9, 10, 11, 12]]
    
    Degree distribution
    {1: 5, 2: 4, 3: 8, 4: 1, 5: 1}
    
    Degrees
    [4, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 5, 1, 1, 1, 1, 1]
    
    Hubs
    [13, 0, 1]
    
    Shortest path
    [2, 0]
    
    Distance
    1
    
    Diameter
    4

Subgraph image:

.. image:: files/network-subgraph.png

Additional functionality
------------------------

Should you need any additional functionality, just tell us. Many things are
trivial to implement in C++ and will be much faster than the corresponding
scripts in Python. (In this regard, minimal span trees, maximal flows, coloring
and shortest path search are, of course, not considered basic functionality. :)

=============================
Community Detection in Graphs
=============================

.. autoclass:: Orange.network.NetworkClustering
   :members:

.. _network-constructor.py: code/network-constructor.py
.. _network-optimization.py: code/network-optimization.py
.. _network-read.py: code/network-read.py
.. _K5.net: code/K5.net
.. _combination.net: code/combination.net
.. _network-widget.py: code/network-widget.py
.. _network-graph-analysis.py: code/network-graph-analysis.py
.. _network-graph.py: code/network-graph.py
.. _network-graph-obj.py: code/network-graph-obj.py

"""
import random
import os

import orangeom
import Orange.core
import Orange.data

from Orange.core import Graph, GraphAsList, GraphAsMatrix, GraphAsTree

class MdsTypeClass():
    def __init__(self):
        self.componentMDS = 0
        self.exactSimulation = 1
        self.MDS = 2

MdsType = MdsTypeClass()

class Network(orangeom.Network):
    
    """Bases: :obj:`Orange.network.GraphAsList`, :obj:`Orange.network.Graph` 
    
    Data structure for representing directed and undirected networks.
    
    Network class holds network structure information and supports basic
    network analysis. Network class is inherited from
    :obj:`Orange.network.GraphAsList`. Refer to
    :obj:`Orange.network.GraphAsList` for more graph analysis tools. See the
    orangeom.Pathfinder class for a way to simplify your network.
    
    .. attribute:: coors
   
        Coordinates for all vertices. They are initialized to random positions.
        You can modify them manually or use one of the optimization algorithms.
        Usage: coors[0][i], coors[1][i]; 0 for x-axis, 1 for y-axis
    
    
    .. attribute:: items
   
        ExampleTable with information about vertices. Number of rows should 
        match the number of vertices.
       
    .. attribute:: links
   
        ExampleTable with information about edges. Number of rows should match 
        the number of edges. Two float attributes named "u" and "v" must be in 
        links table domain to relate the data of an example to an edge. Here, 
        egde is defined by two vertices "u" and "v".

    .. attribute:: optimization
   
        An instance of the NetworkOptimization class. Various network layout 
        optimization methods can be applied to the network through this 
        attribute. 
        
    .. method:: fromDistanceMatrix(matrix, lower, upper, kNN=0):
        
        Creates edges between vertices with the distance within given 
        threshold. The DistanceMatrix dimension should equal the number of 
        vertices.
        
        :param matrix: number of objects in a matrix must match the number 
            of vertices in a network.
        :type matrix: Orange.core.SymMatrix
        :param lower: lower distance bound.
        :type lower: float
        :param upper: upper distance bound.
        :type upper: float
        :param kNN: specifies the minimum number of closest vertices to be 
            connected.
        :type kNN: int
        
    .. method:: hideVertices(vertices)
        
        Remove vertices from optimize list
        
    .. method:: showVertices(vertices)
    
        Add vertices to optimize list
        
    .. method:: showAll()
    
        Add all vertices to optimize list
        
    .. method:: getVisible()
    
        Return optimize list
    
    """
    
    def __init__(self, *args):
        """:param nVertices: number of vertices (default 1)
        :param nEdges: number of edge types (default 1)
        :param directedGraph: directed edges (default True)
        
        """
        #print "orngNetwork.Network"
        self.optimization = NetworkOptimization(self)
        self.clustering = NetworkClustering(self)
        
    def getDistanceMatrixThreshold(self, matrix, ratio):
        """Return lower and upper distance threshold values for the given 
        ratio of edges
        
        """
        values = []
        for i in range(matrix.dim):
            for j in range(i):
                values.append((matrix[i,j], i, j))
                
        values.sort()
        return values[0][0], values[int(ratio*len(values))][0]
        
    def save(self, fileName):
        """Save the network to a Pajek (.net) or GML file format. 
        data.Table items and links are saved automatically if the value is not 
        None. They are saved to "file_items.tab" and "file_links.tab" files.
        
        :param fileName: file path
        :type fileName: string
        
        """
        self.saveNetwork(fileName)
        
    def saveNetwork(self, fileName):        
        try:
            root, ext = os.path.splitext(fileName)
            if ext == '':
                fileName = root + '.net'
            graphFile = open(fileName, 'w+')
        except IOError:
            return 1
            
        root, ext = os.path.splitext(fileName)
        if ext.lower() == ".gml":
            self.saveGML(graphFile)
        else:
            self.savePajek(graphFile)

    def saveGML(self, fp):
        """Save network to GML (.gml) file format.
        
        :param fp: file pointer
        :type fp: file
        
        """
        fp.write("graph\n[\n")
        tabs = "\t"
        fp.write("%slabel\t\"%s\"\n" % (tabs, self.name))
        
        for v in range(self.nVertices):
            try:
                label = self.items[v]['label']
            except:
                label = ""
            
            fp.write("\tnode\n\t[\n\t\tid\t%d\n\t\tlabel\t\"%s\"\n\t]\n" % 
                     (v, label))
        
        for u,v in self.getEdges():
            fp.write("\tedge\n\t[\n\t\tsource\t%d\n\t\ttarget\t%d\n\t\tlabel\t\"%s\"\n\t]\n" % (u, v, ""))
        
        fp.write("]\n")
        
        if self.items != None and len(self.items) > 0:
            (name, ext) = os.path.splitext(fp.name)
            self.items.save(name + "_items.tab")
            
        if hasattr(self, 'links') and self.links != None and \
                                                        len(self.links) > 0:
            (name, ext) = os.path.splitext(fp.name)
            self.links.save(name + "_links.tab")
        
    def savePajek(self, fp):
        """Save network to Pajek (.net) file format.
        
        :param fp: file pointer
        :type fp: file
        
        """
        name = ''
        fp.write('### Generated with Orange Network Visualizer ### \n\n\n')
        if name == '':
            fp.write('*Network ' + '"no name" \n\n')
        else:
            fp.write('*Network ' + str(name) + ' \n\n')

        # print node descriptions
        fp.write('*Vertices %8d %8d\n' % (self.nVertices, self.nEdgeTypes))
        for v in range(self.nVertices):
            fp.write('% 8d ' % (v + 1))
            try:
                label = self.items[v]['label']
                fp.write(str('"' + str(label) + '"') + ' \t')
            except:
                fp.write(str('"' + str(v) + '"') + ' \t')
            
            if hasattr(self, 'coors'):
                x = self.coors[0][v]
                y = self.coors[1][v]
                z = 0.5000
                fp.write('%.4f    %.4f    %.4f\t' % (x, y, z))
            fp.write('\n')

        # print edge descriptions
        # not directed edges
        if self.directed:
            fp.write('*Arcs \n')
            for (i, j) in self.getEdges():
                if len(self[i, j]) > 0:
                    if self.nEdgeTypes > 1:
                        edge_str = str(self[i, j])
                    else:
                        edge_str = "%f" % float(str(self[i, j]))
                    fp.write('%8d %8d %s' % (i + 1, j + 1, edge_str))                    
                    fp.write('\n')
        # directed edges
        else:
            fp.write('*Edges \n')
            writtenEdges = {}
            for (i, j) in self.getEdges():
                if len(self[i, j]) > 0:
                    if i > j: i,j = j,i
                    
                    if not (i,j) in writtenEdges:
                        writtenEdges[(i,j)] = 1
                    else:
                        continue

                    if self.nEdgeTypes > 1:
                        edge_str = str(self[i, j])
                    else:
                        edge_str = "%f" % float(str(self[i, j]))
                    fp.write('%8d %8d %s' % (i + 1, j + 1, edge_str))                    
                    fp.write('\n')

        fp.write('\n')
        
        if self.items != None and len(self.items) > 0:
            (name, ext) = os.path.splitext(fp.name)
            self.items.save(name + "_items.tab")
            
        if hasattr(self, 'links') and self.links != None \
                                                    and len(self.links) > 0:
            (name, ext) = os.path.splitext(fp.name)
            self.links.save(name + "_links.tab")

        return 0
        
    @staticmethod
    def read(fileName, directed=0):
        """Read network. Supported network formats: from Pajek (.net) file, 
        GML.
        
        :param fileName: file path
        :type fileName: string
        :param directed: (default False)
        :type directed: bool
        
        """
        if type(fileName) == file:
            root, ext = os.path.splitext(fileName.name)
            if ext.lower() == ".net":
                net = Network(2,0).parseNetwork(fileName.read(), directed)
                net.optimization = NetworkOptimization(net)
                return net
            else:
                print "invalid network type", fileName.name
                return None
        else:
            root, ext = os.path.splitext(fileName)
            net = None
            if ext.lower() == ".net":
                net = Network(2,0).readPajek(fileName, directed)
            elif ext.lower() == ".gml":
                net = Network(2,0).readGML(fileName)
            else:
                print "Invalid file type %s" % fileName
                
            if net is not None:
                net.optimization = NetworkOptimization(net)
            return net 

class NetworkOptimization(orangeom.NetworkOptimization):
    
    """Perform network layout optimization. Network structure is defined in 
    :obj:`Orange.network.Network` class.
    
    :param network: Network to optimize
    :type network: Orange.network.Network
    
    .. attribute:: graph
    
    Holds the :obj:`Orange.network.Network` object.
    
    .. method:: random()
    
    Random layout optimization.
    
    .. method:: fruchtermanReingold(steps=100, temperature=1000, coolFactor=default, hiddenNodes=[], weighted=False) 
        
    Fruchterman-Reingold spring layout optimization. Set number of iterations 
    with argument steps, start temperature with temperature (for example: 1000) 
    and set list of hidden nodes with argument hidden_nodes.    
        
    .. method:: radialFruchtermanReingold(center, steps=100, temperature=1000)
    
    Radial Fruchterman-Reingold spring layout optimization. Set center node 
    with attribute center, number of iterations with argument steps and start 
    temperature with temperature (for example: 1000).
    
    .. method:: circularOriginal()
    
    Circular layout optimization based on original order.
    
    .. method:: circularRandom()
    
    Circular layout optimization based on random order.
    
    .. method:: circularCrossingReduction()
    
    Circular layout optimization (Michael Baur, Ulrik Brandes) with crossing 
    reduction.
    
    .. method:: closestVertex(x, y)
    
    Return the closest vertex to (x, y) coordinate.  
    
    .. method:: vertexDistances(x, y)
    
    Return distances (list of (distance, vertex) tuples) of all vertices to 
    the given coordinate.
    
    .. method:: getVerticesInRect(x1, y1, x2, y2)
    
    Return a list of all vertices in given rectangle.
    
    """
    
    def __init__(self, network=None, name="None"):
        if network is None:
            network = orangeom.Network(2, 0)
            
        self.setGraph(network)
        self.graph = network
        
        self.maxWidth = 1000
        self.maxHeight = 1000
        
        self.attributeList = {}
        self.attributeValues = {}
        self.vertexDistance = None
        self.mds = None
        
    def setNetwork(network):
        """Set the network object for layout optimization.
    
        :param network: network object for layout optimization
        :type network: Orange.network.Network
        
        """
        self.setGraph(network)
        
    def _computeForces(self):
        """Compute forces for each vertex for force vector visualization."""
        n = self.graph.nVertices
        vertices = set(range(n))
        e_avg = 0
        edges = self.graph.getEdges()
        for u,v in edges:
            u_ = numpy.array([self.graph.coors[0][u], self.graph.coors[1][u]])
            v_ = numpy.array([self.graph.coors[0][v], self.graph.coors[1][v]])
            e_avg += numpy.linalg.norm(u_ - v_)
        e_avg /= len(edges)
        
        forces = []
        maxforce = []
        components = self.graph.getConnectedComponents()
        for component in components:
            outer_vertices = vertices - set(component)
            
            for u in component:
                u_ = numpy.array([self.graph.coors[0][u], 
                                  self.graph.coors[1][u]])
                force = numpy.array([0.0, 0.0])                
                for v in outer_vertices:
                    v_ = numpy.array([self.graph.coors[0][v], 
                                      self.graph.coors[1][v]])
                    d = self.vertexDistance[u, v]
                    norm = numpy.linalg.norm(v_ - u_)
                    force += (d - norm) * (v_ - u_) / norm 
            
                forces.append(force)
                maxforce.append(numpy.linalg.norm(force))
            
        maxforce = max(maxforce)
        rv = []
        for v in range(n):
            force = forces[v]
            v_ = numpy.array([self.graph.coors[0][v], self.graph.coors[1][v]])
            f = force * e_avg / maxforce
            rv.append(([v_[0], v_[0] + f[0]],[v_[1], v_[1] + f[1]]))

        return rv
    
    def collapse(self):
        """Experimental method to group cliques to meta nodes."""
        if len(self.graph.getNodes(1)) > 0:
            nodes = list(set(range(self.graph.nVertices)) - \
                         set(self.graph.getNodes(1)))
                
            if len(nodes) > 0:
                subgraph = orangeom.Network(self.graph.getSubGraph(nodes))
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                    
                for i in range(len(nodes)):
                    self.coors[0][i] = oldcoors[0][nodes[i]]
                    self.coors[1][i] = oldcoors[1][nodes[i]]

        else:
            fullgraphs = self.graph.getLargestFullGraphs()
            subgraph = self.graph
            
            if len(fullgraphs) > 0:
                used = set()
                graphstomerge = list()
                #print fullgraphs
                for fullgraph in fullgraphs:
                    #print fullgraph
                    fullgraph_set = set(fullgraph)
                    if len(used & fullgraph_set) == 0:
                        graphstomerge.append(fullgraph)
                        used |= fullgraph_set
                        
                #print graphstomerge
                #print used
                subgraph = orangeom.Network(
                            subgraph.getSubGraphMergeClusters(graphstomerge))
                                   
                nodescomp = list(set(range(self.graph.nVertices)) - used)
                
                #subgraph.setattr("items", self.graph.items.getitems(nodescomp))
                #subgraph.items.append(self.graph.items[0])
                oldcoors = self.coors
                self.setGraph(subgraph)
                self.graph = subgraph
                for i in range(len(nodescomp)):
                    self.coors[0][i] = oldcoors[0][nodescomp[i]]
                    self.coors[1][i] = oldcoors[1][nodescomp[i]]
                    
                # place meta vertex in center of cluster    
                x, y = 0, 0
                for node in used:
                    x += oldcoors[0][node]
                    y += oldcoors[1][node]
                    
                x = x / len(used)
                y = y / len(used)
                
                self.coors[0][len(nodescomp)] = x
                self.coors[1][len(nodescomp)] = y
            
    def getVars(self):
        """Return a list of features in network items."""
        vars = []
        if (self.graph != None):
            if hasattr(self.graph, "items"):
                if isinstance(self.graph.items, Orange.data.Table):
                    vars[:0] = self.graph.items.domain.variables
                
                    metas = self.graph.items.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
        return vars
    
    def getEdgeVars(self):
        """Return a list of features in network links."""
        vars = []
        if (self.graph != None):
            if hasattr(self.graph, "links"):
                if isinstance(self.graph.links, Orange.data.Table):
                    vars[:0] = self.graph.links.domain.variables
                
                    metas = self.graph.links.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
                        
        return [x for x in vars if str(x.name) != 'u' and str(x.name) != 'v']
    
    def getData(self, i, j):
        import warnings
        warnings.warn("Deprecated.", DeprecationWarning)
        if self.graph.items is Orange.data.Table:
            return self.data[i][j]
        elif self.graph.data is type([]):
            return self.data[i][j]
        
    def nVertices(self):
        import warnings
        warnings.warn("Deprecated.", DeprecationWarning)
        if self.graph:
            return self.graph.nVertices
        
    def rotateVertices(self, components, phi): 
        """Rotate network components for a given angle.
        
        :param components: list of network components
        :type components: list of lists of vertex indices
        :param phi: list of component rotation angles (unit: radians)
        """  
        #print phi 
        for i in range(len(components)):
            if phi[i] == 0:
                continue
            
            component = components[i]
            
            x = self.graph.coors[0][component]
            y = self.graph.coors[1][component]
            
            x_center = x.mean()
            y_center = y.mean()
            
            x = x - x_center
            y = y - y_center
            
            r = numpy.sqrt(x ** 2 + y ** 2)
            fi = numpy.arctan2(y, x)
            
            fi += phi[i]
            #fi += factor * M[i] * numpy.pi / 180
                
            x = r * numpy.cos(fi)
            y = r * numpy.sin(fi)
            
            self.graph.coors[0][component] = x + x_center
            self.graph.coors[1][component] = y + y_center
            
    def rotateComponents(self, maxSteps=100, minMoment=0.000000001, 
                         callbackProgress=None, callbackUpdateCanvas=None):
        """Rotate the network components using a spring model."""
        if self.vertexDistance == None:
            return 1
        
        if self.graph == None:
            return 1
        
        if self.vertexDistance.dim != self.graph.nVertices:
            return 1
        
        self.stopRotate = 0
        
        # rotate only components with more than one vertex
        components = [component for component \
                      in self.graph.getConnectedComponents() \
                      if len(component) > 1]
        vertices = set(range(self.graph.nVertices))
        step = 0
        M = [1]
        temperature = [[30.0, 1] for i in range(len(components))]
        dirChange = [0] * len(components)
        while step < maxSteps and (max(M) > minMoment or min(M) < -minMoment) \
                                                     and not self.stopRotate:
            M = [0] * len(components) 
            
            for i in range(len(components)):
                component = components[i]
                
                outer_vertices = vertices - set(component)
                
                x = self.graph.coors[0][component]
                y = self.graph.coors[1][component]
                
                x_center = x.mean()
                y_center = y.mean()
                
                for j in range(len(component)):
                    u = component[j]

                    for v in outer_vertices:
                        d = self.vertexDistance[u, v]
                        u_x = self.graph.coors[0][u]
                        u_y = self.graph.coors[1][u]
                        v_x = self.graph.coors[0][v]
                        v_y = self.graph.coors[1][v]
                        L = [(u_x - v_x), (u_y - v_y)]
                        R = [(u_x - x_center), (u_y - y_center)]
                        e = math.sqrt((v_x - x_center) ** 2 + \
                                      (v_y - y_center) ** 2)
                        
                        M[i] += (1 - d) / (e ** 2) * numpy.cross(R, L)
            
            tmpM = numpy.array(M)
            #print numpy.min(tmpM), numpy.max(tmpM),numpy.average(tmpM),numpy.min(numpy.abs(tmpM))
            
            phi = [0] * len(components)
            #print "rotating", temperature, M
            for i in range(len(M)):
                if M[i] > 0:
                    if temperature[i][1] < 0:
                        temperature[i][0] = temperature[i][0] * 5 / 10
                        temperature[i][1] = 1
                        dirChange[i] += 1
                        
                    phi[i] = temperature[i][0] * numpy.pi / 180
                elif M[i] < 0:  
                    if temperature[i][1] > 0:
                        temperature[i][0] = temperature[i][0] * 5 / 10
                        temperature[i][1] = -1
                        dirChange[i] += 1
                    
                    phi[i] = -temperature[i][0] * numpy.pi / 180
            
            # stop rotating when phi is to small to notice the rotation
            if max(phi) < numpy.pi / 1800:
                #print "breaking"
                break
            
            self.rotateVertices(components, phi)
            if callbackUpdateCanvas: callbackUpdateCanvas()
            if callbackProgress : callbackProgress(min([dirChange[i] for i \
                                    in range(len(dirChange)) if M[i] != 0]), 9)
            step += 1
    
    def mdsUpdateData(self, components, mds, callbackUpdateCanvas):
        """Translate and rotate the network components to computed positions."""
        component_props = []
        x_mds = []
        y_mds = []
        phi = [None] * len(components)
        self.diag_coors = math.sqrt(( \
                    min(self.graph.coors[0]) - max(self.graph.coors[0]))**2 + \
                    (min(self.graph.coors[1]) - max(self.graph.coors[1]))**2)
        
        if self.mdsType == MdsType.MDS:
            x = [mds.points[u][0] for u in range(self.graph.nVertices)]
            y = [mds.points[u][1] for u in range(self.graph.nVertices)]
            self.graph.coors[0][range(self.graph.nVertices)] =  x
            self.graph.coors[1][range(self.graph.nVertices)] =  y
            if callbackUpdateCanvas:
                callbackUpdateCanvas()
            return
        
        for i in range(len(components)):    
            component = components[i]
            
            if len(mds.points) == len(components):  # if average linkage before
                x_avg_mds = mds.points[i][0]
                y_avg_mds = mds.points[i][1]
            else:                                   # if not average linkage before
                x = [mds.points[u][0] for u in component]
                y = [mds.points[u][1] for u in component]
        
                x_avg_mds = sum(x) / len(x) 
                y_avg_mds = sum(y) / len(y)
                # compute rotation angle
                c = [numpy.linalg.norm(numpy.cross(mds.points[u], \
                            [self.graph.coors[0][u],self.graph.coors[1][u]])) \
                            for u in component]
                n = [numpy.vdot([self.graph.coors[0][u], \
                                 self.graph.coors[1][u]], \
                                 [self.graph.coors[0][u], \
                                  self.graph.coors[1][u]]) for u in component]
                phi[i] = sum(c) / sum(n)
                #print phi
            
            x = self.graph.coors[0][component]
            y = self.graph.coors[1][component]
            
            x_avg_graph = sum(x) / len(x)
            y_avg_graph = sum(y) / len(y)
            
            x_mds.append(x_avg_mds) 
            y_mds.append(y_avg_mds)

            component_props.append((x_avg_graph, y_avg_graph, \
                                    x_avg_mds, y_avg_mds, phi))
        
        w = max(self.graph.coors[0]) - min(self.graph.coors[0])
        h = max(self.graph.coors[1]) - min(self.graph.coors[1])
        d = math.sqrt(w**2 + h**2)
        #d = math.sqrt(w*h)
        e = [math.sqrt((self.graph.coors[0][u] - self.graph.coors[0][v])**2 + 
                  (self.graph.coors[1][u] - self.graph.coors[1][v])**2) for 
                  (u, v) in self.graph.getEdges()]
        
        if self.scalingRatio == 0:
            pass
        elif self.scalingRatio == 1:
            self.mdsScaleRatio = d
        elif self.scalingRatio == 2:
            self.mdsScaleRatio = d / sum(e) * float(len(e))
        elif self.scalingRatio == 3:
            self.mdsScaleRatio = 1 / sum(e) * float(len(e))
        elif self.scalingRatio == 4:
            self.mdsScaleRatio = w * h
        elif self.scalingRatio == 5:
            self.mdsScaleRatio = math.sqrt(w * h)
        elif self.scalingRatio == 6:
            self.mdsScaleRatio = 1
        elif self.scalingRatio == 7:
            e_fr = 0
            e_count = 0
            for i in range(self.graph.nVertices):
                for j in range(i + 1, self.graph.nVertices):
                    x1 = self.graph.coors[0][i]
                    y1 = self.graph.coors[1][i]
                    x2 = self.graph.coors[0][j]
                    y2 = self.graph.coors[1][j]
                    e_fr += math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    e_count += 1
            self.mdsScaleRatio = e_fr / e_count
        elif self.scalingRatio == 8:
            e_fr = 0
            e_count = 0
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    x_avg_graph_i, y_avg_graph_i, x_avg_mds_i, \
                    y_avg_mds_i, phi_i = component_props[i]
                    x_avg_graph_j, y_avg_graph_j, x_avg_mds_j, \
                    y_avg_mds_j, phi_j = component_props[j]
                    e_fr += math.sqrt((x_avg_graph_i-x_avg_graph_j)**2 + \
                                      (y_avg_graph_i-y_avg_graph_j)**2)
                    e_count += 1
            self.mdsScaleRatio = e_fr / e_count       
        elif self.scalingRatio == 9:
            e_fr = 0
            e_count = 0
            for i in range(len(components)):    
                component = components[i]
                x = self.graph.coors[0][component]
                y = self.graph.coors[1][component]
                for i in range(len(x)):
                    for j in range(i + 1, len(y)):
                        x1 = x[i]
                        y1 = y[i]
                        x2 = x[j]
                        y2 = y[j]
                        e_fr += math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        e_count += 1
            self.mdsScaleRatio = e_fr / e_count
        
        diag_mds =  math.sqrt((max(x_mds) - min(x_mds))**2 + (max(y_mds) - \
                                                              min(y_mds))**2)
        e = [math.sqrt((self.graph.coors[0][u] - self.graph.coors[0][v])**2 + 
                  (self.graph.coors[1][u] - self.graph.coors[1][v])**2) for 
                  (u, v) in self.graph.getEdges()]
        e = sum(e) / float(len(e))
        
        x = [mds.points[u][0] for u in range(len(mds.points))]
        y = [mds.points[u][1] for u in range(len(mds.points))]
        w = max(x) - min(x)
        h = max(y) - min(y)
        d = math.sqrt(w**2 + h**2)
        
        if len(x) == 1:
            r = 1
        else:
            if self.scalingRatio == 0:
                r = self.mdsScaleRatio / d * e
            elif self.scalingRatio == 1:
                r = self.mdsScaleRatio / d
            elif self.scalingRatio == 2:
                r = self.mdsScaleRatio / d * e
            elif self.scalingRatio == 3:
                r = self.mdsScaleRatio * e
            elif self.scalingRatio == 4:
                r = self.mdsScaleRatio / (w * h)
            elif self.scalingRatio == 5:
                r = self.mdsScaleRatio / math.sqrt(w * h)
            elif self.scalingRatio == 6:
                r = 1 / math.sqrt(self.graph.nVertices)
            elif self.scalingRatio == 7:
                e_mds = 0
                e_count = 0
                for i in range(len(mds.points)):
                    for j in range(i):
                        x1 = mds.points[i][0]
                        y1 = mds.points[i][1]
                        x2 = mds.points[j][0]
                        y2 = mds.points[j][1]
                        e_mds += math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        e_count += 1
                r = self.mdsScaleRatio / e_mds * e_count
            elif self.scalingRatio == 8:
                e_mds = 0
                e_count = 0
                for i in range(len(components)):
                    for j in range(i + 1, len(components)):
                        x_avg_graph_i, y_avg_graph_i, x_avg_mds_i, \
                        y_avg_mds_i, phi_i = component_props[i]
                        x_avg_graph_j, y_avg_graph_j, x_avg_mds_j, \
                        y_avg_mds_j, phi_j = component_props[j]
                        e_mds += math.sqrt((x_avg_mds_i-x_avg_mds_j)**2 + \
                                           (y_avg_mds_i-y_avg_mds_j)**2)
                        e_count += 1
                r = self.mdsScaleRatio / e_mds * e_count
            elif self.scalingRatio == 9:
                e_mds = 0
                e_count = 0
                for i in range(len(mds.points)):
                    for j in range(i):
                        x1 = mds.points[i][0]
                        y1 = mds.points[i][1]
                        x2 = mds.points[j][0]
                        y2 = mds.points[j][1]
                        e_mds += math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        e_count += 1
                r = self.mdsScaleRatio / e_mds * e_count
                
            #r = self.mdsScaleRatio / d
            #print "d", d, "r", r
            #r = self.mdsScaleRatio / math.sqrt(self.graph.nVertices)
            
        for i in range(len(components)):
            component = components[i]
            x_avg_graph, y_avg_graph, x_avg_mds, \
            y_avg_mds, phi = component_props[i]
            
#            if phi[i]:  # rotate vertices
#                #print "rotate", i, phi[i]
#                r = numpy.array([[numpy.cos(phi[i]), -numpy.sin(phi[i])], [numpy.sin(phi[i]), numpy.cos(phi[i])]])  #rotation matrix
#                c = [x_avg_graph, y_avg_graph]  # center of mass in FR coordinate system
#                v = [numpy.dot(numpy.array([self.graph.coors[0][u], self.graph.coors[1][u]]) - c, r) + c for u in component]
#                self.graph.coors[0][component] = [u[0] for u in v]
#                self.graph.coors[1][component] = [u[1] for u in v]
                
            # translate vertices
            if not self.rotationOnly:
                self.graph.coors[0][component] = \
                (self.graph.coors[0][component] - x_avg_graph) / r + x_avg_mds
                self.graph.coors[1][component] = \
                (self.graph.coors[1][component] - y_avg_graph) / r + y_avg_mds
               
        if callbackUpdateCanvas:
            callbackUpdateCanvas()
    
    def mdsCallback(self, a,b=None):
        """Refresh the UI when running  MDS on network components."""
        if not self.mdsStep % self.mdsRefresh:
            self.mdsUpdateData(self.mdsComponents, 
                               self.mds, 
                               self.callbackUpdateCanvas)
            
            if self.mdsType == MdsType.exactSimulation:
                self.mds.points = [[self.graph.coors[0][i], \
                                    self.graph.coors[1][i]] \
                                    for i in range(len(self.graph.coors))]
                self.mds.freshD = 0
            
            if self.callbackProgress != None:
                self.callbackProgress(self.mds.avgStress, self.mdsStep)
                
        self.mdsStep += 1

        if self.stopMDS:
            return 0
        else:
            return 1
            
    def mdsComponents(self, mdsSteps, mdsRefresh, callbackProgress=None, \
                      callbackUpdateCanvas=None, torgerson=0, \
                      minStressDelta=0, avgLinkage=False, rotationOnly=False, \
                      mdsType=MdsType.componentMDS, scalingRatio=0, \
                      mdsFromCurrentPos=0):
        """Position the network components according to similarities among 
        them.
        
        """

        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            return 1
        
        if self.graph == None:
            return 1
        
        if self.vertexDistance.dim != self.graph.nVertices:
            return 1
        
        self.mdsComponents = self.graph.getConnectedComponents()
        self.mdsRefresh = mdsRefresh
        self.mdsStep = 0
        self.stopMDS = 0
        self.vertexDistance.matrixType = Orange.core.SymMatrix.Symmetric
        self.diag_coors = math.sqrt((min(self.graph.coors[0]) -  \
                                     max(self.graph.coors[0]))**2 + \
                                     (min(self.graph.coors[1]) - \
                                      max(self.graph.coors[1]))**2)
        self.rotationOnly = rotationOnly
        self.mdsType = mdsType
        self.scalingRatio = scalingRatio

        w = max(self.graph.coors[0]) - min(self.graph.coors[0])
        h = max(self.graph.coors[1]) - min(self.graph.coors[1])
        d = math.sqrt(w**2 + h**2)
        #d = math.sqrt(w*h)
        e = [math.sqrt((self.graph.coors[0][u] - self.graph.coors[0][v])**2 + 
                  (self.graph.coors[1][u] - self.graph.coors[1][v])**2) for 
                  (u, v) in self.graph.getEdges()]
        self.mdsScaleRatio = d / sum(e) * float(len(e))
        #print d / sum(e) * float(len(e))
        
        if avgLinkage:
            matrix = self.vertexDistance.avgLinkage(self.mdsComponents)
        else:
            matrix = self.vertexDistance
        
        #if self.mds == None: 
        self.mds = orngMDS.MDS(matrix)
        
        if mdsFromCurrentPos:
            if avgLinkage:
                for u, c in enumerate(self.mdsComponents):
                    x = sum(self.graph.coors[0][c]) / len(c)
                    y = sum(self.graph.coors[1][c]) / len(c)
                    self.mds.points[u][0] = x
                    self.mds.points[u][1] = y
            else:
                for u in range(self.graph.nVertices):
                    self.mds.points[u][0] = self.graph.coors[0][u] 
                    self.mds.points[u][1] = self.graph.coors[1][u]
            
        # set min stress difference between 0.01 and 0.00001
        self.minStressDelta = minStressDelta
        self.callbackUpdateCanvas = callbackUpdateCanvas
        self.callbackProgress = callbackProgress
        
        if torgerson:
            self.mds.Torgerson() 

        self.mds.optimize(mdsSteps, orngMDS.SgnRelStress, self.minStressDelta,\
                          progressCallback=self.mdsCallback)
        self.mdsUpdateData(self.mdsComponents, self.mds, callbackUpdateCanvas)
        
        if callbackProgress != None:
            callbackProgress(self.mds.avgStress, self.mdsStep)
        
        del self.rotationOnly
        del self.diag_coors
        del self.mdsRefresh
        del self.mdsStep
        #del self.mds
        del self.mdsComponents
        del self.minStressDelta
        del self.callbackUpdateCanvas
        del self.callbackProgress
        del self.mdsType
        del self.mdsScaleRatio
        del self.scalingRatio
        return 0

    def mdsComponentsAvgLinkage(self, mdsSteps, mdsRefresh, \
                                callbackProgress=None, \
                                callbackUpdateCanvas=None, torgerson=0, \
                                minStressDelta = 0, scalingRatio=0,\
                                mdsFromCurrentPos=0):
        return self.mdsComponents(mdsSteps, mdsRefresh, callbackProgress, \
                                  callbackUpdateCanvas, torgerson, \
                                  minStressDelta, True, \
                                  scalingRatio=scalingRatio, \
                                  mdsFromCurrentPos=mdsFromCurrentPos)

    def saveNetwork(self, fn):
        import warnings
        warnings.warn("Deprecated. Use Orange.network.Network.save", 
                      DeprecationWarning)
        name = ''
        try:
            root, ext = os.path.splitext(fn)
            if ext == '':
                fn = root + '.net'
            
            graphFile = file(fn, 'w+')
        except IOError:
            return 1

        graphFile.write('### Generated with Orange.network ### \n\n\n')
        if name == '':
            graphFile.write('*Network ' + '"no name" \n\n')
        else:
            graphFile.write('*Network ' + str(name) + ' \n\n')


        #izpis opisov vozlisc
        print "e", self.graph.nEdgeTypes
        graphFile.write('*Vertices %8d %8d\n' % (self.graph.nVertices, \
                                                 self.graph.nEdgeTypes))
        for v in range(self.graph.nVertices):
            graphFile.write('% 8d ' % (v + 1))
#            if verticesParms[v].label!='':
#                self.GraphFile.write(str('"'+ verticesParms[v].label + '"') + ' \t')
#            else:
            try:
                label = self.graph.items[v]['label']
                graphFile.write(str('"' + str(label) + '"') + ' \t')
            except:
                graphFile.write(str('"' + str(v) + '"') + ' \t')
            
            x = self.network.coors[0][v]
            y = self.network.coors[1][v]
            #if x < 0: x = 0
            #if x >= 1: x = 0.9999
            #if y < 0: y = 0
            #if y >= 1: y = 0.9999
            z = 0.5000
            graphFile.write('%.4f    %.4f    %.4f\t' % (x, y, z))
            graphFile.write('\n')

        #izpis opisov povezav
        #najprej neusmerjene
        if self.graph.directed:
            graphFile.write('*Arcs \n')
            for (i, j) in self.graph.getEdges():
                if len(self.graph[i, j]) > 0:
                    graphFile.write('%8d %8d %f' % (i + 1, j + 1, \
                                                float(str(self.graph[i, j]))))
                    graphFile.write('\n')
        else:
            graphFile.write('*Edges \n')
            for (i, j) in self.graph.getEdges():
                if len(self.graph[i, j]) > 0:
                    graphFile.write('%8d %8d %f' % (i + 1, j + 1, \
                                                float(str(self.graph[i, j]))))
                    graphFile.write('\n')

        graphFile.write('\n')
        graphFile.close()
        
        if self.graph.items != None and len(self.graph.items) > 0:
            (name, ext) = os.path.splitext(fn)
            self.graph.items.save(name + "_items.tab")
            
        if self.graph.links != None and len(self.graph.links) > 0:
            (name, ext) = os.path.splitext(fn)
            self.graph.links.save(name + "_links.tab")

        return 0
    
    def readNetwork(self, fn, directed=0):
        import warnings
        warnings.warn("Deprecated. Use Orange.network.Network.read", 
                      DeprecationWarning)
        network = Network(1,directed)
        net = network.readPajek(fn, directed)
        self.setGraph(net)
        self.graph = net
        return net
    
class NetworkClustering():
    
    """A collection of algorithms for community detection in graphs.
    
    :param network: network data for community detection
    :type network: Orange.network.Network
    """ 
    
    random.seed(0)
    
    def __init__(self, network):
        self.net = network
        
        
    def labelPropagation(self, results2items=0, resultHistory2items=0):
        """Label propagation method from Raghavan et al., 2007
        
        :param results2items: append a new feature result to items 
            (Orange.data.Table)
        :type results2items: bool
        :param resultHistory2items: append new features result to items 
            (Orange.data.Table) after each iteration of the algorithm
        :type resultHistory2items: bool
        """
        
        vertices = range(self.net.nVertices)
        labels = range(self.net.nVertices)
        lblhistory = []
        #consecutiveStop = 0
        for i in range(1000):
            random.shuffle(vertices)
            stop = 1
            for v in vertices:
                nbh = self.net.getNeighbours(v)
                if len(nbh) == 0:
                    continue
                
                lbls = [labels[u] for u in nbh]
                lbls = [(len(list(c)), l) for l, c in itertools.groupby(lbls)]
                m = max(lbls)[0]
                mlbls = [l for c, l in lbls if c >= m]
                lbl = random.choice(mlbls)
                
                if labels[v] not in mlbls: stop = 0
                labels[v] = lbl
                
            lblhistory.append([str(l) for l in labels])
            # if stopping condition might be satisfied, check it
            if stop:
                for v in vertices:
                    nbh = self.net.getNeighbours(v)
                    if len(nbh) == 0: continue
                    lbls = [labels[u] for u in nbh]
                    lbls = [(len(list(c)), l) for l, c \
                            in itertools.groupby(lbls)]
                    m = max(lbls)[0]
                    mlbls = [l for c, l in lbls if c >= m]
                    if labels[v] not in mlbls: 
                        stop = 0
                        break
                if stop: break
                    
        if results2items and not resultHistory2items:
            attrs = [Orange.data.feature.Discrete(
                                        'clustering label propagation',
                                        values=list(set([l for l \
                                                        in lblhistory[-1]])))]
            dom = Orange.data.Domain(attrs, 0)
            data = Orange.data.Table(dom, [[l] for l in lblhistory[-1]])
            if self.net.items is None:
                self.net.items = data  
            else: 
                self.net.items = Orange.data.Table([self.net.items, data])
        if resultHistory2items:
            attrs = [Orange.data.feature.Discrete('c'+ str(i),
                values=list(set([l for l in lblhistory[0]]))) for i,labels \
                in enumerate(lblhistory)]
            dom = Orange.data.Domain(attrs, 0)
            # transpose history
            data = map(list, zip(*lblhistory))
            data = Orange.data.Table(dom, data)
            if self.net.items is None:
                self.net.items = data  
            else: 
                self.net.items = Orange.data.Table([self.net.items, data])

        return labels
    
    
