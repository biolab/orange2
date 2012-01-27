""" 
.. index:: network

*********
BaseGraph
*********

BaseGraph is primarily used to work with additional data attached to the 
NetworkX graph. Two types of data can be added to the graph:

* items (:obj:`Orange.data.Table`) - a table containing data about graph nodes. Each row in the table should correspond to a node with ID set to the row index.
* links (:obj:`Orange.data.Table`) - a table containing data about graph edges. Each row in the table corresponds to an edge. Two columns titled "u" and "v" must be given in the table, each containing indices of nodes on the given edge.
    
Some other methods, common to all graph types are also added to BaseGraph class.
    
.. autoclass:: Orange.network.BaseGraph
   :members:

***********
Graph types
***********

The reference in this section is complemented with the original NetworkX 
library reference. For a complete documentation please refer to the 
`NetworkX docs <http://networkx.lanl.gov/reference/>`_. All methods from the
NetworkX package can be used for graph analysis and manipulation with exception
to read and write graph methods. For reading and writing graphs please refer to 
the Orange.network.readwrite docs. 

Graph
=====

.. autoclass:: Orange.network.Graph
   :members:

DiGraph
=======
   
.. autoclass:: Orange.network.DiGraph
   :members:

MultiGraph
==========
   
.. autoclass:: Orange.network.MultiGraph
   :members:
   
MultiDiGraph
============
   
.. autoclass:: Orange.network.MultiDiGraph
   :members:
   
"""

import copy
import math
import numpy
import networkx as nx
import Orange
import orangeom
import readwrite
from networkx import algorithms 
from networkx.classes import function

class MdsTypeClass():
    def __init__(self):
        self.componentMDS = 0
        self.exactSimulation = 1
        self.MDS = 2

MdsType = MdsTypeClass()

def _get_doc(doc):
    tmp = doc.replace('nx.', 'Orange.network.')
    return tmp
    
class BaseGraph():
    """A collection of methods inherited by all graph types (:obj:`Graph`, 
    :obj:`DiGraph`, :obj:`MultiGraph` and :obj:`MultiDiGraph`).
    
    """
    
    def __init__(self):
        self._items = None
        self._links = None
        
    def items(self):
        """Return the :obj:`Orange.data.Table` items with data about network 
        nodes.
        
        """
         
        if self._items is not None and \
                        len(self._items) != self.number_of_nodes():
            print "Warning: items length does not match the number of nodes."
            
        return self._items
    
    def set_items(self, items=None):
        """Set the :obj:`Orange.data.Table` items to the given data. Notice 
        that the number of instances must match the number of nodes.
        
        """
        
        if items is not None:
            if not isinstance(items, Orange.data.Table):
                raise TypeError('items must be of type \'Orange.data.Table\'')
            if len(items) != self.number_of_nodes():
                print "Warning: items length must match the number of nodes."
                
        self._items = items
        
    def links(self):
        """Return the :obj:`Orange.data.Table` links with data about network 
        edges.
        
        """
        
        if self._links is not None \
                    and len(self._links) != self.number_of_edges():
            print "Warning: links length does not match the number of edges."
            
        return self._links
    
    def set_links(self, links=None):
        """Set the :obj:`Orange.data.Table` links to the given data. Notice 
        that the number of instances must match the number of edges.
        
        """
        
        if links is not None:
            if not isinstance(links, Orange.data.Table):
                raise TypeError('links must be of type \'Orange.data.Table\'')
            if len(links) != self.number_of_edges():
                print "Warning: links length must match the number of edges."
        
        self._links = links
        
    def to_orange_network(self):
        """Convert the current network to >>Orange<< NetworkX standard. To use
        :obj:`Orange.network` in Orange widgets, set node IDs to be range 
        [0, no_of_nodes - 1].
        
        """ 
        
        G = self.__class__()
        node_list = sorted(self.nodes())
        node_to_index = dict(zip(node_list, range(self.number_of_nodes())))
        index_to_node = dict(zip(range(self.number_of_nodes()), node_list))
        
        G.add_nodes_from(zip(range(self.number_of_nodes()), [copy.deepcopy(self.node[nid]) for nid in node_list]))
        G.add_edges_from(((node_to_index[u], node_to_index[v], copy.deepcopy(self.edge[u][v])) for u,v in self.edges()))
        
        for id in G.node.keys():
            G.node[id]['old_id'] = index_to_node[id]  
        
        if self.items():
            G.set_items(self.items())

        if self.links():
            G.set_links(self.links())
        
        return G
        
    ### TODO: OVERRIDE METHODS THAT CHANGE GRAPH STRUCTURE, add warning prints
    
    def items_vars(self):
        """Return a list of features in the :obj:`Orange.data.Table` items."""
        
        vars = []
        if (self._items is not None):
            if isinstance(self._items, Orange.data.Table):
                vars = list(self._items.domain.variables)
            
                metas = self._items.domain.getmetas(0)
                vars.extend(var for i, var in metas.iteritems())
        return vars
    
    def links_vars(self):
        """Return a list of features in the :obj:`Orange.data.Table` links."""
        
        vars = []
        if (self._links is not None):
            if isinstance(self._links, Orange.data.Table):
                vars = list(self._links.domain.variables)
            
                metas = self._links.domain.getmetas(0)
                vars.extend(var for i, var in metas.iteritems())
        return [x for x in vars if str(x.name) != 'u' and str(x.name) != 'v']    
    
class Graph(BaseGraph, nx.Graph):
    """Bases: `NetworkX.Graph <http://networkx.lanl.gov/reference/classes.graph.html>`_, 
    :obj:`Orange.network.BaseGraph` 
    
    """
    
    def __init__(self, data=None, name='', **attr):  
        nx.Graph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
    
    def subgraph(self, nbunch):
        G = nx.Graph.subgraph(self, nbunch)
        items = self.items().get_items(G.nodes())
        G = G.to_orange_network()
        G.set_items(items)
        return G
        # TODO: _links
    
    __doc__ += _get_doc(nx.Graph.__doc__)
    __init__.__doc__ = _get_doc(nx.Graph.__init__.__doc__)
     
class DiGraph(BaseGraph, nx.DiGraph):
    """Bases: `NetworkX.DiGraph <http://networkx.lanl.gov/reference/classes.digraph.html>`_, 
    :obj:`Orange.network.BaseGraph` 
    
    """
    
    
    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
    
    __doc__ += _get_doc(nx.DiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.DiGraph.__init__.__doc__)
     
class MultiGraph(BaseGraph, nx.MultiGraph):
    """Bases: `NetworkX.MultiGraph <http://networkx.lanl.gov/reference/classes.multigraph.html>`_, 
    :obj:`Orange.network.BaseGraph` 
    
    """
    
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
    
    __doc__ += _get_doc(nx.MultiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.MultiGraph.__init__.__doc__)
     
class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    """Bases: `NetworkX.MultiDiGraph <http://networkx.lanl.gov/reference/classes.multidigraph.html>`_, 
    :obj:`Orange.network.BaseGraph` 
    
    """
    
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiDiGraph.__init__(self, data, name=name, **attr)
        BaseGraph.__init__(self)
    
    __doc__ += _get_doc(nx.MultiDiGraph.__doc__)
    __init__.__doc__ = _get_doc(nx.MultiDiGraph.__init__.__doc__)
    
class GraphLayout(orangeom.GraphLayout):
    """A class for graph layout optimization. Before using any of the layout
    optimization technique, the class have to be initialized with the :obj:`set_graph`
    method. Also, do not forget to call :obj:`set_graph` again if the graph
    structure changes.
    
    .. attribute:: coors
   
        Coordinates of all vertices. They are initialized to random positions.
        You can modify them manually or use one of the optimization algorithms.
        Usage: coors[0][i], coors[1][i]; 0 for x-axis, 1 for y-axis
        
    
    .. automethod:: Orange.network.GraphLayout.set_graph
    
    **Network optimization**
    
    .. automethod:: Orange.network.GraphLayout.random
    
    .. automethod:: Orange.network.GraphLayout.fr
    
    .. automethod:: Orange.network.GraphLayout.fr_radial
    
    .. automethod:: Orange.network.GraphLayout.circular_original
    
    .. automethod:: Orange.network.GraphLayout.circular_random
    
    .. automethod:: Orange.network.GraphLayout.circular_crossing_reduction
    
    **FragViz**
    
    .. automethod:: Orange.network.GraphLayout.mds_components
    
    .. automethod:: Orange.network.GraphLayout.rotate_components
    
    **Helper methods** 
    
    .. automethod:: Orange.network.GraphLayout.get_vertices_in_rect
    
    .. automethod:: Orange.network.GraphLayout.closest_vertex
    
    .. automethod:: Orange.network.GraphLayout.vertex_distances
    
    .. automethod:: Orange.network.GraphLayout.rotate_vertices
    
    **Examples**
    
    *Network constructor and random layout*
    
    In our first example we create a Network object with a simple full graph (K5). 
    Vertices are initially placed randomly. Graph is visualized using pylabs 
    matplotlib. 
        
    :download:`network-constructor-nx.py <code/network-constructor-nx.py>`
    
    .. literalinclude:: code/network-constructor-nx.py
    
    Executing the above saves a pylab window with the following graph drawing:
    
    .. image:: files/network-K5-random.png
    
    *Network layout optimization*
    
    This example demonstrates how to optimize network layout using one of the
    included algorithms.
    
    part of :download:`network-optimization-nx.py <code/network-optimization-nx.py>`
    
    .. literalinclude:: code/network-optimization-nx.py
        :lines: 14-19
        
    The result of the above script is a spring force layout optimization:
    
    .. image:: files/network-K5-fr.png
    
    """
    
    def __init__(self):
        self.graph = None
        self.items_matrix = None
        
    def set_graph(self, graph=None, positions=None):
        """Init graph structure.
        
        :param graph: Orange network
        :type graph: Orange.netowork.Graph
        
        :param positions: Initial node positions
        :type positions: A list of positions (x, y)
        
        """
        self.graph = graph
        
        if positions is not None and len(positions) == graph.number_of_nodes():
            orangeom.GraphLayout.set_graph(self, graph, positions)
        else:
            orangeom.GraphLayout.set_graph(self, graph)
            
    def random(self):
        """Random graph layout."""
        
        orangeom.GraphLayout.random(self)
        
    def fr(self, steps, temperature, coolFactor=0, weighted=False):
        """Fruchterman-Reingold spring layout optimization. Set number of 
        iterations with argument steps, start temperature with temperature 
        (for example: 1000).
        
        """
        
        return orangeom.GraphLayout.fr(self, steps, temperature, coolFactor, weighted)
        
    def fr_radial(self, center, steps, temperature):
        """Radial Fruchterman-Reingold spring layout optimization. Set center 
        node with attribute center, number of iterations with argument steps 
        and start temperature with temperature (for example: 1000).
        
        """
        
        return orangeom.GraphLayout.fr_radial(self, center, steps, temperature)
    
    def circular_original(self):
        """Circular graph layout with original node order."""
        
        orangeom.GraphLayout.circular_original(self)
    
    def circular_random(self):
        """Circular graph layout with random node order."""
        
        orangeom.GraphLayout.circular_random(self)
    
    def circular_crossing_reduction(self):
        """Circular graph layout with edge crossing reduction (Michael Baur, 
        Ulrik Brandes).
        
        """
        
        orangeom.GraphLayout.circular_crossing_reduction(self)
    
    def get_vertices_in_rect(self, x1, y1, x2, y2):
        """Return a list of nodes in the given rectangle."""
        
        return orangeom.GraphLayout.get_vertices_in_rect(self, x1, y1, x2, y2)
    
    def closest_vertex(self, x, y):
        """Return the closest node to given point."""
        
        return orangeom.GraphLayout.closest_vertex(self, x, y)
    
    def vertex_distances(self, x, y):
        """Return distances (a list of (distance, vertex) tuples) of all nodes 
        to the given position.
        
        """
        
        return orangeom.GraphLayout.vertex_distances(self, x, y)
    
    def rotate_vertices(self, components, phi): 
        """Rotate network components for a given angle.
        
        :param components: list of network components
        :type components: list of lists of vertex indices
        
        :param phi: list of component rotation angles (unit: radians)
        :type phi: float
        
        """  
        #print phi 
        for i in range(len(components)):
            if phi[i] == 0:
                continue
            
            component = components[i]
            
            x = self.coors[0][component]
            y = self.coors[1][component]
            
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
            
            self.coors[0][component] = x + x_center
            self.coors[1][component] = y + y_center
    
    def rotate_components(self, maxSteps=100, minMoment=0.000000001, 
                          callbackProgress=None, callbackUpdateCanvas=None):
        """Rotate the network components using a spring model."""
        
        if self.items_matrix == None:
            return 1
        
        if self.graph == None:
            return 1
        
        if self.items_matrix.dim != self.graph.number_of_nodes():
            return 1
        
        self.stopRotate = 0
        
        # rotate only components with more than one vertex
        components = [component for component \
            in Orange.network.nx.algorithms.components.connected_components(self.graph) \
            if len(component) > 1]
        vertices = set(range(self.graph.number_of_nodes()))
        step = 0
        M = [1]
        temperature = [[30.0, 1] for i in range(len(components))]
        dirChange = [0] * len(components)
        while step < maxSteps and (max(M) > minMoment or \
                                min(M) < -minMoment) and not self.stopRotate:
            M = [0] * len(components) 
            
            for i in range(len(components)):
                component = components[i]
                
                outer_vertices = vertices - set(component)
                
                x = self.coors[0][component]
                y = self.coors[1][component]
                
                x_center = x.mean()
                y_center = y.mean()
                
                for j in range(len(component)):
                    u = component[j]

                    for v in outer_vertices:
                        d = self.items_matrix[u, v]
                        u_x = self.coors[0][u]
                        u_y = self.coors[1][u]
                        v_x = self.coors[0][v]
                        v_y = self.coors[1][v]
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
            
            self.rotate_vertices(components, phi)
            if callbackUpdateCanvas: callbackUpdateCanvas()
            if callbackProgress : callbackProgress(min([dirChange[i] for i \
                                    in range(len(dirChange)) if M[i] != 0]), 9)
            step += 1
    
    def mds_update_data(self, components, mds, callbackUpdateCanvas):
        """Translate and rotate the network components to computed positions."""
        
        component_props = []
        x_mds = []
        y_mds = []
        phi = [None] * len(components)
        self.diag_coors = math.sqrt(( \
                    min(self.coors[0]) - max(self.coors[0]))**2 + \
                    (min(self.coors[1]) - max(self.coors[1]))**2)
        
        if self.mdsType == MdsType.MDS:
            x = [mds.points[u][0] for u in range(self.graph.number_of_nodes())]
            y = [mds.points[u][1] for u in range(self.graph.number_of_nodes())]
            self.coors[0][range(self.graph.number_of_nodes())] =  x
            self.coors[1][range(self.graph.number_of_nodes())] =  y
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
                            [self.coors[0][u],self.coors[1][u]])) \
                            for u in component]
                n = [numpy.vdot([self.coors[0][u], \
                                 self.coors[1][u]], \
                                 [self.coors[0][u], \
                                  self.coors[1][u]]) for u in component]
                phi[i] = sum(c) / sum(n)
                #print phi
            
            x = self.coors[0][component]
            y = self.coors[1][component]
            
            x_avg_graph = sum(x) / len(x)
            y_avg_graph = sum(y) / len(y)
            
            x_mds.append(x_avg_mds) 
            y_mds.append(y_avg_mds)

            component_props.append((x_avg_graph, y_avg_graph, \
                                    x_avg_mds, y_avg_mds, phi))
        
        w = max(self.coors[0]) - min(self.coors[0])
        h = max(self.coors[1]) - min(self.coors[1])
        d = math.sqrt(w**2 + h**2)
        #d = math.sqrt(w*h)
        e = [math.sqrt((self.coors[0][u] - self.coors[0][v])**2 + 
                  (self.coors[1][u] - self.coors[1][v])**2) for 
                  (u, v) in self.graph.edges()]
        
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
            for i in range(self.graph.number_of_nodes()):
                for j in range(i + 1, self.graph.number_of_nodes()):
                    x1 = self.coors[0][i]
                    y1 = self.coors[1][i]
                    x2 = self.coors[0][j]
                    y2 = self.coors[1][j]
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
                x = self.coors[0][component]
                y = self.coors[1][component]
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
        e = [math.sqrt((self.coors[0][u] - self.coors[0][v])**2 + 
                  (self.coors[1][u] - self.coors[1][v])**2) for 
                  (u, v) in self.graph.edges()]
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
                r = 1 / math.sqrt(self.graph.number_of_nodes())
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
            #r = self.mdsScaleRatio / math.sqrt(self.graph.number_of_nodes())
            
        for i in range(len(components)):
            component = components[i]
            x_avg_graph, y_avg_graph, x_avg_mds, \
            y_avg_mds, phi = component_props[i]
            
#            if phi[i]:  # rotate vertices
#                #print "rotate", i, phi[i]
#                r = numpy.array([[numpy.cos(phi[i]), -numpy.sin(phi[i])], [numpy.sin(phi[i]), numpy.cos(phi[i])]])  #rotation matrix
#                c = [x_avg_graph, y_avg_graph]  # center of mass in FR coordinate system
#                v = [numpy.dot(numpy.array([self.coors[0][u], self.coors[1][u]]) - c, r) + c for u in component]
#                self.coors[0][component] = [u[0] for u in v]
#                self.coors[1][component] = [u[1] for u in v]
                
            # translate vertices
            if not self.rotationOnly:
                self.coors[0][component] = \
                (self.coors[0][component] - x_avg_graph) / r + x_avg_mds
                self.coors[1][component] = \
                (self.coors[1][component] - y_avg_graph) / r + y_avg_mds
               
        if callbackUpdateCanvas:
            callbackUpdateCanvas()
    
    def mds_callback(self, a, b=None):
        """Refresh the UI when running  MDS on network components."""
        
        if not self.mdsStep % self.mdsRefresh:
            self.mds_update_data(self.mdsComponentList, 
                               self.mds, 
                               self.callbackUpdateCanvas)
            
            if self.mdsType == MdsType.exactSimulation:
                self.mds.points = [[self.coors[0][i], \
                                    self.coors[1][i]] \
                                    for i in range(len(self.coors))]
                self.mds.freshD = 0
            
            if self.callbackProgress != None:
                self.callbackProgress(self.mds.avg_stress, self.mdsStep)
                
        self.mdsStep += 1

        if self.stopMDS:
            return 0
        else:
            return 1
            
    def mds_components(self, mdsSteps, mdsRefresh, callbackProgress=None, \
                       callbackUpdateCanvas=None, torgerson=0, \
                       minStressDelta=0, avgLinkage=False, rotationOnly=False,\
                       mdsType=MdsType.componentMDS, scalingRatio=0, \
                       mdsFromCurrentPos=0):
        """Position the network components according to similarities among 
        them.
        
        """

        if self.items_matrix == None:
            self.information('Set distance matrix to input signal')
            return 1
        
        if self.graph == None:
            return 1
        
        if self.items_matrix.dim != self.graph.number_of_nodes():
            return 1
        
        self.mdsComponentList = Orange.network.nx.algorithms.components.connected_components(self.graph)
        self.mdsRefresh = mdsRefresh
        self.mdsStep = 0
        self.stopMDS = 0
        self.items_matrix.matrixType = Orange.core.SymMatrix.Symmetric
        self.diag_coors = math.sqrt((min(self.coors[0]) -  \
                                     max(self.coors[0]))**2 + \
                                     (min(self.coors[1]) - \
                                      max(self.coors[1]))**2)
        self.rotationOnly = rotationOnly
        self.mdsType = mdsType
        self.scalingRatio = scalingRatio

        w = max(self.coors[0]) - min(self.coors[0])
        h = max(self.coors[1]) - min(self.coors[1])
        d = math.sqrt(w**2 + h**2)
        #d = math.sqrt(w*h)
        e = [math.sqrt((self.coors[0][u] - self.coors[0][v])**2 + 
                  (self.coors[1][u] - self.coors[1][v])**2) for 
                  (u, v) in self.graph.edges()]
        self.mdsScaleRatio = d / sum(e) * float(len(e))
        #print d / sum(e) * float(len(e))
        
        if avgLinkage:
            matrix = self.items_matrix.avgLinkage(self.mdsComponentList)
        else:
            matrix = self.items_matrix
        
        #if self.mds == None: 
        self.mds = Orange.projection.mds.MDS(matrix)
        
        if mdsFromCurrentPos:
            if avgLinkage:
                for u, c in enumerate(self.mdsComponentList):
                    x = sum(self.coors[0][c]) / len(c)
                    y = sum(self.coors[1][c]) / len(c)
                    self.mds.points[u][0] = x
                    self.mds.points[u][1] = y
            else:
                for u in range(self.graph.number_of_nodes()):
                    self.mds.points[u][0] = self.coors[0][u] 
                    self.mds.points[u][1] = self.coors[1][u]
            
        # set min stress difference between 0.01 and 0.00001
        self.minStressDelta = minStressDelta
        self.callbackUpdateCanvas = callbackUpdateCanvas
        self.callbackProgress = callbackProgress
        
        if torgerson:
            self.mds.Torgerson() 

        self.mds.optimize(mdsSteps, Orange.projection.mds.SgnRelStress, self.minStressDelta,\
                          progress_callback=self.mds_callback)
        self.mds_update_data(self.mdsComponentList, self.mds, callbackUpdateCanvas)
        
        if callbackProgress != None:
            callbackProgress(self.mds.avg_stress, self.mdsStep)
        
        del self.rotationOnly
        del self.diag_coors
        del self.mdsRefresh
        del self.mdsStep
        #del self.mds
        del self.mdsComponentList
        del self.minStressDelta
        del self.callbackUpdateCanvas
        del self.callbackProgress
        del self.mdsType
        del self.mdsScaleRatio
        del self.scalingRatio
        return 0

    def mds_components_avg_linkage(self, mdsSteps, mdsRefresh, \
                                   callbackProgress=None, \
                                   callbackUpdateCanvas=None, torgerson=0, \
                                   minStressDelta = 0, scalingRatio=0,\
                                   mdsFromCurrentPos=0):
        return self.mds_components(mdsSteps, mdsRefresh, callbackProgress, \
                                   callbackUpdateCanvas, torgerson, \
                                   minStressDelta, True, \
                                   scalingRatio=scalingRatio, \
                                   mdsFromCurrentPos=mdsFromCurrentPos)
    
    ##########################################################################
    ### BEGIN: DEPRECATED METHODS (TO DELETE IN ORANGE 3.0)                ###
    ##########################################################################
    
    def map_to_graph(self, graph):
        nodes = sorted(graph.nodes())
        return dict((v, (self.coors[0][i], self.coors[1][i])) for i,v in \
                    enumerate(nodes))
    
class NxView(object):
    """Network View
    
    """
    
    def __init__(self, **attr):
        self._network = None
        self._nx_explorer = None
        
    def set_nx_explorer(self, _nx_explorer):
        self._nx_explorer = _nx_explorer
        
    def init_network(self, graph):
        return graph
        
    def nodes_selected(self):
        pass

    #def node_selection_changed(self):
    #    pass