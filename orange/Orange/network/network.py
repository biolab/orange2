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


class BaseGraph():
    
    def __init__(self):
        self._items = None
        self._links = None
        
    def items(self):
        if self._items is not None and \
                        len(self._items) != self.number_of_nodes():
            print "Warning: items length does not match the number of nodes."
            
        return self._items
    
    def set_items(self, items=None):
        if items is not None:
            if not isinstance(items, Orange.data.Table):
                raise TypeError('items must be of type \'Orange.data.Table\'')
            if len(items) != self.number_of_nodes():
                print "Warning: items length must match the number of nodes."
                
        self._items = items
        
    def links(self):
        if self._links is not None \
                    and len(self._links) != self.number_of_edges():
            print "Warning: links length does not match the number of edges."
            
        return self._links
    
    def set_links(self, links=None):
        if links is not None:
            if not isinstance(links, Orange.data.Table):
                raise TypeError('links must be of type \'Orange.data.Table\'')
            if len(links) != self.number_of_edges():
                print "Warning: links length must match the number of edges."
        
        self._links = links
        
    def to_orange_network(self):
        """Convert the network to Orange NetworkX standard. All node IDs are transformed to range [0, no_of_nodes - 1].""" 
        if isinstance(self, Orange.network.Graph):
            G = Orange.network.Graph()
        elif isinstance(self, Orange.network.DiGraph):
            G = Orange.network.DiGraph()
        elif isinstance(self, Orange.network.MultiGraph):
            G = Orange.network.MultiGraph()
        elif isinstance(self, Orange.network.MultiDiGraph):
            G = Orange.network.DiGraph()
        else:
            raise TypeError('WTF!?')
        
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
        """Return a list of features in network items."""
        vars = []
        if (self._items is not None):
            if isinstance(self._items, Orange.data.Table):
                vars = list(self._items.domain.variables)
            
                metas = self._items.domain.getmetas(0)
                vars.extend(var for i, var in metas.iteritems())
        return vars
    
    def links_vars(self):
        """Return a list of features in network links."""
        vars = []
        if (self._links is not None):
            if isinstance(self._links, Orange.data.Table):
                vars = list(self._links.domain.variables)
            
                metas = self._links.domain.getmetas(0)
                vars.extend(var for i, var in metas.iteritems())
        return [x for x in vars if str(x.name) != 'u' and str(x.name) != 'v']    
    
class Graph(BaseGraph, nx.Graph):
    
    def __init__(self, data=None, name='', **attr):  
        nx.Graph.__init__(self, data, name, **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.Graph.__init__.__doc__
     
class DiGraph(BaseGraph, nx.DiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data, name, **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.DiGraph.__init__.__doc__
     
class MultiGraph(BaseGraph, nx.MultiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data, name, **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.MultiGraph.__init__.__doc__
     
class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiDiGraph.__init__(self, data, name, **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.MultiDiGraph.__init__.__doc__

class GraphLayout(orangeom.GraphLayout):
    
    """A graph layout optimization class."""
    
    def __init__(self):
        self.graph = None
        self.items_matrix = None
        
    def set_graph(self, graph=None, positions=None):
        """Initialize graph structure
        
        :param graph: NetworkX graph
        
        """
        self.graph = graph
        
        if positions is not None and len(positions) == graph.number_of_nodes():
            orangeom.GraphLayout.set_graph(self, graph, positions)
        else:
            orangeom.GraphLayout.set_graph(self, graph)
            
    def random(self):
        orangeom.GraphLayout.random(self)
        
    def fr(self, steps, temperature, coolFactor=0, weighted=False):
        return orangeom.GraphLayout.fr(self, steps, temperature, coolFactor, weighted)
        
    def fr_radial(self, center, steps, temperature):
        return orangeom.GraphLayout.fr_radial(self, center, steps, temperature)
    
    def circular_original(self):
        orangeom.GraphLayout.circular_original(self)
    
    def circular_random(self):
        orangeom.GraphLayout.circular_random(self)
    
    def circular_crossing_reduction(self):
        orangeom.GraphLayout.circular_crossing_reduction(self)
    
    def get_vertices_in_rect(self, x1, y1, x2, y2):
        return orangeom.GraphLayout.get_vertices_in_rect(self, x1, y1, x2, y2)
    
    def closest_vertex(self, x, y):
        return orangeom.GraphLayout.closest_vertex(self, x, y)
    
    def vertex_distances(self, x, y):
        return orangeom.GraphLayout.vertex_distances(self, x, y)
    
    def rotate_vertices(self, components, phi): 
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
    
    
    
    