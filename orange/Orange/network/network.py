import networkx as nx
import readwrite
import Orange
import orangeom

from networkx import algorithms 
from networkx.classes import function

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
        pass
    
    def set_graph(self, graph=None, positions=None):
        """Initialize graph structure
        
        :param graph: NetworkX graph
        
        """
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
    