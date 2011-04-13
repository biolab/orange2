import networkx as nx
import readwrite
import Orange

from networkx import algorithms 
from networkx.classes import function

class BaseGraph():
    
    def __init__(self):
        self._items = None
        self._links = None
        
    def items(self):
        if len(self._items) != self.number_of_nodes():
            print "Warning: items length does not match the number of nodes."
            
        return self._items
    
    def set_items(self, items=None):
        if items:
            if not isinstance(items, Orange.data.Table):
                raise TypeError('items must be of type \'Orange.data.Table\'')
            if len(items) != self.number_of_nodes():
                print "Warning: items length must match the number of nodes."
                
        self._items = items
        
    def links(self):
        if len(self._links) != self.number_of_edges():
            print "Warning: links length does not match the number of edges."
            
        return self._links
    
    def set_links(self, links):
        if links:
            if not isinstance(links, Orange.data.Table):
                raise TypeError('links must be of type \'Orange.data.Table\'')
            if len(items) != self.number_of_edges():
                print "Warning: links length must match the number of edges."
        
        self._links = links
        
    ### TODO: OVERRIDE METHODS THAT CHANGE GRAPH STRUCTURE, add warning prints
    
class Graph(BaseGraph, nx.Graph):
    
    def __init__(self, data=None, name='', **attr):  
        nx.Graph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.Graph.__init__.__doc__
     
class DiGraph(BaseGraph, nx.DiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.DiGraph.__init__.__doc__
     
class MultiGraph(BaseGraph, nx.MultiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.MultiGraph.__init__.__doc__
     
class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiDiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
        
    __init__.__doc__ = nx.MultiDiGraph.__init__.__doc__
