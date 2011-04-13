import networkx as nx
import readwrite

from networkx import algorithms 
from networkx.classes import function

class BaseGraph():
    
    def __init__(self):
        self.items = None
        self.links = None
        
class Graph(BaseGraph, nx.Graph):
    
    def __init__(self, data=None, name='', **attr):
        nx.Graph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
         
class DiGraph(BaseGraph, nx.DiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.DiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
     
class MultiGraph(BaseGraph, nx.MultiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
     
class MultiDiGraph(BaseGraph, nx.MultiDiGraph):
    
    def __init__(self, data=None, name='', **attr):
        nx.MultiDiGraph.__init__(self, data=None, name='', **attr)
        BaseGraph.__init__(self)
