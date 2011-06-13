#####################
Network (``network``)
#####################


Orange network classes are derived from `NetworkX basic graph types <http://networkx.lanl.gov/reference/classes.html>`_ 
and :obj:`Orange.network.BaseGraph`. They provide data structures and methods 
for storing graphs, network analysis and layout optimization.

There are four graph types: :obj:`Orange.network.Graph`, 
:obj:`Orange.network.DiGraph`, :obj:`Orange.network.MultiGraph` and
:obj:`Orange.network.MultiDiGraph`. The choice of graph class depends on the 
structure of the graph you want to represent.

.. toctree::
   :maxdepth: 2

   Orange.network.graphtypes
   Orange.network.readwrite
   Orange.network.layout
   Orange.network.community
   Orange.network.deprecated
