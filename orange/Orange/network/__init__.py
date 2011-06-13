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

import math
import random
import os

import Orange.core
import Orange.data
import Orange.projection

from Orange.core import GraphAsList, GraphAsMatrix, GraphAsTree

try:
    from network import *
except ImportError:
    import warnings
    warnings.warn("Warning: install networkx to use the 'Orange.network' module.") 

import community

from deprecated import *