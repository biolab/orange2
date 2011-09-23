"""

*************************
Network classes in Orange
*************************

Orange network classes are derived from `NetworkX basic graph types <http://networkx.lanl.gov/reference/classes.html>`_ 
and :obj:`Orange.network.BaseGraph`. They provide data structures and methods 
for storing graphs, network analysis and layout optimization.

There are four graph types: :obj:`Orange.network.Graph`, 
:obj:`Orange.network.DiGraph`, :obj:`Orange.network.MultiGraph` and
:obj:`Orange.network.MultiDiGraph`. The choice of graph class depends on the 
structure of the graph you want to represent.

Examples
========

Reading and writing a network
-----------------------------

This example demonstrates reading a network. Network class can read or write 
Pajek (.net) or GML file format.

`network-read-nx.py`_ (uses: `K5.net`_):

.. literalinclude:: code/network-read.py
    :lines: 5-6
    
.. _network-read-nx.py: code/network-read-nx.py
.. _K5.net: code/K5.net

Visualize a network in NetExplorer widget
-----------------------------------------

This example demonstrates how to display a network in NetExplorer.

part of `network-widget.py`_

.. literalinclude:: code/network-widget.py
    :lines: 10-16
    
.. image:: files/network-explorer.png
    :width: 100%

.. _network-widget.py: code/network-widget.py

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
import snap

from deprecated import *