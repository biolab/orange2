"""
.. index:: clustering

Everything about clustering, including agglomerative and hierarchical clustering.

================
Other clustering
================

.. class:: Orange.clustering.ExampleCluster

   To je pa moj dodaten opis ExampleClusterja

   .. method:: fibonaci(n)

      Funkcija, ki izracuna fibonacija.

      :param n: katero fibonacijevo stevilo zelis
      :type n: integer
"""

from __future__ import with_statement

from orange import \
    ExampleCluster, HierarchicalCluster, HierarchicalClusterList, HierarchicalClustering

import math
import sys
import orange
import random
import statc
    
__docformat__ = 'restructuredtext'

#class ExampleCluster2(orange.ExampleCluster):
#    """New example cluster"""
#    pass


