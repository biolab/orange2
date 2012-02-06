"""
.. index:: clustering

Everything about clustering, including agglomerative and hierarchical clustering.
"""

from __future__ import with_statement

from orange import \
    ExampleCluster, HierarchicalCluster, HierarchicalClusterList, HierarchicalClustering

import math
import sys
import orange
import random
from Orange import statc
    
__docformat__ = 'restructuredtext'

