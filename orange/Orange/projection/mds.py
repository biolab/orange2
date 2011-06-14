"""
.. index:: multidimensional scaling (mds)

.. index::
   single: projection; multidimensional scaling (mds)

**********************************
Multidimensional scaling (``mds``)
**********************************

The functionality to perform multidimensional scaling
(http://en.wikipedia.org/wiki/Multidimensional_scaling).

The main class to perform multidimensional scaling is
:class:`Orange.projection.mds.MDS`

.. autoclass:: Orange.projection.mds.MDS
   :members:
   :exclude-members: Torgerson, get_distance, get_stress

Stress functions
================

Stress functions that can be used for MDS have to be implemented as functions
or callable classes:

    .. method:: \ __call__(correct, current, weight=1.0)
       
       Compute the stress using the correct and the current distance value (the
       :obj:`Orange.projection.mds.MDS.distances` and
       :obj:`Orange.projection.mds.MDS.projected_distances` elements).
       
       :param correct: correct (actual) distance between elements, represented by
           the two points.
       :type correct: float
       
       :param current: current distance between the points in the MDS space.
       :type current: float

This module provides the following stress functions:

   * :obj:`SgnRelStress`
   * :obj:`KruskalStress`
   * :obj:`SammonStress`
   * :obj:`SgnSammonStress`

Examples
========

MDS Scatterplot
---------------

The following script computes the Euclidean distance between the data
instances and runs MDS. Final coordinates are plotted with matplotlib
(not included with orange, http://matplotlib.sourceforge.net/).

Example (`mds-scatterplot.py`_, uses `iris.tab`_)

.. literalinclude:: code/mds-scatterplot.py
    :lines: 7-

.. _mds-scatterplot.py: code/mds-scatterplot.py
.. _iris.tab: code/iris.tab

The script produces a file *mds-scatterplot.py.png*. Color denotes
the class. Iris is a relatively simple data set with respect to
classification; to no surprise we see that MDS finds such instance
placement in 2D where instances of different classes are well separated.
Note that MDS has no knowledge of points' classes.

.. image:: files/mds-scatterplot.png


A more advanced example
-----------------------

The following script performs 10 steps of Smacof optimization before computing
the stress. This is suitable if you have a large dataset and want to save some
time.

Example (`mds-advanced.py`_, uses `iris.tab`_)

.. literalinclude:: code/mds-advanced.py
    :lines: 7-

.. _mds-advanced.py: code/mds-advanced.py

A few representative lines of the output are::

    <-0.633911848068, 0.112218663096> [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
    <-0.624193906784, -0.111143872142> [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
    ...
    <0.265250980854, 0.237793982029> [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor']
    <0.208580598235, 0.116296850145> [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']
    ...
    <0.635814905167, 0.238721415401> [6.3, 3.3, 6.0, 2.5, 'Iris-virginica']
    <0.356859534979, -0.175976261497> [5.8, 2.7, 5.1, 1.9, 'Iris-virginica']
    ...


"""


from math import *
from numpy import *
from numpy.linalg import svd

import Orange.core
import orangeom as orangemds
from Orange.misc import deprecated_keywords
from Orange.misc import deprecated_members

KruskalStress = orangemds.KruskalStress()
SammonStress = orangemds.SammonStress()
SgnSammonStress = orangemds.SgnSammonStress()
SgnRelStress = orangemds.SgnRelStress()

PointList = Orange.core.FloatListList
FloatListList = Orange.core.FloatListList

def _mycompare((a,aa),(b,bb)):
    if a == b:
        return 0
    if a < b:
        return -1
    else:
        return 1
            
class PivotMDS(object):
    def __init__(self, distances=None, pivots=50, dim=2, **kwargs):
        self.dst = array([m for m in distances])
        self.n = len(self.dst)

        if type(pivots) == type(1):
            self.k = pivots
            self.pivots = random.permutation(len(self.dst))[:pivots]
            #self.pivots.sort()
        elif type(pivots) == type([]):
            self.pivots = pivots
            #self.pivots.sort()
            self.k = len(self.pivots)
        else:
            raise AttributeError('pivots')
        
    def optimize(self):
#        # Classical MDS (Torgerson)
#        J = identity(self.n) - (1/float(self.n))
#        B = -1/2. * dot(dot(J, self.dst**2), J)
#        w,v = linalg.eig(B)
#        tmp = zip([float(val) for val in w], range(self.n))
#        tmp.sort()
#        w1, w2 = tmp[-1][0], tmp[-2][0]
#        v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
#        return v1 * sqrt(w1), v2 * sqrt(w2) 
        
        # Pivot MDS
        d = self.dst[[self.pivots]].T
        C = d**2
        # double-center d
        cavg = sum(d, axis=0)/(self.k+0.0)      # column sum
        ravg = sum(d, axis=1)/(self.n+0.0)    # row sum
        tavg = sum(cavg)/(self.n+0.0)   # total sum
        # TODO: optimize
        for i in xrange(self.n):
            for j in xrange(self.k):
                C[i,j] += -ravg[i] - cavg[j]
        
        C = -0.5 * (C + tavg)
        w,v = linalg.eig(dot(C.T, C))
        tmp = zip([float(val) for val in w], range(self.n))
        tmp.sort()
        w1, w2 = tmp[-1][0], tmp[-2][0]
        v1, v2 = v[:, tmp[-1][1]], v[:, tmp[-2][1]]
        x = dot(C, v1)
        y = dot(C, v2)
        return x, y
        
        
class MDS(object):
    """
    Main class for performing multidimensional scaling.
    
    :param distances: original dissimilarity - a distance matrix to operate on.
    :type distances: :class:`Orange.core.SymMatrix`
    
    :param dim: dimension of the projected space.
    :type dim: int
    
    :param points: an initial configuration of points (optional)
    :type points: :class:`Orange.core.FloatListList`
    
    An instance of MDS object has the following attributes and functions:
    
    .. attribute:: points
       
       Holds the current configuration of projected points in an
       :class:`Orange.core.FloatListList` object.
       
    .. attribute:: distances
    
       An :class:`Orange.core.SymMatrix` containing the distances that we
       want to achieve (lsmt changes these).
       
    .. attribute:: projected_distances

       An :class:`Orange.core.SymMatrix` containing the distances between
       projected points.
       
    .. attribute:: original_distances

       An :class:`Orange.core.SymMatrix` containing the original distances
       between points.
       
    .. attribute:: stress
       
       An :class:`Orange.core.SymMatrix` holding the stress.
    
    .. attribute:: dim

       An integer holding the dimension of the projected space.
       
    .. attribute:: n

       An integer holding the number of elements (points).
       
    .. attribute:: avg_stress

       A float holding the average stress in the :obj:`stress` matrix.
       
    .. attribute:: progress_callback

       A function that gets called after each optimization step in the
       :func:`run` method.
    
    """
    
    def __init__(self, distances=None, dim=2, **kwargs):
        self.mds=orangemds.MDS(distances, dim, **kwargs)
        self.original_distances=Orange.core.SymMatrix([m for m in self.distances])

    def __getattr__(self, name):
        if name in ["points", "projected_distances", "distances" ,"stress",
                    "progress_callback", "n", "dim", "avg_stress"]:
            #print "rec:",name            
            return self.__dict__["mds"].__dict__[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        #print "setattr"
        if name=="points":
            for i in range(len(value)):
                for j in range(len(value[i])):
                    self.mds.points[i][j]=value[i][j]
            return
            
        if name in ["projected_distances", "distances" ,"stress",
                    "progress_callback"]:
            self.mds.__setattr__(name, value)
        else:
            self.__dict__[name]=value
            
    def __nonzero__(self):
        return True
            
    def smacof_step(self):
        """
        Perform a single iteration of a Smacof algorithm that optimizes
        :obj:`stress` and updates the :obj:`points`.
        """
        self.mds.SMACOFstep()

    def calc_distance(self):
        """
        Compute the distances between points and update the
        :obj:`projected_distances` matrix.
        
        """
        self.mds.get_distance()
        

    @deprecated_keywords({"stressFunc": "stress_func"})
    def calc_stress(self, stress_func=SgnRelStress):
        """
        Compute the stress between the current :obj:`projected_distances` and
        :obj:`distances` matrix using *stress_func* and update the
        :obj:`stress` matrix and :obj:`avgStress` accordingly.
        
        """
        self.mds.getStress(stress_func)

    @deprecated_keywords({"stressFunc": "stress_func"})
    def optimize(self, iter, stress_func=SgnRelStress, eps=1e-3,
                 progress_callback=None):
        self.mds.progress_callback=progress_callback
        self.mds.optimize(iter, stress_func, eps)

    @deprecated_keywords({"stressFunc": "stress_func"})
    def run(self, iter, stress_func=SgnRelStress, eps=1e-3,
            progress_callback=None):
        """
        Perform optimization until stopping conditions are met.
        Stopping conditions are:
           
           * optimization runs for *iter* iterations of smacof_step function, or
           * stress improvement (old stress minus new stress) is smaller than
             eps * old stress.
        
        :param iter: maximum number of optimization iterations.
        :type iter: int
        
        :param stress_func: stress function.
        """
        self.optimize(iter, stress_func, eps, progress_callback)

    def torgerson(self):
        """
        Run the Torgerson algorithm that computes an initial analytical
        solution of the problem.
        
        """
        # Torgerson's initial approximation
        O = array([m for m in self.distances])
        
##        #B = matrixmultiply(O,O)
##        # bug!? B = O**2
##        B = dot(O,O)
##        # double-center B
##        cavg = sum(B, axis=0)/(self.n+0.0)      # column sum
##        ravg = sum(B, axis=1)/(self.n+0.0)    # row sum
##        tavg = sum(cavg)/(self.n+0.0)   # total sum
##        # B[row][column]
##        for i in xrange(self.n):
##            for j in xrange(self.n):
##                B[i,j] += -cavg[j]-ravg[i]
##        B = -0.5*(B+tavg)

        # B = double-center O**2 !!!
        J = identity(self.n) - (1/float(self.n))
        B = -0.5 * dot(dot(J, O**2), J)
        
        # SVD-solve B = ULU'
        #(U,L,V) = singular_value_decomposition(B)
        (U,L,V)=svd(B)
        # X = U(L^0.5)
        # # self.X = matrixmultiply(U,identity(self.n)*sqrt(L))
        # X is n-dimensional, we take the two dimensions with the largest singular values
        idx = argsort(L)[-self.dim:].tolist()
        idx.reverse()
        
        Lt = take(L,idx)   # take those singular values
        Ut = take(U,idx,axis=1) # take those columns that are enabled
        Dt = identity(self.dim)*sqrt(Lt)  # make a diagonal matrix, with squarooted values
        self.points = Orange.core.FloatListList(dot(Ut,Dt))
        self.freshD = 0
        
#        D = identity(self.n)*sqrt(L)  # make a diagonal matrix, with squarooted values
#        X = matrixmultiply(U,D)
#        self.X = take(X,idx,1)
    
    # Kruskal's monotone transformation
    def lsmt(self):
        """
        Execute Kruskal monotone transformation.
        
        """
        # optimize the distance transformation
        # build vector o
        effect = 0
        self.getDistance()
        o = []
        for i in xrange(1,self.n):
            for j in xrange(i):
                o.append((self.original_distances[i,j],(i,j)))
        o.sort(_mycompare)
        # find the ties in o, and construct the d vector sorting in order within ties
        d = []
        td = []
        uv = [] # numbers of consecutively tied o values
        (i,j) = o[0][1]
        distnorm = self.projected_distances[i,j]*self.projected_distances[i,j]
        td = [self.projected_distances[i,j]] # fetch distance
        for l in xrange(1,len(o)):
            # copy now sorted distances in an array
            # but sort distances within a tied o
            (i,j) = o[l][1]
            cd = self.projected_distances[i,j]
            distnorm += self.projected_distances[i,j]*self.projected_distances[i,j]
            if o[l][0] != o[l-1][0]:
                # differing value, flush
                sum = reduce(lambda x,y:x+y,td)+0.0
                d.append([sum,len(td),sum/len(td),td])
                td = []
            td.append(cd)
        sum = reduce(lambda x,y:x+y,td)+0.0
        d.append([sum,len(td),sum/len(td),td])
        ####
        # keep merging non-monotonous areas in d
        monotony = 0
        while not monotony and len(d) > 1:
            monotony = 1
            pi = 0 # index
            n = 1  # n-areas
            nd = []
            r = d[0] # current area
            for i in range(1,len(d)):
                tr = d[i]
                if r[2]>=tr[2]:
                    monotony = 0
                    effect = 1
                    r[0] += tr[0]
                    r[1] += tr[1]
                    r[2] = tr[0]/tr[1]
                    r[3] += tr[3]
                else:
                    nd.append(r)
                    r = tr
            nd.append(r)
            d = nd
        # normalizing multiplier
        sum = 0.0
        for i in d:
            sum += i[2]*i[2]*i[1]
        f = sqrt(distnorm/max(sum,1e-6))
        # transform O
        k = 0
        for i in d:
            for j in range(i[1]):
                (ii,jj) = o[k][1]
                self.distances[ii,jj] = f*i[2]
                k += 1
        assert(len(o) == k)
        self.freshD = 0
        return effect
    
MDS = deprecated_members({"projectedDistances": "projected_distances",
                     "originalDistances": "original_distances",
                     "avgStress": "avg_stress",
                     "progressCallback": "progress_callback",
                     "getStress": "calc_stress",
                     "get_stress": "calc_stress",
                     "calcStress": "calc_stress",
                     "getDistance": "calc_distance",
                     "get_distance": "calc_distance",
                     "calcDistance": "calc_distance",
                     "Torgerson": "torgerson",
                     "SMACOFstep": "smacof_step",
                     "LSMT": "lsmt"})(MDS)
