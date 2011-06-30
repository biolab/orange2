#
# Module Orange Dimension Reduction
# ---------------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
# (Copyright (C)2004 Aleks Jakulin)
#
# Purpose: Dimension reduction
#
# Bibliography: Tom Minka, "36-350: Data Mining, Fall 2003", Lecture Notes, Carnegie Mellon University.
#
# ChangeLog:
#   - 2003/10/28: project initiated
#   - 2003/11/20: returning the parameters of the transform

import numpy, mathutil
import numpy.linalg as LinearAlgebra

# before running PCA, it is helpful to apply the transformation
# operators on individual vectors.
class PCA:
    def __init__(self, data, components=1):
        (u,d,v) = LinearAlgebra.svd(data)
        self.loading = u                # transformed data points
        self.variance = d               # principal components' variance
        self.factors = v                # the principal basis
        d2 = numpy.power(d,2)
        s = numpy.sum(d2)
        if s > 1e-6:
            s = d2/s
        else:
            s = 1.0
        self.R_squared = s # percentage of total variance explained by individual components

def Centering(vector, m = None, inverse=0):
    assert(len(numpy.shape(vector))==1) # this must be a vector
    if m == None:
        m = numpy.average(vector)
    if inverse==0:
        return (vector-m,m)
    else:
        return vector+m

def MaxScaling(vector, param = None):
    if param == None:
        (v,m) = Centering(vector)
        s = max(abs(v))
        if s > 1e-6:
            s = 1.0/s
    else:
        (m,s) = param
        (v,m_) = Centering(vector,m)
    return (v*s,(m,s))

def VarianceScaling(vector,param=None,inverse=0):
    if param == None:
        (v,m) = Centering(vector)
        s = numpy.sqrt(numpy.average(numpy.power(v,2)))
        if s > 1e-6:
            s = 1.0/s
    else:
        (m,s) = param
        if inverse == 0:
            (v,m_) = Centering(vector,m)
        else:
            v = Centering(vector,m,1)
    if inverse == 0:
        return (s*v,(m,s))
    else:
        return s/v

def _BC(vector,lambd):
    if lambd != 0.0:
        return (numpy.power(vector,lambd)-1)/lambd
    else:
        return numpy.log(vector)

class _BCskewness:
    def __init__(self,vector):
        self.v = vector
    def __call__(self,lambd):
        nv = _BC(self.v,lambd)
        mean = numpy.average(nv)
        cv = nv-mean
        skewness = numpy.average(numpy.power(cv,3))/numpy.power(numpy.average(numpy.power(cv,2)),1.5)
        # kurtosis = numpy.average(numpy.power(cv,4))/numpy.power(numpy.average(numpy.power(cv,2)),2)-3
        return skewness**2

def BoxCoxTransform(vector,lambd=None):
    v = -min(vector)+1+vector
    print "shifting by ",-min(vector)+1
    if lambd==None:
        # find the value of lambda that will minimize skew
        lambd = mathutil.minimum(_BCskewness(v))
        print "best-fitting lambda = ",lambd
    return _BC(v,lambd)

def RankConversion(vector,reverse=0):
    assert(len(numpy.shape(vector))==1) # this must be a vector

    newv = numpy.zeros(numpy.size(vector),numpy.float)
    l = []
    for x in xrange(len(vector)):
        l.append((vector[x],x))
    l.sort()
    if reverse:
        l.reverse()
    pi = -1
    pv = 'a'
    idx = []
    pr = 0
    cr = 0
    for (v,i) in l:
        if v != pv:
            r = pr+(cr-pr+1)/2.0
            for j in idx:
                newv[j] = r
            idx = []
            pr = cr
            pv = v
        cr += 1
        idx.append(i)
    r = pr+(cr-pr+1)/2.0
    for j in idx:
        newv[j] = r
    return newv

if __name__== "__main__":
    v = numpy.array([6, 6, 6, 6, 4, 6, 12, 12, 12, 4, 4, 4, 6, 6, 8, 6, 8, 8, 8, 4, 4, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8, 8, 6, 6, 8, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 6, 6, 8, 6, 6, 4, 4, 8, 8, 8, 6, 6, 6, 6, 6, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 8, 4, 6, 6, 6, 6, 6, 6, 4, 6, 4, 4, 6, 6, 6, 6, 8, 6, 6, 4, 6, 6, 6, 8, 8, 8, 5, 5, 6, 6, 10, 8, 12, 12, 12, 8, 6, 6, 8, 8, 6, 4, 8, 8, 6, 6, 6, 8, 8, 8, 8, 4, 4, 4, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 3, 6, 6, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 6, 6, 8, 6, 6, 6, 8, 8, 6, 6, 6, 4, 4, 8, 6, 8, 8, 8, 6, 6, 6, 4, 4, 4, 6, 6, 4, 4, 12, 8, 6, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 6, 8, 6, 6, 6, 6, 8, 6, 8, 8, 4, 8, 8, 6, 6, 6, 4, 6, 4, 4, 4, 4, 4, 6, 6, 4, 6, 4, 6, 6, 6, 6, 4, 6, 4, 4, 8, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 4, 4, 6, 6, 6, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 6, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 6, 4, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 6, 4, 5, 5, 5], numpy.float)
    print "original:"
    print v
    print "rank-transformed:"
    print RankConversion(v)
    print "centered"
    print Centering(v)
    print "minmax scaled"
    print MaxScaling(v)
    print "variance scaling"
    print VarianceScaling(v)
    print "Box-Cox"
    print BoxCoxTransform(v)
