#
# Module Orange Dimension Reduction
# ---------------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
#
# Purpose: Dimension reduction
#
# Bibliography: Tom Minka, "36-350: Data Mining, Fall 2003", Lecture Notes, Carnegie Mellon University.
#
# ChangeLog:
#   - 2003/10/28: project initiated
#   - 2003/11/17: 

import Numeric,LinearAlgebra,mathutil

# before running PCA, it is helpful to apply the transformation
# operators on individual vectors.
class PCA:
    def __init__(self, data, components=1):
        (u,d,v) = LinearAlgebra.singular_value_decomposition(data)
        self.loading = u                # transformed data points
        self.variance = d               # principal components' variance
        self.factors = v                # the principal basis
        d2 = Numeric.power(d,2)
        s = Numeric.sum(d2)
        if s > 1e-6:
            s = d2/s
        else:
            s = 1.0
        self.R_squared = s # percentage of total variance explained by individual components

def Centering(vector):
    assert(len(Numeric.shape(vector))==1) # this must be a vector

    avg = Numeric.average(vector)
    print "shifting by ",-avg
    return vector-avg

def MaxScaling(vector):
    v = Centering(vector)
    s = max(abs(v))
    print "scaling by ",1.0/s
    return v/s

def VarianceScaling(vector):
    v = Centering(vector)
    s = Numeric.sqrt(Numeric.average(Numeric.power(v,2)))
    print "scaling by ",1.0/s
    return v/s

def _BC(vector,lambd):
    if lambd != 0.0:
        return (Numeric.power(vector,lambd)-1)/lambd
    else:
        return Numeric.log(vector)

class _BCskewness:
    def __init__(self,vector):
        self.v = vector
    def __call__(self,lambd):
        nv = _BC(self.v,lambd)
        mean = Numeric.average(nv)
        cv = nv-mean
        skewness = Numeric.average(Numeric.power(cv,3))/Numeric.power(Numeric.average(Numeric.power(cv,2)),1.5)
        # kurtosis = Numeric.average(Numeric.power(cv,4))/Numeric.power(Numeric.average(Numeric.power(cv,2)),2)-3
##        print lambd,skewness
        return skewness**2

def BoxCoxTransform(vector,lambd=None):
    v = -min(vector)+1+vector
    print "shifting by ",-min(vector)+1
    if lambd==None:
        # find the value of lambda that will minimize skew
        lambd = mathutil.minimum(_BCskewness(v))
        print "best-fitting lambda = ",lambd
    return _BC(v,lambd)

def RankConversion(vector):
    assert(len(Numeric.shape(vector))==1) # this must be a vector

    newv = Numeric.zeros(Numeric.size(vector),Numeric.Float)
    l = []
    for x in xrange(len(vector)):
        l.append((vector[x],x))
    l.sort()
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
    v = Numeric.array([6, 6, 6, 6, 4, 6, 12, 12, 12, 4, 4, 4, 6, 6, 8, 6, 8, 8, 8, 4, 4, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8, 8, 6, 6, 8, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 6, 6, 8, 6, 6, 4, 4, 8, 8, 8, 6, 6, 6, 6, 6, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 8, 4, 6, 6, 6, 6, 6, 6, 4, 6, 4, 4, 6, 6, 6, 6, 8, 6, 6, 4, 6, 6, 6, 8, 8, 8, 5, 5, 6, 6, 10, 8, 12, 12, 12, 8, 6, 6, 8, 8, 6, 4, 8, 8, 6, 6, 6, 8, 8, 8, 8, 4, 4, 4, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 3, 6, 6, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 6, 6, 8, 6, 6, 6, 8, 8, 6, 6, 6, 4, 4, 8, 6, 8, 8, 8, 6, 6, 6, 4, 4, 4, 6, 6, 4, 4, 12, 8, 6, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 8, 8, 6, 8, 8, 8, 8, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 6, 8, 6, 6, 6, 6, 8, 6, 8, 8, 4, 8, 8, 6, 6, 6, 4, 6, 4, 4, 4, 4, 4, 6, 6, 4, 6, 4, 6, 6, 6, 6, 4, 6, 4, 4, 8, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 4, 4, 6, 6, 6, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 6, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 6, 4, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 6, 4, 5, 5, 5],Numeric.Float)
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