import orange
import numpy

import scipy.special
import scipy.optimize
import scipy.stats

from pylab import *

def sqrtm(mat):
    """ Retruns the square root of the matrix mat """
    U, S, V = numpy.linalg.svd(mat)
    D = numpy.diag(numpy.sqrt(S))
    return numpy.dot(numpy.dot(U,D),V)

def standardize(mat):
    """ Subtracts means and multiplies by diagonal elements of inverse
        square root of covariance matrix.
    """
    av = numpy.average(mat, axis=0)
    sigma = numpy.corrcoef(mat, rowvar=0)
    srSigma = sqrtm(sigma)
    isrSigma = numpy.linalg.inv(srSigma)
    return (mat-av) * numpy.diag(isrSigma)


def friedman_tmp_func(alpha, Z=numpy.zeros((1,1)), J=5, n=1):
    alpha = numpy.array(alpha)
    pols = [scipy.special.legendre(j) for j in range(0,J+1)]
    vals0 = [numpy.dot(alpha.T, Z[i,:]) for i in range(n)]
    def f_tmp(x): return 2*x-1
    vals = map(f_tmp, map(scipy.stats.zprob, vals0))
    val = [1./n*sum(map(p, vals))**2 for p in pols]
    return vals, pols, - 0.5 * sum([(2*j+1)*v for j, v in enumerate(val)])


class ProjectionPursuit:
    FRIEDMAN = 0
    MOMENT = 1
    SILHUETTE = 2
    HARTINGAN = 3
    
    def __init__(self, data, index = FRIEDMAN, dim=2, maxiter=10):
        self.dim = dim
        if type(data) == orange.ExampleTable:
            self.dataNP = data.toNumpy()[0]         # TODO: check if conversion of discrete values works ok
        else:
            self.dataNP = data
        self.Z = standardize(self.dataNP)
        self.totalSize, self.nVars = numpy.shape(self.Z)
        self.maxiter = maxiter
        self.currentOptimum = None
        self.index = index

    def optimize(self, maxiter = 5, opt_method=scipy.optimize.fmin):
        func = self.getIndex()
        if self.currentOptimum != None:
            x = self.currentOptimum
        else:
            x = numpy.random.rand(self.dim * self.nVars)
        alpha = opt_method(func, x, maxiter=maxiter).reshape(self.dim * self.nVars,1)
        self.currentOptimum = alpha
        print alpha, len(alpha)
        optValue = func(alpha)
        if self.dim == 2:
            alpha1 = alpha[:self.nVars]
            alpha2 = alpha[self.nVars:]
            alpha = numpy.append(alpha1, alpha2, axis=1)
        projectedData = numpy.dot(self.Z, alpha)
        return alpha, optValue, projectedData

    def find_optimum(self, opt_method=scipy.optimize.fmin):
        func = self.getIndex()
        alpha = opt_method(func, \
                           numpy.random.rand(self.dim * self.nVars),\
                           maxiter=self.maxiter).reshape(self.dim * self.nVars,1)
        print alpha, len(alpha)
        optValue = func(alpha)
        if self.dim == 2:
            alpha1 = alpha[:self.nVars]
            alpha2 = alpha[self.nVars:]
            alpha = numpy.append(alpha1, alpha2, axis=1)
        projectedData = numpy.dot(self.Z, alpha)
        return alpha, optValue, projectedData        

    def getIndex(self):
        if self.index == self.FRIEDMAN:
            return self.getFriedmanIndex()
        elif self.index == self.MOMENT:
            return self.getMomentIndex()
        elif self.index == self.SILHUETTE:
            return self.getSilhouetteBasedIndex()
        elif self.index == self.HARTINGAN:
            return self.getHartinganBasedIndex()
        

    def getFriedmanIndex(self, J=5):
        if self.dim == 1:
            def func(alpha, Z=self.Z, J=J, n=self.totalSize):
                vals, pols, val = friedman_tmp_func(alpha, Z=Z, J=J, n=n)
                return val
        elif self.dim == 2:
            def func(alpha, Z=self.Z, J=J, n=self.totalSize):
                alpha1, alpha2 = alpha[:self.nVars], alpha[self.nVars:]
                vals1, pols, val1 = friedman_tmp_func(alpha1, Z=Z, J=J, n=n)
                vals2, pols, val2 = friedman_tmp_func(alpha2, Z=Z, J=J, n=n)
                val12 = - 0.5 * sum([sum([(2*j+1)*(2*k+1)*vals1[j]*vals2[k] for k in range(0, J+1-j)]) \
                             for j in range(0,J+1)])
##                print val1, val2
                return 0.5 * (val1 + val2 + val12)
        return func
        

    def getMomentIndex(self): # lahko dodas faktor 1./12
        if self.dim == 1:
            def func(alpha):
                smpl = numpy.dot(self.Z, alpha)
                return scipy.stats.kstat(smpl, n=3) ** 2 + 0.25 * scipy.stats.kstat(smpl, n=4)
        else:
            print "To do."
        return func

    def getSilhouetteBasedIndex(self, nClusters=5):
        import orngClustering
        def func(alpha, nClusters=nClusters):
            alpha1, alpha2 = alpha[:self.nVars], alpha[self.nVars:]
            alpha1 = alpha1.reshape((self.nVars,1))
            alpha2 = alpha2.reshape(self.nVars,1)
            alpha = numpy.append(alpha1, alpha2, axis=1)
            smpl = numpy.dot(self.Z, alpha)
            smpl = orange.ExampleTable(smpl)
            km = orngClustering.KMeans(smpl, centroids=nClusters)
            score = orngClustering.score_silhouette(km)
            return -score
        import functools
        silhIndex = functools.partial(func, nClusters=nClusters)
        return silhIndex
        

    def getHartinganBasedIndex(self, nClusters=5):
        import orngClustering
        def func(alpha, nClusters=nClusters):
            alpha1, alpha2 = alpha[:self.nVars], alpha[self.nVars:]
            alpha1 = alpha1.reshape((self.nVars,1))
            alpha2 = alpha2.reshape(self.nVars,1)
            alpha = numpy.append(alpha1, alpha2, axis=1)
            smpl = numpy.dot(self.Z, alpha)
            smpl = orange.ExampleTable(smpl)
            km1 = orngClustering.KMeans(smpl, centroids=nClusters)
            km2 = orngClustering.KMeans(smpl, centroids=nClusters)
            
            score = (self.totalSize - nClusters - 1) * (km1.score-km2.score) / (km2.score)
            return -score
        import functools
        hartinganIndex = functools.partial(func, nClusters=nClusters)
        return hartinganIndex
                



def draw_scatter_hist(x,y, fileName="lala.png"):
    from matplotlib.ticker import NullFormatter
    nullfmt   = NullFormatter()         # no labels

    clf()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    figure(1, figsize=(8,8))

    axScatter = axes(rect_scatter)
    axHistx = axes(rect_histx)
    axHisty = axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = numpy.max([numpy.max(np.fabs(x)), numpy.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim( (-lim, lim) )
    axScatter.set_ylim( (-lim, lim) )

    bins = numpy.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    savefig(fileName)


if __name__=="__main__":
##    data = orange.ExampleTable("c:\\Work\\Subgroup discovery\\iris.tab")
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
          
    data = data.select(data.domain.attributes)

    impmin = orange.ImputerConstructor_minimal(data) 
    data = impmin(data) 

    ppy = ProjectionPursuit(data, dim=2, maxiter=100)
    #ppy.friedman_index(J=5)
    #ppy.silhouette_based_index(nClusters=2)

##    import os
##    os.chdir("C:\\Work\\Subgroup discovery")
    #draw_scatter_hist(ppy.friedmanProjData[:,0], ppy.friedmanProjData[:,1])
    #draw_scatter_hist(ppy.silhouetteProjData[:,0], ppy.silhouetteProjData[:,1])

    print ppy.optimize()
