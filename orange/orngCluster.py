## Automatically adapted for numpy.oldnumeric Oct 04, 2007 by 

# ORANGE CLUSTERING
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on Struyf, Hubert, Rousseeuw:
#                Integrating Robust Clustering Techniques in S-PLUS
#                Computational Statistics and Data Analysis, 26, 17-37
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# CVS Status: $Id$
#
# ChangeLog:
#
#   2004/10/4 (aleks)
#       - dendrogram sorting
#
#   2004/4/32 (peter.juvan@fri.uni-lj.si)
#       - added BIC score for MClustering and DMClustering (requires input vectors)
#       - added conversion functions: diss ragged list <-> Numeric.array
#       - fixed bug: DMClustering for k = len(diss) + 1

import math, numpy.oldnumeric as Numeric, statc
import orngCRS

class HClustering:
# in merging pairs occur. negative numbers identify elements in the order
# list, positive the consecutive number (starting from 1) of previous merges.
# Height refers to merges
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
# Methods: 1- average
#          2- single
#          3- complete
#          4- ward
#          5-weighted
#
        def extract_graph(self,data,filename = 'graph.dot'):
                names = []
                for i in data.domain.attributes:
                        t = string.replace(i.name," ","_")
                        names.append(string.replace(i.name,"-","_"))
                f = open(filename,'w')
                f.write("digraph G {\n")
                j = 0
                for i in self.merging:
                        l = "%d"%j
                        j = j+1
                        n = "%d"%j
                        if i[0]<0:
                                a = names[-1-i[0]]
                        else:
                                a = "%d"%(i[0])
                        if i[1]<0:
                                b = names[-1-i[1]]
                        else:
                                b = "%d"%(i[1])
                        f.write("\t%s[ label = \"%s\"];\n"%(n,l))
                        f.write("\t%s -> %s;\n"%(a,n))
                        f.write("\t%s -> %s;\n"%(b,n))
                f.write("}\n")

        def extract_subgroups(self, n):
                # create n best bound sets
                bounds = []
                merges = []
                for i in range(self.n):
                    #merges.append([self.order[self.n-i-1]-1])
                    merges.append([self.n-i-1])
                merges.append("sentry")
                p = self.n
                for i in range(min(n,self.n-1)):
                        merges.append(merges[p+self.merging[i][0]]+merges[p+self.merging[i][1]])
                        bounds.append(merges[-1])
                return bounds

        def extract_hierarchy(self):
                #
                # export the hierarchy in a format understood by induce()
                #
                bounds = []
                merges = []
                for i in range(self.n):
                    merges.append(self.n-i-1)
                merges.append("sentry")
                p = self.n
                c = 0
                for i in range(self.n-1):
                        merges.append(['%d-%.2f'%(c,self.height[i]),merges[p+self.merging[i][0]],merges[p+self.merging[i][1]]])
                        c = c+1
                return merges[-1]
                

        def domapping(self,kk):
                height = 0.0
                merges = []
                for i in range(self.n):
                    #merges.append([self.order[self.n-i-1]-1])
                    merges.append([self.n-i-1])
                merges.append("sentry")
                p = self.n
                deletes = [p]
                for i in range(self.n-kk):
                        height += self.height[i]
                        merges.append(merges[p+self.merging[i][0]]+merges[p+self.merging[i][1]])
                        deletes.append(p+self.merging[i][0])
                        deletes.append(p+self.merging[i][1])
                p = 0
                deletes.sort()
                for i in deletes:
                    del merges[p+i]
                    p = p-1
                self.mapping = [0]*self.n
                for i in range(len(merges)):
                    for j in merges[i]:
                        self.mapping[j] = i+1
                #return height
                    
        def sort(self,seq):
                # try to sort the dendrogram according to a list
                merges = []
                for i in range(self.n):
                    merges.append([self.n-i-1])
                merges.append("sentry")
                p = self.n

                # inverse ordering
                iord = [0]*self.n
                for i in xrange(self.n): # i - position
                        iord[self.order[i]-1] = i # iord[label_i] = position
                
                for i in range(self.n-1):
                        # branches
                        ba = merges[p+self.merging[i][0]]
                        bb = merges[p+self.merging[i][1]]
                        # compute the weight of each branch
                        wa = 0.0
                        wb = 0.0
                        for x in ba:
                                wa += seq[x]
                        wa /= len(ba)
                        for x in bb:
                                wb += seq[x]
                        wb /= len(bb)
                        if wa > wb:
                                la = len(ba)
                                lb = len(bb)
                                for x in ba:
                                        iord[x] += lb # move forward
                                for x in bb:
                                        iord[x] -= la # move backward
                        merges.append(ba+bb)

                # remap
                for i in xrange(self.n): # i - label
                        self.order[iord[i]] = 1+i

        def __init__(self, distrlist, metric=2, method=4):
                assert(metric == 1 or metric == 2)
                assert(method >= 1 or method <= 5)
                if len(distrlist) > 1:
                        (a,b,c,d,e) = orngCRS.HCluster(distrlist,metric, method)
                        self.n = a
                        self.merging = []
                        for i in range(self.n-1):
                            self.merging.append([b[i],b[i+self.n-1]])
                        self.order = c
                        self.height = d
                        self.ac = e
                        self.mapping = []
                        self.values = distrlist
                else:
                        if len(distrlist) == 1:
                                self.n = 1
                                self.merging = []
                                self.order = [1]
                                self.ac = 0.0
                                self.mapping = []
                                self.values = [[1]]
                                self.height = [1.0]
                        else:
                                self.n = 0
                                self.merging = []
                                self.order = []
                                self.ac = 0.0
                                self.mapping = []
                                self.values = []
                                self.height = []
                
class MClustering:
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
    def __init__(self, distrlist,k,metric=2):
        assert(metric == 1 or metric == 2)
        if (len(distrlist) > k):
                (a,b,c,d,e,f) = orngCRS.MCluster(distrlist,k,metric)
                self.n = a
                self.k = b
                self.mapping = c
                self.medoids = d
                self.cdisp = e
                self.disp = f
        else:
                self.n = len(distrlist)
                self.k = self.n
                self.mapping = range(1,self.n+1)
                self.medoids = range(1,self.n+1)
                self.cdisp = [0]*self.n
                self.disp = 0
        self.values = distrlist
        self.metric = metric

    def bic(self):
        return _bic(self.values, self.mapping, self.medoids, self.k)

class FClustering:
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
    def __init__(self, distrlist,k,metric=2):
      assert(metric == 1 or metric == 2)
      if not(k >= 1):
        raise "FClustering: Negative or zero number of clusters"
      if not(k < len(distrlist)/2):
        raise "FClustering: Not enough data for this number of clusters"
      (a,b,c,d,e,f,g,h) = orngCRS.FCluster(distrlist,k,metric)
      self.n = a
      self.k = b
      self.value = c
      self.iterations = d
      self.membership = e
      self.mapping = f
      self.cdisp = g
      self.disp = h

class DHClustering(HClustering):
# in merging pairs occur. negative numbers identify elements in the order
# list, positive the consecutive number (starting from 1) of previous merges.
# Height refers to merges
#
# The diss should contain a dissimilarity matrix, 
#
# Methods: 1- average
#          2- single
#          3- complete
#          4- ward
#          5-weighted
#
    def __init__(self, diss, method=4):
        assert(method >= 1 or method <= 5)
        if len(diss) < 1:
                self.n = 1
                self.merging = []
                self.order = [1]
                self.ac = 0.0
                self.mapping = []
                self.values = [[1]]
                self.height = [1.0]
        else:
                (a,b,c,d,e) = orngCRS.DHCluster(diss, method)
                self.n = a
                self.merging = []
                for i in range(self.n-1):
                    self.merging.append([b[i],b[i+self.n-1]])
                self.order = c
                self.height = d
                self.ac = e
                self.mapping = []
        
class DMClustering:
#
#
        def __init__(self, diss,k):
                if len(diss) <= k-1: #len = 1 if two elements ---- then k can be 2 or less
                        self.n = len(diss)+1
                        self.k = self.n
                        self.mapping = range(1,self.n+1)
                        self.medoids = range(1,self.n+1)
                        self.cdisp = [0]*self.n
                        self.disp = 0
                        self.values = diss
                else:
                        (a,b,c,d,e,f) = orngCRS.DMCluster(diss,k)
                        self.n = a
                        self.k = b
                        self.mapping = c
                        self.medoids = d 
                        self.cdisp = e
                        self.disp = f
                self.diss = diss

        def bic(self, distrlist):
                return _bic(distrlist, self.mapping, self.medoids, self.k)

        def bicMA(self, ma2d):
                """Input: 2d masked array with vectors that were used to calculate self.diss.
                """
                assert type(ma2d) in [Numeric.ArrayType, MA.array]
                return _bicMA(ma2d, self.mapping, self.medoids, self.k)

class DFClustering:
    def __init__(self, diss,k):
        if not(k >= 1):
          raise "DFClustering: Negative or zero number of clusters"
        if not(k <= len(diss)/2):
          raise "DFClustering: Not enough data for this number of clusters"
        (a,b,c,d,e,f,g,h) = orngCRS.DFCluster(diss,k)
        self.n = a
        self.k = b
        self.value = c
        self.iterations = d
        self.membership = e
        self.mapping = f
        self.cdisp = g
        self.disp = h


def bicMC(distrlist, mc):
        assert issubclass(mc.__class__, MClustering) or issubclass(mc.__class__, DMClustering), "mc: MClustering or DMClustering instance expected"
        return _bic(distrlist, mc.mapping, mc.medoids, mc.k)

def _bic(distrlist, mapping, medoids, K):
        """returns Bayesian Information Criteria score of the clustering
        WARNING: cannot evaluate BIC for K == len(distrlist), that is each element represents a cluster
        """
        mapping0 = [m-1 for m in mapping]       # fix indices
        medoids0 = [m-1 for m in medoids]       # fix indices
        M = len(distrlist[0])                   # number of dimensions
        R = len(distrlist) * 1.                 # number of input vectors
        if R == K:
                print "cannot calc. BIC if the number of clusters equals the number of vectors"
                return -1e20
        numFreePar = (M+1) * K * math.log(R, 2) / 2
        Ri = [0] * K
        sumdiffsq = {}
        s2 = 0                                  # max. likelihood of the variance: sigma**2
        for i,c in enumerate(mapping0):
            Ri[c] += 1
            xsumdiffsq = statc.sumdiffsquared(distrlist[i], distrlist[medoids0[c]])
            s2 += xsumdiffsq
            sumdiffsq[i] = xsumdiffsq
        s2 = s2 / (R - K)
        # log-likelihood of the vectors = ld + logf
        logf = R * (-0.5*math.log(2.*math.pi,2) + M/-2.*math.log(s2,2))         # sigma**(-M) == sigma**2**(-M/2)
        ld = 0                                  
        for i,c in enumerate(mapping0):
            ld += math.log(Ri[c] / R, 2) - sumdiffsq[i] / (2 * s2)
        return ld + logf - numFreePar

def _bicMA(ma2d, mapping, medoids, K):
        """returns Bayesian Information Criteria score of the clustering
        WARNING: cannot evaluate BIC for K == len(distrlist), that is each element represents a cluster
        opperates with masked arrays (MA), supports for masked values
        """
        mapping0 = [m-1 for m in mapping]       # fix indices
        medoids0 = [m-1 for m in medoids]       # fix indices
        M = ma2d.shape[1]                       # number of dimensions
        R = ma2d.shape[0]                       # number of input vectors
        if R == K:
                print "cannot calc. BIC if the number of clusters equals the number of vectors"
                return -1e20
        numFreePar = (M+1) * K * math.log(R, 2) / 2
        Ri = [0] * K
        sumdiffsq = MA.zeros(len(mapping0), Numeric.Float)
        for i,c in enumerate(mapping0):
            Ri[c] += 1
            sumdiffsq[i] = MA.add.reduce((ma2d[i] - ma2d[medoids0[c]])**2)
        # max. likelihood of the variance: sigma**2
        s2 = MA.add.reduce(sumdiffsq) / float(R-K)
        # log-likelihood of the vectors = ld + logf
        logf = R * (-0.5*math.log(2.*math.pi,2) + M/-2.*math.log(s2,2))         # sigma**(-M) == sigma**2**(-M/2)
        ld = MA.zeros(len(mapping0), Numeric.Float)
        for i,c in enumerate(mapping0):
            ld[i] = math.log(float(Ri[c]) / R, 2) - sumdiffsq[i] / (2 * s2)
        return MA.add.reduce(ld) + logf - numFreePar


def _fixIndices0(indList):
    """is there any special not to start indices with 0?"""
    return [x-1 for x in indList]


def _fixMCResults0(mc):
        """ before: medoid value of vector i = distrlist[MCluster.medoids[MCluster.mapping[i] - 1] - 1]
            after:  medoid value of vector i = distrlist[MCluster.medoids[MCluster.mapping[i]]]
        """
        mc.mapping = [m-1 for m in mc.mapping]
        mc.medoids = [m-1 for m in mc.medoids]
        


def mtrx2raggedList(mtrx, startEmpty=False):
    """convert a square matrix (2d array) to bottom triangular ragged list
    elements are taken from the bottom part of the matrix
    ragged list equals to distrlist in ?Cluster
    optional empty list at the begining (statEmpty=True)
    """
    rl = [line.tolist()[:i] for i,line in enumerate(mtrx)]
    if startEmpty:
        return rl
    else:
        return rl[1:]


def raggedList2mtrx(raggedList):
    """convert a bottom triangular ragged list (with optional empty list at the beginning) to a square matrix"""
    if len(raggedList[0]) == 0:
        raggedList = raggedList[1:]
    rlLen = len(raggedList) + 1
    m = Numeric.zeros((rlLen,rlLen), Numeric.Float)
    idx = 0
    for lst in raggedList:
        idx += 1
        m[idx, :idx] = Numeric.array(lst, Numeric.Float)
        m[:idx, idx] = Numeric.array(lst, Numeric.Float)
    return m


if __name__ == "__main__":
        import numpy.oldnumeric.random_array as RandomArray
        vecta1 = RandomArray.random((10,2))
        vect1 = vecta1.tolist()
        K = 8
        mcm = orngCluster.MClustering(vect1,k=K,metric=2)
        mce = orngCluster.MClustering(vect1,k=K,metric=1)
        print mcm.bic()
        print mce.bic()
