#
# A Trashy (but working) MDS Library 
#
# CVS Info: $Id$
#
# V 1.01
#   - fixed the scaling bug in constructMatrixFromProx() (H. Harpending)
#
# V 1.0
#
# Aleks Jakulin (jakulin@acm.org) 2001-2002
# Ales Erjavec 2006


import orange
import orangeom as orangemds
from math import *
#from Numeric import *
#from LinearAlgebra import *
from numpy import *
from numpy.linalg import svd


KruskalStress=orangemds.KruskalStress()
SammonStress=orangemds.SammonStress()
SgnSammonStress=orangemds.SgnSammonStress()
SgnRelStress=orangemds.SgnRelStress()

PointList=orange.FloatListList
FloatListList=orange.FloatListList

def _mycompare((a,aa),(b,bb)):
    if a==b:
        return 0
    if a<b:
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
    def __init__(self, distances=None, dim=2, **kwargs):
        self.mds=orangemds.MDS(distances, dim, **kwargs)
        self.originalDistances=orange.SymMatrix([m for m in self.distances])

    def __getattr__(self, name):
        if name in ["points", "projectedDistances", "distances" ,"stress", "progressCallback", "n", "dim", "avgStress"]:
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
            
        if name in ["projectedDistances", "distances" ,"stress", "progressCallback"]:
            self.mds.__setattr__(name, value)
        else:
            self.__dict__[name]=value
            
    def __nonzero__(self):
        return True
            
    def SMACOFstep(self):
        self.mds.SMACOFstep()

    def getDistance(self):
        self.mds.getDistance()

    def getStress(self, stressFunc=SgnRelStress):
        self.mds.getStress(stressFunc)

    def optimize(self, iter, stressFunc=SgnRelStress, eps=1e-3, progressCallback=None):
        self.mds.progressCallback=progressCallback
        self.mds.optimize(iter, stressFunc, eps)

    def run(self, iter, stressFunc=SgnRelStress, eps=1e-3, progressCallback=None):
        self.optimize(iter, stressFunc, eps, progressCallback)

    def Torgerson(self):
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
        self.points = orange.FloatListList(dot(Ut,Dt))
        self.freshD = 0
        
#        D = identity(self.n)*sqrt(L)  # make a diagonal matrix, with squarooted values
#        X = matrixmultiply(U,D)
#        self.X = take(X,idx,1)

    # Kruskal's monotone transformation
    def LSMT(self):
        # optimize the distance transformation
        # build vector o
        effect = 0
        self.getDistance()
        o = []
        for i in xrange(1,self.n):
            for j in xrange(i):
                o.append((self.originalDistances[i,j],(i,j)))
        o.sort(_mycompare)
        # find the ties in o, and construct the d vector sorting in order within ties
        d = []
        td = []
        uv = [] # numbers of consecutively tied o values
        (i,j) = o[0][1]
        distnorm = self.projectedDistances[i,j]*self.projectedDistances[i,j]
        td = [self.projectedDistances[i,j]] # fetch distance
        for l in xrange(1,len(o)):
            # copy now sorted distances in an array
            # but sort distances within a tied o
            (i,j) = o[l][1]
            cd = self.projectedDistances[i,j]
            distnorm += self.projectedDistances[i,j]*self.projectedDistances[i,j]
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

"""
class WMDS(MDS):
    def setWeights(self, weights):
        self.W=weights
        self.Wsum=sum(sum(self.W))

        V = resize(array([0.0]),(self.n,self.n))
        sumv = array([0.0]*self.n)
        for i in range(1,self.n):
            for j in range(i):
                t = (-self.W[i][j]-self.W[j][i])/2.0
                V[i][j] = t
                V[j][i] = t
                sumv[i] -= t
                sumv[j] -= t
        for i in range(self.n):
            V[i][i] = sumv[i]

        # default for no weights
#        for i in range(self.n):
#            for j in range(self.n):
#                if i == j:
#                    V[i][j]=self.n-1
#                else:
#                    V[i][j]=-1
        T = matrixmultiply(transpose(V),V)
        # compute moore-penrose inverse of T
#        (U,L,V) = singular_value_decomposition(T)
#        Lt = identity(self.n)
#        for i in range(self.n):
#            if L[i] > 1e-6:
#                Lt[i][i] /= L[i]
#            else:
#                Lt[i][i] = 0.0
#        MPI = matrixmultiply(transpose(V),matrixmultiply(Lt,transpose(U)))
        self.M = matrixmultiply(generalized_inverse(T),transpose(V))
        self.iV = generalized_inverse(V)
        #print self.M

    def getStress(self, stressf=SgnRelStress):
        total=0.0
        for i in range(self.n):
            for j in range(i):
                r=self.stress[i,j]=stressf(self.distances[i,j], self.projectedDistances[i,j], self.W[i,j])
                total+=abs(r)
        self.avgStress=total/(self.n*self.n)

    def LSMT(self):
        # optimize the distance transformation
        # build vector o
        effect = 0
        self.getDistance()
        o = []
        for i in xrange(1,self.n):
            for j in xrange(i):
                # skip the distances we don't care about
                if self.W[i][j] > 1e-6:
                    o.append((self.originalDistances[i,j],(i,j)))
        o.sort(_mycompare)
        # find the ties in o, and construct the d vector sorting in order within ties
        d = []
        td = []
        uv = [] # numbers of consecutively tied o values
        (i,j) = o[0][1]
        distnorm = self.W[i][j]*self.projectedDistances[i,j]*self.projectedDistances[i,j]
        td = [self.W[i][j]*self.projectedDistances[i,j]] # weighted distance
        ld = [self.W[i][j]] # weights
        for l in xrange(1,len(o)):
            # copy now sorted distances in an array
            # but sort distances within a tied o
            (i,j) = o[l][1]
            cd = self.W[i][j]*self.projectedDistances[i,j]
            distnorm += self.W[i][j]*self.projectedDistances[i,j]*self.projectedDistances[i,j]
            if o[l][0] != o[l-1][0]:
                # differing value, flush
                sum = reduce(lambda x,y:x+y,td)+0.0
                lend = 1.0/(reduce(lambda x,y:x+y,ld)+0.0) # sum of weights
                d.append([sum,lend,sum*lend,td])
                td = []
                ld = []
            td.append(cd)
            ld.append(self.W[i][j])
        lend = 1.0/(reduce(lambda x,y:x+y,ld)+0.0) # sum of weights
        sum = reduce(lambda x,y:x+y,td)+0.0
        d.append([sum,lend,sum*lend,td])
        ####
        # keep merging non-monotonous areas in d
        monotony = 0
        while not monotony and len(d) > 1:
            monotony = 1
            pi = 0 # index
            n = 1  # n-areas
            nd = []
            r = d[0] # current area
            for i in xrange(1,len(d)):
                tr = d[i]
                if r[2]>=tr[2]:
                    monotony = 0
                    effect = 1
                    r[0] += tr[0]  # sum of weighted distances
                    r[1] += tr[1]  # sum of weights
                    r[2] = tr[0]/tr[1] # average distance
                    r[3] += tr[3]  # array of values
                else:
                    nd.append(r)
                    r = tr
            nd.append(r)
            d = nd
        # normalizing multiplier
        sum = 0.0
        for i in d:
            sum += i[2]*i[2]*i[1]
        f = sqrt(distnorm/sum)
        # transform O
        k = 0
        for i in d:
            for j in xrange(len(i[3])): # note the use of the length of the array of distances
                (ii,jj) = o[k][1]
                self.distances[ii,jj] = f*i[2]
                k += 1
        assert(len(o) == k) # all values must be transformed
        self.freshD = 0
        return effect
"""
if __name__=="__main__":
    data=orange.ExampleTable("doc//datasets//iris.tab")
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])
    mds=MDS(matrix, dim=3)
    mds.Torgerson()
    print mds.points[:3]