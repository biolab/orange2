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
#
#

from math import *
from Numeric import *
from LinearAlgebra import *
import RandomArray
import copy
from time import clock
import orangemds

# this function converts a bottom-triangular pseudo-matrix [[1],[2,3]]
# into a proper matrix [[0,1,2],[1,0,3],[2,3,0]]
def constructMatrixFromProx(mat):
    #print "converting..."
    diss = array([0.0])
    ts = len(mat)+1
    diss = resize(diss,(ts,ts))
    mi = 1e500
    for i in range(1,ts):
        for j in range(i):
            r = mat[i-1][j]
            mi = min(r,mi)
            diss[j][i] = diss[i][j] = r
    # normalize
    if mi < 0:
        mi *= 1.1
        diss -= mi
        diss += mi*identity(ts)
    return diss

def _mycompare((a,aa),(b,bb)):
    if a==b:
        return 0
    if a<b:
        return -1
    else:
        return 1


def KruskalStress(correct, current, weight=1):
    d = current-correct
    return weight*d*d

def SammonStress(correct, current, weight=1):
    d = current - correct
    return weight*(d*d)/max(1e-6,current)

# signed Sammon's stress
def SgnSammonStress(correct, current, weight=1):
    d = current-correct
    return weight*d/max(1e-6,current)

# signed relative stress
def SgnRelStress(correct, current, weight=1):
    # negative if too close in diagram, positive if too far in diagram
    d = current-correct
    return weight*d/max(1e-6,correct)

default_stress = SgnRelStress

class MDS:
    # initialize with a dissimilarity matrix
    def __init__(self, dissimilarity_matrix, dimensions = 2):
        self.O = copy.deepcopy(dissimilarity_matrix)
        self.Orig = copy.deepcopy(dissimilarity_matrix)
        self.n = len(self.O)
        self.setDim(dimensions)
        self.W = None
        # initialize the points randomly
        self.X = RandomArray.random(shape=[self.n,dimensions])
        self.dist=RandomArray.random(shape=[self.n, self.n])
        self.stress=RandomArray.random(shape=[self.n, self.n])
        self.freshD = 0

    def setDim(self, dim):
        self.dim = dim

    # update the self.dist distance matrix with euclidean distances
    def getDistance(self):
        if not self.freshD:
            self.freshD = 1
            """
            self.dist = resize(array([0.0]),(self.n,self.n))
            for i in xrange(1,self.n):
                for j in xrange(i):
                    s = 0.0
                    #for k in range(self.dim):
                    #    s += (self.X[i][k]-self.X[j][k])**2
                    v=self.X[i]-self.X[j]
                    s=dot(v,v)
                    s = sqrt(s)
                    self.dist[i][j] = s
                    self.dist[j][i] = s
            """
            orangemds.getDistance(self.X,self.dist)
    
    def getStress(self,stressf=default_stress):
        self.getDistance()
        map={ KruskalStress:0, SammonStress:1, SgnSammonStress:2, SgnRelStress:3}
        try:
            if not isinstance(stressf, int):
                i=map[stressf]
            else:
                i=stressf
            self.arr=orangemds.getStress(self.O,self.dist,self.stress,i)
        except:
            self.stress = resize(array([0.0]),(self.n,self.n))
            self.arr = []
            total = 0.0
            for i in xrange(1,self.n):
                for j in xrange(i):
                    r = stressf(self.O[i][j],self.dist[i][j])
                    self.stress[i][j] = r
                    self.stress[j][i] = r
                    self.arr.append((r,(i,j)))
                    total += abs(r)
            self.arr.sort(_mycompare)
            print "avg. stress: ",total/(self.n*self.n)
        
        
        
    def Torgerson(self):
        # Torgerson's initial approximation
        B = matrixmultiply(self.O,self.O)
        
        # double-center B
        cavg = sum(B)/(self.n+0.0)      # column sum
        ravg = sum(B,1)/(self.n+0.0)    # row sum
        tavg = sum(cavg)/(self.n+0.0)   # total sum
        # B[row][column]
        for i in xrange(self.n):
            for j in xrange(self.n):
                B[i][j] += -cavg[j]-ravg[i]
        B = -0.5*(B+tavg)

        # SVD-solve B = ULU'
        (U,L,V) = singular_value_decomposition(B)
        # X = U(L^0.5)
        # # self.X = matrixmultiply(U,identity(self.n)*sqrt(L))
        # X is n-dimensional, we take the two dimensions with the largest singular values
        idx = argsort(L)[-self.dim:].tolist()
        idx.reverse()
        
        Lt = take(L,idx)   # take those singular values
        Ut = take(U,idx,1) # take those columns that are enabled
        Dt = identity(self.dim)*sqrt(Lt)  # make a diagonal matrix, with squarooted values
        self.X = matrixmultiply(Ut,Dt)
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
                o.append((self.Orig[i][j],(i,j)))
        o.sort(_mycompare)
        # find the ties in o, and construct the d vector sorting in order within ties
        d = []
        td = []
        uv = [] # numbers of consecutively tied o values
        (i,j) = o[0][1]
        distnorm = self.dist[i][j]*self.dist[i][j]
        td = [self.dist[i][j]] # fetch distance
        for l in xrange(1,len(o)):
            # copy now sorted distances in an array
            # but sort distances within a tied o
            (i,j) = o[l][1]
            cd = self.dist[i][j]
            distnorm += self.dist[i][j]*self.dist[i][j]
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
        f = sqrt(distnorm/sum)
        # transform O
        k = 0
        for i in d:
            for j in range(i[1]):
                (ii,jj) = o[k][1]
                self.O[ii][jj] = f*i[2]
                self.O[jj][ii] = f*i[2]
                k += 1
        assert(len(o) == k)
        self.freshD = 0
        return effect

    # perform one step of simple gradient descent SMACOF 
    def SMACOFstep(self):
        # compute the R (n*n) matrix
        self.getDistance()
        st=clock()
        """
        R = resize(array([0.0]),(self.n,self.n))
        sumv = array([0.0]*self.n)
        for i in xrange(self.n):
            for j in xrange(self.n):
                if i != j:
                    if self.dist[i][j] > 1e-6:
                        s = 1.0/self.dist[i][j]
                    else:
                        s = 0.0
                    t = self.O[i][j]*s
                    R[i][j] = -t
                    sumv[i] += t
                    
        for i in range(self.n):
            R[i][i] = sumv[i]
        # compute the iteration step
        self.X = matrixmultiply(R,self.X)/(self.n+0.0)
        """
        orangemds.SMACOFStep(self.dist, self.O, self.X)
        #print "SMACOF", clock()-st
        self.freshD = 0

    def SMACOFstepSlow(self):
        # compute the R (n*n) matrix
        self.getDistance()
        R = resize(array([0.0]),(self.n,self.n))
        sumv = array([0.0]*self.n)
        for i in xrange(self.n):
            for j in xrange(self.n):
                if i != j:
                    if self.dist[i][j] > 1e-6:
                        s = 1.0/self.dist[i][j]
                    else:
                        s = 0.0
                    t = self.O[i][j]*s
                    t = 1+(t-1)*0.1
                    R[i][j] = -t
                    sumv[i] += t
                    
        for i in range(self.n):
            R[i][i] = sumv[i]
        # compute the iteration step
        self.X = matrixmultiply(R,self.X)/(self.n+0.0)
        self.freshD = 0

    def SMACOFstepLocal(self):
        # compute the R (n*n) matrix
        self.getDistance()
        R = resize(array([0.0]),(self.n,self.n))
        sumv = array([0.0]*self.n)
        
        near = array([1e10]*self.n)
        for i in xrange(self.n):
            for j in xrange(self.n):
                if i!=j:
                    near[i]=min(near[i],self.O[i][j])

        for i in xrange(self.n):
            for j in xrange(self.n):
                if i != j:
                    if self.dist[i][j] > 1e-6:
                        s = 1.0/self.dist[i][j]
                    else:
                        s = 0.0
                    t = self.O[i][j]*s
                    t = 1+(t-1)*((near[i]+near[j])/(self.O[i][j]*2.0))
                    R[i][j] = -t
                    sumv[i] += t
                    
        for i in range(self.n):
            R[i][i] = sumv[i]
        # compute the iteration step
        self.X = matrixmultiply(R,self.X)/(self.n+0.0)
        self.freshD = 0


# Weighted MDS
class WMDS(MDS):
    def setWeights(self, wei):
        self.freshD = 0
        self.W = copy.deepcopy(wei)
        self.Wsum = sum(sum(wei))

        # create the prefix matrix for SMACOF
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

    def getStress(self,stressf=default_stress):
        self.getDistance()        
        self.stress = resize(array([0.0]),(self.n,self.n))
        self.arr = []
        total = 0.0
        for i in xrange(1,self.n):
            for j in xrange(i):
                r = stressf(self.O[i][j],self.dist[i][j],self.W[i][j])
                self.stress[i][j] = r
                self.stress[j][i] = r
                self.arr.append((r,(i,j)))
                total += abs(r)
        self.arr.sort(_mycompare)
        print "avg. W stress: ",total/(self.n*self.n)
        #print self.arr
        
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
                    o.append((self.Orig[i][j],(i,j)))
        o.sort(_mycompare)
        # find the ties in o, and construct the d vector sorting in order within ties
        d = []
        td = []
        uv = [] # numbers of consecutively tied o values
        (i,j) = o[0][1]
        distnorm = self.W[i][j]*self.dist[i][j]*self.dist[i][j]
        td = [self.W[i][j]*self.dist[i][j]] # weighted distance
        ld = [self.W[i][j]] # weights
        for l in xrange(1,len(o)):
            # copy now sorted distances in an array
            # but sort distances within a tied o
            (i,j) = o[l][1]
            cd = self.W[i][j]*self.dist[i][j]
            distnorm += self.W[i][j]*self.dist[i][j]*self.dist[i][j]
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
                self.O[ii][jj] = f*i[2]
                self.O[jj][ii] = f*i[2]
                k += 1
        assert(len(o) == k) # all values must be transformed
        self.freshD = 0
        return effect

    def SMACOFstep(self):
        # compute the R (n*n) matrix
        self.getDistance()
        R = resize(array([0.0]),(self.n,self.n))
        sumv = array([0.0]*self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if self.dist[i][j] > 1e-6:
                        s = self.W[i][j]/self.dist[i][j]
                    else:
                        s = 0.0
                    t = self.O[i][j]*s
                    # the closer the value to 1.0, the better it is
                    #t = (self.W[i][j]*self.O[i][j] + (1-self.W[i][j])*self.dist[i][j])/max(1e-6,self.dist[i][j])
                    #t = self.O[i][j]/max(1e-6,self.dist[i][j])
                    R[i][j] = -t
                    sumv[i] += t
                    
        for i in range(self.n):
            R[i][i] = sumv[i]
        # compute the iteration step
        self.X = matrixmultiply(self.iV,matrixmultiply(self.M,matrixmultiply(R,self.X)))
        self.freshD = 0
        self.getDistance()
