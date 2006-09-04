"""
Correspondence analysis is a descriptive/exploratory technique designed to analyze simple two-way and 
multi-way tables containing some measure of correspondence between the rows and columns. 
"""

from numpy import *
import numpy.linalg
import orange
import pylab

def input(filename):
    """ Loads contingency matrix from the file. """
    table = [[],[]]
    try:
        file = open(filename)
        try:
            table = file.readlines()
        finally:
            file.close()
        table = [map(int, el.split()) for el in table]
    except IOError:
        pass
    return table
    
class CA(object):
    """Main class for computation of correspondance analysis"""
    def __init__(self, contingencyTable):
        """ @contingencyTable   instance of "list of lists"
        """     
        #calculating correspondance analysis from the data matrix
        #algorithm described in the book (put reference) is used

        self.__dataMatrix = matrix(contingencyTable)
        sumElem = sum(sum(array(self.__dataMatrix))) * 1.
        
        #corrMatrix is a matrix of relative frequencies of elements in data matrix
        self.__corr = self.__dataMatrix / sumElem        
        self.__colSums = sum(self.__corr)
        self.__rowSums = sum(self.__corr, 1)
        
        self.__colProfiles =  matrix(diag((1. / array(self.__colSums))[0])) * transpose(self.__corr)
        self.__rowProfiles = matrix(diag((1. / array(self.__rowSums).reshape(1,-1))[0])) * self.__corr
    
        self.__a, self.__d, self.__b = self.__calculateSVD();    
        
        self.__f = diag((1. / self.__rowSums).reshape(1,-1).tolist()[0]) * self.__a * self.__d
        self.__g = diag((1. / self.__colSums).tolist()[0]) * self.__b * transpose(self.__d)
        
    def __calculateSVD(self):
        """
            Computes generalized SVD...
            
            This function is used to calculate decomposition A = N * D_mi * M' , where N' * diag(rowSums) * N = I and
            M' * diag(colSums) * M = I. This decomposition is calculated in 4 steps:
            i) B = diag(rowSums)^1/2 * A * diag(colSums)^1/2
            ii) find the ordinary SVD of B: B = U * D * V'
            iii) N = diag(rowSums)^-1/2 * U
                M = diag(colSums)^-1/2 * V
                D_mi = D
            iv) A= N * D_mi * M'
            
            returns (N, D_mi, M)            
        """
        
        a = self.__corr - self.__rowSums * self.__colSums
        b = diag(sqrt((1. / self.__rowSums).reshape(1,-1).tolist()[0])) * a * diag(sqrt((1. / self.__colSums).tolist()[0]))
        u, d, v = numpy.linalg.svd(b, 0)
        N = diag(sqrt(self.__rowSums.reshape(1, -1).tolist()[0])) * u
        M = diag(sqrt(self.__colSums.tolist()[0])) * transpose(v)
        d = diag(d.tolist())
        
        return (N, d, M)       
        
    def getMatrix(self):
        """
            Returns array object that is representation of contingency table.
        """
        return self.__dataMatrix   
        
##    def getCT(self): return self.__ct
        
    dataMatrix = property(getMatrix)

    def getA(self): 
        """
            columns of A defines the principal axes of the column clouds
        """
        return self.__a
    def getD(self): 
        """
            elements on diagonal of D are singular values
        """
        return self.__d
    def getB(self): 
        """
            columns of B defines the principal axes of the row clouds
        """        
        return self.__b
    def getF(self): 
        """
            coordinates of the row profiles with respect to principal axes B
        """
        return self.__f
    def getG(self): 
        """
            coordinates of the column profiles with respect to principal axes A
        """
        return self.__g        
    def getPrincipalRowProfilesCoordinates(self, dim = (0, 1)):
       """Returns principal co-ordinates of row profiles with respect
       to principal axis B.
       Dim defines which principal axes should be taken into account.
       """
       if len(dim) == 0:
           raise Exception("Dim tuple cannot be of lenght zero")
       return array(take(self.__f, dim, 1))
    def getPrincipalColProfilesCoordinates(self, dim = (0, 1)): 
       """Returns principal co-ordinates of column profiles with respect
       to principal axes A.
       Dim defines which principal axes should be taken into account.
       """    
       if len(dim) == 0:
           raise Exception("Dim tuple cannot be of lenght zero")      
       return array(take(self.__g, dim, 1))
    def getStandardRowCoordinates(self):
        dinv = where(self.__d != 0, 1. / self.__d, 0)
        return matrixmultiply(self.__f, transpose(dinv))
    def getStandardColCoordinates(self):
        dinv = where(self.__d != 0, 1. / self.__d, 0)
        return matrixmultiply(self.__g, dinv)
    def DecompositionOfInertia(self, axis = 0):
        """
            axis = 0 decomposition across rows
            axis = 1 decomposition across columns
            
            Columns of this matrix represents contribution of the rows or columns to the inertia of axis.
        """
        if axis == 0:
            return take(array(self.__rowSums) * array(self.__f) * array(self.__f), tuple(range(min(self.__dataMatrix.shape))))
        else:
            return take(array(self.__colSums) * array(self.__g) * array(self.__g), tuple(range(min(self.__dataMatrix.shape))))
    def InertiaOfAxis(self):
        return sum(self.DecompositionOfInertia())
    def PercentageOfInertia(self):
        inertias = sum(self.DecompositionOfInertia())
        return inertias / sum(inertias) * 100
        
    def PlotScreeDiagram(self):
        ## todo: legend, axis, etc
        pylab.plot(range(1, min(self.__dataMatrix.shape) + 1), self.PercentageOfInertia())
        pylab.axis([0, min(self.__dataMatrix.shape) + 1, 0, 100])
        pylab.show()
        
    def Biplot(self, dim = (0, 1)):
        if len(dim) != 2:
           raise Exception("Dim tuple must be of length two")
        pylab.plot(self.getPrincipalRowProfilesCoordinates()[:, dim[0]], self.getPrincipalRowProfilesCoordinates()[:, dim[1]], 'ro',
            self.getPrincipalColProfilesCoordinates()[:, dim[0]], self.getPrincipalColProfilesCoordinates()[:, dim[1]], 'bs')
        pylab.grid()
        pylab.  show()                
    
    A = property(getA)
    B = property(getB)
    D = property(getD)
    F = property(getF)
    G = property(getG)
    
if __name__ == '__main__':
##    a = random.random_integers(0, 100, 100).reshape(10,-1)
##    c = CA(a)
##    c.Biplot()

##    data = matrix([[72,    39,    26,    23 ,    4],
##    [95,    58,    66,    84,    41],
##    [80,    73,    83,     4 ,   96],
##    [79,    93,    35,    73,    63]])
##
##    c = CA(data)



##    data = matrix(random.random_integers(0, 100, 500000).reshape(400,-1))
##    
##    sumElem = sum(sum(data), 1).tolist()[0][0]
##    corr = data * 1. / sumElem
##    colSums = sum(corr)
##    rowSums = sum(corr, 1)
##    colProfiles =  diag((1. / colSums).tolist()[0]) * transpose(corr)
##    rowProfiles = diag((1. / rowSums).reshape(1,-1).tolist()[0]) * corr 
##    a = corr - rowSums * colSums   
##    b = sqrt(diag((1. / rowSums).reshape(1,-1).tolist()[0])) * a * sqrt(diag((1. / colSums).tolist()[0]))    
##    u, d, v = numpy.linalg.svd(b, 0)
##    
##    N = sqrt(diag(rowSums.reshape(1, -1).tolist()[0])) * u
##    M = sqrt(diag(colSums.tolist()[0])) * transpose(v)
##    d = diag(d.tolist()) 
##    
##    print "pero"

##    data = [[9, 11, 4], 
##                [ 3,          5,          3], 
##                [     11,          6,          3], 
##                [24,         73,         48]] 
##    c = CA(data)
    
    import orange
    from orngCA import CA
    
    data = input('doc/datasets/smokers.tab')
    c = CA(data)
    print "Column profiles:"
    print c._CA__colProfiles
    print
    print "Row profiles:"
    print c._CA__rowProfiles
    print 
    print "Singular values: " + str(diag(c.D))
    print "Eigen values: " + str(square(diag(c.D)))
    print "Percentage of Inertia:" + str(c.PercentageOfInertia())
    print 
    print "Principal row coordinates:"
    print c.getPrincipalRowProfilesCoordinates()
    print 
    print "Decomposition Of Inertia:"
    print c.DecompositionOfInertia()
    c.PlotScreeDiagram()
    c.Biplot()
