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
        #private variables in this class
        #
        #used for personal reference
        self.__dataMatrix = array([[],[]])
        self.__corrMatrix = array([[],[]])    
        self.__sumElem = 0;
        self.__rowSums = array([])
        self.__colSums = array([])
        self.__rowProfiles = array([[],[]])
        self.__colProfiles = array([[],[]])
        self.__diagRowInv = array([[],[]])
        self.__diagcolInv = array([[],[]])
        self.__a = array([[],[]])
        self.__d = array([[],[]])
        self.__b = array([[],[]])
        self.__f = array([[],[]])
        self.__g = array([[],[]])
        
        #calculating correspondance analysis from the data matrix
        #algorithm described in the book (put reference) is used

        self.__dataMatrix = contingencyTable

##        if isinstance(contingencyTable, orange.ContingencyAttrAttr):
##            self.__ct = contingencyTable
##            self.__dataMatrix = array([list(a) for a in contingencyTable])
##        elif isinstance(contingencyTable, orange.ExampleTable):
##            self.__ct = contingencyTable
##            keys = contingencyTable.domain.getmetas().keys()
##            self.__dataMatrix = zero((len(data), len(keys)))
##            self.__dataMatrix = array([],[])
            
        #self.__dataMatrix = array(matrix)
        self.__sumElem = sum(sum(self.__dataMatrix))
        
        #corrMatrix is a matrix of relative frequencies of elements in data matrix
        self.__corrMatrix = self.__dataMatrix * 1. / self.__sumElem
        self.__colSums = sum(self.__corrMatrix).reshape(-1,1)
        self.__rowSums = sum(self.__corrMatrix, 1).reshape(-1,1)
        
        #diagRowInv is a diagonal matrix whoose elements are sums of each row of corrMatrix
        invRowSums = 1. / self.__rowSums
        self.__diagRowInv = invRowSums * eye(invRowSums.shape[0])
        
        #diagRowInv is a diagonal matrix whoose elements are sums of each column of corrMatrix
        invcolSums = 1. / self.__colSums
        self.__diagcolInv = invcolSums * eye(invcolSums.shape[0])
        
        self.__rowProfiles = matrixmultiply(self.__diagRowInv, self.__corrMatrix)
        self.__colProfiles = matrixmultiply(self.__diagcolInv, transpose(self.__corrMatrix))
    
        self.__a, self.__d, self.__b = self.__calculateSVD();    
        
        self.__f = matrixmultiply(matrixmultiply(self.__diagRowInv, self.__a), self.__d)
        self.__g = matrixmultiply(matrixmultiply(self.__diagcolInv, self.__b), transpose(self.__d))
        
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
        a = self.__corrMatrix - matrixmultiply(self.__rowSums, transpose(self.__colSums))
        b = matrixmultiply(matrixmultiply(sqrt(self.__diagRowInv), a), sqrt(self.__diagcolInv))
        u, d, v = numpy.linalg.svd(b)
        N = matrixmultiply(sqrt(self.__rowSums * eye(self.__rowSums.shape[0])), u)
        M = matrixmultiply(sqrt(self.__colSums * eye(self.__colSums.shape[0])), transpose(v))
        if a.shape[0] > a.shape[1]:
            list = [0] * (a.shape[1] * (a.shape[0] - a.shape[1]))
            d = concatenate((diag(d), array(list).reshape(a.shape[0] - a.shape[1], -1)))
        elif a.shape[0] < a.shape[1]:
            list = [0] * (a.shape[0] * (a.shape[1] - a.shape[0]))
            d = concatenate((diag(d), array(list).reshape(-1, a.shape[1] - a.shape[0])), 1)
        else:
            d = diag(d)
        
        return (N, d, M)       
        
    def getMatrix(self):
        """
            Returns array object that is representation of contingency table.
        """
        return self.__dataMatrix   
        
##    def getCT(self): return self.__ct
        
    dataMatrix = property(getMatrix)
##    contingencyTable = property(getCT)
    
    def getCorrMatrix(self):
        """
            corrMatrix = dataMatrix / (dataMatrix..)
        """
        return self.__corrMatrix
    def getSumElem(self): return self.__sumElem
    def getRowSums(self): return self.__rowSums
    def getcolSums(self): return self.__colSums
    def getRowProfiles(self): return self.__rowProfiles
    def getColProfiles(self): return self.__colProfiles
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
       return take(self.__f, dim, 1)
    def getPrincipalColProfilesCoordinates(self, dim = (0, 1)): 
       """Returns principal co-ordinates of column profiles with respect
       to principal axes A.
       Dim defines which principal axes should be taken into account.
       """    
       if len(dim) == 0:
           raise Exception("Dim tuple cannot be of lenght zero")      
       return take(self.__g, dim, 1)
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
            return take(self.__rowSums * self.__f * self.__f, tuple(range(min(self.__dataMatrix.shape))))
        else:
            return take(self.__colSums * self.__g * self.__g, tuple(range(min(self.__dataMatrix.shape))))
    def InertiaOfAxis(self):
        return sum(self.DecompositionOfInertia())
    def PercentageOfInertia(self):
        inertias = sum(self.DecompositionOfInertia())
        return inertias / sum(inertias) * 100
        
    def PlotScreeDiagram(self):
        ## todo: legend, axis, etc
        pylab.plot(range(1, min(self.__dataMatrix.shape) + 1), self.PercentageOfInertia())
        pylab.axis([0, min(c.dataMatrix.shape) + 1, 0, 100])
        pylab.show()
        
    def Biplot(self, dim = (0, 1)):
        if len(dim) != 2:
           raise Exception("Dim tuple must be of length two")
        pylab.plot(c.getPrincipalRowProfilesCoordinates()[:, dim[0]], c.getPrincipalRowProfilesCoordinates()[:, dim[1]], 'ro',
            c.getPrincipalColProfilesCoordinates()[:, dim[0]], c.getPrincipalColProfilesCoordinates()[:, dim[1]], 'bs')
        pylab.grid()
        pylab.  show()        
        
    corrMatrix = property(getCorrMatrix)
    sumElem = property(getSumElem)
    rowSums = property(getRowSums)
    colSums = property(getcolSums)
    rowProfiles = property(getRowProfiles)
    colProfiles = property(getColProfiles)
    A = property(getA)
    B = property(getB)
    D = property(getD)
    F = property(getF)
    G = property(getG)
    
if __name__ == '__main__':
##    d = orange.ExampleTable('smokers_ct')
##    c = CA(orange.ContingencyAttrAttr(1, 2, d))
    c = CA(input('smokers.tab'))
    print c.dataMatrix

##    c = CA(None)
##    print c.dataMatrix[0]


    
    
