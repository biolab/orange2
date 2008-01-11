import orange
import numpy
from numpy import dot
from pylab import *
from matplotlib.font_manager import fontManager, FontProperties

def standardize(matrix):
    """ standardizes matrix"""
    mean, std = numpy.mean(matrix, axis=0), numpy.std(matrix, axis=0)
    return (matrix - mean) / std, mean, std

def cummulate(array):
    """ Input:  [a_1,a_2,...,a_n]
        Output: [a_1,a_1+a_2,a_1+a_2+a_3, ... a_1+...a_n] / (a_1+...+a_n)
    """
    cum = [array[0]]
    for i in range(1, len(array)):
        cum.append(cum[i-1] + array[i])
    return cum / cum[-1]

class PCA:
    def __init__(self, data = None, name = 'principal component analysis', nComp = None):
        self.name = name
        self.nComp = nComp

    def __call__(self, data, atts=None, precision=10e-9):
        """ Performs principal component analysis on attributes atts form data"""
        # Z = X*U; U^TU=I; coloumns of U are uncorrelated
        self.data = data
        self.n = len(data)
        if atts == None:
            atts = [a.name for a in data.domain.attributes]
        self.rest = [a.name for a in data.domain.attributes if a.name not in atts]
        dataToTransform = data.select(atts)
        X = dataToTransform.toNumpy()[0]
        self.X = standardize(X)[0]
        X = self.X
        corr = numpy.corrcoef(X, rowvar=0)
        eigvals, eigvec = numpy.linalg.eigh(corr)
        ind = numpy.argsort(-eigvals)
        eigvals = eigvals[ind]
        eigvec = eigvec[:, ind]
        if self.nComp == None:
            self.nComp = len(numpy.where(abs(eigvals) > precision)[0])
        self.eigvals = eigvals[:self.nComp]
        self.alleigvals = eigvals
        self.U = eigvec[:,:self.nComp]
        self.Z = dot(X, self.U)

    def __str__(self, nComp=None):
        """ Prints eigenvalues, cummulatives and loading weights for PCA model."""
        if nComp == None:
            nComp = self.nComp
        cumm = cummulate(self.alleigvals)
            
        print 'N Comp  Eigenvalue  Cummulative'
        for i in range(nComp):
            print 'PC %d' % (i+1), '    %4.3f   ' % self.eigvals[i], ' %4.3f  ' % cumm[i]
        print '\n'
        fmt = 'PC %d   ' * nComp
        print 'Attribute   ' + fmt % tuple(range(1,nComp+1))
        fmt = '%10s ' + '%4.3f  ' * nComp
        for i, a in enumerate(self.data.domain):
            s = tuple([a.name] + list(self.U[i,:nComp]))
            print fmt % s     
            
        return 'PCA'        
            
    def screePlot(self, m=None):
        """ plots scree plot for first m dimensions """
        if m == None:
            m = len(self.eigvals)
        else:
            self.eigvals = self.eigvals[:m]
        indeces = numpy.array(range(1,m+1))
        #plot(indices.T, self.eigvals.T, 'p')
        plot(indeces.T, self.eigvals.T, '-ko')
        title('Scree plot')
        xlabel('Component number')
        ylabel('Eigenvalue')
        xlim(0.9, m+0.1)
        xticks(arange(1,m+1))
        show()

    def plotLoadings(self, compN=0, Show=0):
        """ Plots loading weights for the component number compN for all original attributes.
        Red lines represent positive and black negative weights.
        """
        font = FontProperties()
        font.set_size(7)
        lengths0 = self.U[:,compN]
        m = len(lengths0)
        longest = numpy.max(abs(lengths0))
        lengths1 = abs(lengths0) / longest
        for i in range(m):
            r, phi = lengths1[i], (2. * numpy.pi * i) / m
            if lengths0[i] < 0:
                plot((0.,r*numpy.cos(phi)),(0.,r*numpy.sin(phi)), color = 'r')
            else:
                plot((0.,r*numpy.cos(phi)),(0, r*numpy.sin(phi)), color = 'b')

            if lengths1[i] > Show:
                text(1.25 * numpy.cos(phi), 1.25 * numpy.sin(phi), str(data.domain[i])[14:], fontproperties=font)
                print data.domain.attributes[i], lengths0[i]
                
        t = numpy.arange(0, 2 * numpy.pi + 1, 0.1)
        plot(numpy.cos(t), numpy.sin(t), color = 'black')
        xlim(-1.3, 1.3)
        ylim(-1.3, 1.3)
        title('Loading weights for Principal component ' + str(compN+1))
        show()
        #savefig('test.png')

    def plotInstances(self, Axis=(1,2)):
        """ Plots instances in coordinate system of principal components
            Axis[0] and Axis[1].
        """
        coorX, coorY = self.Z[:,Axis[0]-1], self.Z[:,Axis[1]-1]        
        scatter(coorX, coorY)
        xlabel('Principal component ' + str(Axis[0]))
        ylabel('Principal component ' + str(Axis[1]))
        show()

    def transformed2Orange(self, save=False, fileName=None):
        newnames = [orange.FloatVariable('Z' + str(i+1)) for i in range(self.nComp)]
        pcaDomain = orange.Domain(newnames, 0)
        pcaData = orange.ExampleTable(pcaDomain)
        n = len(pcaData)
        for i in range(n):
            pcaData.append(list(self.Z[i]))
        if self.rest == None:
            newData = orange.ExampleTable(pcaData)
        else:
            print len(self.data.select(self.rest))
            print len(pcaData)
            newData = orange.ExampleTable([self.data.select(self.rest), pcaData])
        self.transformedData = newData
        if save:
            if fileName:
                newData.save(fileName)
            else:
                newData.save('pca.tab')
                print 'Warning! fileName missing. File saved as pca.tab'
        return newData

if __name__ == "__main__":
    
    data = orange.ExampleTable('smoking.tab')

    pca = PCA()
    pca(data)
    print pca
    pca.screePlot()
    pca.plotInstances()
