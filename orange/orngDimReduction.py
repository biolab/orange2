import orange
import numpy
import numpy.ma as MA
import numpy.linalg as linalg


class Base:
    def getSorted(self, values, vectors):
        pairs = [(val, i) for i, val in enumerate(values)]
        pairs.sort()
        pairs.reverse()
        indices = [pair[1] for pair in pairs]
        newValues = [values[i] for i in indices]
        newVectors = MA.take(vectors, indices, axis = 1)
        return newValues, newVectors
        
    # substract average value to get zero mean
    def center(self, data):
        suma = data.sum(axis=0)/float(len(data))
        data -= suma                            
        return data
    
    # normalize variance to 1 for each attribute
    def normalize(self, data):
        data /= MA.std(data, axis=0)      # divide with standard deviation to get deviation = 1 for all attrs
        return data

class PCA(Base):
    def __init__(self, data = None):
        self.setData(data)
        
    def setData(self, data):
        self.data = data
        self.normalizedData = None
        self.eigVectors = None
        self.eigValues = None
                
            
    def compute(self):
        if not self.data:
            return
        if type(self.eigVectors) == MA.MaskedArray and type(self.eigValues) == MA.MaskedArray:
            return
        
        if type(self.data) == orange.ExampleTable:
            data = self.data.toNumpyMA("a")[0]
        else:
            data = self.data
            
        data = self.center(data)
        data = self.normalize(data)
        self.normalizedData = data      # 
        
        covMatrix = MA.dot(data.T, data) / len(data)
        eigVals, eigVectors = linalg.eigh(covMatrix)
        self.eigValues, self.eigVectors = self.getSorted(eigVals, eigVectors)
                
        
    def getCount(self, nPCs = None, varianceExplained = None):
        if nPCs == None and varianceExplained == None:
            nPCs = len(self.eigValues)
        if varianceExplained == None:
            nPCs = min(nPCs, len(self.eigVectors))
        else:
            total = 0; nPCs = 0
            while total < varianceExplained and ind < len(self.eigValues):
                total += self.eigValues[ind]
                nPCs += 1
        return nPCs
        
    def getEigValues(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        nPCs = self.getCount(nPCs, varianceExplained)
        return self.eigValues[:nPCs]


    def getEigVectors(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        nPCs = self.getCount(nPCs, varianceExplained)
        return MA.take(self.eigVectors, range(nPCs), axis=1)
        
        
    def getProjectedData(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        vectors = self.getEigVectors(nPCs, varianceExplained)
        return MA.dot(self.normalizedData, vectors)
    
    def getExampleTable(self, nPCs = None, varianceExplained = None, keepOriginal = 1):
        if not self.data or type(self.data) != orange.ExampleTable: return None
        count = self.getCount(nPCs, varianceExplained)
        projData = self.getProjectedData(nPCs = count)
        newDomain = orange.Domain([orange.FloatVariable("PC %d" % (d+1)) for d in range(count)], 0)
        newTable = orange.ExampleTable(newDomain, projData.data)
        if keepOriginal:
            return orange.ExampleTable([self.data, newTable])
        else:
            if self.data.domain.classVar:
                return orange.ExampleTable([self.data.select([self.data.domain.classVar]), newTable])
            else:
                return newTable
        
# ###########################################################
# Fisher discriminant analysis
class FDA(Base):
    def __init__(self, data = None):
        self.setData(data)
        
    def setData(self, data):
        if data != None and type(data) not in [orange.ExampleTable, tuple]:
            print "invalid data type. Only ExampleTable and tuple types are allowed"
            self.data = None
            return
        self.data = data
        self.normalizedData = None
        self.eigVectors = None
        self.eigValues = None
        
    def compute(self):
        if self.data == None:
            return
        if type(self.eigVectors) == MA.MaskedArray and type(self.eigValues) == MA.MaskedArray:
            return
        
        if type(self.data) == orange.ExampleTable:
            data, classes = self.data.toNumpyMA("a/c")
        elif type(self.data) == tuple:
            data, classes = self.data

        data = self.center(data)
        data = self.normalize(data)
        self.normalizedData = data
        exampleCount, attrCount = data.shape
        classCount = len(set(classes))
        # special case when we have two classes
        if classCount == 2:
            data1 = MA.take(data, numpy.argwhere(classes == 0).flatten(), axis=0)
            data2 = MA.take(data, numpy.argwhere(classes != 0).flatten(), axis=0)
            miDiff = MA.average(data1, axis=1) - MA.average(data2, axis=1)
            covMatrix = (MA.dot(data1.T, data1) + MA.dot(data2.T, data2)) / exampleCount
            self.eigVectors = linalg.inv(covMatrix) * miDiff
            self.eigValues = numpy.array([1])
        else:
            # compute means and average covariances of examples in each class group
            Sw = MA.zeros([attrCount, attrCount])
            for v in set(classes):
                d = MA.take(data, numpy.argwhere(classes == v).flatten(), axis=0)
                d = self.center(d)
                Sw += MA.dot(d.T, d)
            Sw /= exampleCount
            total = MA.dot(data.T, data)/float(exampleCount)
            Sb = total - Sw
                        
            matrix = linalg.inv(Sw)*Sb
            eigVals, eigVectors = linalg.eigh(matrix)
            self.eigValues, self.eigVectors = self.getSorted(eigVals, eigVectors)
            
    def getCount(self, nPCs = None, varianceExplained = None):
        if nPCs == None and varianceExplained == None:
            nPCs = len(self.eigValues)
        if varianceExplained == None:
            nPCs = min(nPCs, len(self.eigVectors))
        else:
            total = 0; nPCs = 0
            while total < varianceExplained and ind < len(self.eigValues):
                total += self.eigValues[ind]
                nPCs += 1
        return nPCs
        
    def getEigValues(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        nPCs = self.getCount(nPCs, varianceExplained)
        return self.eigValues[:nPCs]


    def getEigVectors(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        nPCs = self.getCount(nPCs, varianceExplained)
        return MA.take(self.eigVectors, range(nPCs), axis=1)

            
    def getProjectedData(self, nPCs = None, varianceExplained = None):
        if not self.data: return None
        self.compute()
        
        vectors = self.getEigVectors(nPCs, varianceExplained)
        return MA.dot(self.normalizedData, vectors)
    
    def getExampleTable(self, nPCs = None, varianceExplained = None, keepOriginal = 1):
        if not self.data or type(self.data) != orange.ExampleTable: return None
        count = self.getCount(nPCs, varianceExplained)
        projData = self.getProjectedData(nPCs = count)
        newDomain = orange.Domain([orange.FloatVariable("Component %d" % (d+1)) for d in range(count)], 0)
        newTable = orange.ExampleTable(newDomain, projData.data)
        if keepOriginal:
            return orange.ExampleTable([self.data, newTable])
        else:
            if self.data.domain.classVar:
                return orange.ExampleTable([self.data.select([self.data.domain.classVar]), newTable])
            else:
                return newTable

        
def PCAOnExampleTable(table, keepOriginal = 1, nPCs = -1):
    data = table.toNumpyMA("a")[0]
    projData, vectors, values = pca(data, nPCs)
    newDomain = orange.Domain([orange.FloatVariable("PC %d" % (d+1)) for d in range(len(vectors))], 0)
    newTable = orange.ExampleTable(newDomain, projData.data)
    if keepOriginal:
        return orange.ExampleTable([table, newTable])
    else:
        return newTable
    

def pca(data, nPCs = -1):
    domain = None
    
    suma = data.sum(axis=0)/float(len(data))
    data -= suma       # substract average value to get zero mean
    data /= MA.std(data, axis=0)
    covMatrix = MA.dot(data.T, data) / len(data)

    eigVals, eigVectors = linalg.eigh(covMatrix)
    eigVals = list(eigVals)
    
    if nPCs == -1:
        nPCs = len(eigVals)
    nPCs = min(nPCs, len(eigVals))
    
    pairs = [(val, i) for i, val in enumerate(eigVals)]
    pairs.sort()
    pairs.reverse()
    indices = [pair[1] for pair in pairs[:nPCs]]  # take indices of the wanted number of principal components

    vectors = MA.take(eigVectors, indices, axis = 1)
    values = [eigVals[i] for i in indices]
    projectedData = MA.dot(data, vectors)
    
    return projectedData, vectors, values


if __name__ == "__main__":
    #d = orange.Domain([orange.FloatVariable("X1"), orange.FloatVariable("X2")], 0)
    #data = orange.ExampleTable(d, [[x,x] for x in range(100)])
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    x = PCAOnExampleTable(data)

    data2 = data.toNumpyMA("a")[0]
    projData, vectors, values = pca(data2)
    print vectors 
    print values
    
    instance = PCA(data)
    print instance.getEigValues()
    print instance.getEigVectors()
    print instance.getProjectedData()
    
    fda = FDA(data)
    print fda.getProjectedData()