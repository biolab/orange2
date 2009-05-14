import orange, numpy

def exampleTableToNumpy(data, what="ac"):
    return data.toNumpyMA(what)[0]


def PCAOnExampleTable(table, keepOriginal = 1, nPCs = -1):
    data = exampleTableToNumpy(table, "a")
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
    data /= numpy.ma.std(data, axis=0)
    covMatrix = numpy.ma.dot(data.T, data) / len(data)

    eigVals, eigVectors = numpy.linalg.eigh(covMatrix)
    eigVals = list(eigVals)
    
    if nPCs == -1:
        nPCs = len(eigVals)
    nPCs = min(nPCs, len(eigVals))
    
    pairs = [(val, i) for i, val in enumerate(eigVals)]
    pairs.sort()
    pairs.reverse()
    indices = [pair[1] for pair in pairs[:nPCs]]  # take indices of the wanted number of principal components

    vectors = numpy.take(eigVectors, indices, axis = 1)
    values = [eigVals[i] for i in indices]
    projectedData = numpy.ma.dot(data, vectors)
    
    return projectedData, vectors, values


if __name__ == "__main__":
    #d = orange.Domain([orange.FloatVariable("X1"), orange.FloatVariable("X2")], 0)
    #data = orange.ExampleTable(d, [[x,x] for x in range(100)])
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    x = PCAOnExampleTable(data)

    data = exampleTableToNumpy(data, "a")
    projData, vectors, values = pca(data, 1)
    print vectors, values