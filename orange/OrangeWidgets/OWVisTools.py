import orange, Numeric, LinearAlgebra

"""
# compute how good can we separate classes using fisher discriminant analysis
def computeFisherQuality(exampleTable):    
    exampleTables = {}
    for val in exampleTable.domain.classVar.values:
        data = exampleTable.select({exampleTable.domain.classVar.name: [val]})
        data = orange.Preprocessor_dropMissing(data)  # remove missing values
        numData = data.toNumeric("a", 0, 1, 1e20)[0]
        exampleTables[val] = numData

    attrNum = Numeric.shape(exampleTable1)[1]
    Sw = Numeric.zeros([attrNum, attrNum], Numeric.Float)
    
    for i in range(len(exampleTables.keys())):
        tab = exampleTables[exampleTables.keys()[i]]
        t = tab * tab
        t2 = Numeric.transpose(tab)
        s = Numeric.sum(t)
        for i in range(attrNum):
            Sw[i,i] += s[i]
            for j in range(i+1, attrNum):
                val = t2[i]*t2[j]
                Sw[i,j] += val
                Sw[j,i] += val

    SwInv = Numeric.invert(Sw)
    m1 = Numeric.average(exampleTable1)
    m2 = Numeric.average(exampleTable2)
    w = Numeric.matrixmultiply(SwInv, (m2-m1))      # a line vector representing the discriminant function
    return w


        
        data1 = exampleTables[exampleTables.keys()[i]]
        for j in range(i+1, len(exampleTables.keys())):
            data2 = exampleTables[exampleTables.keys()[j]]
            w = computeFisherDirection(data1, data2)
            projData1 = Numeric.matrixmultiply(data1, w)
            projData2 = Numeric.matrixmultiply(data2, w)
            ave1 = Numeric.average(projData1)
            ave2 = Numeric.average(projData2)
            d1 = projData1 - ave1
            d2 = projData2 - ave2
            dev1 = Numeric.sqrt(Numeric.sum(d1*d1))
            dev2 = Numeric.sqrt(Numeric.sum(d2*d2))
            


# compute direction of the vector that will maximally separate clusters with class values classValue1 and classValue2
# this direction is the solution to the Rayleigh quotient
def computeFisherDirection(exampleTable1, exampleTable2):
    attrNum = Numeric.shape(exampleTable1)[1]
    Sw = Numeric.zeros([attrNum, attrNum], Numeric.Float)

    # for both example tables compute the scatter within matrices and add them
    for tab in [exampleTable1, exampleTable2]:  
        t = tab * tab
        t2 = Numeric.transpose(tab)
        s = Numeric.sum(t)
        for i in range(attrNum):
            Sw[i,i] += s[i]
            for j in range(i+1, attrNum):
                val = t2[i]*t2[j]
                Sw[i,j] += val
                Sw[j,i] += val

    SwInv = Numeric.invert(ret)
    m1 = Numeric.average(exampleTable1)
    m2 = Numeric.average(exampleTable2)
    w = Numeric.matrixmultiply(SwInv, (m2-m1))      # a line vector representing the discriminant function
    return w
"""


# compute how good can we separate classes using fisher discriminant analysis
def computeFisherQuality(exampleTable):    
    exampleTables = {}
    for val in exampleTable.domain.classVar.values:
        data = exampleTable.select({exampleTable.domain.classVar.name: [val]})
        data = orange.Preprocessor_dropMissing(data)  # remove missing values
        if len(data) > 0:
            numData = data.toNumeric("a", 0, 1, 1e20)[0]
            exampleTables[val] = numData

    projVal = 0.0
    for i in range(len(exampleTables.keys())):
        data1 = exampleTables[exampleTables.keys()[i]]
        vals = []
        for j in range(i+1, len(exampleTables.keys())):
            data2 = exampleTables[exampleTables.keys()[j]]
            w = computeFisherDirection(data1, data2)
            projData1 = Numeric.matrixmultiply(data1, w)
            projData2 = Numeric.matrixmultiply(data2, w)
            ave1 = Numeric.average(projData1)
            ave2 = Numeric.average(projData2)
            d1 = projData1 - ave1
            d2 = projData2 - ave2
            dev1 = Numeric.sum(d1*d1)
            dev2 = Numeric.sum(d2*d2)
            if dev1 + dev2 != 0.0:
                vals.append((ave1-ave2)*(ave1-ave2) / (dev1 + dev2))
            else:
                vals.append((ave1-ave2)*(ave1-ave2))
        if vals != []: projVal += min(vals)
            
    return projVal

# compute direction of the vector that will maximally separate clusters with class values classValue1 and classValue2
# this direction is the solution to the Rayleigh quotient
def computeFisherDirection(exampleTable1, exampleTable2):
    attrNum = Numeric.shape(exampleTable1)[1]
    Sw = Numeric.zeros([attrNum, attrNum], Numeric.Float)

    # for both example tables compute the scatter within matrices and add them
    for tab in [exampleTable1, exampleTable2]:
        tab2 = tab - Numeric.average(tab)
        t = tab2 * tab2
        t2 = Numeric.transpose(tab2)
        s = Numeric.sum(t)
        for i in range(attrNum):
            Sw[i,i] += s[i]
            for j in range(i+1, attrNum):
                val = Numeric.sum(t2[i]*t2[j])
                Sw[i,j] += val
                Sw[j,i] += val

    SwInv = LinearAlgebra.inverse(Sw)
    m1 = Numeric.average(exampleTable1)
    m2 = Numeric.average(exampleTable2)
    w = Numeric.matrixmultiply(SwInv, (m2-m1))      # a line vector representing the discriminant function
    return w


if __name__== "__main__":
    """
    data = orange.ExampleTable(r"c:\Development\Python23\Lib\site-packages\Orange\datasets\UCI\iris.tab")
    d2 = data.select(["petal width", "sepal width", "iris"])
    d3 = d2.select({d2.domain.classVar.name: ["Iris-setosa"]})
    d4 = d2.select({d2.domain.classVar.name: ["Iris-virginica"]})
    nd3 = d3.toNumeric("a")[0]
    nd4 = d4.toNumeric("a")[0]
    w = computeFisherDirection(nd3, nd4)
    print w
    """
    #data = orange.ExampleTable(r"c:\Development\Python23\Lib\site-packages\Orange\datasets\microarray\cancer diagnostics\MLL-short-2class.tab" )
    data = orange.ExampleTable(r"c:\Development\Python23\Lib\site-packages\Orange\datasets\test.tab" )
    #data = data.select([0,1,data.domain.classVar.name]) 
    val = computeFisherQuality(data)
