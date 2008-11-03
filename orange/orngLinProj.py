import orangeom, orange
import math, random, numpy, MA
from numpy.linalg import inv, eig      # matrix inverse and eigenvectors
from orngScaleLinProjData import orngScaleLinProjData
import orngVisFuncts

#implementation
FAST_IMPLEMENTATION = 0
SLOW_IMPLEMENTATION = 1
LDA_IMPLEMENTATION = 2

LAW_LINEAR = 0
LAW_SQUARE = 1
LAW_GAUSSIAN = 2
LAW_KNN = 3
LAW_LINEAR_PLUS = 4

DR_PCA = 0
DR_SPCA = 1
DR_PLS = 2

def normalize(x):
    return x / numpy.linalg.norm(x)

def center(matrix):
    '''centers all variables, i.e. subtracts averages in colomns
    and divides them by their standard deviations'''
    n,m = numpy.shape(matrix)
    return (matrix - numpy.multiply(matrix.mean(axis = 0), numpy.ones((n,m))))/numpy.std(matrix, axis = 0)


class FreeViz:
    def __init__(self, graph = None):
        if not graph:
            graph = orngScaleLinProjData()
        self.graph = graph

        self.implementation = 0
        self.attractG = 1.0
        self.repelG = 1.0
        self.law = LAW_LINEAR
        self.restrain = 0
        self.forceBalancing = 0
        self.forceSigma = 1.0
        self.mirrorSymmetry = 1
        self.useGeneralizedEigenvectors = 1

        # s2n heuristics parameters
        self.stepsBeforeUpdate = 10
        self.s2nSpread = 5
        self.s2nPlaceAttributes = 50
        self.s2nMixData = None
        self.autoSetParameters = 1
        self.classPermutationList = None
        self.attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000]
        #attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000, 10000, 50000]

    def clearData(self):
        self.s2nMixData = None
        self.classPermutationList = None

    def setStatusBarText(self, *args):
        pass

    def showAllAttributes(self):
        self.graph.anchorData = [(0,0, a.name) for a in self.graph.dataDomain.attributes]
        self.radialAnchors()

    def getShownAttributeList(self):
        return [anchor[2] for anchor in self.graph.anchorData]

    def radialAnchors(self):
        attrList = self.getShownAttributeList()
        if not attrList: return
        phi = 2*math.pi/len(attrList)
        self.graph.anchorData = [(math.cos(i*phi), math.sin(i*phi), a) for i, a in enumerate(attrList)]


    def randomAnchors(self):
        if not self.graph.haveData: return
        attrList = self.getShownAttributeList()
        if not attrList: return

        if self.restrain == 0:
            def ranch(i, label):
                r = 0.3+0.7*random.random()
                phi = 2*math.pi*random.random()
                return (r*math.cos(phi), r*math.sin(phi), label)

        elif self.restrain == 1:
            def ranch(i, label):
                phi = 2*math.pi*random.random()
                return (math.cos(phi), math.sin(phi), label)

        else:
            def ranch(i, label):
                r = 0.3+0.7*random.random()
                phi = 2*math.pi * i / max(1, len(attrList))
                return (r*math.cos(phi), r*math.sin(phi), label)

        anchors = [ranch(*a) for a in enumerate(attrList)]

        if not self.restrain == 1:
            maxdist = math.sqrt(max([x[0]**2+x[1]**2 for x in anchors]))
            anchors = [(x[0]/maxdist, x[1]/maxdist, x[2]) for x in anchors]

        if not self.restrain == 2 and self.mirrorSymmetry:
            #### Need to rotate and mirror here
            pass

        self.graph.anchorData = anchors

    def optimizeSeparation(self, steps = 10, singleStep = False):
        # check if we have data and a discrete class
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass:
            return
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
        if not attrIndices: return

        if self.implementation == FAST_IMPLEMENTATION:
            return self.optimize_FAST_Separation(steps, singleStep)

        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp
        if singleStep: steps = 1
        if self.implementation == SLOW_IMPLEMENTATION:  impl = self.optimize_SLOW_Separation
        elif self.implementation == LDA_IMPLEMENTATION: impl = self.optimize_LDA_Separation
        XAnchors = None; YAnchors = None

        for c in range((singleStep and 1) or 50):
            for i in range(steps):
                if self.__class__ != FreeViz and self.cancelOptimization == 1: return
                self.graph.anchorData, (XAnchors, YAnchors) = impl(attrIndices, self.graph.anchorData, XAnchors, YAnchors)
            if self.graph.__class__ != FreeViz:
                qApp.processEvents()
                self.graph.updateData()
            #self.recomputeEnergy()

    def optimize_FAST_Separation(self, steps = 10, singleStep = False):
        optimizer = [orangeom.optimizeAnchors, orangeom.optimizeAnchorsRadial, orangeom.optimizeAnchorsR][self.restrain]
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
        if not attrIndices: return

        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        positions = [numpy.array([x[:2] for x in self.graph.anchorData])]
        neededSteps = 0

        validData = self.graph.getValidList(attrIndices)
        if sum(validData) == 0:
            return 0

        data = numpy.compress(validData, self.graph.noJitteringScaledData, axis=1)
        data = numpy.transpose(data).tolist()
        classes = numpy.compress(validData, self.graph.originalData[self.graph.dataClassIndex]).tolist()
        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp

        while 1:
            self.graph.anchorData = optimizer(data, classes, self.graph.anchorData, attrIndices,
                                              attractG = self.attractG, repelG = self.repelG, law = self.law,
                                              sigma2 = self.forceSigma, dynamicBalancing = self.forceBalancing, steps = steps,
                                              normalizeExamples = self.graph.normalizeExamples,
                                              contClass = self.graph.dataHasContinuousClass,
                                              mirrorSymmetry = self.mirrorSymmetry)
            neededSteps += steps

            if self.graph.__class__ != FreeViz:
                qApp.processEvents()
                self.graph.potentialsBmp = None
                self.graph.updateData()

            positions = positions[-49:]+[numpy.array([x[:2] for x in self.graph.anchorData])]
            if len(positions)==50:
                m = max(numpy.sum((positions[0]-positions[49])**2), 0)
                if m < 1e-3: break
            if singleStep or (self.__class__ != FreeViz and self.cancelOptimization):
                break
        return neededSteps

    def optimize_LDA_Separation(self, attrIndices, anchorData, XAnchors = None, YAnchors = None):
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass: 
            return anchorData, (XAnchors, YAnchors)
        classCount = len(self.graph.dataDomain.classVar.values)
        validData = self.graph.getValidList(attrIndices)
        selectedData = numpy.compress(validData, numpy.take(self.graph.noJitteringScaledData, attrIndices, axis = 0), axis = 1)

        if XAnchors == None:
            XAnchors = numpy.array([a[0] for a in anchorData], numpy.float)
        if YAnchors == None:
            YAnchors = numpy.array([a[1] for a in anchorData], numpy.float)

        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, validData = validData, XAnchors = XAnchors, YAnchors = YAnchors, scaleFactor = self.graph.scaleFactor, normalize = self.graph.normalizeExamples, useAnchorData = 1)
        if transProjData == None:
            return anchorData, (XAnchors, YAnchors)

        projData = numpy.transpose(transProjData)
        x_positions, y_positions, classData = projData[0], projData[1], projData[2]

        averages = []
        for i in range(classCount):
            ind = classData == i
            xpos = numpy.compress(ind, x_positions);  ypos = numpy.compress(ind, y_positions)
            xave = numpy.sum(xpos)/len(xpos);         yave = numpy.sum(ypos)/len(ypos)
            averages.append((xave, yave))

        # compute the positions of all the points. we will try to move all points so that the center will be in the (0,0)
        xCenterVector = -numpy.sum(x_positions) / len(x_positions)
        yCenterVector = -numpy.sum(y_positions) / len(y_positions)
        centerVectorLength = math.sqrt(xCenterVector*xCenterVector + yCenterVector*yCenterVector)

        meanDestinationVectors = []

        for i in range(classCount):
            xDir = 0.0; yDir = 0.0; rs = 0.0
            for j in range(classCount):
                if i==j: continue
                r = math.sqrt((averages[i][0] - averages[j][0])**2 + (averages[i][1] - averages[j][1])**2)
                if r == 0.0:
                    xDir += math.cos((i/float(classCount))*2*math.pi)
                    yDir += math.sin((i/float(classCount))*2*math.pi)
                    r = 0.0001
                else:
                    xDir += (1/r**3) * ((averages[i][0] - averages[j][0]))
                    yDir += (1/r**3) * ((averages[i][1] - averages[j][1]))
                #rs += 1/r
            #actualDirAmpl = math.sqrt(xDir**2 + yDir**2)
            #s = abs(xDir)+abs(yDir)
            #xDir = rs * (xDir/s)
            #yDir = rs * (yDir/s)
            meanDestinationVectors.append((xDir, yDir))


        maxLength = math.sqrt(max([x**2 + y**2 for (x,y) in meanDestinationVectors]))
        meanDestinationVectors = [(x/(2*maxLength), y/(2*maxLength)) for (x,y) in meanDestinationVectors]     # normalize destination vectors to some normal values
        meanDestinationVectors = [(meanDestinationVectors[i][0]+averages[i][0], meanDestinationVectors[i][1]+averages[i][1]) for i in range(len(meanDestinationVectors))]    # add destination vectors to the class averages
        #meanDestinationVectors = [(x + xCenterVector/5, y + yCenterVector/5) for (x,y) in meanDestinationVectors]   # center mean values
        meanDestinationVectors = [(x + xCenterVector, y + yCenterVector) for (x,y) in meanDestinationVectors]   # center mean values

        FXs = numpy.zeros(len(x_positions), numpy.float)        # forces
        FYs = numpy.zeros(len(x_positions), numpy.float)

        for c in range(classCount):
            ind = (classData == c)
            numpy.putmask(FXs, ind, meanDestinationVectors[c][0] - x_positions)
            numpy.putmask(FYs, ind, meanDestinationVectors[c][1] - y_positions)

        # compute gradient for all anchors
        GXs = numpy.array([sum(FXs * selectedData[i]) for i in range(len(anchorData))], numpy.float)
        GYs = numpy.array([sum(FYs * selectedData[i]) for i in range(len(anchorData))], numpy.float)

        m = max(max(abs(GXs)), max(abs(GYs)))
        GXs /= (20*m); GYs /= (20*m)

        newXAnchors = XAnchors + GXs
        newYAnchors = YAnchors + GYs

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newXAnchors**2 + newYAnchors**2))
        newXAnchors /= m
        newYAnchors /= m

        #self.parentWidget.updateGraph()

        """
        for a in range(len(anchorData)):
            x = anchorData[a][0]; y = anchorData[a][1];
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [x, x+GXs[a]], yData = [y, y+GYs[a]], forceFilledSymbols = 1, lineWidth=3)

        for i in range(classCount):
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [averages[i][0], meanDestinationVectors[i][0]], yData = [averages[i][1], meanDestinationVectors[i][1]], forceFilledSymbols = 1, lineWidth=3)
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, xData = [averages[i][0], averages[i][0]], yData = [averages[i][1], averages[i][1]], forceFilledSymbols = 1, lineWidth=5)
        """
        #self.parentWidget.graph.repaint()
        #self.graph.anchorData = [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))]
        #self.graph.updateData(attrs, 0)
        return [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))], (newXAnchors, newYAnchors)


    def optimize_SLOW_Separation(self, attrIndices, anchorData, XAnchors = None, YAnchors = None):
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass: 
            return anchorData, (XAnchors, YAnchors)
        validData = self.graph.getValidList(attrIndices)
        selectedData = numpy.compress(validData, numpy.take(self.graph.noJitteringScaledData, attrIndices, axis = 0), axis = 1)

        if XAnchors == None:
            XAnchors = numpy.array([a[0] for a in anchorData], numpy.float)
        if YAnchors == None:
            YAnchors = numpy.array([a[1] for a in anchorData], numpy.float)

        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, validData = validData, XAnchors = XAnchors, YAnchors = YAnchors, scaleFactor = self.graph.scaleFactor, normalize = self.graph.normalizeExamples, useAnchorData = 1)
        if transProjData == None:
            return anchorData, (XAnchors, YAnchors)

        projData = numpy.transpose(transProjData)
        x_positions = projData[0]; x_positions2 = numpy.array(x_positions)
        y_positions = projData[1]; y_positions2 = numpy.array(y_positions)
        classData = projData[2]  ; classData2 = numpy.array(classData)

        FXs = numpy.zeros(len(x_positions), numpy.float)        # forces
        FYs = numpy.zeros(len(x_positions), numpy.float)
        GXs = numpy.zeros(len(anchorData), numpy.float)        # gradients
        GYs = numpy.zeros(len(anchorData), numpy.float)

        rotateArray = range(len(x_positions)); rotateArray = rotateArray[1:] + [0]
        for i in range(len(x_positions)-1):
            x_positions2 = numpy.take(x_positions2, rotateArray)
            y_positions2 = numpy.take(y_positions2, rotateArray)
            classData2 = numpy.take(classData2, rotateArray)
            dx = x_positions2 - x_positions
            dy = y_positions2 - y_positions
            rs2 = dx**2 + dy**2
            rs2 += numpy.where(rs2 == 0.0, 0.0001, 0.0)    # replace zeros to avoid divisions by zero
            rs = numpy.sqrt(rs2)

            F = numpy.zeros(len(x_positions), numpy.float)
            classDiff = numpy.where(classData == classData2, 1, 0)
            numpy.putmask(F, classDiff, 150*self.attractG*rs2)
            numpy.putmask(F, 1-classDiff, -self.repelG/rs2)
            FXs += F * dx / rs
            FYs += F * dy / rs

        # compute gradient for all anchors
        GXs = numpy.array([sum(FXs * selectedData[i]) for i in range(len(anchorData))], numpy.float)
        GYs = numpy.array([sum(FYs * selectedData[i]) for i in range(len(anchorData))], numpy.float)

        m = max(max(abs(GXs)), max(abs(GYs)))
        GXs /= (20*m); GYs /= (20*m)

        newXAnchors = XAnchors + GXs
        newYAnchors = YAnchors + GYs

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newXAnchors**2 + newYAnchors**2))
        newXAnchors /= m
        newYAnchors /= m
        return [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))], (newXAnchors, newYAnchors)


    # ###############################################################
    # S2N HEURISTIC FUNCTIONS
    # ###############################################################

    # if autoSetParameters is set then try different values for parameters and see how good projection do we get
    # if not then just use current parameters to place anchors
    def s2nMixAnchorsAutoSet(self):
        # check if we have data and a discrete class
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass:
            self.setStatusBarText("No data or data without a discrete class") 
            return

        import orngVizRank
        if self.graph.normalizeExamples:
            vizrank = orngVizRank.VizRank(orngVizRank.RADVIZ)
        else:
            vizrank = orngVizRank.VizRank(orngVizRank.LINEAR_PROJECTION)
        vizrank.qualityMeasure = orngVizRank.AVERAGE_CORRECT
        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp

        if self.autoSetParameters:
            results = {}
            self.s2nSpread = 0
            permutations = orngVisFuncts.generateDifferentPermutations(range(len(self.graph.dataDomain.classVar.values)))
            for perm in permutations:
                self.classPermutationList = perm
                for val in self.attrsNum:
                    if self.attrsNum[self.attrsNum.index(val)-1] > len(self.graph.dataDomain.attributes): continue    # allow the computations once
                    self.s2nPlaceAttributes = val
                    if not self.s2nMixAnchors(0):
                        return
                    if self.__class__ != FreeViz:
                        qApp.processEvents()

                    acc, other = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(None, useAnchorData = 1))
                    if results.keys() != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results.keys())))
                    else:                    self.setStatusBarText("Current projection value is %.2f" % (acc))

                    results[acc] = (perm, val)
            if results.keys() == []: return
            self.classPermutationList, self.s2nPlaceAttributes = results[max(results.keys())]
            if self.__class__ != FreeViz:
                qApp.processEvents()
            if not self.s2nMixAnchors(0):        # update the best number of attributes
                return

            results = []
            anchors = self.graph.anchorData
            attributeNameIndex = self.graph.attributeNameIndex
            attrIndices = [attributeNameIndex[val[2]] for val in anchors]
            for val in range(10):
                self.s2nSpread = val
                if not self.s2nMixAnchors(0):
                    return
                acc, other = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(attrIndices, useAnchorData = 1))
                results.append(acc)
                if results != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results)))
                else:             self.setStatusBarText("Current projection value is %.2f" % (acc))
            self.s2nSpread = results.index(max(results))

            self.setStatusBarText("Best projection value is %.2f" % (max(results)))

        # always call this. if autoSetParameters then because we need to set the attribute list in radviz. otherwise because it finds the best attributes for current settings
        self.s2nMixAnchors()


    # place a subset of attributes around the circle. this subset must contain "good" attributes for each of the class values
    def s2nMixAnchors(self, setAttributeListInRadviz = 1):
        # check if we have data and a discrete class
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass: 
            self.setStatusBarText("S2N only works on data with a discrete class value")
            return

        # compute the quality of attributes only once
        if self.s2nMixData == None:
            rankedAttrs, rankedAttrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.graph.rawData, orngVisFuncts.S2NMeasureMix())
            self.s2nMixData = (rankedAttrs, rankedAttrsByClass)
            classCount = len(rankedAttrsByClass)
            attrs = rankedAttrs[:(self.s2nPlaceAttributes/classCount)*classCount]    # select appropriate number of attributes
        else:
            classCount = len(self.s2nMixData[1])
            attrs = self.s2nMixData[0][:(self.s2nPlaceAttributes/classCount)*classCount]

        if len(attrs) == 0:
            self.setStatusBarText("No discrete attributes found")
            return 0

        arr = [0]       # array that will tell where to put the next attribute
        for i in range(1,len(attrs)/2): arr += [i,-i]

        phi = (2*math.pi*self.s2nSpread)/(len(attrs)*10.0)
        anchorData = []; start = []
        arr2 = arr[:(len(attrs)/classCount)+1]
        for cls in range(classCount):
            startPos = (2*math.pi*cls)/classCount
            if self.classPermutationList: cls = self.classPermutationList[cls]
            attrsCls = attrs[cls::classCount]
            tempData = [(arr2[i], math.cos(startPos + arr2[i]*phi), math.sin(startPos + arr2[i]*phi), attrsCls[i]) for i in range(min(len(arr2), len(attrsCls)))]
            start.append(len(anchorData) + len(arr2)/2) # starting indices for each class value
            tempData.sort()
            anchorData += [(x, y, name) for (i, x, y, name) in tempData]

        anchorData = anchorData[(len(attrs)/(2*classCount)):] + anchorData[:(len(attrs)/(2*classCount))]
        self.graph.anchorData = anchorData
        attrNames = [anchor[2] for anchor in anchorData]

        if self.__class__ != FreeViz:
            if setAttributeListInRadviz:
                self.parentWidget.setShownAttributeList(attrNames)
            self.graph.updateData(attrNames)
            self.graph.repaint()
        return 1

    # find interesting linear projection using PCA, SPCA, or PLS
    def findProjection(self, method, attrIndices = None, setAnchors = 0, percentDataUsed = 100):
        if not self.graph.haveData: return
        ai = self.graph.attributeNameIndex
        if attrIndices == None:
            attributes = self.getShownAttributeList()
            attrIndices = [ai[label] for label in attributes]
        if len(attrIndices) == 0: return None

        validData = self.graph.getValidList(attrIndices)
        if sum(validData) == 0: return None

        dataMatrix = numpy.compress(validData, numpy.take(self.graph.noJitteringScaledData, attrIndices, axis = 0), axis = 1)
        if self.graph.dataHasClass:
            classArray = numpy.compress(validData, self.graph.noJitteringScaledData[self.graph.dataClassIndex])

        if percentDataUsed != 100:
            indices = orange.MakeRandomIndices2(self.graph.rawData, 1.0-(float(percentDataUsed)/100.0))
            try:
                dataMatrix = numpy.compress(indices, dataMatrix, axis = 1)
            except:
                pass
            if self.graph.dataHasClass:
                classArray = numpy.compress(indices, classArray)

        vectors = None
        if method == DR_PCA:
            vectors = FreeViz.findSPCAProjection(self, dataMatrix, classArray, SPCA = 0)
        elif method == DR_SPCA and self.graph.dataHasClass:
            vectors = FreeViz.findSPCAProjection(self, dataMatrix, classArray, SPCA = 1)
        elif method == DR_PLS and self.graph.dataHasClass:
            dataMatrix = dataMatrix.transpose()
            classMatrix = numpy.transpose(numpy.matrix(classArray))
            vectors = FreeViz.findPLSProjection(self, dataMatrix, classMatrix, 2)
            vectors = vectors.T

        # test if all values are 0, if there is an invalid number in the array and if there are complex numbers in the array
        if vectors == None or not vectors.any() or False in numpy.isfinite(vectors) or False in numpy.isreal(vectors):
            self.setStatusBarText("Unable to compute anchor positions for the selected attributes")  
            return None

        xAnchors = vectors[0]
        yAnchors = vectors[1]

        m = math.sqrt(max(xAnchors**2 + yAnchors**2))

        xAnchors /= m
        yAnchors /= m
        names = self.graph.attributeNames
        attributes = [names[attrIndices[i]] for i in range(len(attrIndices))]

        if setAnchors:
            self.graph.setAnchors(list(xAnchors), list(yAnchors), attributes)
            self.graph.updateData()
            self.graph.repaint()
        return xAnchors, yAnchors, (attributes, attrIndices)



    def findPLSProjection(self, X,Y,Ncomp):
        '''Predict Y from X using first Ncomp principal components'''

        # data dimensions
        n, mx = numpy.shape(X)
        my = numpy.shape(Y)[1]

        # Z-scores of original matrices
        YMean = Y.mean()
        X,Y = center(X), center(Y)

        P = numpy.empty((mx,Ncomp))
        W = numpy.empty((mx,Ncomp))
        C = numpy.empty((my,Ncomp))
        T = numpy.empty((n,Ncomp))
        U = numpy.empty((n,Ncomp))
        B = numpy.zeros((Ncomp,Ncomp))

        E,F = X,Y

        # main algorithm
        for i in range(Ncomp):

            u = numpy.random.random_sample((n,1))
            w = normalize(numpy.dot(E.T,u))
            t = normalize(numpy.dot(E,w))
            c = normalize(numpy.dot(F.T,t))

            dif = t
            # iterations for loading vector t
            while numpy.linalg.norm(dif) > 10e-16:
                c = normalize(numpy.dot(F.T,t))
                u = numpy.dot(F,c)
                w = normalize(numpy.dot(E.T,u))
                t0 = normalize(numpy.dot(E,w))
                dif = t - t0
                t = t0

            T[:,i] = t.T
            U[:,i] = u.T
            C[:,i] = c.T
            W[:,i] = w.T

            b = numpy.dot(t.T,u)[0,0]
            B[i][i] = b
            p = numpy.dot(E.T,t)
            P[:,i] = p.T
            E = E - numpy.dot(t,p.T)
            xx = b * numpy.dot(t,c.T)
            F = F - xx

        # esimated Y
        #YE = numpy.dot(numpy.dot(T,B),C.T)*numpy.std(Y, axis = 0) + YMean
        #Y = Y*numpy.std(Y, axis = 0)+ YMean
        #BPls = numpy.dot(numpy.dot(numpy.linalg.pinv(P.T),B),C.T)

        return W

    def findSPCAProjection(self, dataMatrix, classArray, SPCA = 1):
        try:
            dataMatrix = numpy.transpose(dataMatrix)

            s = numpy.sum(dataMatrix, axis=0)/float(len(dataMatrix))
            dataMatrix -= s       # substract average value to get zero mean

            # define the Laplacian matrix
            L = numpy.zeros((len(dataMatrix), len(dataMatrix)))
            for i in range(len(dataMatrix)):
                for j in range(i+1, len(dataMatrix)):
                    L[i,j] = -int(classArray[i] != classArray[j])
                    L[j,i] = -int(classArray[i] != classArray[j])

            s = numpy.sum(L, axis=0)      # doesn't matter which axis since the matrix L is symmetrical
            for i in range(len(dataMatrix)):
                L[i,i] = -s[i]

            if self.useGeneralizedEigenvectors:
                covarMatrix = numpy.dot(numpy.transpose(dataMatrix), dataMatrix)
                matrix = inv(covarMatrix)
                matrix = numpy.dot(matrix, numpy.transpose(dataMatrix))
            else:
                matrix = numpy.transpose(dataMatrix)

            # compute dataMatrixT * L * dataMatrix
            if SPCA:
                matrix = numpy.dot(matrix, L)

            matrix = numpy.dot(matrix, dataMatrix)

            vals, vectors = eig(matrix)
            firstInd  = list(vals).index(max(vals))     # save the index of the largest eigenvector
            vals[firstInd] = -1
            secondInd = list(vals).index(max(vals));    # save the index of the second largest eigenvector

            vectors = vectors.transpose()
            return numpy.take(vectors, [firstInd, secondInd], axis = 0)
        except:
            return None


# #############################################################################
# class that represents FreeViz classifier
class FreeVizClassifier(orange.Classifier):
    def __init__(self, data, freeviz):
        self.FreeViz = freeviz

        if self.FreeViz.__class__ != FreeViz:
            self.FreeViz.parentWidget.setData(data)
            self.FreeViz.parentWidget.showAllAttributes = 1
        else:
            self.FreeViz.graph.setData(data)
            self.FreeViz.showAllAttributes()

        #self.FreeViz.randomAnchors()
        self.FreeViz.radialAnchors()
        self.FreeViz.optimizeSeparation()

        graph = self.FreeViz.graph
        ai = graph.attributeNameIndex
        labels = [a[2] for a in graph.anchorData]
        domain = orange.Domain(labels+[graph.dataDomain.classVar], graph.dataDomain)
        indices = [ai[label] for label in labels]
        offsets = [graph.domainDataStat[i].min for i in indices]
        normalizers = [graph.getMinMaxVal(i) for i in indices]
        #averages = [graph.averages[i] for i in indices]
        averages = MA.filled(MA.average(graph.originalData.take(indices, axis =0), 1), 1)

        #self.FreeViz.graph.createProjectionAsNumericArray(indices, useAnchorData = 1)  # Janez: why would you call this function if you don't want its result???
        self.classifier = orange.P2NN(domain,
                                      numpy.transpose(numpy.array([graph.unscaled_x_positions, graph.unscaled_y_positions, graph.originalData[graph.dataClassIndex]])),
                                      graph.anchorData, offsets, normalizers, averages, graph.normalizeExamples, law=self.FreeViz.law)

    # for a given example run argumentation and find out to which class it most often fall
    def __call__(self, example, returnType):
        #example.setclass(0)
        return self.classifier(example, returnType)


class FreeVizLearner(orange.Learner):
    def __init__(self, freeviz = None):
        if not freeviz:
            freeviz = FreeViz()
        self.FreeViz = freeviz
        self.name = "FreeViz Learner"

    def __call__(self, examples, weightID = 0):
        return FreeVizClassifier(examples, self.FreeViz)



# #############################################################################
# class that represents S2N Heuristic classifier
class S2NHeuristicClassifier(orange.Classifier):
    def __init__(self, data, freeviz):
        self.FreeViz = freeviz
        self.data = data

        if self.FreeViz.__class__ != FreeViz:
            self.FreeViz.parentWidget.setData(data)
        else:
            self.FreeViz.setData(data)
            self.FreeViz.graph.setData(data)

        self.FreeViz.s2nMixAnchorsAutoSet()

    def __call__(self, example, returnType):
        table = orange.ExampleTable(example.domain)
        table.append(example)

        if self.FreeViz.__class__ != FreeViz:
            self.FreeViz.parentWidget.subsetdata(table)      # show the example is we use the widget
        else:
            self.FreeViz.graph.setSubsetData(table)

        anchorData = self.FreeViz.graph.anchorData
        attributeNameIndex = self.FreeViz.graph.attributeNameIndex
        scaleFunction = self.FreeViz.graph.scaleExampleValue   # so that we don't have to search the dictionaries each time

        attrListIndices = [attributeNameIndex[val[2]] for val in anchorData]
        attrVals = [scaleFunction(example, index) for index in attrListIndices]

        table = self.FreeViz.graph.createProjectionAsExampleTable(attrListIndices, scaleFactor = self.FreeViz.graph.trueScaleFactor, useAnchorData = 1)
        kValue = int(math.sqrt(len(self.data)))
        knn = orange.kNNLearner(k = kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))

        [xTest, yTest] = self.FreeViz.graph.getProjectedPointPosition(attrListIndices, attrVals, useAnchorData = 1)
        classifier = knn(table, 0)
        (classVal, dist) = classifier(orange.Example(table.domain, [xTest, yTest, "?"]), orange.GetBoth)

        if returnType == orange.GetBoth: return classVal, dist
        else:                            return classVal


class S2NHeuristicLearner(orange.Learner):
    def __init__(self, freeviz = None):
        if not freeviz:
            freeviz = FreeViz()
        self.FreeViz = freeviz
        self.name = "S2N Feature Selection Learner"

    def __call__(self, examples, weightID = 0):
        return S2NHeuristicClassifier(examples, self.FreeViz)

