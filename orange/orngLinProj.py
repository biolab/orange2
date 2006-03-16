import orangeom, orange
import math, random, Numeric, LinearAlgebra
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
        self.rawdata = None

        # s2n heuristics parameters
        self.stepsBeforeUpdate = 10
        self.s2nSpread = 5
        self.s2nPlaceAttributes = 50
        self.s2nMixData = None
        self.autoSetParameters = 1
        self.classPermutationList = None
        self.attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000]
        #attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000, 10000, 50000]
        
    def setData(self, data):
        self.rawdata = data
        self.s2nMixData = None
        self.classPermutationList = None
        
    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subsetdata):
        self.subsetdata = subsetdata
    

    def showAllAttributes(self):
        self.graph.anchorData = [(0,0, a.name) for a in self.graph.rawdata.domain.attributes]
        self.radialAnchors()

    def getShownAttributeList(self):
        return [anchor[2] for anchor in self.graph.anchorData]

    def radialAnchors(self):
        attrList = self.getShownAttributeList()
        phi = 2*math.pi/len(attrList)
        self.graph.anchorData = [(math.cos(i*phi), math.sin(i*phi), a) for i, a in enumerate(attrList)]


    def randomAnchors(self):
        if not self.graph.rawdata: return

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
                phi = 2*math.pi * i / n
                return (r*math.cos(phi), r*math.sin(phi), label)

        anchors = [ranch(*a) for a in enumerate(self.getShownAttributeList())]

        if not self.restrain == 1:
            maxdist = math.sqrt(max([x[0]**2+x[1]**2 for x in anchors]))
            anchors = [(x[0]/maxdist, x[1]/maxdist, x[2]) for x in anchors]

        if not self.restrain == 2 and self.mirrorSymmetry:
            #### Need to rotate and mirror here
            pass
            
        self.graph.anchorData = anchors

    def optimizeSeparation(self, steps = 10, singleStep = False):
        if self.implementation == FAST_IMPLEMENTATION:
            return self.optimize_FAST_Separation(steps, singleStep)
        else:
            if singleStep: steps = 1
            if self.implementation == SLOW_IMPLEMENTATION:  impl = self.optimize_SLOW_Separation
            elif self.implementation == LDA_IMPLEMENTATION: impl = self.optimize_LDA_Separation
            ai = self.graph.attributeNameIndex
            attrIndices = [ai[label] for label in self.getShownAttributeList()]
            XAnchors = None; YAnchors = None
            if self.__class__ != FreeViz: from qt import qApp

            for c in range((singleStep and 1) or 50):                
                for i in range(steps):
                    if self.__class__ != FreeViz and self.cancelOptimization == 1: return
                    self.graph.anchorData, (XAnchors, YAnchors) = impl(attrIndices, self.graph.anchorData, XAnchors, YAnchors)
                if self.graph.__class__ != orngScaleLinProjData:
                    qApp.processEvents()
                    self.graph.updateData()
                #self.recomputeEnergy()

    def optimize_FAST_Separation(self, steps = 10, singleStep = False):
        classes = [int(x.getclass()) for x in self.graph.rawdata]
        optimizer = [orangeom.optimizeAnchors, orangeom.optimizeAnchorsRadial, orangeom.optimizeAnchorsR][self.restrain]
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
        if self.__class__ != FreeViz: from qt import qApp
       
        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        positions = [Numeric.array([x[:2] for x in self.graph.anchorData])]
        neededSteps = 0
        while 1:
            self.graph.anchorData = optimizer(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices,
                                              attractG = self.attractG, repelG = self.repelG, law = self.law,
                                              sigma2 = self.forceSigma, dynamicBalancing = self.forceBalancing, steps = steps,
                                              normalizeExamples = self.graph.normalizeExamples,
                                              contClass = self.graph.rawdata.domain.classVar.varType == orange.VarTypes.Continuous,
                                              mirrorSymmetry = self.mirrorSymmetry)
            neededSteps += steps

            if self.graph.__class__ != orngScaleLinProjData:
                qApp.processEvents()
                self.graph.potentialsBmp = None
                self.graph.updateData()
                
            positions = positions[-49:]+[Numeric.array([x[:2] for x in self.graph.anchorData])]
            if len(positions)==50:
                m = max(Numeric.sum((positions[0]-positions[49])**2, 1))
                if m < 1e-3: break
            if singleStep or (self.__class__ != FreeViz and self.cancelOptimization):
                break
        return neededSteps

    def optimize_LDA_Separation(self, attrIndices, anchorData, XAnchors = None, YAnchors = None):
        dataSize = len(self.graph.rawdata)
        classCount = len(self.graph.rawdata.domain.classVar.values)
        validData = self.graph.getValidList(attrIndices)
        selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))

        if not XAnchors: XAnchors = Numeric.array([a[0] for a in anchorData], Numeric.Float)
        if not YAnchors: YAnchors = Numeric.array([a[1] for a in anchorData], Numeric.Float)
        
        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, settingsDict = {"validData": validData, "XAnchors": XAnchors, "YAnchors": YAnchors, "scaleFactor": self.graph.scaleFactor, "normalize": self.graph.normalizeExamples, "useAnchorData": 1})
        projData = Numeric.transpose(transProjData)
        x_positions = projData[0]; y_positions = projData[1]; classData = projData[2]

        averages = []
        for i in range(classCount):
            ind = classData == i
            xpos = Numeric.compress(ind, x_positions);  ypos = Numeric.compress(ind, y_positions)
            xave = Numeric.sum(xpos)/len(xpos);         yave = Numeric.sum(ypos)/len(ypos)
            averages.append((xave, yave))

        # compute the positions of all the points. we will try to move all points so that the center will be in the (0,0)
        xCenterVector = -Numeric.sum(x_positions) / len(x_positions)   
        yCenterVector = -Numeric.sum(y_positions) / len(y_positions)
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

        FXs = Numeric.zeros(len(x_positions), Numeric.Float)        # forces
        FYs = Numeric.zeros(len(x_positions), Numeric.Float)
        
        for c in range(classCount):
            ind = (classData == c)
            Numeric.putmask(FXs, ind, meanDestinationVectors[c][0] - x_positions)
            Numeric.putmask(FYs, ind, meanDestinationVectors[c][1] - y_positions)
            
        # compute gradient for all anchors
        GXs = Numeric.array([sum(FXs * selectedData[i]) for i in range(len(anchorData))], Numeric.Float)
        GYs = Numeric.array([sum(FYs * selectedData[i]) for i in range(len(anchorData))], Numeric.Float)

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
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x, x+GXs[a]], yData = [y, y+GYs[a]], forceFilledSymbols = 1, lineWidth=3)
        
        for i in range(classCount):
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [averages[i][0], meanDestinationVectors[i][0]], yData = [averages[i][1], meanDestinationVectors[i][1]], forceFilledSymbols = 1, lineWidth=3)
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtCurve.Lines, xData = [averages[i][0], averages[i][0]], yData = [averages[i][1], averages[i][1]], forceFilledSymbols = 1, lineWidth=5)
        """
        #self.parentWidget.graph.repaint()
        #self.graph.anchorData = [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))]
        #self.graph.updateData(attrs, 0)
        return [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))], (newXAnchors, newYAnchors)

            
    def optimize_SLOW_Separation(self, attrIndices, anchorData, XAnchors = None, YAnchors = None):
        dataSize = len(self.graph.rawdata)
        validData = self.graph.getValidList(attrIndices)
        selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))

        if not XAnchors: XAnchors = Numeric.array([a[0] for a in anchorData], Numeric.Float)
        if not YAnchors: YAnchors = Numeric.array([a[1] for a in anchorData], Numeric.Float)
        
        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, settingsDict = {"validData": validData, "XAnchors": XAnchors, "YAnchors": YAnchors, "scaleFactor": self.graph.scaleFactor, "normalize": self.graph.normalizeExamples, "useAnchorData": 1})
        projData = Numeric.transpose(transProjData)
        x_positions = projData[0]; x_positions2 = Numeric.array(x_positions)
        y_positions = projData[1]; y_positions2 = Numeric.array(y_positions)
        classData = projData[2]  ; classData2 = Numeric.array(classData)

        FXs = Numeric.zeros(len(x_positions), Numeric.Float)        # forces
        FYs = Numeric.zeros(len(x_positions), Numeric.Float)
        GXs = Numeric.zeros(len(anchorData), Numeric.Float)        # gradients
        GYs = Numeric.zeros(len(anchorData), Numeric.Float)
        
        rotateArray = range(len(x_positions)); rotateArray = rotateArray[1:] + [0]
        for i in range(len(x_positions)-1):
            x_positions2 = Numeric.take(x_positions2, rotateArray)
            y_positions2 = Numeric.take(y_positions2, rotateArray)
            classData2 = Numeric.take(classData2, rotateArray)
            dx = x_positions2 - x_positions
            dy = y_positions2 - y_positions
            rs2 = dx**2 + dy**2
            rs2 += Numeric.where(rs2 == 0.0, 0.0001, 0.0)    # replace zeros to avoid divisions by zero
            rs = Numeric.sqrt(rs2)
            
            F = Numeric.zeros(len(x_positions), Numeric.Float)
            classDiff = Numeric.where(classData == classData2, 1, 0)
            Numeric.putmask(F, classDiff, 150*self.attractG*rs2)
            Numeric.putmask(F, 1-classDiff, -self.repelG/rs2)
            FXs += F * dx / rs
            FYs += F * dy / rs

        # compute gradient for all anchors
        GXs = Numeric.array([sum(FXs * selectedData[i]) for i in range(len(anchorData))], Numeric.Float)
        GYs = Numeric.array([sum(FYs * selectedData[i]) for i in range(len(anchorData))], Numeric.Float)

        m = max(max(abs(GXs)), max(abs(GYs)))
        GXs /= (20*m); GYs /= (20*m)
        
        newXAnchors = XAnchors + GXs
        newYAnchors = YAnchors + GYs

        # normalize so that the anchor most far away will lie on the circle        
        m = math.sqrt(max(newXAnchors**2 + newYAnchors**2))
        newXAnchors /= m
        newYAnchors /= m
        return [(newXAnchors[i], newYAnchors[i], anchorData[i][2]) for i in range(len(anchorData))], (newXAnchors, newYAnchors)

            
##    def recomputeEnergy(self, newEnergy = None):
##        if not newEnergy:
##            classes = [int(x.getclass()) for x in self.graph.rawdata]
##            ai = self.graph.attributeNameIndex
##            attrIndices = [ai[label] for label in self.parentWidget.getShownAttributeList()]
##            newEnergy = orangeom.computeEnergy(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG)
##        if self.__class__ != FreeViz:
##            self.energyLabel.setText("Energy: %.3f" % newEnergy)
##            self.energyLabel.repaint()

    def findSPCAProjection(self, attrIndices = None, setGraphAnchors = 1):
        try:
            ai = self.graph.attributeNameIndex
            if not attrIndices:
                attributes = self.getShownAttributeList()
                attrIndices = [ai[label] for label in attributes]
                
            validData = self.graph.getValidList(attrIndices)
            self.graph.normalizeExamples = 0
            
            selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))
            classData = Numeric.compress(validData, self.graph.noJitteringScaledData[ai[self.graph.rawdata.domain.classVar.name]])
            selectedData = Numeric.transpose(selectedData)
            
            s = Numeric.sum(selectedData)/float(len(selectedData))  
            selectedData -= s       # substract average value to get zero mean

            # define the Laplacian matrix
            L = Numeric.zeros((len(selectedData), len(selectedData)))
            for i in range(len(selectedData)):
                for j in range(i+1, len(selectedData)):
                    L[i,j] = -(classData[i] != classData[j])
                    L[j,i] = -(classData[i] != classData[j])
            
            s = Numeric.sum(L)
            for i in range(len(selectedData)):
                L[i,i] = -s[i]

            if self.useGeneralizedEigenvectors:
                covarMatrix = Numeric.matrixmultiply(Numeric.transpose(selectedData), selectedData)
                matrix = LinearAlgebra.inverse(covarMatrix)
                matrix = Numeric.matrixmultiply(matrix, Numeric.transpose(selectedData))
            else:
                matrix = Numeric.transpose(selectedData)
            
            # compute selectedDataT * L * selectedData
            matrix = Numeric.matrixmultiply(matrix, L)
            matrix = Numeric.matrixmultiply(matrix, selectedData)

            vals, vectors = LinearAlgebra.eigenvectors(matrix)
            """
            if vals.typecode() in Numeric.typecodes["Complex"]:     # the eigenvalues are complex numbers -> singluar covariance matrix
                names = self.graph.attributeNames
                attributes = [names[attrIndices[i]] for i in range(len(attrIndices))]
                anchors = self.graph.createAnchors(len(attributes), attributes)
                if setGraphAnchors: self.graph.anchorData = self.graph.createAnchors(len(attributes), attributes)
                return [anchors[i][0] for i in range(len(attributes))], [anchors[i][1] for i in range(len(attributes))], (attributes, attrIndices)
            """ 
            firstInd  = list(vals).index(max(vals)); vals[firstInd] = -1   # save the index of the largest eigenvector
            secondInd = list(vals).index(max(vals));                       # save the index of the second largest eigenvector

            xAnchors = vectors[firstInd]
            yAnchors = vectors[secondInd]

            lengthArr = xAnchors**2 + yAnchors**2
            m = math.sqrt(max(lengthArr))
            xAnchors /= m
            yAnchors /= m
            names = self.graph.attributeNames
            attributes = [names[attrIndices[i]] for i in range(len(attrIndices))]

            """
            temp = [(lengthArr[i], i) for i in range(len(lengthArr))]
            temp.sort()

            newXAnchors = []; newYAnchors = []; newAttributes = []; newIndices = []
            for i in range(len(temp))[::-1]:        # move from the longest attribute to the shortest
                newXAnchors.append(xAnchors[temp[i][1]])
                newYAnchors.append(yAnchors[temp[i][1]])
                newAttributes.append(attributes[temp[i][1]])
                newIndices.append(attrIndices[temp[i][1]])

            if setGraphAnchors:
                self.graph.setAnchors(newXAnchors, newYAnchors, newAttributes)
            #print attrIndices, newXAnchors, newYAnchors

            return newXAnchors, newYAnchors, (newAttributes, newIndices)
            """
            if setGraphAnchors:
                self.graph.setAnchors(xAnchors, yAnchors, attributes)
            return xAnchors, yAnchors, (attributes, attrIndices)
        except:
            #print "unable to compute the inverse of a singular matrix."
            names = self.graph.attributeNames
            attributes = [names[attrIndices[i]] for i in range(len(attrIndices))]
            anchors = self.graph.createAnchors(len(attributes), attributes)
            if setGraphAnchors: self.graph.anchorData = self.graph.createAnchors(len(attributes), attributes)
            return [anchors[i][0] for i in range(len(attributes))], [anchors[i][1] for i in range(len(attributes))], (attributes, attrIndices)

    # ###############################################################
    # S2N HEURISTIC FUNCTIONS
    # ###############################################################
    
    # if autoSetParameters is set then try different values for parameters and see how good projection do we get
    # if not then just use current parameters to place anchors
    def s2nMixAnchorsAutoSet(self):
        if self.__class__ != FreeViz:
            import qt
                
        if not self.rawdata.domain.classVar or not self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            if self.__class__ != FreeViz:
                qt.QMessageBox.critical( None, "Error", 'This heuristic works only in data sets with a discrete class value.', QMessageBox.Ok)
            else:
                print "S2N heuristic works only in data sets with a discrete class value"
            return

        import orngVizRank
        vizrank = orngVizRank.VizRank(orngVizRank.RADVIZ, graph = self.graph)
        vizrank.qualityMeasure = orngVizRank.AVERAGE_CORRECT
        vizrank.setData(self.rawdata)
        
        if self.autoSetParameters:
            results = {}
            self.s2nSpread = 0
            permutations = orngVisFuncts.generateDifferentPermutations(range(len(self.rawdata.domain.classVar.values)))
            for perm in permutations:
                self.classPermutationList = perm
                for val in self.attrsNum:
                    if self.attrsNum[self.attrsNum.index(val)-1] > len(self.rawdata.domain.attributes): continue    # allow the computations once
                    self.s2nPlaceAttributes = val
                    self.s2nMixAnchors(0)
                    if self.__class__ != FreeViz:
                        qt.qApp.processEvents()
                        
                    acc, other, resultsByFolds = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(None, settingsDict = {"useAnchorData": 1}))
                    if hasattr(self, "setStatusBarText"):
                        if results.keys() != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results.keys())))
                        else:                    self.setStatusBarText("Current projection value is %.2f" % (acc))
                                                             
                    results[acc] = (perm, val)
            if results.keys() == []: return
            self.classPermutationList, self.s2nPlaceAttributes = results[max(results.keys())]
            if self.__class__ != FreeViz:
                qt.qApp.processEvents()
            self.s2nMixAnchors(0)        # update the best number of attributes

            results = []
            anchors = self.graph.anchorData
            attributeNameIndex = self.graph.attributeNameIndex
            attrIndices = [attributeNameIndex[val[2]] for val in anchors]
            for val in range(10):
                self.s2nSpread = val
                self.s2nMixAnchors(0)
                acc, other, resultsByFolds = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(attrIndices, settingsDict = {"useAnchorData": 1}))
                results.append(acc)
                if hasattr(self, "setStatusBarText"):
                    if results != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results)))
                    else:             self.setStatusBarText("Current projection value is %.2f" % (acc))
            self.s2nSpread = results.index(max(results))

            if hasattr(self, "setStatusBarText"):
                self.setStatusBarText("Best projection value is %.2f" % (max(results)))

        # always call this. if autoSetParameters then because we need to set the attribute list in radviz. otherwise because it finds the best attributes for current settings
        self.s2nMixAnchors()


    # place a subset of attributes around the circle. this subset must contain "good" attributes for each of the class values
    def s2nMixAnchors(self, setAttributeListInRadviz = 1):
        if self.__class__ != FreeViz:
            import qt
            if not self.rawdata.domain.classVar or not self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                qt.QMessageBox.critical( None, "Error", 'This heuristic works only in data sets with a discrete class value.', QMessageBox.Ok)
                return
        
        # compute the quality of attributes only once
        if self.s2nMixData == None:
            rankedAttrs, rankedAttrsByClass = orngVisFuncts.findAttributeGroupsForRadviz(self.rawdata, orngVisFuncts.S2NMeasureMix())
            self.s2nMixData = (rankedAttrs, rankedAttrsByClass)
            classCount = len(rankedAttrsByClass)
            attrs = rankedAttrs[:(self.s2nPlaceAttributes/classCount)*classCount]    # select appropriate number of attributes
        else:
            classCount = len(self.s2nMixData[1])
            attrs = self.s2nMixData[0][:(self.s2nPlaceAttributes/classCount)*classCount]
            
        arr = [0]       # array that will tell where to put the next attribute
        for i in range(1,len(attrs)/2): arr += [i,-i]

        if len(attrs) == 0: return
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
                self.parentWidget.setShownAttributeList(self.rawdata, attrNames)
            self.graph.updateData(attrNames)
            self.graph.repaint()




# #############################################################################
# class that represents FreeViz classifier 
class FreeVizClassifier(orange.Classifier):
    def __init__(self, data, freeviz):
        self.FreeViz = freeviz

        if self.FreeViz.__class__ != FreeViz:
            self.FreeViz.parentWidget.cdata(data)
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
        domain = orange.Domain(labels+[graph.rawdata.domain.classVar], graph.rawdata.domain)
        indices = [ai[label] for label in labels]
        offsets = [graph.offsets[i] for i in indices]
        normalizers = [graph.normalizers[i] for i in indices]
        averages = [graph.averages[i] for i in indices]

        self.FreeViz.graph.createProjectionAsNumericArray(indices, settingsDict = {"useAnchorData": 1})
        self.classifier = orange.P2NN(domain,
                                      Numeric.transpose(Numeric.array([graph.unscaled_x_positions, graph.unscaled_y_positions, [float(ex.getclass()) for ex in graph.rawdata]])),
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
            self.FreeViz.parentWidget.cdata(data)
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
                
        table = self.FreeViz.graph.createProjectionAsExampleTable(attrListIndices, settingsDict = {"scaleFactor": self.FreeViz.graph.trueScaleFactor, "useAnchorData": 1})
        kValue = int(math.sqrt(len(self.data)))
        knn = orange.kNNLearner(k = kValue, rankWeight = 0, distanceConstructor = orange.ExamplesDistanceConstructor_Euclidean(normalize=0))

        [xTest, yTest] = self.FreeViz.graph.getProjectedPointPosition(attrListIndices, attrVals, settingsDict = {"useAnchorData":1})
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

