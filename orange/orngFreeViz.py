import orangeom, orange
import math, random, Numeric
from orngScaleRadvizData import orngScaleRadvizData

#implementation
FAST_IMPLEMENTATION = 0
SLOW_IMPLEMENTATION = 1
LDA_IMPLEMENTATION = 2

LAW_LINEAR = 0
LAW_SQUARE = 1
LAW_EXPONENTIAL = 2

class FreeViz:
    def __init__(self, graph = None):
        if not graph:
            graph = orngScaleRadvizData()
        self.graph = graph

        self.implementation = 0
        self.attractG = 1.0
        self.repelG = 1.0
        self.law = LAW_LINEAR
        self.lockToCircle = 0

    def showAllAttributes(self):
        self.graph.anchorData = [(0,0, a.name) for a in self.graph.rawdata.domain.attributes]
        self.radialAnchors()

    def getShownAttributeList(self):
        return [anchor[2] for anchor in self.graph.anchorData]

    def radialAnchors(self):
        attrList = self.getShownAttributeList()
        phi = 2*math.pi/len(attrList)
        self.graph.anchorData = [(math.cos(i*phi), math.sin(i*phi), a) for i, a in enumerate(attrList)]

    def ranch(self, label):
        r = self.lockToCircle and 1.0 or 0.3+0.7*random.random()
        phi = 2*math.pi*random.random()
        return (r*math.cos(phi), r*math.sin(phi), label)

    def randomAnchors(self):
        anchors = [self.ranch(a) for a in self.getShownAttributeList()]
        if not self.lockToCircle:
            maxdist = math.sqrt(max([x[0]**2+x[1]**2 for x in anchors]))
            anchors = [(x[0]/maxdist, x[1]/maxdist, x[2]) for x in anchors]
        self.graph.anchorData = anchors

    def optimizeSeparation(self, steps = 10, singleStep = False):

        if self.implementation == FAST_IMPLEMENTATION:
            self.optimize_FAST_Separation(steps, singleStep)
        else:
            if singleStep: steps = 1
            if self.implementation == SLOW_IMPLEMENTATION:  impl = self.optimize_SLOW_Separation
            elif self.implementation == LDA_IMPLEMENTATION: impl = self.optimize_LDA_Separation
            ai = self.graph.attributeNameIndex
            attrIndices = [ai[label] for label in self.getShownAttributeList()]
            XAnchors = None; YAnchors = None
            if self.__class__ != FreeViz: from qt import qApp
            
            for c in range(50):                
                for i in range(steps):
                    if self.__class__ != FreeViz and self.cancelOptimization == 1: return
                    self.graph.anchorData, (XAnchors, YAnchors) = impl(attrIndices, self.graph.anchorData, XAnchors, YAnchors)
                if self.graph.__class__ != orngScaleRadvizData:
                    qApp.processEvents()
                    self.graph.updateData()
                self.recomputeEnergy()

    def optimize_FAST_Separation(self, steps = 10, singleStep = False):
        classes = [int(x.getclass()) for x in self.graph.rawdata]
        optimizer = self.lockToCircle and orangeom.optimizeAnchorsRadial or orangeom.optimizeAnchors
        ai = self.graph.attributeNameIndex
        attrIndices = [ai[label] for label in self.getShownAttributeList()]
        contClass = self.graph.rawdata.domain.classVar.varType == orange.VarTypes.Continuous
        if self.__class__ != FreeViz: from qt import qApp
       
        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        positions = [Numeric.array([x[:2] for x in self.graph.anchorData])]
        while 1:
            self.graph.anchorData, E = optimizer(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG, self.law, steps, self.graph.normalizeExamples, contClass)
            if self.graph.__class__ != orngScaleRadvizData:
                qApp.processEvents()
                self.graph.potentialsBmp = None
                self.graph.updateData()
            self.recomputeEnergy(E)
                
            positions = positions[-49:]+[Numeric.array([x[:2] for x in self.graph.anchorData])]
            if len(positions)==50:
                m = max(Numeric.sum((positions[0]-positions[49])**2, 1))
                if m < 1e-2: break
            if singleStep or (self.__class__ != FreeViz and self.cancelOptimization): break


    def optimize_LDA_Separation(self, attrIndices, anchorData, XAnchors = None, YAnchors = None):
        dataSize = len(self.graph.rawdata)
        classCount = len(self.graph.rawdata.domain.classVar.values)
        validData = self.graph.getValidList(attrIndices)
        selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))

        if not XAnchors: XAnchors = Numeric.array([a[0] for a in anchorData], Numeric.Float)
        if not YAnchors: YAnchors = Numeric.array([a[1] for a in anchorData], Numeric.Float)
        
        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, validData = validData, XAnchors = XAnchors, YAnchors = YAnchors, scaleFactor = self.graph.scaleFactor, normalize = self.graph.normalizeExamples, useAnchorData = 1)
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
        dataSize = len(self.rawdata)
        validData = self.graph.getValidList(attrIndices)
        selectedData = Numeric.compress(validData, Numeric.take(self.graph.noJitteringScaledData, attrIndices))

        if not XAnchors: XAnchors = Numeric.array([a[0] for a in anchorData], Numeric.Float)
        if not YAnchors: YAnchors = Numeric.array([a[1] for a in anchorData], Numeric.Float)
        
        transProjData = self.graph.createProjectionAsNumericArray(attrIndices, validData = validData, XAnchors = XAnchors, YAnchors = YAnchors, scaleFactor = self.graph.scaleFactor, normalize = self.graph.normalizeExamples, useAnchorData = 1)
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

            
    def recomputeEnergy(self, newEnergy = None):
        if not newEnergy:
            classes = [int(x.getclass()) for x in self.graph.rawdata]
            ai = self.graph.attributeNameIndex
            attrIndices = [ai[label] for label in self.getShownAttributeList()]
            newEnergy = orangeom.computeEnergy(Numeric.transpose(self.graph.scaledData).tolist(), classes, self.graph.anchorData, attrIndices, self.attractG, -self.repelG)
        if self.__class__ != FreeViz:
            self.energyLabel.setText("Energy: %.3f" % newEnergy)
            self.energyLabel.repaint()
        return newEnergy


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
            
        self.FreeViz.randomAnchors()
        #self.radvizWidget.radialAnchors()
        self.FreeViz.optimizeSeparation()
        
        graph = self.FreeViz.graph
        ai = graph.attributeNameIndex
        labels = [a[2] for a in graph.anchorData]
        domain = orange.Domain(labels+[graph.rawdata.domain.classVar], graph.rawdata.domain)
        indices = [ai[label] for label in labels]
        offsets = [graph.offsets[i] for i in indices]
        normalizers = [graph.normalizers[i] for i in indices]
        averages = [graph.averages[i] for i in indices]

        self.FreeViz.graph.createProjectionAsNumericArray(indices, useAnchorData = 1)
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
