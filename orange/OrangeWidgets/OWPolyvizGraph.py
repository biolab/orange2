#
# OWPolyvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy, deepcopy
import time

# ####################################################################
# get a list of all different permutations
def getPermutationList(elements, tempPerm, currList, checkReverse):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        getPermutationList(elements, tempPerm, currList, checkReverse)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return


        if checkReverse == 1:
            # also try the reverse permutation
            temp.reverse()
            for i in range(len(temp)):
                el = temp.pop()
                temp.insert(0, el)
                if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)

def fact(i):
        ret = 1
        while i > 1:
            ret = ret*i
            i -= 1
        return ret
    
# return number of combinations where we select "select" from "total"
def combinations(select, total):
    return fact(total)/ (fact(total-select)*fact(select))


###########################################################################################
##### CLASS : OWPolyvizGRAPH
###########################################################################################
class OWPolyvizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.attrLocalValues = {}
        self.lineLength = 2*0.05
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.percentDataUsed = 100
        self.minExamples = 0
        self.enhancedTooltips = 1

        self.dataMap = {}		# each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.validData = []
        self.kNeighbours = 1
        self.statusBar = None
        self.showLegend = 1

    def setEnhancedTooltips(self, enhanced):
        self.enhancedTooltips = enhanced
        self.dataMap = {}

    def setLineLength(self, len):
        self.lineLength = len*0.05

    def createAnchors(self, anchorNum):
        anchors = [[],[]]
        for i in range(anchorNum):
            x = math.cos(2*math.pi * float(i) / float(anchorNum)); strX = "%.5f" % (x)
            y = math.sin(2*math.pi * float(i) / float(anchorNum)); strY = "%.5f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))
        return anchors

    def calculateAttrValuesSum(self, data, datalen, indices, validData):
        if datalen == 0: return []
        
        sum=[]
        for i in range(datalen):
            if validData[i] == 0:
                sum.append(1.0)
                continue
            temp = 0
            for j in range(len(indices)): temp += data[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum.append(temp)
        return sum

    #
    # if we use globalScaling we must also create a copy of localy scaled data
    #
    def setData(self, data):
        # first call the original function to scale data
        OWVisGraph.setData(self, data)

        for index in range(len(data.domain)):
            if data.domain[index].varType == orange.VarTypes.Discrete:
                values = [0, len(data.domain[index].values)-1]
            else:
                values = [self.domainDataStat[index].min, self.domainDataStat[index].max]
            self.attrLocalValues[data.domain[index].name] = values

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, attributeReverse, **args):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()
    
        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)


        # we must have at least 3 attributes to be able to show anything
        if len(labels) < 3: return
        if len(self.rawdata) == 0 or len(labels) == 0: self.updateLayout(); return

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        
        if self.showLegend: self.setAxisScale(QwtPlot.xBottom, -1.20, 1.25, 1)
        else:               self.setAxisScale(QwtPlot.xBottom, -1.20, 1.20, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.20, 1.20, 1)

        length = len(labels)
        dataSize = len(self.rawdata)
        self.dataMap = {}
        indices = []
        self.anchorData = []
        polyvizLineCoordsX = []; polyvizLineCoordsY = []    # if class is discrete we will optimize drawing by storing computed values and adding less data curves to plot
        classIsDiscrete = 0
        classNameIndex = -1
        
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        # ##########
        # create anchor for every attribute
        for i in range(length):
            x1 = math.cos(2*math.pi * float(i) / float(length)); strX1 = "%.4f" % (x1)
            y1 = math.sin(2*math.pi * float(i) / float(length)); strY1 = "%.4f" % (y1)
            x2 = math.cos(2*math.pi * float(i+1) / float(length)); strX2 = "%.4f" % (x2)
            y2 = math.sin(2*math.pi * float(i+1) / float(length)); strY2 = "%.4f" % (y2)
            self.anchorData.append((float(strX1), float(strY1), float(strX2), float(strY2), labels[i]))
        
        
        # if we don't want coloring
        if self.className == "(One color)" or self.showKNNModel:      
            valLen = 1
            if self.showKNNModel == 1:
                # variables and domain for the table
                domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain[self.className]])
                table = orange.ExampleTable(domain)
            
            for i in range(len(labels)):
                polyvizLineCoordsX.append([[]])
                polyvizLineCoordsY.append([[]])                

        # if we have a discrete class
        elif self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
            classIsDiscrete = 1
            classNameIndex = self.attributeNames.index(self.className)
            valLen = len(self.rawdata.domain[self.className].values)
            classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)
            for i in range(len(labels)):
                tempX = []; tempY = []
                for j in range(len(classValueIndices)):
                    tempX.append([]); tempY.append([])
                polyvizLineCoordsX.append(tempX)
                polyvizLineCoordsY.append(tempY)
        else:
            valLen = 0
            classNameIndex = self.attributeNames.index(self.className)
            
        # ######################
        # compute valid data examples
        self.validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": self.validData[i] = 0
                if classNameIndex >= 0 and self.scaledData[classNameIndex][i] == "?": self.validData[i] = 0


        # ##########
        # draw text at lines
        for i in range(length):
            # print attribute name
            self.addMarker(labels[i], 0.6*(self.anchorData[i][0]+ self.anchorData[i][2]), 0.6*(self.anchorData[i][1]+ self.anchorData[i][3]), Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)

            if self.rawdata.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = self.getVariableValuesSorted(self.rawdata, labels[i])
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    if attributeReverse[labels[i]] == 0:
                        self.addMarker(values[j], k*(1-pos)*self.anchorData[i][0]+k*pos*self.anchorData[i][2], k*(1-pos)*self.anchorData[i][1]+k*pos*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)
                    else:
                        self.addMarker(values[j], k*pos*self.anchorData[i][0]+k*(1-pos)*self.anchorData[i][2], k*pos*self.anchorData[i][1]+k*(1-pos)*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)

            else:
                # min and max value
                names = ["%.3f" % (self.attrLocalValues[labels[i]][0]), "%.3f" % (self.attrLocalValues[labels[i]][1])]
                if attributeReverse[labels[i]] == 1: names.reverse()
                self.addMarker(names[0],0.95*self.anchorData[i][0]+0.15*self.anchorData[i][2], 0.95*self.anchorData[i][1]+0.15*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)
                self.addMarker(names[1], 0.15*self.anchorData[i][0]+0.95*self.anchorData[i][2], 0.15*self.anchorData[i][1]+0.95*self.anchorData[i][3], Qt.AlignHCenter + Qt.AlignVCenter)


        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y
        contData = []    # list to store color, x and y position of data items in case of continuous class

        sum = self.calculateAttrValuesSum(self.scaledData, len(self.rawdata), indices, self.validData)

        
        # ##########
        #  create data curves
        RECT_SIZE = 0.01    # size of tooltip rectangle in percents of graph size
        for i in range(dataSize):
            if self.validData[i] == 0: continue
            
            # #########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            xDataAnchors = []; yDataAnchors = []
            for j in range(length):
                index = indices[j]
                val = self.noJitteringScaledData[index][i]
                if attributeReverse[labels[j]] == 1: val = 1-val
                xDataAnchor = self.anchorData[j][0]*(1-val) + self.anchorData[j][2]*val
                yDataAnchor = self.anchorData[j][1]*(1-val) + self.anchorData[j][3]*val
                x_i += xDataAnchor * (self.scaledData[index][i] / sum[i])
                y_i += yDataAnchor * (self.scaledData[index][i] / sum[i])
                xDataAnchors.append(xDataAnchor)
                yDataAnchors.append(yDataAnchor)

            # scale data according to scale factor
            x_i = x_i * self.scaleFactor
            y_i = y_i * self.scaleFactor
                

            # #########
            # we add a tooltip for this point
            text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices + [self.className])
            r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
            self.tips.addToolTip(r, text)

            lineColor = QColor(0,0,0)
            if self.showKNNModel == 1:
                table.append(orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]]))
            elif valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
                lineColor.setHsv(0, 255, 255)
            elif classIsDiscrete:
                index = classValueIndices[self.rawdata[i][self.className].value]
                curveData[index][0].append(x_i)
                curveData[index][1].append(y_i)
                lineColor.setHsv(self.coloringScaledData[classNameIndex][i] * 360, 255, 255)
            else:
                contData.append([x_i, y_i, self.coloringScaledData[classNameIndex][i] * 360]) # store data for drawing
                lineColor.setHsv(self.coloringScaledData[classNameIndex][i] * 360, 255, 255)


            # compute the lines from anchors
            for j in range(length):
                dist = EuclDist([x_i, y_i], [xDataAnchors[j] , yDataAnchors[j]])
                if dist == 0: continue
                kvoc = float(self.lineLength) / dist
                lineX1 = x_i; lineY1 = y_i

                # we don't make extrapolation
                if kvoc > 1: lineX2 = lineX1; lineY2 = lineY1
                else:
                    lineX2 = (1.0 - kvoc)*xDataAnchors[j] + kvoc * lineX1
                    lineY2 = (1.0 - kvoc)*yDataAnchors[j] + kvoc * lineY1


                if self.showKNNModel:
                    polyvizLineCoordsX[j][0] += [xDataAnchors[j], lineX2]
                    polyvizLineCoordsY[j][0] += [yDataAnchors[j], lineY2]
                elif valLen == 1:
                    polyvizLineCoordsX[j][0] += [xDataAnchors[j], lineX2, xDataAnchors[j]]
                    polyvizLineCoordsY[j][0] += [yDataAnchors[j], lineY2, yDataAnchors[j]]
                elif classIsDiscrete:
                    index = classValueIndices[self.rawdata[i][self.className].value]
                    polyvizLineCoordsX[j][index] += [xDataAnchors[j], lineX2, xDataAnchors[j]]
                    polyvizLineCoordsY[j][index] += [yDataAnchors[j], lineY2, yDataAnchors[j]]
                else:
                    self.addCurve('line' + str(i), lineColor, lineColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = [xDataAnchors[j], lineX2], yData = [yDataAnchors[j], lineY2])


            if self.enhancedTooltips == 1:
                # create a dictionary value so that tooltips will be shown faster
                data = self.rawdata[i]
                dictValue = "%.1f-%.1f"%(x_i, y_i)
                if not self.dataMap.has_key(dictValue):
                    self.dataMap[dictValue] = []
                self.dataMap[dictValue].append((x_i, y_i, xDataAnchors, yDataAnchors, lineColor, data))
             

        ###### SHOW KNN MODEL QUALITY/ERROR
        if self.showKNNModel:                       
            vals = []
            knn = orange.kNNLearner(table, k=self.kNeighbours, rankWeight = 0)
            if self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
                classValues = list(self.rawdata.domain[self.className].values)
                for j in range(len(table)):
                    out = knn(table[j], orange.GetProbabilities)
                    prob = out[table[j].getclass()]
                    if self.showCorrect == 1: prob = 1.0 - prob
                    vals.append(prob)
            else:
                for j in range(len(table)):
                    vals.append(pow(table[j][2].value - knn(table[j]), 2))
                maxError = max(vals)
                if self.showCorrect == 1:
                    vals = [val/maxError for val in vals]
                else:
                    vals = [1.0 - val/maxError for val in vals]

            for j in range(len(table)):
                newColor = QColor(55+vals[j]*200, 55+vals[j]*200, 55+vals[j]*200)
                key = self.addCurve(str(j), newColor, newColor, self.pointWidth, xData = [table[j][0].value], yData = [table[j][1].value])
                for i in range(len(polyvizLineCoordsX)):
                    self.addCurve('line' + str(i), newColor, newColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = polyvizLineCoordsX[i][0][2*j:2*j+2], yData = polyvizLineCoordsY[i][0][2*j:2*j+2])

        ###### ONE COLOR OR DISCRETE CLASS ATTRIBUTE
        elif valLen == 1 or classIsDiscrete:        
            # create data curves for dots
            for i in range(valLen):
                newColor = QColor()
                newColor.setHsv(self.colorHueValues[i]*360, 255, 255)
                self.addCurve(str(i), newColor, newColor, self.pointWidth, xData = curveData[i][0], yData = curveData[i][1])
                for j in range(len(labels)):
                    self.addCurve("lines" + str(i), newColor, newColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = polyvizLineCoordsX[j][i], yData = polyvizLineCoordsY[j][i])

        ###### CONTINUOUS CLASS ATTRIBUTE
        else:                                       
            for i in range(len(contData)):
                newColor = QColor()
                newColor.setHsv(contData[i][2], 255, 255)
                self.addCurve(str(i), newColor, newColor, self.pointWidth, xData = [contData[i][0]], yData = [contData[i][1]])

        
        # ##########
        # draw polygon
        xdata = []; ydata = []
        for i in range(len(labels)+1):
            xdata.append(math.cos(2*math.pi * float(i) / float(len(labels))))
            ydata.append(math.sin(2*math.pi * float(i) / float(len(labels))))

        newCurveKey = self.addCurve("polygon", QColor(0,0,0), QColor(0,0,0), 0, QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata, yData = ydata)
        pen = self.curve(newCurveKey).pen(); pen.setWidth(2); self.curve(newCurveKey).setPen(pen)
        

        #################
        # draw the legend
        if self.className != "(One color)" and self.showLegend:
            # show legend for discrete class
            if classIsDiscrete:
                self.addMarker(self.className, 0.87, 1.06, Qt.AlignLeft)
                classVariableValues = self.getVariableValuesSorted(self.rawdata, self.className)
                for index in range(len(classVariableValues)):
                    newColor = QColor()
                    if len(classVariableValues) < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[index]*360, 255, 255)
                    else:                                                   newColor.setHsv((index*360)/len(classVariableValues), 255, 255)
                    y = 1.0 - index * 0.05
                    self.addCurve(str(i), newColor, newColor, self.pointWidth, xData= [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)
            # show legend for continuous class
            else:
                x0 = 1.20; x1 = 1.24
                for i in range(1000):
                    y = -1.0 + i*2.0/1000.0
                    newCurveKey = self.insertCurve(str(i))
                    newColor = QColor()
                    newColor.setHsv(float(i*self.MAX_HUE_VAL)/1000.0, 255, 255)
                    self.setCurvePen(newCurveKey, QPen(newColor))
                    self.setCurveData(newCurveKey, [x0,x1], [y,y])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.className]
                self.addMarker("%s = %.3f" % (self.className, minVal), x0 - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %.3f" % (self.className, maxVal), x0 - 0.02, +1.0 - 0.04, Qt.AlignLeft)


    ##########################
    ## do we want advanced tooltips
    def onMouseMoved(self, e):
        for key in self.tooltipCurveKeys:  self.removeCurve(key)
        for marker in self.tooltipMarkers: self.removeMarker(marker)
        self.tooltipCurveKeys = []
        self.tooltipMarkers = []
            
        x = self.invTransform(QwtPlot.xBottom, e.x())
        y = self.invTransform(QwtPlot.yLeft, e.y())
        dictValue = "%.1f-%.1f"%(x, y)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            dist = 100.0
            nearestPoint = ()
            for (x_i, y_i, xAnchors, yAnchors, color, data) in points:
                if abs(x-x_i)+abs(y-y_i) < dist:
                    dist = abs(x-x_i)+abs(y-y_i)
                    nearestPoint = (x_i, y_i, xAnchors, yAnchors, color, data)
           
            if dist < 0.05:
                x_i = nearestPoint[0]; y_i = nearestPoint[1]; xAnchors = nearestPoint[2]; yAnchors = nearestPoint[3]; color = nearestPoint[4]; data = nearestPoint[5]
                for i in range(len(self.anchorData)):
                    (xAnchor1, yAnchor1, xAnchor2, yAnchor2, label) = self.anchorData[i]

                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, xData = [x_i, xAnchors[i]], yData = [y_i, yAnchors[i]])
                    self.tooltipCurveKeys.append(key)
                    
                    # draw text
                    marker = self.addMarker(str(data[self.attributeNames.index(label)].value),(x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    self.tooltipMarkers.append(marker)

        OWVisGraph.onMouseMoved(self, e)
        self.update()
        # -----------------------------------------------------------
        # -----------------------------------------------------------

    def generateAttrReverseLists(self, attrList, fullAttribList, tempList):
        if attrList == []: return tempList
        tempList2 = deepcopy(tempList)
        index = fullAttribList.index(attrList[0])
        for list in tempList2: list[index] = 1
        return self.generateAttrReverseLists(attrList[1:], fullAttribList, tempList + tempList2)


    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, attributeReverse):
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            indices.append(self.attributeNames.index(label))

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[self.className]])

        # which data items have all values valid
        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        

        # create anchor for every attribute
        anchors = self.createAnchors(attrListLength)
        
        # store all sums
        sum = self.calculateAttrValuesSum(self.noJitteringScaledData, len(self.rawdata), indices, validData)

        table = orange.ExampleTable(domain)

        # calculate projections
        for i in range(dataSize):
            if validData[i] == 0: continue
            
            x_i = 0.0; y_i = 0.0
            for j in range(attrListLength):
                val = self.noJitteringScaledData[indices[j]][i]
                if attributeReverse[attrList[j]] == 1: val = 1-val
                xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%attrListLength]*val
                yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%attrListLength]*val
                x_i += xDataAnchor * (self.noJitteringScaledData[indices[j]][i] / sum[i])
                y_i += yDataAnchor * (self.noJitteringScaledData[indices[j]][i] / sum[i])
               
            example = orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]])
            table.append(example)

        tempPermValue = 0.0        
        knn = orange.kNNLearner(table, k=self.kNeighbours, rankWeight = 0)
        
        if table.domain.classVar.varType == orange.VarTypes.Discrete:
            # use knn on every example and compute its accuracy
            classValues = list(self.rawdata.domain[self.className].values)
            for j in range(len(table)):
                index = classValues.index(table[j][2].value)
                tempPermValue += knn(table[j], orange.GetProbabilities)[index]
            print "k = %3.d, Accuracy: %2.2f%%" % (self.kNeighbours, tempPermValue*100.0/float(len(table)) )
            return tempPermValue*100.0/float(len(table))
        else:
            for j in range(len(table)):
                tempPermValue += pow(table[j][2].value - knn(table[j]), 2)
            tempPermValue /= float(len(table))
            print "k = %3.d, MSE: %2.2f" % (self.kNeighbours, tempPermValue)
            return tempPermValue
    
        
    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalSeparation(self, attrList, attrReverseDict, printTime = 1, progressBar = None):
        if self.className == "(One color)":
            print "Unable to compute optimal ordering. Please select class attribute first."
            return []

        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            indices.append(self.attributeNames.index(label))

        # if we want global value scaling, we must first compute it
        if self.globalValueScaling == 1:
            selectedGlobScaledData = []
            for i in range(len(self.rawdata.domain)): selectedGlobScaledData.append([])
        
            (min, max) =  self.getMinMaxValDomain(self.rawdata, attrList)
            
            for attr in attrList:
                index = self.attributeNames.index(attr)
                scaled, values = self.scaleData(self.rawdata, index, min, max, jitteringEnabled = 0)
                selectedGlobScaledData[index] = scaled
        else:
            selectedGlobScaledData = self.noJitteringScaledData

        print "----------------------------"
        print "generating permutations. Please wait"
        indPermutations = {}
        getPermutationList(indices, [], indPermutations, attrReverseDict == None)

        attrReverse = []
        if attrReverseDict != None: # if we received a dictionary, then we don't reverse attributes
            temp = [0] * len(self.rawdata.domain)
            for val in attrReverseDict.keys():
                temp[self.attributeNames.index(val)] = attrReverseDict[val]
            attrReverse.append(temp)
        else:
            attrReverse = self.generateAttrReverseLists(attrList, self.attributeNames,[[0]*len(self.rawdata.domain)])

        fullList = []
        permutationIndex = 0 # current permutation index
        totalPermutations = len(indPermutations.values())*len(attrReverse)

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[self.className]])

        # which data items have all values valid
        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        print "Nr. of examples: ", str(count)
        if count < self.minExamples:
            print "not enough examples in example table. Ignoring permutation."
            print "------------------------------"
            return []

        # create anchor for every attribute
        anchors = self.createAnchors(attrListLength)
        
        # store all sums
        sum = self.calculateAttrValuesSum(selectedGlobScaledData, len(self.rawdata), indices, validData)

        t = time.time()

        if progressBar:
            progressBar.setTotalSteps(len(indPermutations.values())*len(attrReverse))
            progressBar.setProgress(0)

        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            for attrOrder in attrReverse:
                permutationIndex += 1

                if progressBar != None:
                    progressBar.setProgress(progressBar.progress()+1)

                table = orange.ExampleTable(domain)

                # calculate projections
                for i in range(dataSize):
                    if validData[i] == 0: continue
                    
                    x_i = 0.0; y_i = 0.0
                    for j in range(attrListLength):
                        index = permutation[j]
                        val = selectedGlobScaledData[index][i]
                        if attrOrder[index] == 0:
                            xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%attrListLength]*val
                            yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%attrListLength]*val
                        else:
                            xDataAnchor = anchors[0][j]*val + anchors[0][(j+1)%attrListLength]*(1-val)
                            yDataAnchor = anchors[1][j]*val + anchors[1][(j+1)%attrListLength]*(1-val)
                        x_i += xDataAnchor * (selectedGlobScaledData[index][i] / sum[i])
                        y_i += yDataAnchor * (selectedGlobScaledData[index][i] / sum[i])
                       
                    example = orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]])
                    table.append(example)

                tempPermValue = 0
                experiments = 0
                selection = orange.MakeRandomIndices2(table, 1.0-float(self.percentDataUsed)/100.0)
                knn = orange.kNNLearner(table, k=self.kNeighbours, rankWeight = 0)

                if table.domain.classVar.varType == orange.VarTypes.Discrete:
                    if selection[j] == 0: continue
                    classValues = list(self.rawdata.domain[self.className].values)
                    for j in range(len(table)):
                        index = classValues.index(table[j][2].value)
                        tempPermValue += knn(table[j], orange.GetProbabilities)[index]
                        experiments += 1
                    tempPermValue = tempPermValue*100.0/float(experiments)
                    print "permutation %6d / %d. Accuracy: %2.2f%%" % (permutationIndex, totalPermutations, tempPermValue )
                else:
                    for j in range(len(table)):
                        if selection[j] == 0: continue
                        tempPermValue += pow(table[j][2].value - knn(table[j]), 2)
                        experiments += 1
                    tempPermValue /= float(experiments)
                    print "permutation %6d / %d. MSE: %2.2f" % (permutationIndex, totalPermutations, tempPermValue) 

                # save the permutation
                tempList = []
                for i in permutation:
                    tempList.append(self.attributeNames[i])
                fullList.append((tempPermValue, len(table), tempList, attrOrder))

        if printTime:
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            print "------------------------------"

        return fullList
                
    def getOptimalSubsetSeparation(self, attrList, attrReverseDict, numOfAttr, maxResultsLen, progressBar = None):
        full = []
        
        self.totalPossibilities = 0
        self.startTime = time.time()
        for i in range(numOfAttr, 2, -1):
            self.totalPossibilities += combinations(i, len(attrList))

        if progressBar:
            progressBar.setTotalSteps(self.totalPossibilities)
            progressBar.setProgress(0)
                
        for i in range(numOfAttr, 2, -1):
            full1 = self.getOptimalExactSeparation(attrList, [], attrReverseDict, i, maxResultsLen, progressBar)
            full = full + full1
            """
            while len(full) > maxResultsLen:
                el = min(full)
                full.remove(el)
            """
        return full

    def getOptimalExactSeparation(self, attrList, subsetList, attrReverseDict, numOfAttr, maxResultsLen, progressBar = None):
        if attrList == [] or numOfAttr == 0:
            if len(subsetList) < 3 or numOfAttr != 0: return []
            if progressBar:
                progressBar.setProgress(progressBar.progress()+1)
           
            print subsetList
            if self.totalPossibilities > 0 and self.triedPossibilities > 0:
                secs = int(time.time() - self.startTime)
                totalExpectedSecs = int(float(self.totalPossibilities*secs)/float(self.triedPossibilities))
                restSecs = totalExpectedSecs - secs
                print "Used time: %d:%02d:%02d, Remaining time: %d:%02d:%02d (total experiments: %d, rest: %d)" %(secs /3600, (secs-((secs/3600)*3600))/60, secs%60, restSecs /3600, (restSecs-((restSecs/3600)*3600))/60, restSecs%60, self.totalPossibilities, self.totalPossibilities-self.triedPossibilities)
            self.triedPossibilities += 1
            return self.getOptimalSeparation(subsetList, attrReverseDict, printTime = 0)

        full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, attrReverseDict, numOfAttr, maxResultsLen, progressBar)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, attrReverseDict, numOfAttr-1, maxResultsLen, progressBar)

        # find max values in booth lists
        full = full1 + full2
        shortList = []
        if self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete: funct = max
        else: funct = min
        for i in range(min(maxResultsLen, len(full))):
            item = funct(full)
            shortList.append(item)
            full.remove(item)

        return shortList

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWPolyvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
