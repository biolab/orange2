#
# OWPolyvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy       # used to copy arrays

###########################################################################################
##### FUNCTIONS FOR CALCULATING PERMUTATIONS, DISTANCES, ...
###########################################################################################

# calculate Euclidean distance between two points
def EuclDist(v1, v2):
    val = 0
    for i in range(len(v1)):
        val += (v1[i]-v2[i])**2
    return sqrt(val)
        

# add val to sorted list list. if len > maxLen delete last element
def addToList(list, val, maxLen):
    i = 0
    for i in range(len(list)):
        if val < list[i]:
            list.insert(i, val)
            if len(list) > maxLen:
                list.remove(list[maxLen])
            return
    if len(list) < maxLen:
        list.insert(len(list), val)

# get a list of all different permutations
def getPermutationList(elements, tempPerm, currList):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        getPermutationList(elements, tempPerm, currList)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            try:
                index = currList[str(temp)]
                return
            except: pass
            
        # also try the reverse permutation
        temp.reverse()
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            try:
                index = currList[str(temp)]
                return
            except: pass
        currList[str(tempPerm)] = copy(tempPerm)
    


###########################################################################################
##### CLASS : OWPolyvizGRAPH
###########################################################################################
class OWPolyvizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.localScaledData = []
        self.attrLocalValues = {}
        self.lineLength = 2*0.05

    def setLineLength(self, len):
        self.lineLength = len*0.05

    #
    # if we use globalScaling we must also create a copy of localy scaled data
    #
    def setData(self, data):
        # first call the original function to scale data
        OWVisGraph.setData(self, data)

        if data == None: return

        self.localScaledData = []        
        if self.jitteringType != 'none':
            for index in range(len(data.domain)):
                scaled, values = self.scaleData(data, index, jitteringEnabled = 0)
                self.localScaledData.append(scaled)
                self.attrLocalValues[data.domain[index].name] = values

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className, statusBar):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        # we must have at least 3 attributes to be able to show anything
        if len(labels) < 3: return

        self.statusBar = statusBar        

        if len(self.scaledData) == 0 or len(self.localScaledData) == 0 or len(labels) == 0: self.updateLayout(); return

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setTickLength(1, 1, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setTickLength(1, 1, 0)
        
        self.setAxisScale(QwtPlot.xBottom, -1.1, 1.1, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.1, 1.1, 1)

        length = len(labels)
        indices = []
        xs = []

        ###########
        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        ###########
        # create anchor for two edges of every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        ###########
        # draw polygon
        xData = []; yData = []
        for i in range(len(labels)+1):
            x = math.cos(2*math.pi * float(i) / float(len(labels)))
            y = math.sin(2*math.pi * float(i) / float(len(labels)))
            xData.append(x)
            yData.append(y)
        newCurveKey = self.insertCurve("polygon")
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.setCurveStyle(newCurveKey, QwtCurve.Lines)
        self.setCurveData(newCurveKey, xData, yData) 

        ###########
        # draw text at lines
        for i in range(length):
            # attribute name
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(0.65*(anchors[0][i]+anchors[0][(i+1)%length]))
            self.marker(mkey).setYValue(0.65*(anchors[1][i]+anchors[1][(i+1)%length]))
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)

            if self.rawdata.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = self.getVariableValuesSorted(self.rawdata, labels[i])
                count = len(values)
                k = 1.15
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    mkey = self.insertMarker(values[j])
                    self.marker(mkey).setXValue(k*(1-pos)*anchors[0][i]+k*pos*anchors[0][(i+1)%length])
                    self.marker(mkey).setYValue(k*(1-pos)*anchors[1][i]+k*pos*anchors[1][(i+1)%length])
                    self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
            else:
                # min value
                name = "%.3f" % (self.attrLocalValues[labels[i]][0])
                mkey = self.insertMarker(name)
                self.marker(mkey).setXValue(0.95*anchors[0][i]+0.15*anchors[0][(i+1)%length])
                self.marker(mkey).setYValue(0.95*anchors[1][i]+0.15*anchors[1][(i+1)%length])
                self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
                # max value
                name = "%.3f" % (self.attrLocalValues[labels[i]][1])
                mkey = self.insertMarker(name)
                self.marker(mkey).setXValue(0.15*anchors[0][i]+0.95*anchors[0][(i+1)%length])
                self.marker(mkey).setYValue(0.15*anchors[1][i]+0.95*anchors[1][(i+1)%length])
                self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)


        #self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        #self.updateLayout()


        # if we don't want coloring
        if className == "(One color)":      
            valLen = 1

        # if we have a discrete class
        elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:    
            valLen = len(self.rawdata.domain[className].values)
            # we create a hash table of variable values and their indices
            classValueIndices = self.getVariableValueIndices(self.rawdata, className)

        # if we have a continuous class
        else:
            valLen = 0
            if className != "(One color)" and className != '':
                scaledClassData, vals = self.scaleData(self.rawdata, className, forColoring = 1) # scale class data for coloring

        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y
        contData = []    # list to store color, x and y position of data items in case of continuous class
        
        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------
        RECT_SIZE = 0.01    # size of tooltip rectangle in percents of graph size
        for i in range(dataSize):
            sum_i = 0.0
            for j in range(length):
                sum_i += self.scaledData[indices[j]][i]

            if sum_i == 0.0: sum_i = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero

            ##########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            xDataAnchors = []; yDataAnchors = []
            for j in range(length):
                index = indices[j]
                val = self.localScaledData[index][i]
                xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%length]*val
                yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%length]*val
                x_i += xDataAnchor * (self.scaledData[index][i] / sum_i)
                y_i += yDataAnchor * (self.scaledData[index][i] / sum_i)
                xDataAnchors.append(xDataAnchor)
                yDataAnchors.append(yDataAnchor)
                

            ##########
            # we add a tooltip for this point
            text= self.getExampleText(self.rawdata, self.rawdata[i])
            r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
            self.tips.addToolTip(r, text)

            lineColor = QColor(0,0,0)
            if valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
                lineColor.setHsv(0, 255, 255)
            elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][className].value]][1].append(y_i)
                lineColor.setHsv(classValueIndices[self.rawdata[i][className].value] * 360/(valLen), 255, 255)
            else:
                contData.append([x_i, y_i, scaledClassData[i] * 360]) # store data for drawing
                lineColor.setHsv(scaledClassData[i] * 360, 255, 255)

            # draw the data line
            for j in range(length):
                dist = EuclDist([x_i, y_i], [xDataAnchors[j] , yDataAnchors[j]])
                if dist == 0: kvoc = 0
                else: kvoc = float(self.lineLength) / dist
                if kvoc > 1:    # we don't make extrapolation
                    x_j = x_i
                    y_j = y_i
                else:
                    x_j = (1.0 - kvoc)*xDataAnchors[j] + kvoc*x_i
                    y_j = (1.0 - kvoc)*yDataAnchors[j] + kvoc*y_i
                key = self.addCurve('line' + str(i), lineColor, lineColor, 0, QwtCurve.Lines, symbol = QwtSymbol.None)
                self.setCurveData(key, [xDataAnchors[j], x_j], [yDataAnchors[j], y_j])
                

        if valLen == 1 or self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                newColor = QColor()
                newColor.setHsv(i*360/(valLen), 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, curveData[i][0], curveData[i][1])
        else:
            for i in range(len(contData)):
                newColor = QColor()
                newColor.setHsv(contData[i][2], 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, [contData[i][0]], [contData[i][1]])

        #################
        # draw the legend
        if className != "(One color)" and self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            classVariableValues = self.getVariableValuesSorted(self.rawdata, className)
            for index in range(len(classVariableValues)):
                newColor = QColor()
                newColor.setHsv(index*360/(valLen), 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                y = 1.08 - index * 0.05
                self.setCurveData(key, [0.95, 0.95], [y, y])
                mkey = self.insertMarker(classVariableValues[index])
                self.marker(mkey).setXValue(0.90)
                self.marker(mkey).setYValue(y)
                self.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)


        # -----------------------------------------------------------
        # -----------------------------------------------------------
        

    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalAttrOrder(self, attrList, className):
        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Continuous:
            print "incorrect class name for computing optimal ordering. A discrete class must be selected."
            return attrList

        # we have to create a copy of scaled data, because we don't know if the data in self.scaledData was made with jittering
        selectedLocScaledData = []; selectedGlobScaledData = []
        for i in range(len(self.rawdata.domain)): selectedLocScaledData.append([]); selectedGlobScaledData.append([])

        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValsCount = len(self.rawdata.domain[className].values)

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)
            scaled, vals = self.scaleData(self.rawdata, index, jitteringEnabled = 0)
            selectedLocScaledData[index] = scaled

        if self.globalValueScaling == 1:
            min = -1; max = -1
            for attr in attrList:
                if self.rawdata.domain[attr].varType == orange.VarTypes.Discrete: continue
                index = self.scaledDataAttributes.index(attr)
                (minVal, maxVal) = self.getMinMaxVal(self.rawdata, index)
                if attr == attrList[0]:
                    min = minVal; max = maxVal
                else:
                    if minVal < min: min = minVal
                    if maxVal > max: max = maxVal

            for attr in attrList:
                index = self.scaledDataAttributes.index(attr)
                scaled, values = self.scaleData(self.rawdata, index, min, max, jitteringEnabled = 0)
                selectedGlobScaledData[index] = scaled
        else:
            selectedGlobScaledData = selectedLocScaledData

        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(attrListLength):
            x = math.cos(2*math.pi * float(i) / float(attrListLength)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(attrListLength)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        # we create a hash table of variable values and their indices
        classValueIndices = self.getVariableValueIndices(self.rawdata, className)

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            temp = 0
            for j in range(attrListLength):
                temp += selectedGlobScaledData[indices[j]][i]
            sum_i.append(temp)

        # create all possible circular permutations of this indices
        indPermutations = {}
        getPermutationList(indices, [], indPermutations)

        bestPerm = []; bestPermValue = 10000000000  # we search for minimum bestPermValue
        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            curveData = []
            for i in range(classValsCount): curveData.append([ [] , [] ])   # we create classValsCount empty lists with sublists for x and y
            
            for i in range(dataSize):
                ##########
                # calculate projections
                x_i = 0.0; y_i = 0.0
                for j in range(attrListLength):
                    index = permutation[j]
                    val = selectedGlobScaledData[index][i]
                    xDataAnchor = anchors[0][j]*val + anchors[0][j%attrListLength]*(1.0-val)
                    yDataAnchor = anchors[1][j]*val + anchors[1][j%attrListLength]*(1.0-val)
                    x_i += xDataAnchor * (selectedGlobScaledData[index][i] / sum_i[i])
                    y_i += yDataAnchor * (selectedGlobScaledData[index][i] / sum_i[i])

                    
                curveData[classValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][className].value]][1].append(y_i)
            
            sumSameClass = 0.0; sumDiffClass = 0.0
            K_NEIGHB = 5
            for attrValInd in range(classValsCount):
                # sum shortest distances for within class instances
                for i1 in range(len(curveData[attrValInd])):
                    sameClass = [];
                    for i2 in range(len(curveData[attrValInd])):
                        if i1 == i2: continue
                        val = EuclDist([curveData[attrValInd][0][i1], curveData[attrValInd][1][i1]], [curveData[attrValInd][0][i2], curveData[attrValInd][1][i2]])
                        addToList(sameClass, val, 1);
                    for item in sameClass: sumSameClass += math.log(10*item)

                # sum shortest distances between instances in different classes
                for i1 in range(len(curveData[attrValInd])):
                    diffClass = [];
                    for attrValInd2 in range(classValsCount):
                        if attrValInd == attrValInd2: continue
                        for i2 in range(len(curveData[attrValInd2])):
                            val = EuclDist([curveData[attrValInd][0][i1], curveData[attrValInd][1][i1]], [curveData[attrValInd2][0][i2], curveData[attrValInd2][1][i2]])
                            addToList(diffClass, val, K_NEIGHB)
                    for item in diffClass: sumDiffClass += math.log(10*item)

            #if (sumSameClass / sumDiffClass) < bestPermValue:
            if (1 / sumDiffClass) < bestPermValue:
                #bestPermValue = sumSameClass / 5*sumDiffClass
                bestPermValue = 1 / sumDiffClass
                bestPerm = permutation
                print bestPermValue ," - " ,str(bestPerm)

        # return best permutation
        retList = []
        for i in bestPerm:
            retList.append(self.scaledDataAttributes[i])
        return retList

    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWPolyvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
