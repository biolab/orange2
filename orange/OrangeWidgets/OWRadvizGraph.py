#
# OWRadvizGraph.py
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
def addToList(list, val, ind, maxLen):
    i = 0
    for i in range(len(list)):
        (val2, ind2) = list[i]
        if val < val2:
            list.insert(i, (val, ind))
            if len(list) > maxLen:
                list.remove(list[maxLen])
            return
    if len(list) < maxLen:
        list.insert(len(list), (val, ind))

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
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.kNeighbours = 1

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className, statusBar):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        self.statusBar = statusBar        

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return

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
        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        ###########
        # draw "circle"
        xData = []; yData = []
        circResol = 100
        for i in range(circResol+1):
            x = math.cos(2*math.pi * float(i) / float(circResol))
            y = math.sin(2*math.pi * float(i) / float(circResol))
            xData.append(x)
            yData.append(y)
        newCurveKey = self.insertCurve("circle")
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.setCurveStyle(newCurveKey, QwtCurve.Lines)
        self.setCurveData(newCurveKey, xData, yData) 

        ###########
        # draw dots at anchors
        newCurveKey = self.insertCurve("dots")
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.setCurveStyle(newCurveKey, QwtCurve.NoCurve)
        self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(10, 10)))
        self.setCurveData(newCurveKey, anchors[0]+[anchors[0][0]], anchors[1]+[anchors[1][0]]) 

        ###########
        # draw text at anchors
        for i in range(length):
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(anchors[0][i]*1.04)
            self.marker(mkey).setYValue(anchors[1][i]*1.04)
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)


        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------
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
                scaledClassData, vals = self.scaleData(self.rawdata, className, forColoring = 1)

        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y

        RECT_SIZE = 0.01    # size of rectangle
        for i in range(dataSize):
            sum_i = 0.0
            for j in range(length):
                sum_i += self.scaledData[indices[j]][i]

            if sum_i == 0.0: sum_i = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero

            ##########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            for j in range(length):
                index = indices[j]
                x_i += anchors[0][j]*(self.scaledData[index][i] / sum_i)
                y_i += anchors[1][j]*(self.scaledData[index][i] / sum_i)

            ##########
            # we add a tooltip for this point
            text= self.getExampleText(self.rawdata, self.rawdata[i])
            r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
            self.tips.addToolTip(r, text)


            if valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
            elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][className].value]][1].append(y_i)
            else:
                newColor = QColor()
                newColor.setHsv(scaledClassData[i] * 360, 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, [x_i], [y_i])

        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                newColor = QColor()
                newColor.setHsv(i*360/(valLen), 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, curveData[i][0], curveData[i][1])

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
    def getOptimalSeparation(self, attrList, className, kNeighbours):
        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Continuous:
            print "incorrect class name for computing optimal ordering"
            return attrList

        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValsCount = len(self.rawdata.domain[className].values)
        attr = self.rawdata.domain[className]

        # we have to create a copy of scaled data, because we don't know if the data in self.scaledData was made with jittering
        selectedScaledData = []
        for i in range(len(self.rawdata.domain)): selectedScaledData.append([])        

        # if global value scaling is selected, compute min and max values
        min = -1; max = -1; first = TRUE
        if self.globalValueScaling == 1:
            for attr in attrList:
                if self.rawdata.domain[attr].varType == orange.VarTypes.Discrete: continue
                (minVal, maxVal) = self.getMinMaxVal(self.rawdata, attr)
                if first == TRUE:
                    min = minVal; max = maxVal
                    first = FALSE
                else:
                    if minVal < min: min = minVal
                    if maxVal > max: max = maxVal

        # compute scaled data
        indices = [];
        for label in attrList:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)
            scaled, vals = self.scaleData(self.rawdata, index, jitteringEnabled = 0)
            selectedScaledData[index] = scaled

        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(attrListLength):
            x = math.cos(2*math.pi * float(i) / float(attrListLength)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(attrListLength)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            temp = 0
            for j in range(attrListLength):
                temp += selectedScaledData[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum_i.append(temp)

        # create all possible circular permutations of this indices
        indPermutations = {}
        getPermutationList(indices, [], indPermutations)

        print "all permutations: ", str(len(indPermutations.values()))

        bestPerm = []; bestPermValue = 0  # we search for maximum bestPermValue
        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            curveData = []
            xData = []
            yData = []
            tempPermValue = 0
           
            for i in range(dataSize):
                # calculate projections
                x_i = 0.0; y_i = 0.0
                for j in range(attrListLength):
                    index = permutation[j]
                    x_i = x_i + anchors[0][j]*(selectedScaledData[index][i] / sum_i[i])
                    y_i = y_i + anchors[1][j]*(selectedScaledData[index][i] / sum_i[i])
                
                #curveData.append([x_i, y_i])
                xData.append(x_i)
                yData.append(y_i)

            xVar = orange.FloatVariable("xVar")
            yVar = orange.FloatVariable("yVar")
            domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])
            table = orange.ExampleTable(domain)
            for i in range(len(xData)):
                example = orange.Example(domain, [xData[i], yData[i], self.rawdata[i][className]])
                table.append(example)

            classValues = list(self.rawdata.domain[className].values)
            knn = orange.kNNLearner(table, k = kNeighbours)
            for i in range(len(table)):
                out = knn(table[i], orange.GetProbabilities)
                index = classValues.index(self.rawdata[i][className].value)
                if knn(table[i]) == table[i][2]:
                    tempPermValue += out[index]

            """
            #for every point we find k nearest neighbours and calculate major class value. If this value
            # is the same as the tested value then ok, else we made a mistake.
            # In all possible permutations we choose the one that made the least mistakes
            for i1 in range(len(curveData)):
                neighbours = []
                for i2 in range(len(curveData)):
                    if i1 == i2: continue
                    val = EuclDist([curveData[i1][0], curveData[i1][1]], [curveData[i2][0], curveData[i2][1]])
                    addToList(neighbours, val, i2, kNeighbours);

                # calculate the major class
                classes = [0]*classValsCount
                for (val, ind) in neighbours:
                    classes[self.rawdata.domain[className].values.index(self.rawdata[ind][className].value)]+=1
                max = 0
                for val in classes:
                    if max < val: max = val
                
                #classes.sort(); classes.reverse()
                if classes.count(max) == 1:
                    if self.rawdata[i1][className].value == list(self.rawdata.domain[className].values)[classes.index(max)]: tempPermValue+=1
                else:
                    count = classes.count(max)
                    ind = list(self.rawdata.domain[className].values).index(self.rawdata[i1][className].value)
                    if classes[ind] == max: tempPermValue += 1.0/float(count)
            """

            #for ind in permutation:
            #    print self.rawdata.domain[ind].name
            print "permutation value :", str(tempPermValue)

            if tempPermValue > bestPermValue:
                bestPermValue = tempPermValue
                bestPerm = permutation
                                                   
        # return best permutation
        retList = []
        for i in bestPerm:
            retList.append(self.scaledDataAttributes[i])
        return (retList, bestPermValue)

    def getOptimalSubsetSeparation(self, attrList, subsetList, className, kNeighbours):
        if attrList == []: return ([], 0)
        (list1, v1) = self.getOptimalSubsetSeparation(attrList[1:], subsetList, className, kNeighbours)
        subsetList.insert(0, attrList[0])
        (list2, v2) = self.getOptimalSubsetSeparation(attrList[1:], subsetList, className, kNeighbours)
        if (v1 > v2): return (list1, v1)
        else:         return (list2, v2)
    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
