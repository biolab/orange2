#
# OWRadvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy

# ####################################################################
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
            if str(temp) in currList: return
            
        # also try the reverse permutation
        temp.reverse()
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)
    





###########################################################################################
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
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
        
        self.setAxisScale(QwtPlot.xBottom, -1.2, 1.2, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.1, 1.1, 1)

        length = len(labels)
        indices = []
        xs = []

        # ##########
        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        # ##########
        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        # ##########
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

        # ##########
        # draw dots at anchors
        newCurveKey = self.insertCurve("dots")
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.setCurveStyle(newCurveKey, QwtCurve.NoCurve)
        self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(10, 10)))
        self.setCurveData(newCurveKey, anchors[0]+[anchors[0][0]], anchors[1]+[anchors[1][0]]) 

        # ##########
        # draw text at anchors
        for i in range(length):
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(anchors[0][i]*1.1)
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

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        RECT_SIZE = 0.01    # size of rectangle
        for i in range(dataSize):
            if validData[i] == 0: continue
            
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

        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(attrListLength):
            x = math.cos(2*math.pi * float(i) / float(attrListLength)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(attrListLength)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))


        indices = []
        for attr in attrList:
            indices.append(self.attributeNames.index(attr))


        # create all possible circular permutations of this indices
        print "----------------------------"
        print "generating permutations. Please wait"
        indPermutations = {}
        getPermutationList(indices, [], indPermutations)

        print "Total permutations: ", str(len(indPermutations.values()))

        bestPerm = []; bestPermValue = 0  # we search for maximum bestPermValue
        fullList = []

        permutationIndex = 0 # current permutation index
        totalPermutations = len(indPermutations.values())

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            if validData[i] == 0:
                sum_i.append(1.0)
                continue

            temp = 0    
            for j in range(attrListLength):
                temp += self.noJitteringScaledData[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum_i.append(temp)

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])
        
        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            permutationIndex += 1
            curveData = []
            tempPermValue = 0

            table = orange.ExampleTable(domain)
           
            for i in range(dataSize):
                if validData[i] == 0: continue
                
                # calculate projections
                x_i = 0.0; y_i = 0.0
                for j in range(attrListLength):
                    index = permutation[j]
                    x_i = x_i + anchors[0][j]*(self.noJitteringScaledData[index][i] / sum_i[i])
                    y_i = y_i + anchors[1][j]*(self.noJitteringScaledData[index][i] / sum_i[i])
                
                example = orange.Example(domain, [x_i, y_i, self.rawdata[i][className]])
                table.append(example)

            #orange.saveTabDelimited("E:\\temp\\data.tab", table)

            classValues = list(self.rawdata.domain[className].values)
            classValNum = len(classValues)
            
            exampleDist = orange.ExamplesDistanceConstructor_Euclidean()
            near = orange.FindNearestConstructor_BruteForce(table, distanceConstructor = exampleDist)
            euclidean = orange.ExamplesDistance_Euclidean()
            euclidean.normalizers = [1,1]   # our table has attributes x,y, and class
            for i in range(len(table)):
                prob = [0]*classValNum
                neighbours = near(kNeighbours, table[i])
                for neighbour in neighbours:
                    dist = euclidean(table[i], neighbour)
                    val = math.exp(-(dist*dist))
                    index = classValues.index(neighbour.getclass().value)
                    prob[index] += val

                # calculate sum for normalization
                sum = 0
                for val in prob: sum += val
                
                index = classValues.index(table[i].getclass().value)
                tempPermValue += float(prob[index])/float(sum)

            """
            # to bo delalo, ko bo popravljen orangov kNNLearner
            classValues = list(self.rawdata.domain[className].values)
            print kNeighbours
            knn = orange.kNNLearner(table, k=40)
            print knn.k, len(table)
            for j in range(len(table)):
                out = knn(table[i], orange.GetProbabilities)
                index = classValues.index(table[i][2].value)
                if knn(table[j]) == table[j][2]:  tempPermValue += out[index]  #tempPermValue += 1
            """
            
            print "permutation %6d / %d. Value : %.2f (Accuracy: %2.2f)" % (permutationIndex, totalPermutations, tempPermValue, tempPermValue*100.0/float(len(table)) )

            if tempPermValue > bestPermValue:
                bestPermValue = tempPermValue
                bestPerm = permutation

            # save the permutation
            tempList = []
            for i in permutation:
                tempList.append(self.attributeNames[i])
            fullList.append(((tempPermValue*100.0/float(len(table)), len(table)), tempList))

        # return best permutation
        retList = []
        for i in bestPerm:
            retList.append(self.attributeNames[i])
        return (retList, bestPermValue, fullList)

    def getOptimalSubsetSeparation(self, attrList, subsetList, className, kNeighbours):
        if attrList == []:
            if len(subsetList) < 2: return ([], 0, [])
            return self.getOptimalSeparation(subsetList, className, kNeighbours)
        (list1, v1, full1) = self.getOptimalSubsetSeparation(attrList[1:], subsetList, className, kNeighbours)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        (list2, v2, full2) = self.getOptimalSubsetSeparation(attrList[1:], subsetList2, className, kNeighbours)

        # find max values in booth lists
        full = full1 + full2
        small = []
        for i in range(min(100, len(full))):
            (val, list) = max(full)
            small.append((val, list))
            full.remove((val, list))
            
        if (v1 > v2): return (list1, v1, small)
        else:         return (list2, v2, small)



if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
