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
        self.localScaledData = []
        self.attrLocalValues = {}
        self.lineLength = 2*0.05

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
    def updateData(self, labels, className, attributeReverse, statusBar):
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
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        
        self.setAxisScale(QwtPlot.xBottom, -1.20, 1.20, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.20, 1.20, 1)

        length = len(labels)
        indices = []
        xs = []

        ###########
        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        # create anchor for two edges of every attribute
        anchors = self.createAnchors(len(labels))
        

        # ##########
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

        # ##########
        # draw text at lines
        for i in range(length):
            # print attribute name
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(0.6*(anchors[0][i]+anchors[0][(i+1)%length]))
            self.marker(mkey).setYValue(0.6*(anchors[1][i]+anchors[1][(i+1)%length]))
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
            font = self.marker(mkey).font(); font.setBold(1); self.marker(mkey).setFont(font)

            if self.rawdata.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = self.getVariableValuesSorted(self.rawdata, labels[i])
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    mkey = self.insertMarker(values[j])
                    if attributeReverse[labels[i]] == 0:
                        self.marker(mkey).setXValue(k*(1-pos)*anchors[0][i]+k*pos*anchors[0][(i+1)%length])
                        self.marker(mkey).setYValue(k*(1-pos)*anchors[1][i]+k*pos*anchors[1][(i+1)%length])
                    else:
                        self.marker(mkey).setXValue(k*pos*anchors[0][i]+k*(1-pos)*anchors[0][(i+1)%length])
                        self.marker(mkey).setYValue(k*pos*anchors[1][i]+k*(1-pos)*anchors[1][(i+1)%length])
                    self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
            else:
                # min value
                names = ["%.3f" % (self.attrLocalValues[labels[i]][0]), "%.3f" % (self.attrLocalValues[labels[i]][1])]
                if attributeReverse[labels[i]] == 1: names.reverse()
                mkey = self.insertMarker(names[0])
                self.marker(mkey).setXValue(0.95*anchors[0][i]+0.15*anchors[0][(i+1)%length])
                self.marker(mkey).setYValue(0.95*anchors[1][i]+0.15*anchors[1][(i+1)%length])
                self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
                # max value
                mkey = self.insertMarker(names[1])
                self.marker(mkey).setXValue(0.15*anchors[0][i]+0.95*anchors[0][(i+1)%length])
                self.marker(mkey).setYValue(0.15*anchors[1][i]+0.95*anchors[1][(i+1)%length])
                self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)


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

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        sum = self.calculateAttrValuesSum(self.scaledData, len(self.rawdata), indices, validData)

        # ##########
        #  create data curves
        RECT_SIZE = 0.01    # size of tooltip rectangle in percents of graph size
        for i in range(dataSize):
            if validData[i] == 0: continue
            
            # #########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            xDataAnchors = []; yDataAnchors = []
            for j in range(length):
                index = indices[j]
                val = self.localScaledData[index][i]
                if attributeReverse[labels[j]] == 1: val = 1-val
                xDataAnchor = anchors[0][j]*(1-val) + anchors[0][(j+1)%length]*val
                yDataAnchor = anchors[1][j]*(1-val) + anchors[1][(j+1)%length]*val
                x_i += xDataAnchor * (self.scaledData[index][i] / sum[i])
                y_i += yDataAnchor * (self.scaledData[index][i] / sum[i])
                xDataAnchors.append(xDataAnchor)
                yDataAnchors.append(yDataAnchor)
                

            # #########
            # we add a tooltip for this point
            text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices)
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
            mkey = self.insertMarker(className)
            self.marker(mkey).setXValue(0.87)
            self.marker(mkey).setYValue(1.06)
            self.marker(mkey).setLabelAlignment(Qt.AlignLeft)
            
            classVariableValues = self.getVariableValuesSorted(self.rawdata, className)
            for index in range(len(classVariableValues)):
                newColor = QColor()
                newColor.setHsv(index*360/(valLen), 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                y = 1.0 - index * 0.05
                self.setCurveData(key, [0.95, 0.95], [y, y])
                mkey = self.insertMarker(classVariableValues[index])
                self.marker(mkey).setXValue(0.90)
                self.marker(mkey).setYValue(y)
                self.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)


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
    def getOptimalSeparation(self, attrList, attrReverseDict, className, kNeighbours, printTime = 1, progressBar = None):
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
        classValueIndices = self.getVariableValueIndices(self.rawdata, className)

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            index = self.attributeNames.index(label)
            indices.append(index)
            scaled, vals = self.scaleData(self.rawdata, index, jitteringEnabled = 0)
            selectedLocScaledData[index] = scaled

        if self.globalValueScaling == 1:
            (min, max) =  self.getMinMaxValDomain(self.rawdata, attrList)
            
            for attr in attrList:
                index = self.attributeNames.index(attr)
                scaled, values = self.scaleData(self.rawdata, index, min, max, jitteringEnabled = 0)
                selectedGlobScaledData[index] = scaled
        else:
            selectedGlobScaledData = selectedLocScaledData

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
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])

        # which data items have all values valid
        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        print "Nr. of examples: ", str(count)

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

                tempPermValue = 0
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
                       
                    example = orange.Example(domain, [x_i, y_i, self.rawdata[i][className]])
                    table.append(example)

                """
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
                    sumVal = 0
                    for val in prob: sumVal += val
                    
                    index = classValues.index(table[i].getclass().value)
                    tempPermValue += float(prob[index])/float(sumVal)
                """

                # to bo delalo, ko bo popravljen orangov kNNLearner
                classValues = list(self.rawdata.domain[className].values)
                knn = orange.kNNLearner(table, k=kNeighbours)
                for j in range(len(table)):
                    out = knn(table[j], orange.GetProbabilities)
                    index = classValues.index(table[j][2].value)
                    tempPermValue += out[index]

                print "permutation %6d / %d. Accuracy: %2.2f%%" % (permutationIndex, totalPermutations, tempPermValue*100.0/float(len(table)) )

                # save the permutation
                tempList = []
                for i in permutation:
                    tempList.append(self.attributeNames[i])
                fullList.append((tempPermValue*100.0/float(len(table)), len(table), tempList, attrOrder))

        if printTime:
            print "------------------------------"
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)

        return fullList
                
    def getOptimalSubsetSeparation(self, attrList, attrReverseDict, className, kNeighbours, numOfAttr, maxResultsLen, progressBar = None):
        full = []
        
        totalPossibilities = 0
        for i in range(numOfAttr, 2, -1):
            totalPossibilities += combinations(i, len(attrList))

        if progressBar:
            progressBar.setTotalSteps(totalPossibilities)
            progressBar.setProgress(0)
                
        for i in range(numOfAttr, 2, -1):
            full1 = self.getOptimalExactSeparation(attrList, [], attrReverseDict, className, kNeighbours, i, maxResultsLen, progressBar)
            full = full + full1
            while len(full) > maxResultsLen:
                el = min(full)
                full.remove(el)
            
        return full

    def getOptimalExactSeparation(self, attrList, subsetList, attrReverseDict, className, kNeighbours, numOfAttr, maxResultsLen, progressBar = None):
        if attrList == [] or numOfAttr == 0:
            if len(subsetList) < 3 or numOfAttr != 0: return []
            if progressBar:
                progressBar.setProgress(progressBar.progress()+1)
                print progressBar.progress()
            
            print subsetList
            return self.getOptimalSeparation(subsetList, attrReverseDict, className, kNeighbours, printTime = 0)

        full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, attrReverseDict, className, kNeighbours, numOfAttr, maxResultsLen, progressBar)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, attrReverseDict, className, kNeighbours, numOfAttr-1, maxResultsLen, progressBar)

        # find max values in booth lists
        full = full1 + full2
        shortList = []
        for i in range(min(maxResultsLen, len(full))):
            item = max(full)
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
