#
# OWRadvizGraph.py
#
# the base for all parallel graphs

import sys
import math
import orange
import os.path
from OWGraph import *
from OWDistributions import *
from qt import *
from OWTools import *
from qwt import *
from Numeric import *
from copy import copy


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
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.pointWidth = 4
        self.scaledData = []
        self.scaledDataAttributes = []
        self.jitteringType = 'none'
        self.showDistributions = 0
        self.graphCanvasColor = str(Qt.white.name())

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)        
        self.curveIndex = 0

    def setShowDistributions(self, showDistributions):
        self.showDistributions = showDistributions

    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType

    def setPointWidth(self, width):
        self.pointWidth = width
    
    #
    # scale data at index index to the interval 0 - 1
    #
    def scaleData(self, data, index):
        attr = data.domain[index]
        temp = [];
        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = {}
            for i in range(len(attr.values)):
                variableValueIndices[attr.values[i]] = i

            count = float(len(attr.values))
            for i in range(len(data)):
                val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count) + 0.1 * self.rndCorrection(1.0/count)
                temp.append(val)

                    
        # is the attribute continuous
        else:
            # first find min and max value
            i = 0
            while data[i][attr].isSpecial() == 1: i+=1
            min = data[i][attr].value
            max = data[i][attr].value
            for item in data:
                if item[attr].isSpecial() == 1: continue
                if item[attr].value < min:
                    min = item[attr].value
                elif item[attr].value > max:
                    max = item[attr].value

            diff = max - min
            # create new list with values scaled from 0 to 1
            for i in range(len(data)):
                temp.append((data[i][attr].value - min) / diff)
        return temp

    #
    # set new data and scale its values
    #
    def setData(self, data):
        self.rawdata = data
        self.scaledData = []
        self.scaledDataAttributes = []
        
        if data == None: return

        self.distributions = []; self.totals = []
        for index in range(len(data.domain)):
            attr = data.domain[index]
            self.scaledDataAttributes.append(attr.name)
            scaled = self.scaleData(data, index)
            self.scaledData.append(scaled)

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className):
        self.removeCurves()
        self.removeMarkers()

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

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

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

        # draw dots at anchors
        newCurveKey = self.insertCurve("dots")
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        self.setCurveStyle(newCurveKey, QwtCurve.NoCurve)
        self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(10, 10)))
        self.setCurveData(newCurveKey, anchors[0]+[anchors[0][0]], anchors[1]+[anchors[1][0]]) 

        # draw text at anchors
        for i in range(length):
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(anchors[0][i]*1.04)
            self.marker(mkey).setYValue(anchors[1][i]*1.04)
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)


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
            variableValueIndices = {}
            attr = self.rawdata.domain[className]
            for i in range(len(attr.values)):
                variableValueIndices[attr.values[i]] = i
        # if we have a continuous class
        else:
            valLen = 0
            scaledClassData = []
            if className != "(One color)" and className != '':
                ex_jitter = self.jitteringType
                self.setJitteringOption('none')
                scaledClassData = self.scaleData(self.rawdata, className)
                self.setJitteringOption(ex_jitter)

        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y
        
        for i in range(dataSize):
            sum_i = 0.0
            for j in range(length):
                sum_i = sum_i + self.scaledData[indices[j]][i]

            if sum_i == 0.0: continue

            x_i = 0.0
            y_i = 0.0
            for j in range(length):
                index = indices[j]
                x_i = x_i + anchors[0][j]*(self.scaledData[index][i] / sum_i)
                y_i = y_i + anchors[1][j]*(self.scaledData[index][i] / sum_i)

            if valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
            elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
                curveData[variableValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[variableValueIndices[self.rawdata[i][className].value]][1].append(y_i)
            else:
                newCurveKey = self.insertCurve(str(i))
                newColor = QColor()
                newColor.setHsv(scaledClassData[i]*360, 255, 255)
                self.setCurveStyle(newCurveKey, QwtCurve.Dots)
                self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(self.pointWidth, self.pointWidth)))
                self.setCurveData(newCurveKey, [x_i, x_i], [y_i, y_i])

        self.curveColors = []
        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                newCurveKey = self.insertCurve(str(i))
                newColor = QColor()
                newColor.setHsv(i*360/(valLen), 255, 255)
                self.curveColors.append(newColor)
                self.setCurveStyle(newCurveKey, QwtCurve.Dots)
                self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(self.pointWidth, self.pointWidth)))
                self.setCurveData(newCurveKey, curveData[i][0], curveData[i][1])

        # draw the legend
        if className != "(One color)" and self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for index in range(valLen):
                newCurveKey = self.insertCurve(str(index))
                newColor = QColor()
                newColor.setHsv(index*360/(valLen), 255, 255)
                self.setCurveStyle(newCurveKey, QwtCurve.Dots)
                self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(self.pointWidth, self.pointWidth)))
                y = 1.08 - index * 0.05
                self.setCurveData(newCurveKey, [0.95, 0.95], [y, y])
                mkey = self.insertMarker(self.rawdata.domain[className].values[index])
                self.marker(mkey).setXValue(0.90)
                self.marker(mkey).setYValue(y)
                self.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)



        # -----------------------------------------------------------
        # -----------------------------------------------------------
        

    def rndCorrection(self, max):
        """
        returns a number from -max to max, self.jitteringType defines which distribution is to be used.
        function is used to plot data points for categorical variables
        """    
        if self.jitteringType == 'none': 
            return 0.0
        elif self.jitteringType  == 'uniform': 
            return (random() - 0.5)*2*max
        elif self.jitteringType  == 'triangle': 
            b = (1 - betavariate(1,1)) ; return choice((-b,b))*max
        elif self.jitteringType  == 'beta': 
            b = (1 - betavariate(1,2)) ; return choice((-b,b))*max


    # #######################################
    # try to find the optimal attribute order
    def getOptimalAttrOrder(self, attrList, className):
        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Continuous:
            print "incorrect class name for computing optimal ordering"
            return attrList

        # we have to create a copy of scaled data, because we don't know if the data in self.scaledData was made with jittering
        selectedScaledData = []
        for i in range(len(self.rawdata.domain)): selectedScaledData.append([])

        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)
        classValsCount = len(self.rawdata.domain[className].values)
        variableValueIndices = {}
        attr = self.rawdata.domain[className]

        # create a table of indices that stores the sequence of variable indices        
        indices = [];
        for label in attrList:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)
            scaled = self.scaleData(self.rawdata, index)
            selectedScaledData[index] = scaled

        # create anchor for every attribute
        anchors = [[],[]]
        for i in range(attrListLength):
            x = math.cos(2*math.pi * float(i) / float(attrListLength)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(attrListLength)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        # we create a hash table of variable values and their indices
        for i in range(classValsCount):
            variableValueIndices[attr.values[i]] = i

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            temp = 0
            for j in range(attrListLength):
                temp += selectedScaledData[indices[j]][i]
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
                if sum_i[i] == 0.0: continue

                # calculate projections
                x_i = 0.0; y_i = 0.0
                for j in range(attrListLength):
                    index = permutation[j]
                    x_i = x_i + anchors[0][j]*(selectedScaledData[index][i] / sum_i[i])
                    y_i = y_i + anchors[1][j]*(selectedScaledData[index][i] / sum_i[i])
                
                curveData[variableValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[variableValueIndices[self.rawdata[i][className].value]][1].append(y_i)
            
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
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
