#
# OWParallelGraph.py
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

class OWParallelGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.scaledData = []
        self.scaledDataAttributes = []
        self.jitteringType = 'none'
        self.showDistributions = 0
        self.GraphCanvasColor = str(Qt.white.name())

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)        
        self.curveIndex = 0

    def setShowDistributions(self, showDistributions):
        self.showDistributions = showDistributions

    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType


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
            if len(attr.values) > 1: num = float(len(attr.values)-1)
            else: num = float(1)

            for i in range(len(data)):
                if data[i][index].isSpecial(): temp.append(0)
                else:
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count) + 0.2 * self.rndCorrection(1.0/count)
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
                if data[i][attr].isSpecial() == 1:  temp.append(0)
                else:                               temp.append((data[i][attr].value - min) / diff)
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
        for index in range(len(data.data.domain)):
            attr = data.data.domain[index]
            self.scaledDataAttributes.append(attr.name)
            scaled = self.scaleData(data.data, index)
            self.scaledData.append(scaled)

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className):
        self.removeCurves()
        self.axesKeys = []
        self.curveKeys = []

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return
        
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        if self.showDistributions == 1 and self.rawdata.data.domain[labels[len(labels)-1]].varType == orange.VarTypes.Discrete:
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels)-0.5, 1)
        else:                           self.setAxisScale(QwtPlot.xBottom, 0, len(labels)-1.0, 1)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisMaxMinor(QwtPlot.yLeft, 0)
        self.setAxisMaxMajor(QwtPlot.yLeft, 1)

        length = len(labels)
        indices = []
        xs = []

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        # create a table of class values that will be used for coloring the lines
        scaledClassData = []
        if className != "(One color)" and className != '':
            ex_jitter = self.jitteringType
            self.setJitteringOption('none')
            scaledClassData = self.scaleData(self.rawdata.data, className)
            self.setJitteringOption(ex_jitter)

        xs = range(length)
        dataSize = len(self.scaledData[0])        
        for i in range(dataSize):
            newCurveKey = self.insertCurve(str(i))
            self.curveKeys.append(newCurveKey)
            newColor = QColor()
            if scaledClassData != []:
                newColor.setHsv(scaledClassData[i]*255, 255, 255)
            self.setCurvePen(newCurveKey, QPen(newColor))
            ys = []
            for index in indices:
                ys.append(self.scaledData[index][i])
            self.setCurveData(newCurveKey, xs, ys)

        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.axesKeys.append(newCurveKey)
            self.setCurveData(newCurveKey, [i,i], [0,1])

        # do we want to show distributions with discrete attributes
        if self.showDistributions and className != "(One color)" and className != "" and self.rawdata.data.domain[className].varType == orange.VarTypes.Discrete:
            self.showDistributionValues(className, self.rawdata.data, indices)
            
        

    def showDistributionValues(self, className, data, indices):
        # get index of class         
        classNameIndex = 0
        for i in range(len(data.domain)):
            if data.domain[i].name == className: classNameIndex = i

        # create color table            
        count = float(len(data.domain[className].values))
        if count < 1:
            count = 1.0

        colors = []
        for i in range(count): colors.append(float(1+2*i)/float(2*count))

        classValueIndices = {}
        # we create a hash table of possible class values (IF we have a discrete class)
        for i in range(count):
            classValueIndices[list(data.domain[className].values)[i]] = i

        for graphAttrIndex in range(len(indices)):
            index = indices[graphAttrIndex]
            if data.domain[index].varType == orange.VarTypes.Discrete:
                attr = data.domain[index]
                attrLen = len(attr.values)
                
                values = []
                totals = [0] * attrLen

                # we create a hash table of variable values and their indices
                variableValueIndices = {}
                for i in range(attrLen):
                    variableValueIndices[attr.values[i]] = i
                
                for i in range(count):
                    values.append([0] * attrLen)

                for i in range(len(data)):
                    if not data[i][index].isSpecial():
                        # processing for distributions
                        attrIndex = variableValueIndices[data[i][index].value]
                        classIndex = classValueIndices[data[i][classNameIndex].value]
                        totals[attrIndex] += 1
                        values[classIndex][attrIndex] = values[classIndex][attrIndex] + 1

                maximum = 0
                for i in range(len(values)):
                    for j in range(len(values[i])):
                        if values[i][j] > maximum: maximum = values[i][j]
                        
                # create bar curve
                for i in range(count):
                    curve = subBarQwtPlotCurve(self)
                    newColor = QColor()
                    newColor.setHsv(colors[i]*255, 255, 255)
                    curve.color = newColor
                    xData = []
                    yData = []
                    for j in range(attrLen):
                        #width = float(values[i][j]*0.5) / float(totals[j])
                        width = float(values[i][j]*0.5) / float(maximum)
                        interval = 1.0/float(2*attrLen)
                        yOff = float(1.0 + 2.0*j)/float(2*attrLen)
                        height = 0.7/float(count*attrLen)

                        yLowBott = yOff - float(count*height)/2.0 + i*height
                        xData.append(graphAttrIndex)
                        xData.append(graphAttrIndex + width)
                        yData.append(yLowBott)
                        yData.append(yLowBott + height)

                    ckey = self.insertCurve(curve)
                    self.setCurveStyle(ckey, QwtCurve.UserCurve)
                    self.setCurveData(ckey, xData, yData)


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
             
    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWParallelGraph()
    c.setCoordinateAxes(['red','green','blue','light blue', 'dark blue', 'yellow', 'orange', 'magenta'])
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
