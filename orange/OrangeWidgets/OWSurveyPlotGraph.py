#
# OWParallelGraph.py
#
# the base for all parallel graphs

import sys
import math
import orange
import os.path
from OWGraph import *
from qt import *
from OWTools import *
from qwt import *
from Numeric import *

class OWSurveyPlotGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.scaledData = []
        self.scaledDataAttributes = []
        self.GraphCanvasColor = str(Qt.white.name())

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)        
        self.curveIndex = 0

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
                val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count)
                temp.append(val)
                    
        # is the attribute continuous
        else:
            # first find min and max value
            min = data[0][attr].value
            max = data[0][attr].value
            for item in data:
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
        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return
        
        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.setAxisScale(QwtPlot.xBottom, 0, len(labels), 1)
        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisMaxMinor(QwtPlot.yLeft, 0)
        self.setAxisMaxMajor(QwtPlot.yLeft, len(self.rawdata))

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
            scaledClassData = self.scaleData(self.rawdata, className)

        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.setCurveData(newCurveKey, [i,i], [0,1])
            
        xs = range(length)
        count = len(self.rawdata)
        for i in range(count):
            curve = subBarQwtPlotCurve(self)
            newColor = QColor()
            if scaledClassData != []:
                newColor.setHsv(scaledClassData[i]*360, 255, 255)
            else:
                newColor.setRgb(0,0,0)
            curve.color = newColor
            curve.penColor = newColor
            xData = []; yData = []
            for j in range(length):
                width = self.scaledData[indices[j]][i] * 0.45
                xData.append(j-width)
                xData.append(j+width)
                yData.append(i)
                yData.append(i+1)

            ckey = self.insertCurve(curve)
            self.setCurveStyle(ckey, QwtCurve.UserCurve)
            self.setCurveData(ckey, xData, yData)
            
        


if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWSurveyPlotGraph()
            
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
