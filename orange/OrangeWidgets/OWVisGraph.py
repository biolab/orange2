#
# OWVisGraph.py
#
# extension for the base graph class that is used in all visualization widgets
from OWGraph import *
from Numeric import *
import sys
import math
import orange
import os.path
from OWTools import *


class OWVisGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.rawdata = []
        self.pointWidth = 5
        self.scaledData = []
        self.scaledDataAttributes = []
        self.jitteringType = 'none'
        self.globalValueScaling = 0
        self.GraphCanvasColor = str(Qt.white.name())

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)
        self.tips = DynamicToolTipFloat()
        self.statusBar = None
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.plotMouseMoved)

    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType

    def setPointWidth(self, pointWidth):
        self.pointWidth = pointWidth

    def setGlobalValueScaling(self, globalScale):
        self.globalValueScaling = globalScale
        
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
                     

    def addCurve(self, name, brushColor, penColor, size, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, enableLegend = 0):
        newCurveKey = self.insertCurve(name)
        newSymbol = QwtSymbol(symbol, QBrush(brushColor), QPen(penColor), QSize(size, size))
        self.setCurveSymbol(newCurveKey, newSymbol)
        self.setCurveStyle(newCurveKey, style)
        self.setCurvePen(newCurveKey, QPen(penColor))
        self.enableLegend(enableLegend, newCurveKey)
        return newCurveKey

    # return string with attribute names and their values for example example
    def getExampleText(self, data, example):
        text = ""
        for i in range(len(data.domain)):
            if data.domain[i].varType == orange.VarTypes.Discrete:
                text = "%s%s = %s ; " % (text, data.domain[i].name, str(example[i].value))
            else:
                text = "%s%s = %.3f ; " % (text, data.domain[i].name, example[i].value)
        return text
    
    # return a list of sorted values for attribute at index index
    def getVariableValuesSorted(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValuesSorted"
            return []
        
        values = list(data.domain[index].values)
        intValues = []
        i = 0
        # do all attribute values containt integers?
        try:
            while i < len(values):
                temp = int(values[i])
                intValues.append(temp)
                i += 1
        except: pass

        # if all values were intergers, we first sort them ascendently
        if i == len(values):
            intValues.sort()
            values = intValues
        out = []
        for i in range(len(values)):
            out.append(str(values[i]))

        return out

    # create a dictionary with variable at index index. Keys are variable values, key values are indices (transform from string to int)
    # in case all values are integers, we also sort them
    def getVariableValueIndices(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValueIndices"
            return {}

        values = self.getVariableValuesSorted(data, index)

        dict = {}
        for i in range(len(values)):
            dict[values[i]] = i
        return dict

    #
    # get min and max value of data attribute at index index
    #
    def getMinMaxVal(self, data, index):
        attr = data.domain[index]

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            count = float(len(attr.values))
            return (0, count-1)
                    
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
            return (min, max)
        print "incorrect attribute type for scaling"
        return (0, 1)
        
    #
    # scale data at index index to the interval 0 to 1
    #
    def scaleData(self, data, index, min = -1, max = -1, forColoring = 0, jitteringEnabled = 1):
        attr = data.domain[index]
        temp = []; values = []

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = self.getVariableValueIndices(data, index)

            count = float(len(attr.values))
            if len(attr.values) > 1: num = float(len(attr.values)-1)
            else: num = float(1)

            if forColoring == 1:
                for i in range(len(data)):
                    val = float(variableValueIndices[data[i][index].value]) / float(count)
                    temp.append(val)
            elif jitteringEnabled == 1:
                for i in range(len(data)):
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count) + self.rndCorrection(0.2/count)
                    temp.append(val)
            else:
                for i in range(len(data)):
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / float(2*count)
                    temp.append(val)
                    
        # is the attribute continuous
        else:
            # if we don't use global normalisation then we first find min and max value
            if min == -1 and max == -1:
                i = 0
                while data[i][attr].isSpecial() == 1: i+=1
                min = data[i][attr].value
                max = data[i][attr].value

                for item in data:
                    if item[attr].isSpecial() == 1: continue
                    if item[attr].value < min:   min = item[attr].value
                    elif item[attr].value > max: max = item[attr].value
            
            diff = max - min
            values = [min, max]

            if forColoring == 1:
                for i in range(len(data)):
                    temp.append((data[i][attr].value - min)*0.85 / diff)        # we make color palette smaller, because red is in the begining and ending of hsv
            else:
                for i in range(len(data)):
                    temp.append((data[i][attr].value - min) / diff)
        return (temp, values)


    def rescaleAttributesGlobaly(self, data, attrList):
        min = -1; max = -1; first = TRUE
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue
            index = self.scaledDataAttributes.index(attr)
            (minVal, maxVal) = self.getMinMaxVal(data, index)
            if first == TRUE:
                min = minVal; max = maxVal
                first = FALSE
            else:
                if minVal < min: min = minVal
                if maxVal > max: max = maxVal

        for attr in attrList:
            index = self.scaledDataAttributes.index(attr)
            scaled, values = self.scaleData(data, index, min, max)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values

    #
    # set new data and scale its values
    #
    def setData(self, data):
        self.rawdata = data
        self.scaledData = []
        self.attrValues = {}
        self.scaledDataAttributes = []
        
        if data == None: return

        min = -1; max = -1; first = TRUE
        if self.globalValueScaling == 1:
            for index in range(len(data.domain)):
                if data.domain[index].varType == orange.VarTypes.Discrete: continue
                (minVal, maxVal) = self.getMinMaxVal(data, index)
                if first == TRUE:
                    min = minVal; max = maxVal
                    first = FALSE
                else:
                    if minVal < min: min = minVal
                    if maxVal > max: max = maxVal

        for index in range(len(data.domain)):
            attr = data.domain[index]
            self.scaledDataAttributes.append(attr.name)
            scaled, values = self.scaleData(data, index, min, max)
            self.scaledData.append(scaled)
            self.attrValues[attr.name] = values

    def plotMouseMoved(self, e):
        x = e.x()
        y = e.y()
        """
        found = 0
        p = QPoint(x,y)
        for i in range(len(self.tips.rects)):
            if self.tips.rects[i].contains(p):
                found = 1
                if self.statusBar != None:
                    self.statusBar.message(self.tips.texts[i])
                    return
        if found == 0 and self.statusBar != None:
            self.statusBar.message("")
        """
        fx = self.invTransform(QwtPlot.xBottom, x)
        fy = self.invTransform(QwtPlot.yLeft, y)
        if self.statusBar != None:
            text = self.tips.maybeTip(fx,fy)
            self.statusBar.message(text)
        #print "fx = " + str(fx) + " ; fy = " + str(fy) + " ; text = " + text