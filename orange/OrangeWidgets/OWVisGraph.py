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

# ####################################################################
# calculate Euclidean distance between two points
def EuclDist(v1, v2):
    val = 0
    for i in range(len(v1)):
        val += (v1[i]-v2[i])**2
    return sqrt(val)
        

# ####################################################################
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


class OWVisGraph(OWGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)

        self.rawdata = []                   # input data
        self.scaledData = []                # scaled data to the interval 0-1
        self.attributeNames = []      # list of attribute names from self.rawdata
        self.domainDataStat = []
        self.pointWidth = 5
        self.jitteringType = 'none'
        self.jitterSize = 10
        self.globalValueScaling = 0         # do we want to scale data globally
        self.setCanvasColor(QColor(Qt.white.name()))
        self.xpos = 0   # we have to initialize values, since we might get onMouseRelease event before onMousePress
        self.ypos = 0
        self.zoomStack = []
        self.noJitteringScaledData = []

        self.enableGridX(FALSE)
        self.enableGridY(FALSE)

        self.blankClick = 0
        self.noneSymbol = QwtSymbol()
        self.noneSymbol.setStyle(QwtSymbol.None)
        self.tips = DynamicToolTipFloat()
        self.statusBar = None
        self.canvas().setMouseTracking(1)
        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)
        self.zoomStack = []
        self.connect(self, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.onMousePressed)
        self.connect(self, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)


    #####################################################################
    #####################################################################
    # set new data and scale its values
    def setData(self, data):
        self.rawdata = data
        self.domainDataStat = orange.DomainBasicAttrStat(data)
        self.scaledData = []
        self.noJitteringScaledData = []
        self.attrValues = {}
        self.attributeNames = []
        for attr in data.domain: self.attributeNames.append(attr.name)
        if data == None: return
        
        if self.globalValueScaling == 1:
            self.rescaleAttributesGlobaly(data, self.attributeNames)
        else:
            for index in range(len(data.domain)):
                scaled, values = self.scaleData(data, index)
                self.scaledData.append(scaled)
                self.attrValues[data.domain[index].name] = values
        self.scaleDataNoJittering()
    #####################################################################
    #####################################################################


    
    # ####################################################################
    # compute min and max value for a list of attributes 
    def getMinMaxValDomain(self, data, attrList):
        first = TRUE
        min = -1; max = -1
        for attr in attrList:
            if data.domain[attr].varType == orange.VarTypes.Discrete: continue
            (minVal, maxVal) = self.getMinMaxVal(data, attr)
            if first == TRUE:
                min = minVal; max = maxVal
                first = FALSE
            else:
                if minVal < min: min = minVal
                if maxVal > max: max = maxVal
        return (min, max)

    

    # ####################################################################
    # get min and max value of data attribute at index index
    def getMinMaxVal(self, data, index):
        attr = data.domain[index]

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            print "warning. Computing min, max value for discrete attribute."
            return (0, float(len(attr.values))-1)
        else:
            return (self.domainDataStat[index].min, self.domainDataStat[index].max)
        
    # ####################################################################
    # scale data at index index to the interval 0 to 1
    # min, max - if booth -1 --> scale to interval 0 to 1, else scale inside interval [min, max]
    # forColoring - if TRUE we don't scale from 0 to 1 but a little less (e.g. [0.2, 0.8]) so that the colours won't overlap
    # jitteringEnabled - jittering enabled or not
    # ####################################################################
    def scaleData(self, data, index, min = -1, max = -1, forColoring = 0, jitteringEnabled = 1):
        attr = data.domain[index]
        temp = []; values = []

        # is the attribute discrete
        if attr.varType == orange.VarTypes.Discrete:
            # we create a hash table of variable values and their indices
            variableValueIndices = self.getVariableValueIndices(data, index)
            count = float(len(attr.values))
            values = [0, count-1]
            countx2 = float(2*count)	# we compute this value here, so that we don't have to compute it in the loop
            count100 = float(100.0*count) # same

            if forColoring == 1:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = float(variableValueIndices[data[i][index].value]) / float(count)
                    temp.append(val)
            elif jitteringEnabled == 1:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / countx2 + self.rndCorrection(self.jitterSize/count100)
                    temp.append(val)
            else:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    val = (1.0 + 2.0*float(variableValueIndices[data[i][index].value])) / countx2
                    temp.append(val)
                    
        # is the attribute continuous
        else:
            if min == max == -1:
                min = self.domainDataStat[index].min
                max = self.domainDataStat[index].max
            diff = max - min
            values = [min, max]

            if forColoring == 1:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    temp.append((data[i][attr].value - min)*0.85 / diff)        # we make color palette smaller, because red is in the begining and ending of hsv
            else:
                for i in range(len(data)):
                    if data[i][index].isSpecial() == 1: temp.append("?"); continue
                    temp.append((data[i][attr].value - min) / diff)
        return (temp, values)

    # ####################################################################
    # scale data with no jittering
    def scaleDataNoJittering(self):
        # we have to create a copy of scaled data, because we don't know if the data in self.scaledData was made with jittering
        self.noJitteringScaledData = []
        for i in range(len(self.rawdata.domain)):
            scaled, vals = self.scaleData(self.rawdata, i, jitteringEnabled = 0)
            self.noJitteringScaledData.append(scaled)


    def rescaleAttributesGlobaly(self, data, attrList, jittering = 1):
        if len(attrList) == 0: return
        # find min, max values
        (Min, Max) = self.getMinMaxValDomain(data, attrList)

        # scale data values inside min and max
        for attr in attrList:
            index = self.attributeNames.index(attr)
            scaled, values = self.scaleData(data, index, Min, Max, jitteringEnabled = jittering)
            self.scaledData[index] = scaled
            self.attrValues[attr] = values


    # ####################################################################    
    # return a list of sorted values for attribute at index index
    # EXPLANATION: if variable values have values 1,2,3,4,... then their order in orange depends on when they appear first
    # in the data. With this function we get a sorted list of values
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

    # ####################################################################
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


    def setJitteringOption(self, jitteringType):
        self.jitteringType = jitteringType

    def setJitterSize(self, size):
        self.jitterSize = size

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

    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleText(self, data, example):
        text = ""
        for i in range(len(data.domain)):
            if data.domain[i].varType == orange.VarTypes.Discrete:
                if example[i].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[i].name)
                else:
                    text = "%s%s = %s; " % (text, data.domain[i].name, str(example[i].value))
            else:
                if example[i].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[i].name)
                else:
                    text = "%s%s = %.3f; " % (text, data.domain[i].name, example[i].value)
        return text

    # ####################################################################
    # return string with attribute names and their values for example example
    def getShortExampleText(self, data, example, indices):
        text = ""
        for i in range(len(indices)):
            index = indices[i]
            if data.domain[index].varType == orange.VarTypes.Discrete:
                if example[index].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[index].name)
                else:
                    text = "%s%s = %s; " % (text, data.domain[index].name, str(example[index].value))
            else:
                if example[i].isSpecial():
                    text = "%s%s = ?; " % (text, data.domain[index].name)
                else:
                    text = "%s%s = %.3f; " % (text, data.domain[index].name, example[index].value)
        return text

    # ###############################################
    # HANDLING MOUSE EVENTS
    # ###############################################
    def onMousePressed(self, e):
        if Qt.LeftButton == e.button():
            # Python semantics: self.pos = e.pos() does not work; force a copy
            self.xpos = e.pos().x()
            self.ypos = e.pos().y()
            self.enableOutline(1)
            self.setOutlinePen(QPen(Qt.black))
            self.setOutlineStyle(Qwt.Rect)
            self.zooming = 1
            if self.zoomStack == []:
                self.zoomState = (
                    self.axisScale(QwtPlot.xBottom).lBound(),
                    self.axisScale(QwtPlot.xBottom).hBound(),
                    self.axisScale(QwtPlot.yLeft).lBound(),
                    self.axisScale(QwtPlot.yLeft).hBound(),
                    )
        elif Qt.RightButton == e.button():
            self.zooming = 0
        # fake a mouse move to show the cursor position
        self.onMouseMoved(e)
        self.event(e)

    def onMouseReleased(self, e):
        if Qt.LeftButton == e.button():
            xmin = min(self.xpos, e.pos().x())
            xmax = max(self.xpos, e.pos().x())
            ymin = min(self.ypos, e.pos().y())
            ymax = ymin + ((xmax-xmin)*self.height())/self.width()  # compute the last value so that the picture remains its w/h ratio
            #ymax = max(self.ypos, e.pos().y())
            self.setOutlineStyle(Qwt.Cross)
            xmin = self.invTransform(QwtPlot.xBottom, xmin)
            xmax = self.invTransform(QwtPlot.xBottom, xmax)
            ymin = self.invTransform(QwtPlot.yLeft, ymin)
            ymax = self.invTransform(QwtPlot.yLeft, ymax)
            if xmin == xmax or ymin == ymax:
                return
            self.blankClick = 0
            self.zoomStack.append(self.zoomState)
            self.zoomState = (xmin, xmax, ymin, ymax)
            self.enableOutline(0)
        elif Qt.RightButton == e.button():
            if len(self.zoomStack):
                xmin, xmax, ymin, ymax = self.zoomStack.pop()
            else:
                self.blankClick = 1 # we just clicked and released the button at the same position. This is used in OWSmartVisualization
                return

        self.setAxisScale(QwtPlot.xBottom, xmin, xmax)
        self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
        self.replot()
        self.event(e)

    def onMouseMoved(self, e):
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
        self.event(e)