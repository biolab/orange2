#
# OWRadvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy
import time
from operator import add
from math import *

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
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.minExamples = 0
        self.percentDataUsed = 100
        self.p = None
        self.anchorData =[]	    # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.dataMap = {}		# each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.kNeighbours = 1
        self.statusBar = None
        self.showLegend = 1

    
    def drawGnuplot(self, labels):
        import Gnuplot

        length = len(labels)
        self.p = Gnuplot.Gnuplot()
        self.p('set noborder')
        
        # circle
        xs = []; ys = []
        for phi in range(0, 361):
             xs.append(cos(phi*math.pi/180.0))
             ys.append(sin(phi*math.pi/180.0))

        self.p.replot(Gnuplot.Data(xs,ys,with='lines'))

        """        
        gplt.hold('on')
        gplt.xaxis((-1.22, 1.22))
        gplt.yaxis((-1.13, 1.13))
        gplt.plot(xy,ys)
        gplt.grid('off')
        """
        
        #anchors
        xs = []; ys = []
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))
            xs.append(x); ys.append(y)
            
        self.p.replot(Gnuplot.Data(xs, ys, with='points 8'))

        # ##########
        # create a table of indices that stores the sequence of variable indices
        indices = []
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------
        # if we don't want coloring
        if self.className == "(One color)":      
            valLen = 1
        elif self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:    
            valLen = len(self.rawdata.domain[self.className].values)
            # we create a hash table of variable values and their indices
            classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)

        # if we have a continuous class
        else:
            valLen = 0
            if self.className != "(One color)" and self.className != '':
                scaledClassData, vals = self.scaleData(self.rawdata, self.className, forColoring = 1)

        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

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

            if valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
            elif self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][self.className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][self.className].value]][1].append(y_i)

        if self.className == "(One color)" or self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                self.p.replot(Gnuplot.Data(curveData[i][0], curveData[i][1], with='point 7'))

    def saveGnuplot(self, labels):
        if self.p == None:
            self.drawGnuplot(labels)

        qfileName = QFileDialog.getSaveFileName("graph.eps","Enhanced post script (*.EPS)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        if ext == "": fileName += ".eps"
        self.p.hardcopy(fileName, enhanced = 1, color = 1)
      

    def setEnhancedTooltips(self, enhanced):
        self.enhancedTooltips = enhanced
        self.dataMap = {}
        self.anchorData = []



    def saveProjectionAsTabData(self, labels, fileName):
        if len(self.scaledData) == 0 or len(labels) == 0: return

        length = len(labels)
        indices = []
        xs = []

   
        # ##########
        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)
        exAnchorData = []
        dataMap = {}

        # ##########
        # create anchor for every attribute
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            exAnchorData.append((float(strX), float(strY), labels[i]))


        valLen = len(self.rawdata.domain[self.className].values)
        classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)	# we create a hash table of variable values and their indices            
    
        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[self.className]])
        table = orange.ExampleTable(domain)

        for i in range(dataSize):
            if validData[i] == 0: continue
            
            sum_i = 0.0
            for j in range(length):
                sum_i += self.noJitteringScaledData[indices[j]][i]
            if sum_i == 0.0: sum_i = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero

            
            ##########
            # calculate the position of the data point
            x_i = 0.0; y_i = 0.0
            for j in range(length):
                index = indices[j]
                x_i += exAnchorData[j][0]*(self.noJitteringScaledData[index][i] / sum_i)
                y_i += exAnchorData[j][1]*(self.noJitteringScaledData[index][i] / sum_i)

            example = orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]])
            table.append(example)

        orange.saveTabDelimited(fileName, table)

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, **args):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
                
        self.setAxisScale(QwtPlot.xBottom, -1.22, 1.22, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)

        length = len(labels)
        xs = []
        self.dataMap = {}
        indices = []
        self.anchorData = []

        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)

        # ##########
        # create anchor for every attribute
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            self.anchorData.append((float(strX), float(strY), labels[i]))


        # ##########
        # draw "circle"
        xdata = []; ydata = []
        for i in range(101):
            xdata.append(math.cos(2*math.pi * float(i) / 100.0))
            ydata.append(math.sin(2*math.pi * float(i) / 100.0))
        self.addCurve("circle", QColor(0,0,0), QColor(0,0,0), 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata, yData = ydata)

        # ##########
        # draw dots at anchors
        xArray = []; yArray = []
        for (x,y,label) in self.anchorData:
            xArray.append(x), yArray.append(y)
        xArray.append(self.anchorData[0][0])
        yArray.append(self.anchorData[0][1])
        self.addCurve("dots", QColor(140,140,140), QColor(140,140,140), 10, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = xArray, yData = yArray)

        # ##########
        # draw text at anchors
        for i in range(length):
            self.addMarker(labels[i], self.anchorData[i][0]*1.1, self.anchorData[i][1]*1.04, Qt.AlignHCenter + Qt.AlignVCenter, bold = 1)


        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------

        if self.className == "(One color)":      # if we don't want coloring
            valLen = 1
        elif self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:    	# if we have a discrete class
            classNameIndex = self.attributeNames.index(self.className)
            valLen = len(self.rawdata.domain[self.className].values)
            classValueIndices = self.getVariableValueIndices(self.rawdata, self.className)	# we create a hash table of variable values and their indices            
        else:	# if we have a continuous class
            valLen = 0
            classNameIndex = self.attributeNames.index(self.className)


        if self.showKNNModel == 1:
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain[self.className]])
            table = orange.ExampleTable(domain)
            

        dataSize = len(self.rawdata)
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        RECT_SIZE = 0.01    # size of rectangle
        newColor = QColor(0,0,0)
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
                x_i += self.anchorData[j][0]*(self.scaledData[index][i] / sum_i)
                y_i += self.anchorData[j][1]*(self.scaledData[index][i] / sum_i)

            # scale data according to scale factor
            x_i = x_i * self.scaleFactor
            y_i = y_i * self.scaleFactor

            ##########
            # we add a tooltip for this point
            text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices)
            r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
            self.tips.addToolTip(r, text)


            if self.showKNNModel == 1:
                table.append(orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]]))
            elif valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
            elif self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][self.className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][self.className].value]][1].append(y_i)
                newColor = QColor()
                newColor.setHsv(self.coloringScaledData[classNameIndex][i] * 360, 255, 255)
            else:
                newColor = QColor()
                newColor.setHsv(self.coloringScaledData[classNameIndex][i] * 360, 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, [x_i], [y_i])

            if self.enhancedTooltips == 1:            
                # create a dictionary value so that tooltips will be shown faster
                data = self.rawdata[i]
                dictValue = "%.1f-%.1f"%(x_i, y_i)
                if not self.dataMap.has_key(dictValue):
                    self.dataMap[dictValue] = []
                self.dataMap[dictValue].append((x_i, y_i, newColor, data))

    
        #################
        if self.showKNNModel == 1:
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
                
        # we add computed data in curveData as curves and show it
        elif self.className == "(One color)" or self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                newColor = QColor()
                if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[i]*360, 255, 255)
                else:                                 newColor.setHsv((i*360)/valLen, 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, xData = curveData[i][0], yData = curveData[i][1])
                #index = classValueIndices[self.rawdata.domain.classVar.values[i]]
                #key = self.addCurve(str(index), newColor, newColor, self.pointWidth)
                #self.setCurveData(key, curveData[index][0], curveData[index][1])


        #################
        # draw the legend
        if self.className != "(One color)" and self.showLegend:
            # show legend for discrete class
            if self.rawdata.domain[self.className].varType == orange.VarTypes.Discrete:
                self.addMarker(self.className, 0.87, 1.06, Qt.AlignLeft)
                    
                classVariableValues = self.getVariableValuesSorted(self.rawdata, self.className)
                for index in range(len(classVariableValues)):
                    newColor = QColor()
                    if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[index]*360, 255, 255)
                    else:                                 newColor.setHsv((index*360)/valLen, 255, 255)
                    y = 1.0 - index * 0.05
                    self.addCurve(str(index), newColor, newColor, self.pointWidth, xData = [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)
            # show legend for continuous class
            else:
                x0 = 1.15; x1 = 1.20
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
            for (x_i, y_i, color, data) in points:
                if abs(x-x_i)+abs(y-y_i) < dist:
                    dist = abs(x-x_i)+abs(y-y_i)
                    nearestPoint = (x_i, y_i, color, data)
           
            if dist < 0.05:
                x_i = nearestPoint[0]; y_i = nearestPoint[1]; color = nearestPoint[2]; data = nearestPoint[3]
                for (xAnchor,yAnchor,label) in self.anchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = self.addMarker(str(data[self.attributeNames.index(label)].value), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    self.tooltipMarkers.append(marker)
                    

        OWVisGraph.onMouseMoved(self, e)
        self.update()



    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, **args):
        self.__dict__.update(args)
        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)

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

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        # store all sums
        sum_i=[]
        for i in range(dataSize):
            if validData[i] == 0: sum_i.append(1.0); continue

            temp = 0    
            for j in range(attrListLength): temp += self.noJitteringScaledData[indices[j]][i]
            if temp == 0.0: temp = 1.0    # we set sum to 1 because it won't make a difference and we prevent division by zero
            sum_i.append(temp)

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[self.className]])

        table = orange.ExampleTable(domain)
                 
        for i in range(dataSize):
            if validData[i] == 0: continue
            
            # calculate projections
            x_i = 0.0; y_i = 0.0
            for j in range(attrListLength):
                x_i = x_i + anchors[0][j]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])
                y_i = y_i + anchors[1][j]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])
            
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
    def getOptimalSeparation(self, attrList, printTime = 1, progressBar = None):
        if self.className == "(One color)":
            print "Unable to compute optimal ordering. Please select class attribute first."
            return []

        # define lenghts and variables
        attrListLength = len(attrList)
        dataSize = len(self.rawdata)

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

        print "Total permutations: ", len(indPermutations.values())

        fullList = []
        permutationIndex = 0 # current permutation index
        totalPermutations = len(indPermutations.values())

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(attrListLength):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        ###################
        # print total number of valid examples
        count = 0
        for i in range(dataSize):
            if validData[i] == 1: count+=1
        print "Nr. of examples: ", str(count)
        if count < self.minExamples:
            print "not enough examples in example table. Ignoring permutation."
            print "------------------------------"
            return []

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
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[self.className]])

        t = time.time()

        if progressBar:
            progressBar.setTotalSteps(len(indPermutations.values()))
            progressBar.setProgress(0)
        
        # for every permutation compute how good it separates different classes            
        for permutation in indPermutations.values():
            permutationIndex += 1
            
            if progressBar != None: progressBar.setProgress(progressBar.progress()+1)           
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
                
                example = orange.Example(domain, [x_i, y_i, self.rawdata[i][self.className]])
                table.append(example)

            tempPermValue = 0.0
            experiments = 0            
            knn = orange.kNNLearner(table, k=self.kNeighbours, rankWeight = 0)
            selection = orange.MakeRandomIndices2(table, 1.0-float(self.percentDataUsed)/100.0)
            
            if table.domain.classVar.varType == orange.VarTypes.Discrete:
                if selection[j] == 0: continue
                classValues = list(self.rawdata.domain[self.className].values)
                for j in range(len(table)):
                    index = classValues.index(table[j][2].value)
                    tempPermValue += knn(table[j], orange.GetProbabilities)[index]
                    experiments += 1
                tempPermValue = tempPermValue*100.0/float(experiments)
                print "permutation %6d / %d. Accuracy: %2.2f%%" % (permutationIndex, totalPermutations, tempPermValue*100.0/float(experiments) )
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
            fullList.append((tempPermValue, len(table), tempList))

        if printTime:
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            print "------------------------------"

        return fullList

    
    # try all possibilities with numOfAttr attributes or less
    # attrList = list of attributes to choose from
    # maxResultLen = max length of returning list
    def getOptimalSubsetSeparation(self, attrList, numOfAttr, maxResultsLen, progressBar = None):
        full = []

        self.totalPossibilities = 0
        self.startTime = time.time()
        for i in range(numOfAttr, 2, -1):
            self.totalPossibilities += combinations(i, len(attrList))
            
        if progressBar:
            progressBar.setTotalSteps(self.totalPossibilities)
            progressBar.setProgress(0)
                
        for i in range(numOfAttr, 2, -1):
            full1 = self.getOptimalExactSeparation(attrList, [], i, maxResultsLen, progressBar)
            full = full + full1
            
        return full

    # try all posibilities with exactly numOfAttr attributes
    def getOptimalExactSeparation(self, attrList, subsetList, numOfAttr, maxResultsLen, progressBar = None):
        if attrList == [] or numOfAttr == 0:
            if len(subsetList) < 3 or numOfAttr != 0: return []
            if progressBar: progressBar.setProgress(progressBar.progress()+1)
            print subsetList
            if self.totalPossibilities > 0 and self.triedPossibilities > 0:
                secs = int(time.time() - self.startTime)
                totalExpectedSecs = int(float(self.totalPossibilities*secs)/float(self.triedPossibilities))
                restSecs = totalExpectedSecs - secs
                print "Used time: %d:%02d:%02d, Remaining time: %d:%02d:%02d (total experiments: %d, rest: %d)" %(secs /3600, (secs-((secs/3600)*3600))/60, secs%60, restSecs /3600, (restSecs-((restSecs/3600)*3600))/60, restSecs%60, self.totalPossibilities, self.totalPossibilities-self.triedPossibilities)
            self.triedPossibilities += 1
            return self.getOptimalSeparation(subsetList)

        full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, numOfAttr, maxResultsLen, progressBar)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, numOfAttr-1, maxResultsLen, progressBar)

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
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
