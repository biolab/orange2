#
# OWRadvizGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from copy import copy
import time
from operator import add

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
        self.exLabelData = [[],[]] 	# form: [[labels],[indices]]
        self.exAnchorData =[]	# form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.dataMap = {}		# each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []

    
    def drawGnuplot(self, labels, className):
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
        if className == "(One color)":      
            valLen = 1
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
            elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][className].value]][1].append(y_i)

        if className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                self.p.replot(Gnuplot.Data(curveData[i][0], curveData[i][1], with='point 7'))

    def saveGnuplot(self, labels, className):
        if self.p == None:
            self.drawGnuplot(labels, className)

        qfileName = QFileDialog.getSaveFileName("graph.eps","Enhanced post script (*.EPS)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        if ext == "": fileName += ".eps"
        self.p.hardcopy(fileName, enhanced = 1, color = 1)
      

    def setEnhancedTooltips(self, enhanced):
        self.enhancedTooltips = enhanced
        self.dataMap = {}
        self.exLabelData = [[],[]]
        self.exAnchorData = []



    def saveProjectionAsTabData(self, labels, className, fileName):
        if len(self.scaledData) == 0 or len(labels) == 0: return

        length = len(labels)
        indices = []
        xs = []

   
        # ##########
        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.attributeNames.index(label)
            indices.append(index)
        exLabelData = [labels, indices]
        exAnchorData = []
        dataMap = {}

        # ##########
        # create anchor for every attribute
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            exAnchorData.append((float(strX), float(strY), labels[i]))


        valLen = len(self.rawdata.domain[className].values)
        classValueIndices = self.getVariableValueIndices(self.rawdata, className)	# we create a hash table of variable values and their indices            
    
        dataSize = len(self.scaledData[0])
        curveData = []
        for i in range(valLen): curveData.append([ [] , [] ])   # we create valLen empty lists with sublists for x and y

        validData = [1] * dataSize
        for i in range(dataSize):
            for j in range(length):
                if self.scaledData[indices[j]][i] == "?": validData[i] = 0

        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])
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

            example = orange.Example(domain, [x_i, y_i, self.rawdata[i][className]])
            table.append(example)

        orange.saveTabDelimited(fileName, table)


    def getVariableValueIndices(self, data, index):
        if data.domain[index].varType == orange.VarTypes.Continuous:
            print "Invalid index for getVariableValueIndices"
            return {}

        values = self.getVariableValuesSorted(data, index)

        dict = {}
        for i in range(len(values)):
            dict[values[i]] = i
        return dict

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, className, statusBar, showKNNModel = 0, kNeighbours = -1, showCorrect = 1):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        self.statusBar = statusBar        

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
        classNameIndex = self.attributeNames.index(className)
        self.dataMap = {}

        if self.exLabelData[0] != labels:
            # ##########
            # create a table of indices that stores the sequence of variable indices
            indices = []
            for label in labels:
                index = self.attributeNames.index(label)
                indices.append(index)

            # ##########
            # create a table of indices that stores the sequence of variable indices
            self.exLabelData = [labels, indices]
            self.exAnchorData = []

            # ##########
            # create anchor for every attribute
            for i in range(length):
                x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
                y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
                self.exAnchorData.append((float(strX), float(strY), labels[i]))
        else:
            indices = self.exLabelData[1]

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
        newColor.setRgb(140, 140, 140)
        self.setCurveStyle(newCurveKey, QwtCurve.NoCurve)
        self.setCurveSymbol(newCurveKey, QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(10, 10)))
        xArray = []; yArray = []
        for (x,y,label) in self.exAnchorData:
            xArray.append(x), yArray.append(y)
        xArray.append(self.exAnchorData[0][0])
        yArray.append(self.exAnchorData[0][1])
        self.setCurveData(newCurveKey, xArray, yArray) 

        # ##########
        # draw text at anchors
        for i in range(length):
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(self.exAnchorData[i][0]*1.1)
            self.marker(mkey).setYValue(self.exAnchorData[i][1]*1.04)
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)
            font = self.marker(mkey).font(); font.setBold(1); self.marker(mkey).setFont(font)


        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        # -----------------------------------------------------------
        #  create data curves
        # -----------------------------------------------------------
        
        if className == "(One color)":      # if we don't want coloring
            valLen = 1
        elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:    	# if we have a discrete class
            valLen = len(self.rawdata.domain[className].values)
            classValueIndices = self.getVariableValueIndices(self.rawdata, className)	# we create a hash table of variable values and their indices            
        else:	# if we have a continuous class
            valLen = 0


        if showKNNModel == 1:
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain[className]])
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
                x_i += self.exAnchorData[j][0]*(self.scaledData[index][i] / sum_i)
                y_i += self.exAnchorData[j][1]*(self.scaledData[index][i] / sum_i)

            # scale data according to scale factor
            x_i = x_i * self.scaleFactor
            y_i = y_i * self.scaleFactor

            ##########
            # we add a tooltip for this point
            text= self.getShortExampleText(self.rawdata, self.rawdata[i], indices)
            r = QRectFloat(x_i-RECT_SIZE, y_i-RECT_SIZE, 2*RECT_SIZE, 2*RECT_SIZE)
            self.tips.addToolTip(r, text)


            if showKNNModel == 1:
                table.append(orange.Example(domain, [x_i, y_i, self.rawdata[i][className]]))
            elif valLen == 1:
                curveData[0][0].append(x_i)
                curveData[0][1].append(y_i)
            elif self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
                curveData[classValueIndices[self.rawdata[i][className].value]][0].append(x_i)
                curveData[classValueIndices[self.rawdata[i][className].value]][1].append(y_i)
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
        if showKNNModel == 1:
            classValues = list(self.rawdata.domain[className].values)
            knn = orange.kNNLearner(table, k=kNeighbours, rankWeight = 0)

            for j in range(len(table)):
                out = knn(table[j], orange.GetProbabilities)
                prob = out[table[j].getclass()]
                if showCorrect == 1:
                    prob = 1.0 - prob
                    newColor = QColor(prob*255, prob*255, prob*255)
                else:
                    newColor = QColor(prob*255, prob*255, prob*255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, xData = [table[j][0].value], yData = [table[j][1].value])
        # we add computed data in curveData as curves and show it
        elif className == "(One color)" or self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            for i in range(valLen):
                newColor = QColor()
                if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[i]*360, 255, 255)
                else:                                 newColor.setHsv((i*360)/valLen, 255, 255)
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth)
                self.setCurveData(key, curveData[i][0], curveData[i][1])
                #index = classValueIndices[self.rawdata.domain.classVar.values[i]]
                #key = self.addCurve(str(index), newColor, newColor, self.pointWidth)
                #self.setCurveData(key, curveData[index][0], curveData[index][1])


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
                if valLen < len(self.colorHueValues): newColor.setHsv(self.colorHueValues[index]*360, 255, 255)
                else:                                 newColor.setHsv((index*360)/valLen, 255, 255)
                key = self.addCurve(str(index), newColor, newColor, self.pointWidth)
                y = 1.0 - index * 0.05
                self.setCurveData(key, [0.95, 0.95], [y, y])
                mkey = self.insertMarker(classVariableValues[index])
                self.marker(mkey).setXValue(0.90)
                self.marker(mkey).setYValue(y)
                self.marker(mkey).setLabelAlignment(Qt.AlignLeft + Qt.AlignHCenter)

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
                for (xAnchor,yAnchor,label) in self.exAnchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1)
                    self.setCurveStyle(key, QwtCurve.Lines)
                    self.tooltipCurveKeys.append(key)
                    self.setCurveData(key, [x_i, xAnchor], [y_i, yAnchor])

                    # draw text
                    marker = self.insertMarker(str(data[self.attributeNames.index(label)].value))
                    self.tooltipMarkers.append(marker)
                    self.marker(marker).setXValue((x_i + xAnchor)/2.0)
                    self.marker(marker).setYValue((y_i + yAnchor)/2.0)
                    self.marker(marker).setLabelAlignment(Qt.AlignVCenter + Qt.AlignHCenter)
                    font = self.marker(marker).font(); font.setBold(1); self.marker(marker).setFont(font)
                    

        OWVisGraph.onMouseMoved(self, e)
        self.update()



    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, className, kNeighbours):
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
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])

        table = orange.ExampleTable(domain)
                 
        for i in range(dataSize):
            if validData[i] == 0: continue
            
            # calculate projections
            x_i = 0.0; y_i = 0.0
            for j in range(attrListLength):
                x_i = x_i + anchors[0][j]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])
                y_i = y_i + anchors[1][j]*(self.noJitteringScaledData[indices[j]][i] / sum_i[i])
            
            example = orange.Example(domain, [x_i, y_i, self.rawdata[i][className]])
            table.append(example)

        # use knn on every example and compute its accuracy
        classValues = list(self.rawdata.domain[className].values)
        knn = orange.kNNLearner(table, k=kNeighbours, rankWeight = 0)

        tempPermValue = 0.0        
        for j in range(len(table)):
            out = knn(table[j], orange.GetProbabilities)
            index = classValues.index(table[j][2].value)
            tempPermValue += out[index]
            #if out[index] >= 1.0/len(classValues): tempPermValue += 1

        print "k = %3.d, Accuracy: %2.2f%%" % (kNeighbours, tempPermValue*100.0/float(len(table)) )

        return tempPermValue*100.0/float(len(table))



    # #######################################
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getOptimalSeparation(self, attrList, className, kNeighbours, printTime = 1, progressBar = None):
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
        domain = orange.Domain([xVar, yVar, self.rawdata.domain[className]])

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
                
                example = orange.Example(domain, [x_i, y_i, self.rawdata[i][className]])
                table.append(example)


            # use knn on every example and compute its accuracy
            classValues = list(self.rawdata.domain[className].values)
            knn = orange.kNNLearner(table, k=kNeighbours, rankWeight = 0)

            experiments = 0            
            selection = orange.MakeRandomIndices2(table, 1.0-float(self.percentDataUsed)/100.0)
            for j in range(len(table)):
                if selection[j] == 0: continue
                out = knn(table[j], orange.GetProbabilities)
                index = classValues.index(table[j][2].value)
                tempPermValue += out[table[j].getclass()]
                #if out[index] >= 1.0/len(classValues): tempPermValue += 1
                experiments += 1

            print "permutation %6d / %d. Accuracy: %2.2f%%" % (permutationIndex, totalPermutations, tempPermValue*100.0/float(experiments) )

            # save the permutation
            tempList = []
            for i in permutation:
                tempList.append(self.attributeNames[i])
            fullList.append((tempPermValue*100.0/float(experiments), len(table), tempList))

        if printTime:
            secs = time.time() - t
            print "Used time: %d min, %d sec" %(secs/60, secs%60)
            print "------------------------------"

        return fullList

    
    # try all possibilities with numOfAttr attributes or less
    # attrList = list of attributes to choose from
    # kNeighbours = number of neighbours to test
    # maxResultLen = max length of returning list
    def getOptimalSubsetSeparation(self, attrList, className, kNeighbours, numOfAttr, maxResultsLen, progressBar = None):
        full = []

        self.totalPossibilities = 0
        for i in range(numOfAttr, 2, -1):
            self.totalPossibilities += combinations(i, len(attrList))
            
        if progressBar:
            progressBar.setTotalSteps(self.totalPossibilities)
            progressBar.setProgress(0)
                
        for i in range(numOfAttr, 2, -1):
            full1 = self.getOptimalExactSeparation(attrList, [], className, kNeighbours, i, maxResultsLen, progressBar)
            full = full + full1
            
        return full

    # try all posibilities with exactly numOfAttr attributes
    def getOptimalExactSeparation(self, attrList, subsetList, className, kNeighbours, numOfAttr, maxResultsLen, progressBar = None):
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
            return self.getOptimalSeparation(subsetList, className, kNeighbours)

        full1 = self.getOptimalExactSeparation(attrList[1:], subsetList, className, kNeighbours, numOfAttr, maxResultsLen, progressBar)
        subsetList2 = copy(subsetList)
        subsetList2.insert(0, attrList[0])
        full2 = self.getOptimalExactSeparation(attrList[1:], subsetList2, className, kNeighbours, numOfAttr-1, maxResultsLen, progressBar)

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
    c = OWRadvizGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
