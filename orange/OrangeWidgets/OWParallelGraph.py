#
# OWParallelGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *
from OWDistributions import *
from qt import *
from OWTools import *
from qwt import *
from Numeric import *

class OWParallelGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)

        self.showDistributions = 0
        self.hidePureExamples = 1
        self.metaid = -1
        

    def setShowDistributions(self, showDistributions):
        self.showDistributions = showDistributions

    def setShowAttrValues(self, showAttrValues):
        self.showAttrValues = showAttrValues

    def setHidePureExamples(self, hide):
        self.hidePureExamples = hide

    def setData(self, data):
        OWVisGraph.setData(self, data)
        self.metaid = -1
        
    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, className):
        self.removeCurves()
        self.removeMarkers()
        self.axesKeys = []
        self.curveKeys = []

        if len(self.scaledData) == 0 or len(labels) == 0: self.updateLayout(); return

        self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        if (self.showDistributions == 1 or self.showAttrValues == 1) and self.rawdata.domain[labels[len(labels)-1]].varType == orange.VarTypes.Discrete:
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels)-0.5, 1)
        else:   self.setAxisScale(QwtPlot.xBottom, 0, len(labels)-1.0, 1)

        if self.showAttrValues == 1:
            self.setAxisScale(QwtPlot.yLeft, -0.03, 1.03, 1)
        else:
            self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)

        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setTickLength(1, 1, 0)

        self.setAxisMaxMajor(QwtPlot.xBottom, len(labels)-1.0)        
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        
        length = len(labels)
        indices = []
        xs = []

        # create a table of indices that stores the sequence of variable indices
        for label in labels:
            index = self.scaledDataAttributes.index(label)
            indices.append(index)

        # create a table of class values that will be used for coloring the lines
        scaledClassData = []
        classValues = []
        if className != "(One color)" and className != '':
            scaledClassData, classValues = self.scaleData(self.rawdata, className, -1,-1, 1)

        xs = range(length)
        dataSize = len(self.scaledData[0])        

        #############################################
        # if self.hidePureExamples == 1 we have to calculate where to stop drawing lines
        # we do this by adding a integer meta attribute, that for each example stores attribute index, where we stop drawing lines
        dataStop = []
        lastIndex = indices[length-1]
        for i in range(dataSize): dataStop.append(lastIndex)
        classIndex = self.scaledDataAttributes.index(className)
        if self.hidePureExamples == 1 and self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            if self.metaid == -1:
                self.metaid = orange.newmetaid()
                metavar = orange.IntVariable("ItemIndex")
                self.rawdata.domain.addmeta(self.metaid, metavar)
                for i in range(dataSize): self.rawdata[i].setmeta(self.metaid, i)

            for i in range(length-1,-1,-1):
                if self.rawdata.domain[indices[i]].varType != orange.VarTypes.Discrete or labels[i] == className: continue

                attr = self.rawdata.domain[indices[i]]                
                for attrVal in attr.values:
                    tempData = self.rawdata.select({attr.name:attrVal})
                    ind = 0
                    while ind < len(tempData):
                        if tempData[0][classIndex] != tempData[ind][classIndex]: break
                        ind += 1
                    # if all examples belong to one class we repair the meta variable values
                    if ind == len(tempData):
                        val = indices[i]
                        for item in tempData:
                            index = int(item.getmeta(self.metaid))
                            dataStop[index] = val


        #############################################
        # draw the data
        for i in range(dataSize):
            newCurveKey = self.insertCurve(str(i))
            self.curveKeys.append(newCurveKey)
            newColor = QColor()
            if scaledClassData != []:
                newColor.setHsv(scaledClassData[i]*360, 255, 255)
            self.setCurvePen(newCurveKey, QPen(newColor))
            ys = []
            for index in indices:
                ys.append(self.scaledData[index][i])
                if index == dataStop[i]: break
            self.setCurveData(newCurveKey, xs, ys)


        #############################################
        # do we want to show distributions with discrete attributes
        if self.showDistributions and className != "(One color)" and className != "" and self.rawdata.domain[className].varType == orange.VarTypes.Discrete:
            self.showDistributionValues(className, self.rawdata, indices)
            

        curve = subBarQwtPlotCurve(self)
        newColor = QColor()
        newColor.setRgb(0, 0, 0)
        curve.color = newColor
        curve.setBrush(QBrush(QBrush.NoBrush))
        ckey = self.insertCurve(curve)
        self.setCurveStyle(ckey, QwtCurve.UserCurve)
        self.setCurveData(ckey, [1,1], [2,2])



        #############################################
        # draw vertical lines that represent attributes
        for i in range(len(labels)):
            newCurveKey = self.insertCurve(labels[i])
            self.axesKeys.append(newCurveKey)
            self.setCurveData(newCurveKey, [i,i], [0,1])
            if self.showAttrValues == 1:
                attr = self.rawdata.domain[labels[i]]
                if attr.varType == orange.VarTypes.Continuous:
                    strVal = "%.2f" % (self.attrValues[attr.name][0])
                    mkey1 = self.insertMarker(strVal)
                    self.marker(mkey1).setXValue(i)
                    self.marker(mkey1).setYValue(0.0)
                    strVal = "%.2f" % (self.attrValues[attr.name][1])
                    mkey2 = self.insertMarker(strVal)
                    self.marker(mkey2).setXValue(i)
                    self.marker(mkey2).setYValue(1.0)
                    if i == 0:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignRight + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignRight + Qt.AlignTop)
                    elif i == len(labels)-1:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignLeft + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignLeft + Qt.AlignTop)
                    else:
                        self.marker(mkey1).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
                        self.marker(mkey2).setLabelAlignment(Qt.AlignCenter + Qt.AlignTop)
                elif attr.varType == orange.VarTypes.Discrete:
                    attrVals = self.getVariableValuesSorted(self.rawdata, labels[i])
                    valsLen = len(attrVals)
                    for pos in range(len(attrVals)):
                        mkey = self.insertMarker(attrVals[pos])
                        self.marker(mkey).setXValue(i)
                        self.marker(mkey).setYValue(float(1+2*pos)/float(2*valsLen))
                        self.marker(mkey).setLabelAlignment(Qt.AlignRight + Qt.AlignHCenter)
                    
                    

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
        #for i in range(count): colors.append(float(1+2*i)/float(2*count))
        for i in range(count): colors.append(float(i)/float(count))

        # we create a hash table of possible class values (happens only if we have a discrete class)
        classValueIndices = self.getVariableValueIndices(data, className)
        
        for graphAttrIndex in range(len(indices)):
            index = indices[graphAttrIndex]
            if data.domain[index].varType == orange.VarTypes.Discrete:
                attr = data.domain[index]
                attrLen = len(attr.values)
                
                values = []
                totals = [0] * attrLen

                # we create a hash table of variable values and their indices
                variableValueIndices = self.getVariableValueIndices(data, index)
                
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
                    newColor.setHsv(colors[i]*360, 255, 255)
                    curve.color = newColor
                    xData = []
                    yData = []
                    for j in range(attrLen):
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

    
    
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWParallelGraph()
    c.setCoordinateAxes(['red','green','blue','light blue', 'dark blue', 'yellow', 'orange', 'magenta'])
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
