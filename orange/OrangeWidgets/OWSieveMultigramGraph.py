#
# OWSieveMultigramGraph.py
#
# the base for all parallel graphs

from OWVisGraph import *

###########################################################################################
##### CLASS : OWSieveMultigram graph
###########################################################################################
class OWSieveMultigramGraph(OWVisGraph):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.maxLineWidth = 5
        self.pearsonMinRes = 2
        self.pearsonMaxRes = 10

    def setSettings(self, maxLineWidth, independenceKvoc, pearsonMinRes, pearsonMaxRes):
        self.independenceKvoc = independenceKvoc
        self.maxLineWidth = maxLineWidth
        self.pearsonMaxRes = pearsonMaxRes
        self.pearsonMinRes = pearsonMinRes

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, data, labels, probabilities, criteria, statusBar):
        self.removeCurves()
        self.removeMarkers()
        self.tips.removeAll()

        # we must have at least 3 attributes to be able to show anything
        if len(labels) < 3: return

        self.statusBar = statusBar        

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setTickLength(1, 1, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setTickLength(1, 1, 0)
        
        self.setAxisScale(QwtPlot.xBottom, -1.15, 1.15, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.15, 1.15, 1)

        length = len(labels)
        indices = []
        xs = []

        attrNameList = []
        for attr in data.domain.attributes: attrNameList.append(attr.name)
    
        ###########
        # create a table of indices that stores the sequence of variable indices
        for label in labels: indices.append(attrNameList.index(label))

        ###########
        # create anchor for two edges of every attribute
        anchors = [[],[]]
        for i in range(length):
            x = math.cos(2*math.pi * float(i) / float(length)); strX = "%.4f" % (x)
            y = math.sin(2*math.pi * float(i) / float(length)); strY = "%.4f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))

        ###########
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

        ###########
        # draw text at lines
        for i in range(length):
            # attribute name
            mkey = self.insertMarker(labels[i])
            self.marker(mkey).setXValue(0.57*(anchors[0][i]+anchors[0][(i+1)%length]))
            self.marker(mkey).setYValue(0.57*(anchors[1][i]+anchors[1][(i+1)%length]))
            self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)

            if data.domain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = data.domain[labels[i]].values
                count = len(values)
                k = 1.07
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    mkey = self.insertMarker(values[j])
                    self.marker(mkey).setXValue(k*(1-pos)*anchors[0][i]+k*pos*anchors[0][(i+1)%length])
                    self.marker(mkey).setYValue(k*(1-pos)*anchors[1][i]+k*pos*anchors[1][(i+1)%length])
                    self.marker(mkey).setLabelAlignment(Qt.AlignHCenter + Qt.AlignVCenter)

        # -----------------------------------------------------------
        #  create data lines
        # -----------------------------------------------------------

        for attrXindex in range(len(labels)):
            attrXName = labels[attrXindex]

            for attrYindex in range(attrXindex+1, len(labels)):
                attrYName = labels[attrYindex]

                for valXindex in range(len(data.domain[attrXName].values)):
                    valX = data.domain[attrXName].values[valXindex]

                    for valYindex in range(len(data.domain[attrYName].values)):
                        valY = data.domain[attrYName].values[valYindex]

                        ((nameX, countX),(nameY, countY), actual, sum) = probabilities['%s+%s:%s+%s' %(attrXName, valX, attrYName, valY)]

                        # calculate starting and ending coordinates for lines
                        val = (1.0 + 2.0*float(valXindex)) / float(2*len(data.domain[attrXName].values))                    
                        attrXDataAnchorX = anchors[0][attrXindex]*(1-val) + anchors[0][(attrXindex+1)%length]*val
                        attrXDataAnchorY = anchors[1][attrXindex]*(1-val) + anchors[1][(attrXindex+1)%length]*val

                        val = (1.0 + 2.0*float(valYindex)) / float(2*len(data.domain[attrYName].values))                    
                        attrYDataAnchorX = anchors[0][attrYindex]*(1-val) + anchors[0][(attrYindex+1)%length]*val
                        attrYDataAnchorY = anchors[1][attrYindex]*(1-val) + anchors[1][(attrYindex+1)%length]*val

                        #if criteria == "Attribute independence": self.addLineIndependent([attrXDataAnchorX, attrYDataAnchorX], [attrXDataAnchorY, attrYDataAnchorY], countX, countY, actual, sum)
                        #elif criteria == "Attribute independence (Pearson residuals)":
                        self.addLinePearson([attrXDataAnchorX, attrYDataAnchorX], [attrXDataAnchorY, attrYDataAnchorY], countX, countY, actual, sum)

    """                        
    def addLineIndependent(self, xDataList, yDataList, countX, countY, actual, sum):
        independentProb = float(countX*countY)/float(sum*sum)
        m = 2.
        actualProb = (independentProb*m + actual)/float(sum+m)
        print countX, "\t", countY, "\t", actualProb, "\t", independentProb, "\t", sum
        #actualProb = float(actual)/float(sum)

        #print countX, countY, actual
        if (countX*countY < 5) and (actual < 5): return   # in case we have too little examples we don't estimate the deviation from independence

        # compute 2 constants
        constA = -205.0 / float(self.independenceKvoc)
        constB = 255 - constA

        # set color
        if actualProb > independentProb:
            pen = QPen(QColor(0,0,255))
            b = 255
            if independentProb == 0: r = g = constA*actualProb*sum+ 255
            else:                r = g = constA*actualProb/independentProb + 255 - constA   # if actual/independent = 10 --> r=g=255; actual==independent --> r=g=0
            r = g = max(r, 50)   # if actual/independent > 10 --> r=g=50     -- we don't go under 50
            penWidth = int(min((actualProb/independentProb)*(self.maxLineWidth/options.independenceKvoc), self.maxLineWidth))
        else:
            pen = QPen(QColor(255,0,0))
            r = 255
            if actualProb == 0: g = b = constA*independentProb*sum + 255  
            else:           g = b = constA*independentProb/actualProb + 255 - constA   # if independent/actual= 10 --> g=b=255; actual==independent --> r=g=0
            g = b = max(g, 50)  # if actual/independent > 10 --> b=g=50     -- we don't go under 50
            penWidth = int(min((independentProb/actualProb)*(self.maxLineWidth/options.independenceKvoc), self.maxLineWidth))
        color = QColor(r,g,b)

        #print penWidth
        key = self.addCurve('line', color, color, 0, QwtCurve.Lines, symbol = QwtSymbol.None)
        pen = QPen(color, penWidth)
        self.setCurvePen(key, pen)
        self.setCurveData(key, xDataList, yDataList)
        
    """
    def addLinePearson(self, xDataList, yDataList, countX, countY, actual, sum):
        expected = float(countX*countY)/float(sum)
        pearson = (actual - expected) / sqrt(expected)

        if pearson > -self.pearsonMinRes and pearson < self.pearsonMinRes: return # we don't want to draw white lines
        
        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = min(floor(pearson), self.pearsonMaxRes)
            b = 255
            r = g = 255 - intPearson*200.0/float(self.pearsonMaxRes)
            r = g = max(r, 55)  #
            penWidth = int(float(intPearson*self.maxLineWidth)/float(self.pearsonMaxRes))
        elif pearson < 0:
            intPearson = max(ceil(pearson), -self.pearsonMaxRes)
            r = 255
            b = g = 255 + intPearson*200.0/float(self.pearsonMaxRes)
            b = g = max(b, 55)
            penWidth = int(float(intPearson*self.maxLineWidth)/float(-self.pearsonMaxRes))
        color = QColor(r,g,b)
        
        #print penWidth
        key = self.addCurve('line', color, color, 0, QwtCurve.Lines, symbol = QwtSymbol.None)
        pen = QPen(color, penWidth)
        self.setCurvePen(key, pen)
        self.setCurveData(key, xDataList, yDataList)

            
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWSieveMultigramGraph()
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
