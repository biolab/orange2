# <name>2D Misclassified</name>
# <description>2D visualization of correct and incorrect classifications)</description>
# <category>Visualization</category>
# <icon>icons\2DMisclassified.png</icon>
#
# OW2DMisClassified.py
#
# 2D Visualization of correct and incorrect classifications
# of data
# 
from OWWidget import *
from OWGraph import *
from OData import *
from OW2DInteractions import *

class OW2DMisclassified(OW2DInteractions):
    def __init__(self,parent=None):
        OW2DInteractions.__init__(self, parent)
        self.myclassifier = None
        self.inputs.remove("cdata")
        self.addInput("cdata")
        self.addInput("classifier")
	########################################################
    # process classifier signal - set new classifier and redraw
    ########################################################
    def classifier(self, Classifier):
        self.setOutcomeNames(["correct", "incorrect"])
        self.myclassifier = Classifier
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()

	########################################################
    # calculate 2 arrays for each classification - correct and incorrect
    ########################################################
    def calcCurves(self):
        if self.data == None:
            return

        if self.myclassifier == None:
            return

        # initialize the data point sets
        curveDataPoints = []
        for curveIndex in range(len(self.outcomenames)):
            curveDataPoints.append( {'x':[], 'y':[]} )

        # calculate data points
        for item in self.data.table:
            if item[self.outcome].isSpecial():
                continue

            # is the classification correct
            ins = 1
            if self.myclassifier(item) == item.getclass():
                ins = 0

            if item[self.xAxis].isSpecial():
                continue
            if item[self.yAxis].isSpecial():
                continue

            curveDataPoints[ins]['x'].append( item[self.xAxis] + self.rndCorrection() )
            curveDataPoints[ins]['y'].append( item[self.yAxis] + self.rndCorrection() )

        # put data into curves
        for curveIndex in range(len(self.curveKeys)):
            self.graph.setCurveData(self.curveKeys[curveIndex], curveDataPoints[curveIndex]['x'], curveDataPoints[curveIndex]['y'])


	########################################################
	# cdata signal processing
	########################################################
    def cdata(self, data):
        self.data = data
        if self.data == None:
            self.setMainGraphTitle('')
            self.setComboBoxes([])
            self.setOutcomeNames([])
            self.setXAxis(None)
            self.setYAxis(None)
            self.repaint()
            return

        self.setMainGraphTitle(self.data.title)
        self.setComboBoxes(self.data.getVarNames())
        self.setOutcomeNames([])
        self.setXAxis(self.data.getVarNames()[0])
        self.setYAxis(self.data.getVarNames()[1])
        list = self.data.getPotentialOutcomes()
        self.outcome = self.data.getVarNames(None,TRUE).index(list[-1])
        self.calcCurves()
        self.refreshVisibleCurves()
        self.graph.replot()
        self.repaint()


	########################################################
    # set 2 different classifications - correct and incorrect
    ########################################################
    def setOutcomeNames(self, list):
        self.outcomenames = ["correct", "incorrect"]

        # create the appropriate number of curves, one for every outcomename
        # first remove old curves and self.outcomes(QListBox) items if any
        self.graph.removeCurves()
        self.outcomesQLB.clear()
        if len(self.outcomenames) == 0: return

        # create new curves and QListBox items
        self.curveKeys = []
        self.curveColors = []
        for curveIndex in range(2):
            # insert QListBox item
            newColor = QColor()
            newColor.setHsv(curveIndex*360/2, 255, 255)
            self.outcomesQLB.insertItem(ColorPixmap(newColor), self.outcomenames[curveIndex])

            # insert curve
            self.curveColors.append(newColor)
            newSymbol = QwtSymbol(QwtSymbol.Ellipse, QBrush(newColor), QPen(newColor), QSize(self.PointWidth, self.PointWidth))
            newCurveKey = self.graph.insertCurve(self.outcomenames[curveIndex])
            self.curveKeys.append(newCurveKey)
            self.graph.setCurveStyle(newCurveKey, QwtCurve.Dots)
            self.graph.setCurveSymbol(newCurveKey, newSymbol)
        self.outcomesQLB.selectAll(TRUE)


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OW2DMisclassified()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
