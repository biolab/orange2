"""
<name>LinViz</name>
<description>TO DO</description>
<icon>icons/LinViz.png</icon>
<priority>5200</priority>
"""
# LinViz.py
#
# 

from OWWidget import *
#from random import betavariate 
from OWLinVizGraph import *
#from qt import *
#import orange, orngDimRed
import orngLinVis

###########################################################################################
##### WIDGET : LinViz visualization
###########################################################################################
class OWLinViz(OWWidget):
    settingsList = []
       
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "LinViz", TRUE)

        self.inputs = [("Examples", ExampleTable, self.newdata, 1), ("Classifier", orange.Classifier, self.learner, 0)]
        self.outputs = [] 

        #load settings
        self.loadSettings()
        self.learners = {}  # dictionary of learners
        self.data = None
        self.visualisers = []

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWLinVizGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
                
        self.statusBar.message("")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
       
        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)

 
    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        pass        
             
    def setScaleFactor(self, n):
        self.scaleFactor = float(self.scaleFactorList[n])
        self.graph.scaleFactor = self.scaleFactor
        self.updateGraph()

       
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.jitterSize = self.jitterSize
        self.graph.setData(self.data)
        self.updateGraph()

    def setShowFilledSymbols(self):
        self.showFilledSymbols = not self.showFilledSymbols
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()


    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    # #####################

    def updateGraph(self):
        if self.learners.keys() == [] or self.data == None: return
                
        self.graph.removeCurves()
        self.graph.removeMarkers()

        # init graph
        #self.graph.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        #self.graph.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        #scaleDraw = self.graph.axisScaleDraw(QwtPlot.xBottom)
        #scaleDraw.setOptions(0) 
        #scaleDraw.setTickLength(0, 0, 0)
        #scaleDraw = self.graph.axisScaleDraw(QwtPlot.yLeft)
        #scaleDraw.setOptions(0) 
        #scaleDraw.setTickLength(0, 0, 0)
                
        #self.setAxisScale(QwtPlot.xBottom, -1.22, 1.22, 1)
        #self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)

        for vis in self.visualisers:
            j=0
            for i in range(len(vis.coeff_names)):
                x=0; y=0
                for name in vis.coeff_names[i][1:]:
                    self.graph.addCurve(str(j), QColor(0,0,0), QColor(0,0,0), 2, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x, x+vis.basis_c[j][0]], yData = [y, y+vis.basis_c[j][1]])
                    x += vis.basis_c[j][0]
                    y += vis.basis_c[j][1]
                    self.graph.addMarker(vis.coeff_names[i][0] + " " + name, x, y, Qt.AlignCenter, bold = 1)
                    j += 1

                """
                #self.graph.addCurve(str(i), QColor(0,0,0), QColor(0,0,0), 2, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [0, vis.basis_c[j]], yData = [0, vis.basis_c[j]])
                self.graph.addCurve(str(i), QColor(0,0,0), QColor(0,0,0), 2, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x, vis.basis_c[j][0]], yData = [y, vis.basis_c[j][1]])
                x += vis.basis_c[j][0]
                y += vis.basis_c[j][1]
                self.graph.addMarker(vis.coeff_names[i][0], x, y, Qt.AlignCenter, bold = 1)
                j += 1
                """

            
            xs = [[] for i in range(len(self.data.domain.classVar.values))]
            ys = [[] for i in range(len(self.data.domain.classVar.values))]
            classValues = getVariableValuesSorted(self.data, self.data.domain.classVar.name)
            for i in range(len(vis.example_c)):
                index = classValues.index(self.data[i].getclass().value)
                xs[index].append(vis.example_c[i][0])
                ys[index].append(vis.example_c[i][1])

            colors = ColorPaletteHSV(len(xs))
            for i in range(len(xs)):
                self.graph.addCurve(self.data.domain.classVar.name + " = " + classValues[i], colors.getColor(i), colors.getColor(i), 4, symbol = QwtSymbol.Ellipse, xData = xs[i], yData = ys[i], enableLegend = 1)
            
        #self.graph.update()
        #self.repaint()


    
    # ###### DATA signal ################################
    # receive new data and update all fields
    def newdata(self, data):
        if len(data.domain.classVar.values) != 2:
            print "The domain does not have a binary class. Binary class is required."
            return
        
        # remove missing values
        self.data = orange.Preprocessor_dropMissing(data)
        #self.graph.setData(self.data)
        
        self.compute()        
        self.updateGraph()
    
    def learner(self, learner, id=0):
        print "learner"
        self.learners[id] = learner
        self.compute(learner)

        self.updateGraph()
        
    # ################################################

    def compute(self, learner = None):
        if not self.data: return
        if not learner: learners = self.learners
        else:           learners = {0:learner}

        self.visualisers = []

        for key in learners.keys():
            visualiser = orngLinVis.Visualizer(self.data, learners[key])
            self.visualisers.append(visualiser)

        self.updateGraph()            

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWLinViz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
