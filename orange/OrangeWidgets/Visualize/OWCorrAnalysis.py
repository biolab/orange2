"""
<name>Correspondence Analysis</name>
<description>Takes a ExampleTable and makes correspondence analysis</description>
<icon>icons/CorrespondenceAnalysis.png</icon>
<priority>3300</priority>
"""

from qt import *
from qttable import *
from OWWidget import *
#from OWScatterPlotGraph import *
from OWCorrAnalysisGraph import *
import OWGUI, OWToolbars, OWDlgs
import orngCA
from numpy import *
from OWToolbars import ZoomSelectToolbar

textCorpusModul = 1

try:
  from orngTextCorpus import CategoryDocument, checkFromText
except ImportError:
  textCorpusModul = 0

import os

class OWCorrAnalysis(OWWidget):
    settingsList = ['graph.pointWidth', "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale",
                    "graph.showLegend", 'autoSendSelection', "graph.showFilledSymbols", 'toolbarSelection',
                    "colorSettings", "percRadius"]
                    
    contextHandlers = {"": DomainContextHandler("", ["attrRow", "attrCol"])}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'CorrAnalysis')

        self.inputs = [("Data", ExampleTable, self.dataset)]
        self.outputs = []
        
        self.data = None
        self.CA = None
        self.colors = ColorPaletteHSV(2)
        
        #Locals
        self.showGridlines = 0
        self.autoSendSelection = 0
        self.toolbarSelection = 0
        self.percRadius = 5
        
        
        self.colorSettings = None

        # GUI
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
        layout = QVBoxLayout(self.mainArea)
        self.tabsMain = QTabWidget(self.mainArea, 'tabWidgetMain')
        
        layout.addWidget(self.tabsMain)

        # ScatterPlot
        self.graph = OWCorrAnalysisGraph(None, "ScatterPlot")
        self.tabsMain.insertTab(self.graph, "Scatter Plot") 

        self.icons = self.createAttributeIconDict()
        
        self.textData = False
        if textCorpusModul:
          OWGUI.checkBox(self.GeneralTab, self, 'textData', 'Textual data', callback = self.initAttrValues)
        
        #col attribute
        self.attrCol = ""
        self.attrColCombo = OWGUI.comboBox(self.GeneralTab, self, "attrCol", " Column Table Attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)

        # row attribute
        self.attrRow = ""
        self.attrRowCombo = OWGUI.comboBox(self.GeneralTab, self, "attrRow", "Row Table Attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)
       
        #x principal axis
        self.attrX = 0
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " Principal axis X ", callback = self.contributionBox, sendSelectedValue = 1, valueType = str)
        
        #y principal axis
        self.attrY = 0
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Principal axis Y ", callback = self.contributionBox, sendSelectedValue = 1, valueType = str)
        
        contribution = QVGroupBox('Contribution to inertia', self.GeneralTab)
        self.firstAxis = OWGUI.widgetLabel(contribution, 'Axis 1: 10%')
        self.secondAxis = OWGUI.widgetLabel(contribution, 'Axis 2: 10%')
        
        sliders = QVGroupBox('Percentage of points', self.GeneralTab)
        OWGUI.widgetLabel(sliders, 'Row points')
        self.percRow = 100        
        self.rowSlider = OWGUI.hSlider(sliders, self, 'percRow', minValue=1, maxValue=100, step=10, callback = self.updateGraph)
        OWGUI.widgetLabel(sliders, 'Column points')
        self.percCol = 100        
        self.colSlider = OWGUI.hSlider(sliders, self, 'percCol', minValue=1, maxValue=100, step=10, callback = self.updateGraph)

        
        #zooming
        self.zoomSelectToolbar = ZoomBrowseSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
        
        self.apply = False
        OWGUI.button(self.GeneralTab, self, 'Update Graph', self.buttonUpdate)
            
        # ####################################
        # SETTINGS TAB
        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'graph.pointWidth', box=' Point Size ', minValue=1, maxValue=20, step=1, callback = self.replotCurves)
        
        # general graph settings
        box4 = OWGUI.widgetBox(self.SettingsTab, " General Graph Settings ")
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'X axis title', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showYLaxisTitle', 'Y axis title', callback = self.updateGraph)
##        OWGUI.checkBox(box4, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showLegend', 'Show legend', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showFilledSymbols', 'Show filled symbols', callback = self.updateGraph)        
        OWGUI.checkBox(box4, self, 'showGridlines', 'Show gridlines', callback = self.setShowGridlines)
##        OWGUI.checkBox(box4, self, 'graph.showClusters', 'Show clusters', callback = self.updateGraph, tooltip = "Show a line boundary around a significant cluster")        

        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, " Colors ", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)
        
        #browsing radius
        OWGUI.hSlider(self.SettingsTab, self, 'percRadius', box=' Browsing Curve Size ', minValue = 0, maxValue=100, step=5, callback = self.calcRadius)
        

        self.activateLoadedSettings()
        self.resize(700, 800)        
        
    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette()
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.setGridPen(QPen(dlg.getColor("Grid")))
                
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection, self.zoomSelectToolbar.actionBrowse, self.zoomSelectToolbar.actionBrowseCircle][self.toolbarSelection], []) 
        
    def dataset(self, dataset):
        self.closeContext()
        if dataset:
            self.data = dataset       
            if textCorpusModul:
              self.textData = checkFromText(self.data)
            else:
              self.textData = False
            self.initAttrValues()            
        else:
            self.data = None
            self.initAttrValues() 
            
        self.openContext("", dataset)
        self.buttonUpdate()
            
    def initAttrValues(self):
        self.attrRowCombo.clear()
        self.attrColCombo.clear()
 
        if self.data == None: return 
        
        if self.textData:
            self.attrRowCombo.insertItem('document')
            self.attrRowCombo.insertItem('category')
            self.attrColCombo.insertItem('words')
        else:        
            for attr in self.data.domain:
                if attr.varType == orange.VarTypes.Discrete: self.attrRowCombo.insertItem(self.icons[attr.varType], attr.name)
                if attr.varType == orange.VarTypes.Discrete: self.attrColCombo.insertItem(self.icons[attr.varType], attr.name)

        self.attrRow = str(self.attrRowCombo.text(0))
        if self.attrColCombo.count() > 1: 
            self.attrCol = str(self.attrColCombo.text(1))
        else:                           
            self.attrCol = str(self.attrColCombo.text(0))
            
        self.updateTables()
        
    def updateTables(self):
        if self.textData:
            if textCorpusModul:
              data = (self.attrRow == 'document' and [self.data] or [CategoryDocument(self.data).dataCD])[0]
            else:
              data = self.data
            metas = data.domain.getmetas()
            lenMetas = len(metas)
            caList = []
            for ex in data:
                cur = [0] * lenMetas
                for i, m in zip(range(lenMetas), metas.keys()):
                    try:
                        cur[i] = ex[m].native()
                    except:
                        cur[i] = 0
                caList.append(cur)
            self.CA = orngCA.CA(caList)
            try:
                self.tipsR = [ex['meta'].native() for ex in data]
            except:
                self.tipsR = [ex.name for ex in data]
            self.tipsC = [a.name for a in data.domain.getmetas().values()]
        else:            
            ca = orange.ContingencyAttrAttr(self.attrRow, self.attrCol, self.data)
            caList = [[col for col in row] for row in ca]
            self.CA = orngCA.CA(caList)
            self.tipsR = [s for s, v in ca.outerDistribution.items()]
            self.tipsC = [s for s, v in ca.innerDistribution.items()]
            del ca
               
        self.rowSlider.setMinValue(1)
        self.rowSlider.setMaxValue(len(self.tipsR))
        self.percRow = len(self.tipsR) > 100 and 0.5 * len(self.tipsR) or len(self.tipsR)
        self.colSlider.setMinValue(1)
        self.colSlider.setMaxValue(len(self.tipsC))
        self.percCol = len(self.tipsC) > 100 and 0.5 * len(self.tipsC) or len(self.tipsC)
        
        
        self.initAxesValues()
        self.tabsMain.showPage(self.graph)
        self.calcRadius()
        
        del caList
        
    def initAxesValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        
        if self.data == None: return 
            
        for i in range(1, min(self.CA.D.shape) + 1):
            self.attrXCombo.insertItem(str(i))
            self.attrYCombo.insertItem(str(i))        
        
        self.attrX = str(self.attrXCombo.text(0))
        if self.attrYCombo.count() > 1: 
            self.attrY = str(self.attrYCombo.text(1))
        else:                           
            self.attrY = str(self.attrYCombo.text(0))
            
        self.contributionBox()
        
    def contributionBox(self):
        self.firstAxis.setText ('Axis %d: %f%%' % (int(self.attrX), self.CA.InertiaOfAxis(1)[int(self.attrX)-1]))
        self.secondAxis.setText ('Axis %d: %f%%' % (int(self.attrY), self.CA.InertiaOfAxis(1)[int(self.attrY)-1]))            
        
        self.updateGraph()
        
    def buttonUpdate(self):
        self.apply = True
        self.updateGraph()
        self.apply = False
    def updateGraph(self):
        self.graph.zoomStack = []
        if not self.data:
            return        
        if not self.apply:
            return
            
        self.graph.removeAllSelections()
##        self.graph.removeBrowsingCurve()        
        self.graph.removeCurves()
        self.graph.removeMarkers()
        self.graph.tips.removeAll()
        
        if self.graph.showXaxisTitle == 1: self.graph.setXaxisTitle("Axis " + self.attrX)
        else: self.graph.setXaxisTitle(None)

        if self.graph.showYLaxisTitle == 1: self.graph.setYLaxisTitle("Axis " + self.attrY)
        else: self.graph.setYLaxisTitle(None)        
        
        cor = self.CA.getPrincipalRowProfilesCoordinates((int(self.attrX)-1, int(self.attrY)-1))
        numCor = int(self.percRow)
        indices = self.CA.PointsWithMostInertia(rowColumn = 0, axis = (int(self.attrX)-1, int(self.attrY)-1))[:numCor]
        cor = [cor[i] for i in indices]
        tipsR = [self.tipsR[i] for i in indices]
        self.plotPoint(cor, 0, tipsR, "Row points", self.graph.showFilledSymbols)            
            
        cor = self.CA.getPrincipalColProfilesCoordinates((int(self.attrX)-1, int(self.attrY)-1))      
        numCor = int(self.percCol)
        indices = self.CA.PointsWithMostInertia(rowColumn = 1, axis = (int(self.attrX)-1, int(self.attrY)-1))[:numCor]
        cor = [cor[i] for i in indices]
        tipsC = [self.tipsC[i] for i in indices]
        self.plotPoint(cor, 1, tipsC, "Column points", self.graph.showFilledSymbols)

        self.graph.enableLegend(1)
        self.graph.replot()
    
        
    def plotPoint(self, cor, color, tips, curveName = "", showFilledSymbols = 1):
        fillColor = self.colors[color]
        edgeColor = self.colors[color]
               
        cor = array(cor)
        key = self.graph.addCurve(curveName, fillColor, edgeColor, self.graph.pointWidth, xData = list(cor[:, 0]), yData = list(cor[:, 1]), showFilledSymbols = showFilledSymbols)                 
        
        for i in range(len(cor)):
            x = cor[i][0]
            y = cor[i][1]           
         
            self.graph.tips.addToolTip(x, y, tips[i])   

    def sendSelections(self):
        pass
        
        
    def replotCurves(self):
        for key in self.graph.curveKeys():
            symbol = self.graph.curveSymbol(key)
            self.graph.setCurveSymbol(key, QwtSymbol(symbol.style(), symbol.brush(), symbol.pen(), QSize(self.graph.pointWidth, self.graph.pointWidth)))
        self.graph.repaint()
        
    def setShowGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)        
        
    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_loop():
            self.colorSettings = dlg.getColorSchemas()
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette()
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridPen(QPen(dlg.getColor("Grid")))
            self.updateGraph()

    def createColorDialog(self):
        c = OWDlgs.ColorPalette(self, "Color Palette")
        c.createDiscretePalette(" Discrete Palette ")
        c.createContinuousPalette("contPalette", " Continuous palette ")
        box = c.createBox("otherColors", " Other Colors ")
        c.createColorButton(box, "Canvas", "Canvas color", Qt.white)
        box.addSpace(5)
        c.createColorButton(box, "Grid", "Grid color", Qt.black)
        box.addSpace(5)
        box.adjustSize()
        c.setColorSchemas(self.colorSettings)
        return c    
    
    def calcRadius(self):
        self.graph.radius =  (self.graph.axisScale(QwtPlot.xBottom).hBound() - self.graph.axisScale(QwtPlot.xBottom).lBound()) * self.percRadius / 100.0;
        
class ZoomBrowseSelectToolbar(ZoomSelectToolbar):
    def __init__(self, widget, parent, graph, autoSend = 0):
        ZoomSelectToolbar.__init__(self, widget, parent, graph, autoSend)
        self.widget = widget
        group = QHButtonGroup("Browsing", parent)
        self.buttonBrowse = OWToolbars.createButton(group, "Browsing tool - Rectangle", self.actionBrowse, QPixmap(OWToolbars.dlg_browseRectangle), toggle = 1)
        self.buttonBrowseCircle = OWToolbars.createButton(group, "Browsing tool - Circle", self.actionBrowseCircle, QPixmap(OWToolbars.dlg_browseCircle), toggle = 1)        
        
    def actionZooming(self):
        ZoomSelectToolbar.actionZooming(self)
        if 'buttonBrowse' in dir(self): self.buttonBrowse.setOn(0)
        if 'buttonBrowseCircle' in dir(self): self.buttonBrowseCircle.setOn(0)

    def actionRectangleSelection(self):
        ZoomSelectToolbar.actionRectangleSelection(self)
        if 'buttonBrowse' in dir(self): self.buttonBrowse.setOn(0)
        if 'buttonBrowseCircle' in dir(self): self.buttonBrowseCircle.setOn(0)

    def actionPolygonSelection(self):
        ZoomSelectToolbar.actionPolygonSelection(self)
        if 'buttonBrowse' in dir(self): self.buttonBrowse.setOn(0)
        if 'buttonBrowseCircle' in dir(self): self.buttonBrowseCircle.setOn(0)
        
    def actionBrowse(self):
        state = self.buttonBrowse.isOn()
        self.buttonBrowse.setOn(state)
        self.graph.activateBrowsing(state)
        if state:
            self.buttonBrowseCircle.setOn(0)
            self.buttonZoom.setOn(0)
            self.buttonSelectRect.setOn(0)
            self.buttonSelectPoly.setOn(0)   
            if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 3
            self.widget.calcRadius()
        else:
            self.buttonZoom.setOn(1)            
            if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 0
            
    def actionBrowseCircle(self):
        state = self.buttonBrowseCircle.isOn()
        self.buttonBrowseCircle.setOn(state)
        self.graph.activateBrowsingCircle(state)
        if state:
            self.buttonBrowse.setOn(0)
            self.buttonZoom.setOn(0)
            self.buttonSelectRect.setOn(0)
            self.buttonSelectPoly.setOn(0)
            if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 4
            self.widget.calcRadius()
        else:
            self.buttonZoom.setOn(1)
            if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 0

if __name__=="__main__":
    from orngTextCorpus import *
    import cPickle
##    os.chdir("/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/")
    appl = QApplication(sys.argv) 
    ow = OWCorrAnalysis() 
    appl.setMainWidget(ow) 
    ow.show() 
##    dataset = orange.ExampleTable('/home/mkolar/Docs/Diplomski/repository/orange/doc/datasets/iris.tab') 

##    lem = lemmatizer.FSALemmatization('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/engleski_rjecnik.fsa')
##    for word in loadWordSet('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/engleski_stoprijeci.txt'):
##        lem.stopwords.append(word)       
##    a = TextCorpusLoader('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/reuters-exchanges-small.xml', lem = lem)

    #a = orange.ExampleTable('../../doc/datasets/smokers_ct.tab')
    f=open('../../CDFSallDataCW', 'r')
    a =cPickle.load(f)
    f.close()
    ow.dataset(a) 
    appl.exec_loop()            
