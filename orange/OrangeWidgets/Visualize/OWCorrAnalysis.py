"""
<name>Correspondence analysis</name>
<description>Takes a ExampleTable and makes correspondence analysis</description>
<icon>icons/ca.png</icon>
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

class OWCorrAnalysis(OWWidget):
    settingsList = ['graph.pointWidth', "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale",
                    "graph.showLegend", 'autoSendSelection', "graph.showFilledSymbols", 'toolbarSelection',
                    "colorSettings", "percRadius"]
                    
    contextHandlers = {"": DomainContextHandler("", ["attrCol","attrRow","attrX", "attrY"])}
    
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
        self.percRadius = 10
        
        
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
        
        #col attribute
        self.attrCol = ""
        self.attrColCombo = OWGUI.comboBox(self.GeneralTab, self, "attrCol", " Column Table Attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)

        # row attribute
        self.attrRow = ""
        self.attrRowCombo = OWGUI.comboBox(self.GeneralTab, self, "attrRow", "Row Table Attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)
       
        #x principal axis
        self.attrX = 0
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " Principal axis X ", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)
        
        #y principal axis
        self.attrY = 0
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Principal axis Y ", callback = self.updateGraph, sendSelectedValue = 1, valueType = str)
        
        #zooming
        self.zoomSelectToolbar = ZoomBrowseSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)
        
        # Data Table
        self.DataTab = QTable(None)
        self.DataTab.setSelectionMode(QTable.NoSelection)          
        self.tabsMain.insertTab(self.DataTab, "Data Table")       
        self.DataTab.show()
        
        # Correspondence Table
        self.CorrTab = QTable(None)
        self.CorrTab.setSelectionMode(QTable.NoSelection)
        self.tabsMain.insertTab(self.CorrTab, "Correspondence Table")       
        self.CorrTab.show()
  
        # Row profiles 
        self.RowProfilesTab = QTable(None)
        self.RowProfilesTab.setSelectionMode(QTable.NoSelection)          
        self.tabsMain.insertTab(self.RowProfilesTab, "Row profiles")       
        self.RowProfilesTab.show()
        
        # Column profiles 
        self.ColProfilesTab = QTable(None)
        self.ColProfilesTab.setSelectionMode(QTable.NoSelection)          
        self.tabsMain.insertTab(self.ColProfilesTab, "Column profiles")       
        self.ColProfilesTab.show() 
        
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
        self.calcRadius()

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
            self.initAttrValues()            
        else:
            self.data = None
            self.initAttrValues() 
            
        self.openContext("", dataset)
        self.updateGraph()
            
    def fill_table(self, matrix, table):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                OWGUI.tableItem(table, i, j, '%.4f' % matrix[i][j], editType=QTableItem.Never, background = Qt.white)
        
    def fill_headers(self, ca, table, numRows = None, numCols = None):        
        if not numRows:
            table.setNumCols(len(ca.innerDistribution.items()))
            table.setNumRows(len(ca.outerDistribution.items()))
        else:
            table.setNumCols(numCols)
            table.setNumRows(numRows)
        
        hheader = table.horizontalHeader()
        for i,var in enumerate(ca.innerDistribution.items()):
            hheader.setLabel(i, var[0])
            
        vheader = table.verticalHeader()
        for i, var in enumerate(ca.outerDistribution.items()):
            vheader.setLabel(i, var[0])
            
        table.show()
    
    def fill_dataTable(self, ca, c):
        dim = c.dataMatrix.shape
        #data tab
        self.fill_headers(ca, self.DataTab, dim[0] + 1, dim[1] + 1)
        self.DataTab.horizontalHeader().setLabel(dim[1], "Sum")
        self.DataTab.verticalHeader().setLabel(dim[0], "Sum")            
        self.fill_table(c.dataMatrix, self.DataTab)
        #filling row and col sum
        rowSums = sum(c.dataMatrix, 1).reshape(-1,1)
        colSums = sum(c.dataMatrix).reshape(-1,1)
        for i in range(dim[0]):
            OWGUI.tableItem(self.DataTab, i, dim[1], '%.4f' % rowSums[i][0], editType=QTableItem.Never, background = Qt.green)
        for i in range(dim[1]):
            OWGUI.tableItem(self.DataTab, dim[0], i, '%.4f' % colSums[i][0], editType=QTableItem.Never, background = Qt.green)               
        OWGUI.tableItem(self.DataTab, dim[0], dim[1], '%.4f' % sum(colSums), editType=QTableItem.Never, background = Qt.red)          
        
    def fill_corrTable(self, ca, c):  
        dim = c.dataMatrix.shape      
        #contingency tab
        self.fill_headers(ca, self.CorrTab, dim[0] + 1, dim[1] + 1)
        self.CorrTab.horizontalHeader().setLabel(dim[1], "Sum")
        self.CorrTab.verticalHeader().setLabel(dim[0], "Sum")            
        self.fill_table(c.corrMatrix, self.CorrTab)
        #filling row and col sum
        for i in range(dim[0]):
            OWGUI.tableItem(self.CorrTab, i, dim[1], '%.4f' % c.rowSums[i][0], editType=QTableItem.Never, background = Qt.green)
        for i in range(dim[1]):
            OWGUI.tableItem(self.CorrTab, dim[0], i, '%.4f' % c.colSums[i][0], editType=QTableItem.Never, background = Qt.green)          
            
    def fill_rowProfiles(self, c):
            dim = c.dataMatrix.shape    
            #row profiles tab
            self.RowProfilesTab.setNumCols(dim[1])
            self.RowProfilesTab.setNumRows(dim[0])
            self.fill_table(c.rowProfiles, self.RowProfilesTab)    
     
    def fill_colProfiles(self, c):
        dim = c.dataMatrix.shape    
        #column profiles tab
        self.ColProfilesTab.setNumCols(dim[0])
        self.ColProfilesTab.setNumRows(dim[1])
        self.fill_table(c.colProfiles, self.ColProfilesTab)         
        
    def initAttrValues(self):
        self.attrRowCombo.clear()
        self.attrColCombo.clear()
 
        if self.data == None: return 
            
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
        ca = orange.ContingencyAttrAttr(self.attrRow, self.attrCol, self.data)
        self.CA = orngCA.CA(ca)            
        
        self.fill_dataTable(ca, self.CA) 
        self.fill_corrTable(ca, self.CA)
        self.fill_rowProfiles(self.CA)
        self.fill_colProfiles(self.CA)
        
        self.initAxesValues()
        
        
    def initAxesValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        
        if self.data == None: return 
            
        for i in range(min(self.CA.D.shape)):
            self.attrXCombo.insertItem(str(i))
            self.attrYCombo.insertItem(str(i))
        
        
        self.attrX = str(self.attrXCombo.text(0))
        if self.attrYCombo.count() > 1: 
            self.attrY = str(self.attrYCombo.text(1))
        else:                           
            self.attrY = str(self.attrYCombo.text(0))
        self.updateGraph()
        
    def updateGraph(self):
        self.graph.zoomStack = []
        if not self.data:
            return        
            
        self.graph.removeAllSelections()
        self.graph.removeBrowsingCurve()        
        self.graph.removeCurves()
        self.graph.removeMarkers()
        
##        self.graph.enableLegend(self.graph.showLegend)
        
        if self.graph.showXaxisTitle == 1: self.graph.setXaxisTitle("Axis " + self.attrX)
        else: self.graph.setXaxisTitle(None)

        if self.graph.showYLaxisTitle == 1: self.graph.setYLaxisTitle("Axis " + self.attrY)
        else: self.graph.setYLaxisTitle(None)        
        
        cor = self.CA.getPrincipalRowProfilesCoordinates((int(self.attrX), int(self.attrY)))        
        tips = [s for s, v in self.CA.contingencyTable.outerDistribution.items()]
        self.plotPoint(cor, 0, tips, "Row points", self.graph.showFilledSymbols)            
            
        cor = self.CA.getPrincipalColProfilesCoordinates((int(self.attrX), int(self.attrY)))        
        tips = [s for s, v in self.CA.contingencyTable.innerDistribution.items()]
        self.plotPoint(cor, 1, tips, "Column points", self.graph.showFilledSymbols)

        self.graph.enableLegend(1)
        self.graph.replot()
        
##        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.showColorLegend, self.attrLabel)
##        self.graph.repaint()        
        
    def plotPoint(self, cor, color, tips, curveName = "", showFilledSymbols = 1):
        fillColor = self.colors[color]
        edgeColor = self.colors[color]
        
##        key = self.graph.addCurve("row" + str(i), fillColor, edgeColor, self.graph.pointWidth, xData = [x], yData = [y])         
        key = self.graph.addCurve(curveName, fillColor, edgeColor, self.graph.pointWidth, xData = list(cor[:, 0]), yData = list(cor[:, 1]), showFilledSymbols = showFilledSymbols)                 
        
        for i in range(len(cor)):
            x = cor[i][0]
            y = cor[i][1]           
         
            self.graph.tips.addToolTip(x, y, tips[i])   
##            self.graph.addMarker(tips[i], x, y)

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
        
        group = QHButtonGroup("Browsing", parent)
        self.buttonBrowse = OWToolbars.createButton(group, "Browsing tool - Rectangle", self.actionBrowse, QPixmap(OWToolbars.dlg_zoom), toggle = 1)
        self.buttonBrowseCircle = OWToolbars.createButton(group, "Browsing tool - Circle", self.actionBrowseCircle, QPixmap(OWToolbars.dlg_zoom), toggle = 1)        
        
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
        else:
            self.buttonZoom.setOn(1)
            if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 0

if __name__=="__main__": 
    appl = QApplication(sys.argv) 
    ow = OWCorrAnalysis() 
    appl.setMainWidget(ow) 
    ow.show() 
    dataset = orange.ExampleTable('smokers_ct.tab') 
    ow.dataset(dataset) 
    appl.exec_loop()            
