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
import OWGUI, OWToolbars
import orngCA
from numpy import *

class OWCorrAnalysis(OWWidget):
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'CorrAnalysis')

        self.inputs = [("Data", ExampleTable, self.dataset)]
        self.outputs = []
        
        self.data = None
        self.CA = None
        self.colors = ColorPaletteHSV(2)

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
##        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
##        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        #browsing
        self.browseButton = OWToolbars.createButton(self.GeneralTab, "Browsing tool", self.actionBrowse, QPixmap(OWToolbars.dlg_zoom), toggle = 1)
        
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

        self.resize(700, 800)
        
    def dataset(self, dataset):
        if dataset:
            self.data = dataset            
            self.initAttrValues()            
        else:
            self.data = None
            self.initAttrValues() 
            
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
        self.graph.removeCurves()
        self.graph.removeMarkers()
##        self.graph.removeTooltips()
        
        cor = self.CA.getPrincipalRowProfilesCoordinates((int(self.attrX), int(self.attrY)))        
        tips = [s for s, v in self.CA.contingencyTable.outerDistribution.items()]
        self.plotPoint(cor, 0, tips, "Row points")            
            
        cor = self.CA.getPrincipalColProfilesCoordinates((int(self.attrX), int(self.attrY)))        
        tips = [s for s, v in self.CA.contingencyTable.innerDistribution.items()]
        self.plotPoint(cor, 1, tips, "Column points")

        self.graph.enableLegend(1)
        self.graph.replot()
        
##        self.tabsMain.setCurrentPage(1)
        
    def plotPoint(self, cor, color, tips, curveName = ""):
        fillColor = self.colors[color]
        edgeColor = self.colors[color]
        
##        key = self.graph.addCurve("row" + str(i), fillColor, edgeColor, self.graph.pointWidth, xData = [x], yData = [y])         
        key = self.graph.addCurve(curveName, fillColor, edgeColor, self.graph.pointWidth, xData = list(cor[:, 0]), yData = list(cor[:, 1]))                 
        
        for i in range(len(cor)):
            x = cor[i][0]
            y = cor[i][1]           
         
            self.graph.tips.addToolTip(x, y, tips[i])   
##            self.graph.addMarker(tips[i], x, y)

    def actionBrowse(self):
        state = self.browseButton.isOn()
        self.browseButton.setOn(state)
        self.graph.activateBrowsing(state)

##        self.browseButton.setOn(1)
##        self.graph.activateBrowsing(1)

if __name__=="__main__": 
    appl = QApplication(sys.argv) 
    ow = OWCorrAnalysis() 
    appl.setMainWidget(ow) 
    ow.show() 
    dataset = orange.ExampleTable('smokers_ct.tab') 
    ow.dataset(dataset) 
    appl.exec_loop()            
