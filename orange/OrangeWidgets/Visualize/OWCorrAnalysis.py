"""
<name>Correspondence Analysis</name>
<description>Takes a ExampleTable and makes correspondence analysis</description>
<icon>icons/CorrespondenceAnalysis.png</icon>
<priority>3300</priority>
"""

from qttable import *
from OWWidget import *
#from OWScatterPlotGraph import *
from OWCorrAnalysisGraph import *
import OWGUI, OWToolbars, OWColorPalette
import orngCA
from numpy import *
from OWToolbars import ZoomSelectToolbar
try:
    import orngText
except:
    pass

textCorpusModul = 1

#try:
#  from orngTextCorpus import CategoryDocument, checkFromText
#except ImportError:
#  textCorpusModul = 0


import os

def checkFromText(data):

    if not isinstance(data, orange.ExampleTable):
        return False
    if len(data.domain.attributes) < 10 and len(data.domain.getmetas(orngText.TEXTMETAID)) > 15:
        return True
    elif len(data.domain.attributes) * 2 < len(data.domain.getmetas(orngText.TEXTMETAID)):
        return True
    return False

class OWCorrAnalysis(OWWidget):
    settingsList = ['graph.pointWidth', "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines", "graph.showAxisScale",
                    "graph.showLegend", 'autoSendSelection', "graph.showFilledSymbols", 'toolbarSelection',
                    "colorSettings", "percRadius", "recentFiles"]

    contextHandlers = {"": DomainContextHandler("", ["attrRow", "attrCol"])}

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'CorrAnalysis')
        self.callbackDeposit = []

        self.inputs = [("Data", ExampleTable, self.dataset)]
        self.outputs = [("Selected data", ExampleTable)]
        self.recentFiles=[]

        self.data = None
        self.CA = None
        self.CAloaded = False
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
        self.SettingsTab = QVGroupBox(self)
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
        self.attrColCombo = OWGUI.comboBox(self.GeneralTab, self, "attrCol", " Column table attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)

        # row attribute
        self.attrRow = ""
        self.attrRowCombo = OWGUI.comboBox(self.GeneralTab, self, "attrRow", "Row table attribute ", callback = self.updateTables, sendSelectedValue = 1, valueType = str)

        #x principal axis
        self.attrX = 0
        self.attrXCombo = OWGUI.comboBox(self.GeneralTab, self, "attrX", " Principal axis X ", callback = self.contributionBox, sendSelectedValue = 1, valueType = str)

        #y principal axis
        self.attrY = 0
        self.attrYCombo = OWGUI.comboBox(self.GeneralTab, self, "attrY", " Principal axis Y ", callback = self.contributionBox, sendSelectedValue = 1, valueType = str)

        contribution = QVGroupBox('Contribution to inertia', self.GeneralTab)
        self.firstAxis = OWGUI.widgetLabel(contribution, 'Axis %d: %f%%' % (1, 10))
        self.secondAxis = OWGUI.widgetLabel(contribution, 'Axis %d: %f%%' % (2, 10))

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
        self.connect(self.graph, SIGNAL('plotMousePressed(const QMouseEvent&)'), self.sendSelections)

        self.apply = False
        OWGUI.button(self.GeneralTab, self, 'Update Graph', self.buttonUpdate)

        OWGUI.button(self.GeneralTab, self, 'Save graph', self.graph.saveToFile)
        OWGUI.button(self.GeneralTab, self, 'Save CA', self.saveCA)
        OWGUI.button(self.GeneralTab, self, 'Load CA', self.loadCA)
        self.chosenSelection = []
        self.selections = []
        OWGUI.listBox(self.GeneralTab, self, "chosenSelection", "selections", box="Feature selection used", selectionMode = QListWidget.MultiSelection, callback = None)
        # ####################################
        # SETTINGS TAB
        # point width
        OWGUI.hSlider(self.SettingsTab, self, 'graph.pointWidth', box=' Point size ', minValue=1, maxValue=20, step=1, callback = self.replotCurves)

        # general graph settings
        box4 = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box4, self, 'graph.showXaxisTitle', 'X-axis title', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showYLaxisTitle', 'Y-axis title', callback = self.updateGraph)
##        OWGUI.checkBox(box4, self, 'graph.showAxisScale', 'Show axis scale', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showLegend', 'Show legend', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showFilledSymbols', 'Show filled symbols', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'showGridlines', 'Show gridlines', callback = self.setShowGridlines)
##        OWGUI.checkBox(box4, self, 'graph.showClusters', 'Show clusters', callback = self.updateGraph, tooltip = "Show a line boundary around a significant cluster")
        OWGUI.checkBox(box4, self, 'graph.showRowLabels', 'Show row labels', callback = self.updateGraph)
        OWGUI.checkBox(box4, self, 'graph.showColumnLabels', 'Show column labels', callback = self.updateGraph)


        self.colorButtonsBox = OWGUI.widgetBox(self.SettingsTab, " Colors ", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)

        #browsing radius
        OWGUI.hSlider(self.SettingsTab, self, 'percRadius', box=' Browsing curve size ', minValue = 0, maxValue=100, step=5, callback = self.calcRadius)

        #font size
        OWGUI.hSlider(self.SettingsTab, self, 'graph.labelSize', box=' Set font size for labels ', minValue = 8, maxValue=48, step=1, callback = self.updateGraph)

        OWGUI.hSlider(self.SettingsTab, self, 'graph.maxPoints', box=' Maximum number of points ', minValue = 10, maxValue=40, step=1, callback = None)

        self.resultsTab = QVGroupBox(self, "Results")
        self.tabsMain.insertTab(self.resultsTab, "Results")
        self.chosenDoc = []
        self.docs = self.graph.docs
        OWGUI.listBox(self.resultsTab, self, "chosenDoc", "docs", box="Documents", callback = None)
        self.chosenFeature = []
        self.features = self.graph.features
        OWGUI.listBox(self.resultsTab, self, "chosenFeature", "features", box="Features", selectionMode = QListWidget.MultiSelection, callback = None)
        OWGUI.button(self.resultsTab, self, "Save selected features", callback = self.saveFeatures)
        OWGUI.button(self.resultsTab, self, "Reconstruct words from letter ngrams", callback = self.reconstruct)
        self.chosenWord = []
        self.words = []
        OWGUI.listBox(self.resultsTab, self, "chosenWord", "words", box="Suggested words", callback = None)

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        self.graph.setGridColor(QPen(dlg.getColor("Grid")))

        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection, self.zoomSelectToolbar.actionBrowse, self.zoomSelectToolbar.actionBrowseCircle][self.toolbarSelection], [])

        self.resize(700, 800)


    def loadCA(self):
        import cPickle
        try:
            if self.recentFiles:
                lastPath = os.path.split(self.recentFiles[0])[0]
            else:
                lastPath = "."

            fn = str(QFileDialog.getOpenFileName(None, "Load CA data", lastPath, "Text files (*.*)"))
            if not fn:
                return

            fn = os.path.abspath(fn)

            f = open(fn,'rb')
            self.CA = cPickle.load(f)
            f.close()

            fn = str(QFileDialog.getOpenFileName(None, "Save CA data", lastPath, "Text files (*.*)"))
            if not fn:
                return

            fn = os.path.abspath(fn)
            if fn in self.recentFiles: # if already in list, remove it
                self.recentFiles.remove(fn)
            self.recentFiles.insert(0, fn)

            f = open(fn,'rb')
            data = cPickle.load(f)
            f.close()
            self.CAloaded = True
            self.dataset(data)
        except e:
            print e
            self.CA = None



    def saveCA(self):
        from time import time

        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."

        fn = str(QFileDialog.getSaveFileName(None, "Save CA data", lastPath, "Text files (*.*)"))
        if not fn:
            return

        fn = os.path.abspath(fn)
        f = open(fn, 'wb')
        import cPickle
        cPickle.dump(self.CA, f, 1)
        f.close()
        fn = str(QFileDialog.getSaveFileName(None, "Save text data", lastPath, "Text files (*.*)"))
        if not fn:
            return

        fn = os.path.abspath(fn)

        f = open(fn,'wb')
        cPickle.dump(self.data, f, 1)
        f.close()


    def saveFeatures(self):
        """Saves the features in a file called features.txt"""
        f = open("features.txt", 'a')
        f.write('=========================\n')
        for fn in self.chosenFeature:
            f.write(self.features[fn]+'\n')
        f.close()


    def reconstruct(self):
        if self.textData:
            import tmt
            tokens = set([])
            for ex in self.data:
                tmp = tmt.tokenizeNonWords(ex['text'].value.decode('utf-8','ignore').encode('cp1250','ignore'))
                for t in tmp:
                    tokens.add(' ' + t + ' ')
            del tmp
            wordList = dict.fromkeys(tokens, 0)
            ngrams = set(self.features)
            n = len(self.features[0])
            for token in tokens:
                i = 0
                while i < len(token) - n:
                    if token[i:i+n] in ngrams:
                        wordList[token] += 1.0
                    i += 1
            tmp = [(k, v) for k, v in wordList.items() if v]
            tmp.sort(lambda x,y: -cmp(x[1], y[1]))
            self.words = [i[0] + '  ' + str(i[1]) for i in tmp]


    def dataset(self, dataset):
        self.closeContext()
        if dataset:
            self.data = dataset
            if textCorpusModul:
              self.textData = checkFromText(self.data)
            else:
              self.textData = False
            try:
                self.selections = self.data.selection
            except AttributeError:
                self.selections = []
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
            self.attrRowCombo.addItem('document')
            self.attrRowCombo.addItem('category')
            self.attrColCombo.addItem('words')
        else:
            for attr in self.data.domain:
                if attr.varType == orange.VarTypes.Discrete: self.attrRowCombo.addItem(QListWidgetItem(self.icons[attr.varType], attr.name))
                if attr.varType == orange.VarTypes.Discrete: self.attrColCombo.addItem(QListWidgetItem(self.icons[attr.varType], attr.name))

        self.attrRow = str(self.attrRowCombo.itemText(0))
        if self.attrColCombo.count() > 1:
            self.attrCol = str(self.attrColCombo.itemText(1))
        else:
            self.attrCol = str(self.attrColCombo.itemText(0))

        self.updateTables()

    def updateTables(self):
        if self.textData:
            if textCorpusModul:
              data = (self.attrRow == 'document' and [self.data] or [CategoryDocument(self.data).dataCD])[0]
            else:
              data = self.data
            metas = data.domain.getmetas(orngText.TEXTMETAID)
            lenMetas = len(metas)
            caList = []
            for ex in data:
                cur = [0] * lenMetas
                for i, m in zip(range(lenMetas), metas.keys()):
                    try:
                        cur[i] = float(ex[m].native())
                    except:
                        cur[i] = 0
                caList.append(cur)
            if not self.CAloaded:
                self.CA = orngCA.CA(caList)
            hasNameAttribute = 'name' in [i.name for i in data.domain.attributes]
            hasCategoryAttribute = 'category' in [i.name for i in data.domain.attributes]
            if not hasCategoryAttribute:
                if not hasNameAttribute:
                    self.tipsR = [ex['text'].value[:35] for ex in data]
                    self.rowCategories = [(ex['text'].value[:35], "Row points") for ex in data]
                    self.catColors = {"Row points": 0}
                else:
                    self.tipsR = [ex['name'].native() for ex in data]
                    self.rowCategories = [(ex['name'].native(), "Row points") for ex in data]
                    self.catColors = {"Row points": 0}
            try:
                if hasCategoryAttribute:
                    if not hasNameAttribute:
                        self.tipsR = [ex['text'].value[:35] for ex in data]
                        self.rowCategories = [(ex['text'].value[:35], ex['category'].native()) for ex in data]
                    else:
                        self.tipsR = [ex['name'].native() for ex in data]
                        self.rowCategories = [(ex['name'].native(), ex['category'].native()) for ex in data]
                    self.catColors = {}
                    col = 0
                    colors = [0, 2, 3, 5, 6, 12]
                    for ex in data:
                        if ex['category'].native() not in self.catColors.keys():
                            self.catColors[ex['category'].native()] = colors[col]
                            col = (col + 1) % len(colors)
            except:
                if hasCategoryAttribute:
                    if not hasNameAttribute:
                        self.tipsR = [ex['text'].value[:35] for ex in data]
                        self.rowCategories = [(ex['text'].value[:35], ex[-1].native()) for ex in data]
                    else:
                        self.tipsR = [ex.name for ex in data]
                        self.rowCategories = [(ex.name, ex[-1].native()) for ex in data]
                    self.catColors = {}
                    col = 0
                    colors = [0, 2, 3, 5, 6, 12]
                    for ex in data:
                        if ex['category'].native() not in self.catColors.keys():
                            self.catColors[ex['category'].native()] = colors[col]
                            col = (col + 1) % len(colors)
            self.tipsC = [a.name for a in data.domain.getmetas(orngText.TEXTMETAID).values()]
        else:
            ca = orange.ContingencyAttrAttr(self.attrRow, self.attrCol, self.data)
            caList = [[col for col in row] for row in ca]
            if not self.CAloaded:
                self.CA = orngCA.CA(caList)
            self.tipsR = [s for s, v in ca.outerDistribution.items()]
            self.tipsC = [s for s, v in ca.innerDistribution.items()]
            self.rowCategories = [(ex[1].native(), "Row points") for ex in self.data]
            self.rowCategories = list(dict(self.rowCategories).items())
            self.catColors = {"Row points": 0}
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

        arr = [str(i) for i in range(1, min(self.CA.D.shape) + 1)]
        self.attrXCombo.addItems(arr)
        self.attrYCombo.addItems(arr)

        self.attrX = str(self.attrXCombo.itemText(0))
        if self.attrYCombo.count() > 1:
            self.attrY = str(self.attrYCombo.itemText(1))
        else:
            self.attrY = str(self.attrYCombo.itemText(0))

        self.contributionBox()

    def contributionBox(self):
        self.firstAxis.setText ('Axis %d: %f%%' % (int(self.attrX), self.CA.InertiaOfAxis(1)[int(self.attrX)-1]))
        self.secondAxis.setText ('Axis %d: %f%%' % (int(self.attrY), self.CA.InertiaOfAxis(1)[int(self.attrY)-1]))

        self.updateGraph()

    def buttonUpdate(self):
        self.apply = True
        self.graph.state = ZOOMING
        self.zoomSelectToolbar.buttonBrowse.setOn(0)
        self.zoomSelectToolbar.buttonBrowseCircle.setOn(0)
        self.zoomSelectToolbar.buttonZoom.setOn(1)
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
        else: self.graph.setXaxisTitle("")

        if self.graph.showYLaxisTitle == 1: self.graph.setYLaxisTitle("Axis " + self.attrY)
        else: self.graph.setYLaxisTitle("")

        cor = self.CA.getPrincipalRowProfilesCoordinates((int(self.attrX)-1, int(self.attrY)-1))
        numCor = int(self.percRow)
        indices = self.CA.PointsWithMostInertia(rowColumn = 0, axis = (int(self.attrX)-1, int(self.attrY)-1))[:numCor]
        cor = [cor[i] for i in indices]
        tipsR = [self.tipsR[i] + 'R' for i in indices]
        #if not self.graph.showRowLabels: tipsR = ['' for i in indices]
        labelDict = dict(zip(tipsR, cor))
        rowCategories = [self.rowCategories[i] for i in indices]
        for cat, col in self.catColors.items():
            newtips = [c[0] + 'R' for c in rowCategories if c[1] == cat]
            newcor = [labelDict[f] for f in newtips]
            if not self.graph.showRowLabels: newtips = ['' for i in indices]
            self.plotPoint(newcor, col, newtips, cat or "Row points", self.graph.showFilledSymbols)

        cor = self.CA.getPrincipalColProfilesCoordinates((int(self.attrX)-1, int(self.attrY)-1))
        numCor = int(self.percCol)
        indices = self.CA.PointsWithMostInertia(rowColumn = 1, axis = (int(self.attrX)-1, int(self.attrY)-1))[:numCor]
        cor = [cor[i] for i in indices]
        tipsC = [self.tipsC[i] + 'C' for i in indices]
        if not self.graph.showColumnLabels: tipsC = ['' for i in indices]
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

    def sendSelections(self, e):
        self.docs = self.graph.docs
        self.features = self.graph.features
        hasNameAttribute = 'name' in [i.name for i in self.data.domain.attributes]
        examples = []
        if not hasNameAttribute:
            for ex in self.data:
                if ex['text'].value[:35] in self.docs:
                    examples.append(ex)
        else:
            for ex in self.data:
                if ex['name'].native() in self.docs:
                    examples.append(ex)
        newMetas = {}
        for ex in examples:
            for k, v in ex.getmetas(orngText.TEXTMETAID).items():
                if k not in newMetas.keys():
                    newMetas[k] = self.data.domain[k]
        newDomain = orange.Domain(self.data.domain.attributes, 0)
        newDomain.addmetas(newMetas, orngText.TEXTMETAID)
        newdata = orange.ExampleTable(newDomain, examples)
        self.send("Selected data", newdata)

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
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridColor(QPen(dlg.getColor("Grid")))
            self.updateGraph()

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color palette")
        c.createDiscretePalette("discPalette", "Discrete palette")
        c.createContinuousPalette("contPalette", "Continuous palette")
        box = c.createBox("otherColors", "Other colors")
        c.createColorButton(box, "Canvas", "Canvas color", QColor(Qt.white))
        box.addSpace(5)
        c.createColorButton(box, "Grid", "Grid color", QColor(Qt.black))
        box.addSpace(5)
        box.adjustSize()
        c.setColorSchemas(self.colorSettings)
        return c

    def calcRadius(self):
        self.graph.radius =  (self.graph.axisScale(QwtPlot.xBottom).interval().maxValue() - self.graph.axisScale(QwtPlot.xBottom).interval().minValue()) * self.percRadius / 100.0;

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
        state = self.buttonBrowse.isChecked()
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
        state = self.buttonBrowseCircle.isChecked()
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
    #from orngTextCorpus import *
    import cPickle, orngText
##    os.chdir("/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/")
    appl = QApplication(sys.argv)
    ow = OWCorrAnalysis()

    #owb = OWBagofWords.OWBagofWords()
    t = orngText.loadFromXML(r'c:\test\orange\msnbc.xml')
    #owb.data = t
    #owb.show()
    stop = orngText.loadWordSet(r'C:\tmtorange\common\en_stopwords.txt')
    p = orngText.Preprocess(language = 'hr')
    print 'Done with loading'
    t1 = orngText.extractLetterNGram(t, 2)
    #t1 = orngText.extractWordNGram(t, stopwords = stop, measure = 'MI', threshold = 7, n=2)
    #t1 = orngText.extractWordNGram(t1, stopwords = stop, measure = 'MI', threshold = 10, n=3)
    #t1 = orngText.extractNamedEntities(t, stopwords = stop)
    #t1 = orngText.bagOfWords(t1, stopwords = stop)
    print len(t1.domain.getmetas(orngText.TEXTMETAID))
    print 'Done with extracting'
    #t2 = orngText.FSS(t1, 'TF', 'MIN', 0.98)
    #print len(t2.domain.getmetas())
    print 'Done with feature selection'
    appl.setMainWidget(ow)
    #t3 = orngText.DSS(t2, 'WF', 'MIN', 1)
    #print 'Done with document selection'
    ow.dataset(t1)
    print 'Done'
    ow.show()
##    dataset = orange.ExampleTable('/home/mkolar/Docs/Diplomski/repository/orange/doc/datasets/iris.tab')

##    lem = lemmatizer.FSALemmatization('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/engleski_rjecnik.fsa')
##    for word in loadWordSet('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/engleski_stoprijeci.txt'):
##        lem.stopwords.append(word)
##    a = TextCorpusLoader('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/reuters-exchanges-small.xml', lem = lem)

##    #a = orange.ExampleTable('../../doc/datasets/smokers_ct.tab')
##    f=open('../../CDFSallDataCW', 'r')
##    a =cPickle.load(f)
##    f.close()
##    ow.dataset(a)

    appl.exec_()
