"""
<name>Data Table</name>
<description>Shows data in a spreadsheet.</description>
<icon>icons/DataTable.png</icon>
<priority>100</priority>
<contact>Peter Juvan (peter.juvan@fri.uni-lj.si)</contact>
"""

# OWDataTable.py
#
# wishes:
# ignore attributes, filter examples by attribute values, do
# all sorts of preprocessing (including discretization) on the table,
# output a new table and export it in variety of formats.

from OWWidget import *
import OWGUI
import math
from orngDataCaching import *
from PyQt4 import *

##############################################################################

OrangeValueRole = Qt.UserRole + 1

class OWDataTable(OWWidget):
    settingsList = ["showDistributions", "showMeta", "distColorRgb", "showAttributeLabels"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Table")

        self.inputs = [("Examples", ExampleTable, self.dataset, Multiple + Default)]
        self.outputs = []

        self.data = {}          # key: id, value: ExampleTable
        self.showMetas = {}     # key: id, value: (True/False, columnList)
        self.showMeta = 1
        self.showAttributeLabels = 1
        self.showDistributions = 1
        self.distColorRgb = (220,220,220, 255)
        self.distColor = QColor(*self.distColorRgb)
        self.locale = QLocale()

        self.loadSettings()

        # info box
        infoBox = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoEx = OWGUI.widgetLabel(infoBox, 'No data on input.')
        self.infoMiss = OWGUI.widgetLabel(infoBox, ' ')
        OWGUI.widgetLabel(infoBox, ' ')
        self.infoAttr = OWGUI.widgetLabel(infoBox, ' ')
        self.infoMeta = OWGUI.widgetLabel(infoBox, ' ')
        OWGUI.widgetLabel(infoBox, ' ')
        self.infoClass = OWGUI.widgetLabel(infoBox, ' ')
        infoBox.setMinimumWidth(200)
        OWGUI.separator(self.controlArea)

        # settings box
        boxSettings = OWGUI.widgetBox(self.controlArea, "Settings")
        self.cbShowMeta = OWGUI.checkBox(boxSettings, self, "showMeta", 'Show meta attributes', callback = self.cbShowMetaClicked)
        self.cbShowMeta.setEnabled(False)
        self.cbShowAttLbls = OWGUI.checkBox(boxSettings, self, "showAttributeLabels", 'Show attribute labels (if any)', callback = self.cbShowAttLabelsClicked)
        self.cbShowAttLbls.setEnabled(True)
        self.cbShowDistributions = OWGUI.checkBox(boxSettings, self, "showDistributions", 'Visualize continuous values', callback = self.cbShowDistributions)
        colBox = OWGUI.indentedBox(boxSettings, orientation = "horizontal")
        OWGUI.widgetLabel(colBox, "Color: ")
        self.colButton = OWGUI.toolButton(colBox, self, self.changeColor, width=20, height=20, debuggingEnabled = 0)
        OWGUI.rubber(colBox)

        resizeColsBox = OWGUI.widgetBox(boxSettings, 0, "horizontal", 0)
        OWGUI.label(resizeColsBox, self, "Resize columns: ")
        OWGUI.button(resizeColsBox, self, "+", self.increaseColWidth, tooltip = "Increase the width of the columns", width=30)
        OWGUI.button(resizeColsBox, self, "-", self.decreaseColWidth, tooltip = "Decrease the width of the columns", width=30)
        OWGUI.rubber(resizeColsBox)

        self.btnResetSort = OWGUI.button(boxSettings, self, "Restore Order of Examples", callback = self.btnResetSortClicked, tooltip = "Show examples in the same order as they appear in the file")

        OWGUI.rubber(self.controlArea)

        # GUI with tabs
        self.tabs = OWGUI.tabWidget(self.mainArea)
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        self.connect(self.tabs,SIGNAL("currentChanged(QWidget*)"),self.tabClicked)

        self.updateColor()

    def changeColor(self):
        color = QColorDialog.getColor(self.distColor, self)
        if color.isValid():
            self.distColorRgb = color.getRgb()
            self.updateColor()

    def updateColor(self):
        self.distColor = QColor(*self.distColorRgb)
        w = self.colButton.width()-8
        h = self.colButton.height()-8
        pixmap = QPixmap(w, h)
        painter = QPainter()
        painter.begin(pixmap)
        painter.fillRect(0,0,w,h, QBrush(self.distColor))
        painter.end()
        self.colButton.setIcon(QIcon(pixmap))

    def increaseColWidth(self):
        table = self.tabs.currentWidget()
        if table:
            for col in range(table.columnCount()):
                w = table.columnWidth(col)
                table.setColumnWidth(col, w + 10)

    def decreaseColWidth(self):
        table = self.tabs.currentWidget()
        if table:
            for col in range(table.columnCount()):
                w = table.columnWidth(col)
                minW = table.sizeHintForColumn(col)
                table.setColumnWidth(col, max(w - 10, minW))


    def dataset(self, data, id=None):
        """Generates a new table and adds it to a new tab when new data arrives;
        or hides the table and removes a tab when data==None;
        or replaces the table when new data arrives together with already existing id."""
        if data != None:  # can be an empty table!
            if self.data.has_key(id):
                # remove existing table
                self.data.pop(id)
                self.showMetas.pop(id)
                self.id2table[id].hide()
                self.tabs.removeTab(self.tabs.indexOf(self.id2table[id]))
                self.table2id.pop(self.id2table.pop(id))
            self.data[id] = data
            self.showMetas[id] = (True, [])

            table = OWGUI.table(None, 0,0)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)

            self.id2table[id] = table
            self.table2id[table] = id
            if data.name:
                tabName = "%s " % data.name
            else:
                tabName = ""
            tabName += "(" + str(id[1]) + ")"
            if id[2] != None:
                tabName += " [" + str(id[2]) + "]"
            self.tabs.addTab(table, tabName)

            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()
            self.tabs.setCurrentIndex(self.tabs.indexOf(table))
            self.setInfo(data)
            self.cbShowMeta.setEnabled(len(self.showMetas[id][1])>0)        # enable showMetas checkbox only if metas exist

        elif self.data.has_key(id):
            table = self.id2table[id]
            self.data.pop(id)
            self.showMetas.pop(id)
            table.hide()
            self.tabs.removeTab(self.tabs.indexOf(table))
            self.table2id.pop(self.id2table.pop(id))
            self.setInfo(self.data.get(self.table2id.get(self.tabs.currentWidget(),None),None))

        # disable showMetas checkbox if there is no data on input
        if len(self.data) == 0:
            self.cbShowMeta.setEnabled(False)

    def sendReport(self):
        qTableInstance = self.tabs.currentWidget()
        id = self.table2id.get(qTableInstance, None)
        data = self.data.get(id, None)
        self.reportData(data)
        table = self.id2table[id]
        import OWReport
        self.reportRaw(OWReport.reportTable(table))
        
        
    # Writes data into table, adjusts the column width.
    def setTable(self, table, data):
        if data==None:
            return
        qApp.setOverrideCursor(Qt.WaitCursor)
        vars = data.domain.variables
        m = data.domain.getmetas(False)
        ml = [(k, m[k]) for k in m]
        ml.sort(lambda x,y: cmp(y[0], x[0]))
        metas = [x[1] for x in ml]
        metaKeys = [x[0] for x in ml]

        mo = data.domain.getmetas(True).items()
        if mo:
            mo.sort(lambda x,y: cmp(x[1].name.lower(),y[1].name.lower()))
#            metas.append(None)
#            metaKeys.append(None)

        varsMetas = vars + metas

        numVars = len(data.domain.variables)
        numMetas = len(metas)
        numVarsMetas = numVars + numMetas
        numEx = len(data)
        numSpaces = int(math.log(max(numEx,1), 10))+1

        table.clear()
        table.oldSortingIndex = -1
        table.oldSortingOrder = 1
        table.setColumnCount(numVarsMetas)
        table.setRowCount(numEx)

        table.dist = getCached(data, orange.DomainBasicAttrStat, (data,))
        
        table.setItemDelegate(TableItemDelegate(self, table))
        table.variableNames = [var.name for var in varsMetas]
        table.data = data
        id = self.table2id.get(table, None)

        # set the header (attribute names)

        self.drawAttributeLabels(table)

        #table.hide()
        clsColor = QColor(160,160,160)
        metaColor = QColor(220,220,200)
        white = QColor(Qt.white)
        for j,(key,attr) in enumerate(zip(range(numVars) + metaKeys, varsMetas)):
            self.progressBarSet(j*100.0/numVarsMetas)
            if attr == data.domain.classVar:
                bgColor = clsColor
            elif attr in metas or attr is None:
                bgColor = metaColor
                self.showMetas[id][1].append(j) # store indices of meta attributes
            else:
                bgColor = white

            for i in range(numEx):
##                table.setItem(i, j, TableWidgetItem(data[i][key]
##                OWGUI.tableItem(table, i,j, str(data[i][key]), backColor = bgColor)
                if data.domain[key].varType == orange.VarTypes.Continuous and not data[i][key].isSpecial():
                    item = OWGUI.tableItem(table, i,j, float(str(data[i][key])), backColor = bgColor)
                else:
                    item = OWGUI.tableItem(table, i,j, str(data[i][key]), backColor = bgColor)
##                item.setData(OrangeValueRole, QVariant(str(data[i][key])))
 

        table.resizeRowsToContents()
        table.resizeColumnsToContents()

        self.connect(table.horizontalHeader(), SIGNAL("sectionClicked(int)"), self.sortByColumn)
        #table.verticalHeader().setMovable(False)

        qApp.restoreOverrideCursor()
        #table.setCurrentCell(-1,-1)
        #table.show()
 

    def setCornerText(self, table, text):
        """
        Set table corner text. As this is an ugly hack, do everything in
        try - except blocks, as it may stop working in newer Qt.
        """

        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QAbstractButton)

                class efc(QObject):
                    def eventFilter(self, o, e):
                        if (e.type() == QEvent.Paint):
                            if isinstance(o, QAbstractButton):
                                btn = o
                                #paint by hand (borrowed from QTableCornerButton)
                                opt = QStyleOptionHeader()
                                opt.init(btn)
                                state = QStyle.State_None;
                                if (btn.isEnabled()):
                                    state |= QStyle.State_Enabled;
                                if (btn.isActiveWindow()):
                                    state |= QStyle.State_Active;
                                if (btn.isDown()):
                                    state |= QStyle.State_Sunken;
                                opt.state = state;
                                opt.rect = btn.rect();
                                opt.text = btn.text();
                                opt.position = QStyleOptionHeader.OnlyOneSection;
                                painter = QStylePainter(btn);
                                painter.drawControl(QStyle.CE_Header, opt);
                                return True # eat evebt
                        return False
                
                table.efc = efc()
                btn.installEventFilter(table.efc)
                table.btn = btn
            except:
                table.btnfailed = True

        if hasattr(table, "btn"):
            try:
                btn = table.btn
                btn.setText(text)
                opt = QStyleOptionHeader()
                opt.text = btn.text()
                s = btn.style().sizeFromContents(QStyle.CT_HeaderSection, opt, QSize(), btn).expandedTo(QApplication.globalStrut())
                if s.isValid():
                    table.verticalHeader().setMinimumWidth(s.width())
            except:
                pass

    def sortByColumn(self, index):
        table = self.tabs.currentWidget()
        table.horizontalHeader().setSortIndicatorShown(1)
        header = table.horizontalHeader()
        if index == table.oldSortingIndex:
            order = table.oldSortingOrder == Qt.AscendingOrder and Qt.DescendingOrder or Qt.AscendingOrder
        else:
            order = Qt.AscendingOrder
        table.sortByColumn(index, order)
        table.oldSortingIndex = index
        table.oldSortingOrder = order
        #header.setSortIndicator(index, order)

    def tabClicked(self, qTableInstance):
        """Updates the info box and showMetas checkbox when a tab is clicked.
        """
        id = self.table2id.get(qTableInstance,None)
        self.setInfo(self.data.get(id,None))
        show_col = self.showMetas.get(id,None)
        if show_col:
            self.cbShowMeta.setChecked(show_col[0])
            self.cbShowMeta.setEnabled(len(show_col[1])>0)

    def cbShowMetaClicked(self):
        table = self.tabs.currentWidget()
        id = self.table2id.get(table, None)
        if self.showMetas.has_key(id):
            show,col = self.showMetas[id]
            self.showMetas[id] = (not show,col)
        if show:
            for c in col:
                table.hideColumn(c)
        else:
            for c in col:
                table.showColumn(c)
                table.resizeColumnToContents(c)

    def drawAttributeLabels(self, table):
        table.setHorizontalHeaderLabels(table.variableNames)
        if self.showAttributeLabels:
            labelnames = set()
            for a in table.data.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(list(labelnames))
            if len(labelnames):
                table.setHorizontalHeaderLabels([table.variableNames[i] + "\n" + "\n".join(["%s" % a.attributes.get(lab, "") for lab in labelnames]) for (i, a) in enumerate(table.data.domain.attributes)])
                self.setCornerText(table, "\n".join([""] + labelnames))
        else:
            self.setCornerText(table, "")

    def cbShowAttLabelsClicked(self):
        for table in self.table2id.keys():
            self.drawAttributeLabels(table)

    def cbShowDistributions(self):
        table = self.tabs.currentWidget()
        if table:
            table.reset()

    # show data in the default order
    def btnResetSortClicked(self):
        table = self.tabs.currentWidget()
        if table:
            id = self.table2id[table]
            data = self.data[id]
            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()

    def setInfo(self, data):
        """Updates data info.
        """
        def sp(l, capitalize=False):
            n = len(l)
            if n == 0:
                if capitalize:
                    return "No", "s"
                else:
                    return "no", "s"
            elif n == 1:
                return str(n), ''
            else:
                return str(n), 's'

        if data == None:
            self.infoEx.setText('No data on input.')
            self.infoMiss.setText('')
            self.infoAttr.setText('')
            self.infoMeta.setText('')
            self.infoClass.setText('')
        else:
            self.infoEx.setText("%s example%s," % sp(data))
            missData = orange.Preprocessor_takeMissing(data)
            self.infoMiss.setText('%s (%.1f%s) with missing values.' % (len(missData), len(data) and 100.*len(missData)/len(data), "%"))
            self.infoAttr.setText("%s attribute%s," % sp(data.domain.attributes,True))
            self.infoMeta.setText("%s meta attribute%s." % sp(data.domain.getmetas()))
            if data.domain.classVar:
                if data.domain.classVar.varType == orange.VarTypes.Discrete:
                    self.infoClass.setText('Discrete class with %s value%s.' % sp(data.domain.classVar.values))
                elif data.domain.classVar.varType == orange.VarTypes.Continuous:
                    self.infoClass.setText('Continuous class.')
                else:
                    self.infoClass.setText("Class is neither discrete nor continuous.")
            else:
                self.infoClass.setText('Classless domain.')
                

class TableItemDelegate(QItemDelegate):
    def __init__(self, widget = None, table = None):
        QItemDelegate.__init__(self, widget)
        self.table = table
        self.widget = widget

    def paint(self, painter, option, index):
        painter.save()
        self.drawBackground(painter, option, index)
        value, ok = index.data(Qt.DisplayRole).toDouble()

        if ok:        # in case we get "?" it is not ok
            if self.widget.showDistributions:
                col = index.column()
                if col < len(self.table.dist) and self.table.dist[col]:        # meta attributes and discrete attributes don't have a key
                    dist = self.table.dist[col]
                    smallerWidth = option.rect.width() * (dist.max - value) / (dist.max-dist.min or 1)
                    painter.fillRect(option.rect.adjusted(0,0,-smallerWidth,0), self.widget.distColor)
##            text = self.widget.locale.toString(value)    # we need this to convert doubles like 1.39999999909909 into 1.4
##        else:
        text = index.data(Qt.DisplayRole).toString()
        ##text = index.data(OrangeValueRole).toString()

        self.drawDisplay(painter, option, option.rect, text)
        painter.restore()




if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()

    #d1 = orange.ExampleTable(r'..\..\doc\datasets\auto-mpg')
    #d2 = orange.ExampleTable('test-labels')
    #d3 = orange.ExampleTable(r'..\..\doc\datasets\sponge.tab')
    #d4 = orange.ExampleTable(r'..\..\doc\datasets\wpbc.csv')
    #d5 = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
    #d5 = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab")
    d5 = orange.ExampleTable("adult_sample")
    d5 = orange.ExampleTable("/home/marko/tdw")
    #d5 = orange.ExampleTable(r"e:\Development\Orange Datasets\Cancer\SRBCT.tab")
    ow.show()
    #ow.dataset(d1,"auto-mpg")
    #ow.dataset(d2,"voting")
    #ow.dataset(d4,"wpbc")
    ow.dataset(d5,"adult_sample")
    a.exec_()
    ow.saveSettings()
