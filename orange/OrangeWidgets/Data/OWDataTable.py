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
import OWColorPalette

##############################################################################

def safe_call(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception, ex:
            print >> sys.stderr, func.__name__, "call error", ex 
            return QVariant()
    return wrapper
    

class ExampleTableModel(QAbstractItemModel):
    def __init__(self, examples, dist, *args):
        QAbstractItemModel.__init__(self, *args)
        self.examples = examples
        self.dist = dist
        self.attributes = list(self.examples.domain.attributes)
        self.classVar = self.examples.domain.classVar
        self.metas = self.examples.domain.getmetas().values()
        self.all_attrs = self.attributes + ([self.classVar] if self.classVar else []) + self.metas
        self.clsColor = QColor(160,160,160)
        self.metaColor = QColor(220,220,200)
        self.sorted_map = range(len(self.examples))
        
        self.attrLabels = sorted(reduce(set.union, [attr.attributes for attr in self.all_attrs], set()))
        self._other_data = {}
        
    showAttrLabels = pyqtProperty("bool", 
                                  fget=lambda self: getattr(self, "_showAttrLabels", False),
                                  fset=lambda self, val: (self.emit(SIGNAL("layoutAboutToBeChanged()")),
                                                          setattr(self, "_showAttrLabels", val),
                                                          self.emit(SIGNAL("headerDataChanged(Qt::Orientation, int, int)"), Qt.Horizontal, 0, len(self.all_attrs)-1),
                                                          self.emit(SIGNAL("layoutChanged()")),
                                                          self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(0,0),
                                                                    self.index(len(self.examples) - 1, len(self.all_attrs) - 1))
                                                          ) or None
                                  )
    
    @safe_call
    def data(self, index, role):
        row, col = self.sorted_map[index.row()], index.column()
        example, attr = self.examples[row], self.all_attrs[col]
        val = example[attr]
        domain = self.examples.domain
        if role == Qt.DisplayRole:
                return QVariant(str(val))
        elif role == Qt.BackgroundRole:
            if attr == self.classVar and col == len(domain.attributes) and domain.classVar: #check if attr is actual class or a duplication in the meta attributes
                return QVariant(self.clsColor)
            elif attr in self.metas:
                return QVariant(self.metaColor)
        elif role == OWGUI.TableBarItem.BarRole and val.varType == orange.VarTypes.Continuous \
                    and not val.isSpecial() and attr not in self.metas:
            dist = self.dist[col]
            return QVariant((float(val) - dist.min) / (dist.max - dist.min or 1))
        elif role == OWGUI.TableValueRole: # The actual value
            return QVariant(val)
        elif role == OWGUI.TableClassValueRole: # The class value for the row's example
            return QVariant(example.get_class())
        elif role == OWGUI.TableVariable: # The variable descriptor for column
            return QVariant(val.variable)
        
        return self._other_data.get((index.row(), index.column(), role), QVariant())
        
    def setData(self, index, variant, role):
        self._other_data[index.row(), index.column(), role] = variant
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), index, index)
        
    def index(self, row, col, parent=QModelIndex()):
        return self.createIndex(row, col, 0)
    
    def parent(self, index):
        return QModelIndex()
    
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return max([len(self.examples)] + [row for row, _, _ in self._other_data.keys()])
        
    def columnCount(self, index=QModelIndex()):
        return max([len(self.all_attrs)] + [col for _, col, _ in self._other_data.keys()])
    
    @safe_call
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal:
            attr = self.all_attrs[section]
            if role ==Qt.DisplayRole:
                values = [attr.name] + ([str(attr.attributes.get(label, "")) for label in self.attrLabels] if self.showAttrLabels else [])
                return QVariant("\n".join(values))
            if role == Qt.ToolTipRole:
                pairs = [(key, str(attr.attributes[key])) for key in self.attrLabels if key in attr.attributes]
                tip = "<b>%s</b>" % attr.name
                tip = "<br>".join([tip] + ["%s = %s" % pair for pair in pairs])
                return QVariant(tip)  
        else:
            if role == Qt.DisplayRole:
                return QVariant(section + 1)
        return QVariant()
    
    def sort(self, column, order=Qt.AscendingOrder):
        self.emit(SIGNAL("layoutAboutToBeChanged()"))
        attr = self.all_attrs[column] 
        values = [(ex[attr], i) for i, ex in enumerate(self.examples)]
        values = sorted(values, key=lambda t: t[0] if not t[0].isSpecial() else sys.maxint, reverse=(order!=Qt.AscendingOrder))
        self.sorted_map = [v[1] for v in values]
        self.emit(SIGNAL("layoutChanged()"))
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(0,0),
                  self.index(len(self.examples) - 1, len(self.all_attrs) - 1))
            

class OWDataTable(OWWidget):
    settingsList = ["showDistributions", "showMeta", "distColorRgb", "showAttributeLabels", "autoCommit", "selectedSchemaIndex", "colorByClass"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Table")

        self.inputs = [("Data", ExampleTable, self.dataset, Multiple + Default)]
        self.outputs = [("Selected Data", ExampleTable, Default), ("Other Data", ExampleTable)]

        self.data = {}          # key: id, value: ExampleTable
        self.showMetas = {}     # key: id, value: (True/False, columnList)
        self.showMeta = 1
        self.showAttributeLabels = 1
        self.showDistributions = 1
        self.distColorRgb = (220,220,220, 255)
        self.distColor = QColor(*self.distColorRgb)
        self.locale = QLocale()
        self.autoCommit = False
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.colorByClass = True
        
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
        boxSettings = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        self.cbShowMeta = OWGUI.checkBox(boxSettings, self, "showMeta", 'Show meta attributes', callback = self.cbShowMetaClicked)
        self.cbShowMeta.setEnabled(False)
        self.cbShowAttLbls = OWGUI.checkBox(boxSettings, self, "showAttributeLabels", 'Show attribute labels (if any)', callback = self.cbShowAttLabelsClicked)
        self.cbShowAttLbls.setEnabled(True)

        box = OWGUI.widgetBox(self.controlArea, "Colors")
        OWGUI.checkBox(box, self, "showDistributions", 'Visualize continuous values', callback = self.cbShowDistributions)
        OWGUI.checkBox(box, self, "colorByClass", 'Color by class value', callback = self.cbShowDistributions)
        OWGUI.button(box, self, "Set colors", self.setColors, tooltip = "Set the canvas background color and color palette for coloring continuous variables", debuggingEnabled = 0)

        resizeColsBox = OWGUI.widgetBox(boxSettings, 0, "horizontal", 0)
        OWGUI.label(resizeColsBox, self, "Resize columns: ")
        OWGUI.toolButton(resizeColsBox, self, "+", self.increaseColWidth, tooltip = "Increase the width of the columns", width=20, height=20)
        OWGUI.toolButton(resizeColsBox, self, "-", self.decreaseColWidth, tooltip = "Decrease the width of the columns", width=20, height=20)
        OWGUI.rubber(resizeColsBox)

        self.btnResetSort = OWGUI.button(boxSettings, self, "Restore Order of Examples", callback = self.btnResetSortClicked, tooltip = "Show examples in the same order as they appear in the file")
        
        OWGUI.separator(self.controlArea)
        selectionBox = OWGUI.widgetBox(self.controlArea, "Selection")
        self.sendButton = OWGUI.button(selectionBox, self, "Send selections", self.commit, default=True)
        cb = OWGUI.checkBox(selectionBox, self, "autoCommit", "Commit on any change", callback=self.commitIf)
        OWGUI.setStopper(self, self.sendButton, cb, "selectionChangedFlag", self.commit)

        OWGUI.rubber(self.controlArea)

        dlg = self.createColorDialog()
        self.discPalette = dlg.getDiscretePalette("discPalette")

        # GUI with tabs
        self.tabs = OWGUI.tabWidget(self.mainArea)
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        self.connect(self.tabs, SIGNAL("currentChanged(QWidget*)"), self.tabClicked)
        
        self.selectionChangedFlag = False
        

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Default", "Default color", QColor(Qt.white))
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.distColorRgb = dlg.getColor("Default")

    def increaseColWidth(self):
        table = self.tabs.currentWidget()
        if table:
            for col in range(table.model().columnCount(QModelIndex())):
                w = table.columnWidth(col)
                table.setColumnWidth(col, w + 10)

    def decreaseColWidth(self):
        table = self.tabs.currentWidget()
        if table:
            for col in range(table.model().columnCount(QModelIndex())):
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

            table = QTableView()
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSortingEnabled(True)
            table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
            table.horizontalHeader().setMovable(True)
            table.horizontalHeader().setClickable(True)
            table.horizontalHeader().setSortIndicatorShown(False)
            
            option = table.viewOptions()
            size = table.style().sizeFromContents(QStyle.CT_ItemViewItem, option, QSize(20, 20), table) #QSize(20, QFontMetrics(option.font).lineSpacing()), table)
            
            table.verticalHeader().setDefaultSectionSize(size.height() + 2) #int(size.height() * 1.25) + 2)

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
            self.sendButton.setEnabled(not self.autoCommit)

        elif self.data.has_key(id):
            table = self.id2table[id]
            self.data.pop(id)
            self.showMetas.pop(id)
            table.hide()
            self.tabs.removeTab(self.tabs.indexOf(table))
            self.table2id.pop(self.id2table.pop(id))
            self.setInfo(self.data.get(self.table2id.get(self.tabs.currentWidget(),None),None))

        if len(self.data) == 0:
            self.sendButton.setEnabled(False)

        self.setCbShowMeta()

    def setCbShowMeta(self):
        for ti in range(self.tabs.count()):
            if len(self.tabs.widget(ti).model().metas)>0:
                self.cbShowMeta.setEnabled(True)
                break
        else:
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
            metas.append(None)
            metaKeys.append(None)

        varsMetas = vars + metas

        numVars = len(data.domain.variables)
        numMetas = len(metas)
        numVarsMetas = numVars + numMetas
        numEx = len(data)
        numSpaces = int(math.log(max(numEx,1), 10))+1

#        table.clear()
        table.oldSortingIndex = -1
        table.oldSortingOrder = 1
#        table.setColumnCount(numVarsMetas)
#        table.setRowCount(numEx)

        dist = getCached(data, orange.DomainBasicAttrStat, (data,))
        
        datamodel = ExampleTableModel(data, dist, self)
        
#        proxy = QSortFilterProxyModel(self)
#        proxy.setSourceModel(datamodel)
        
        color_schema = self.discPalette if self.colorByClass else None
        table.setItemDelegate(OWGUI.TableBarItem(self, color=self.distColor, color_schema=color_schema) \
                              if self.showDistributions else QStyledItemDelegate(self)) #TableItemDelegate(self, table))
        
        table.setModel(datamodel)
        def p():
            try:
                table.updateGeometries()
                table.viewport().update()
            except RuntimeError:
                pass
        
        size = table.verticalHeader().sectionSizeHint(0)
        table.verticalHeader().setDefaultSectionSize(size)
        
        self.connect(datamodel, SIGNAL("layoutChanged()"), lambda *args: QTimer.singleShot(50, p))
        
        id = self.table2id.get(table, None)

        # set the header (attribute names)

        self.drawAttributeLabels(table)

        self.showMetas[id][1].extend([i for i, attr in enumerate(table.model().all_attrs) if attr in table.model().metas])
        self.connect(table.horizontalHeader(), SIGNAL("sectionClicked(int)"), self.sortByColumn)
        self.connect(table.selectionModel(), SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.updateSelection)
        #table.verticalHeader().setMovable(False)

        qApp.restoreOverrideCursor() 

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
        self.updateSelection()

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
#        table.setHorizontalHeaderLabels(table.variableNames)
        table.model().showAttrLabels = bool(self.showAttributeLabels)
        if self.showAttributeLabels:
            labelnames = set()
            for a in table.model().examples.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(list(labelnames))
#            if len(labelnames):
#                table.setHorizontalHeaderLabels([table.variableNames[i] + "\n" + "\n".join(["%s" % a.attributes.get(lab, "") for lab in labelnames]) for (i, a) in enumerate(table.data.domain.attributes)])
            self.setCornerText(table, "\n".join([""] + labelnames))
        else:
            self.setCornerText(table, "")
        table.repaint()

    def cbShowAttLabelsClicked(self):
        for table in self.table2id.keys():
            self.drawAttributeLabels(table)

    def cbShowDistributions(self):
        for ti in range(self.tabs.count()):
            color_schema = self.discPalette if self.colorByClass else None
            delegate = OWGUI.TableBarItem(self, color=self.distColor,
                                          color_schema=color_schema) \
                       if self.showDistributions else QStyledItemDelegate(self)
            self.tabs.widget(ti).setItemDelegate(delegate)
        tab = self.tabs.currentWidget()
        if tab:
            tab.reset()

    # show data in the default order
    def btnResetSortClicked(self):
        table = self.tabs.currentWidget()
        if table:
            id = self.table2id[table]
            data = self.data[id]
            table.horizontalHeader().setSortIndicatorShown(False)
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

    def updateSelection(self, *args):
        self.sendButton.setEnabled(bool(self.getCurrentSelection()) and not self.autoCommit)
        self.commitIf()
            
    def getCurrentSelection(self):
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model()
            new = table.selectionModel().selectedIndexes()
            return sorted(set([model.sorted_map[ind.row()] for ind in new]))
        
    def commitIf(self):
        if self.autoCommit:
            self.commit()
        else:
            self.selectionChangedFlag = True
            
    def commit(self):
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model()
            selected = self.getCurrentSelection()
            selection = [1 if i in selected else 0 for i in range(len(model.examples))]
            data = model.examples.select(selection)
            self.send("Selected Data", data if len(data) > 0 else None)
            data = model.examples.select(selection, 0)
            self.send("Other Data", data if len(data) > 0 else None)
        else:
            self.send("Selected Data", None)
            self.send("Other Data", None)
            
        self.selectionChangedFlag = False
            
        

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()

    #d1 = orange.ExampleTable(r'..\..\doc\datasets\auto-mpg')
    #d2 = orange.ExampleTable('test-labels')
    #d3 = orange.ExampleTable(r'..\..\doc\datasets\sponge.tab')
    #d4 = orange.ExampleTable(r'..\..\doc\datasets\wpbc.csv')
    d5 = orange.ExampleTable('../../doc/datasets/adult_sample.tab')
    #d5 = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab")
#    d5 = orange.ExampleTable("adult_sample")
#    d5 = orange.ExampleTable("/home/marko/tdw")
    #d5 = orange.ExampleTable(r"e:\Development\Orange Datasets\Cancer\SRBCT.tab")
    ow.show()
    #ow.dataset(d1,"auto-mpg")
    #ow.dataset(d2,"voting")
    #ow.dataset(d4,"wpbc")
    ow.dataset(d5,"adult_sample")
    a.exec_()
    ow.saveSettings()
