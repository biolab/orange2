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

from qttable import *
from OWWidget import *
import OWGUI
import math

##############################################################################

class OWDataTable(OWWidget):
    settingsList = []

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Data Table")

        self.inputs = [("Examples", ExampleTable, self.dataset, Multiple + Default)]
        self.outputs = []
        
        self.data = {}          # key: id, value: ExampleTable
        self.showMetas = {}     # key: id, value: (True/False, columnList)

        # info box
        #self.controlArea.layout().setResizeMode(QLayout.Minimum)
        infoBox = QVGroupBox("Info", self.controlArea)
        self.infoEx = QLabel('No data on input.', infoBox)
        self.infoMiss = QLabel('', infoBox)
        QLabel('', infoBox)
        self.infoAttr = QLabel('', infoBox)
        self.infoMeta = QLabel('', infoBox)
        QLabel('', infoBox)
        self.infoClass = QLabel('', infoBox)
        infoBox.setMinimumWidth(200)
        #infoBox.setMaximumHeight(infoBox.sizeHint().height())

        # settings box        
        boxSettings = QVGroupBox("Settings", self.controlArea)
        self.cbShowMeta = QCheckBox('Show meta attributes', boxSettings)
        self.cbShowMeta.setChecked(True)
        self.cbShowMeta.setEnabled(False)
        self.connect(self.cbShowMeta, SIGNAL("clicked()"), self.cbShowMetaClicked)
        self.btnResetSort = QPushButton("Restore Original Order", boxSettings)
        self.connect(self.btnResetSort, SIGNAL("clicked()"), self.btnResetSortClicked)
        boxSettings.setMaximumHeight(boxSettings.sizeHint().height())
        
        # GUI with tabs
        layout=QVBoxLayout(self.mainArea)
        self.tabs = QTabWidget(self.mainArea, 'tabWidget')
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        layout.addWidget(self.tabs)
        self.connect(self.tabs,SIGNAL("currentChanged(QWidget*)"),self.tabClicked)
        

    def dataset(self, data, id=None):
        """Generates a new table and adds it to a new tab when new data arrives;
        or hides the table and removes a tab when data==None;
        or replaces the table when new data arrives together with already existing id.
        """
        if data != None:  # can be an empty table!
            if self.data.has_key(id):
                # remove existing table
                self.data.pop(id)
                self.showMetas.pop(id)
                self.id2table[id].hide()
                self.tabs.removePage(self.id2table[id])
                self.table2id.pop(self.id2table.pop(id))
            self.data[id] = data
            self.showMetas[id] = (True, [])
            self.progressBarInit()
            table=MyTable(None)
            table.setSelectionMode(QTable.NoSelection)
            self.id2table[id] = table
            self.table2id[table] = id
            if data.name:
                tabName = "%s " % data.name
            else:
                tabName = ""
            tabName += "(" + str(id[1]) + ")"
            if id[2] != None:
                tabName += " [" + str(id[2]) + "]"
##            tabName = data.name + " "+str(id)
            self.tabs.insertTab(table, tabName)
            self.set_table(table, data)
            self.tabs.showPage(table)
            self.progressBarFinished()
            self.set_info(data)
            # enable showMetas checkbox only if metas exist
            self.cbShowMeta.setEnabled(len(self.showMetas[id][1])>0)
        elif self.data.has_key(id):
            self.data.pop(id)
            self.showMetas.pop(id)
            self.id2table[id].hide()
            self.tabs.removePage(self.id2table[id])
            self.table2id.pop(self.id2table.pop(id))
            self.set_info(self.data.get(self.table2id.get(self.tabs.currentPage(),None),None))
        # disable showMetas checkbox if there is no data on input
        if len(self.data) == 0:
            self.cbShowMeta.setEnabled(False)


    def tabClicked(self, qTableInstance):
        """Updates the info box and showMetas checkbox when a tab is clicked.
        """
        id = self.table2id.get(qTableInstance,None)
        self.set_info(self.data.get(id,None))
        show_col = self.showMetas.get(id,None)
        if show_col:
            self.cbShowMeta.setChecked(show_col[0])
            self.cbShowMeta.setEnabled(len(show_col[1])>0)

    def cbShowMetaClicked(self):
        table = self.tabs.currentPage()
        id = self.table2id.get(table, None)
        if self.showMetas.has_key(id):
            show,col = self.showMetas[id]
            self.showMetas[id] = (not(show),col)
        if show:
            for c in col:
                table.hideColumn(c)
        else:
            for c in col:
                table.showColumn(c)
                # we need to readjust the column width
                table.adjustColumn(c)
                table.setColumnWidth(c, table.columnWidth(c)+22)


    def btnResetSortClicked(self):
        """Sort the data by the last (hidden) column.
        """
        self.sortby = 0
        table = self.tabs.currentPage()
        if table:
            self.sort(table.numCols()-1)


    def set_info(self, data):
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
                    self.infoClass.setText("Class neither discrete nor continuous.")
            else:
                self.infoClass.setText('Classless domain.')


    def set_table(self, table, data):
        """Writes data into table, adjusts the column width.
        """
        qApp.setOverrideCursor(QWidget.waitCursor)
        if data==None:
            return
        vars = data.domain.variables
        m = data.domain.getmetas() # getmetas returns a dictionary
        ml = [(k, m[k]) for k in m]
        ml.sort(lambda x,y: cmp(y[0], x[0]))
        metas = [x[1] for x in ml]
        metaKeys = [x[0] for x in ml]
        varsMetas = vars + metas
        numVars = len(data.domain.variables)
        numMetas = len(metas)
        numVarsMetas = numVars + numMetas
        numEx = len(data)
        numSpaces = int(math.log(max(numEx,1), 10))+1

        table.setNumCols(numVarsMetas+1)
        table.setNumRows(numEx)
        id = self.table2id.get(table,None)

        # set the header (attribute names)
        hheader=table.horizontalHeader()
        for i,var in enumerate(varsMetas):
            hheader.setLabel(i, var.name)
        hheader.setLabel(numVarsMetas, "")
        
        # set the contents of the table (values of attributes)
        # iterate variables
        table.disableUpdate=True
        table.hide()
        table.ranks={}
        table.values={}
        table.setDelayColumnAdjust(numVarsMetas>200)
        for j,(key,attr) in enumerate(zip(range(numVars) + metaKeys, varsMetas)):
            #table.setNumCols(j+1)
            #hheader.setLabel(j, attr.name)
            self.progressBarSet(j*100.0/numVarsMetas)
            if attr == data.domain.classVar:
                bgColor = QColor(160,160,160)
            elif attr in metas:
                bgColor = QColor(220,220,200)
                self.showMetas[id][1].append(j) # store indices of meta attributes
            else:
                bgColor = Qt.white
            # generate list of tuples (attribute value, instance index) and sort by attrVal
            valIdx = [(str(ex[key]),idx) for idx,ex in enumerate(data)]
            table.values[j]=[v[0]+" " for v in valIdx]
            valIdx.sort()
            # generate a dictionary where key: instance index, value: rank
            idx2rankDict = dict(zip([x[1] for x in valIdx], range(numEx)))
            table.ranks[j]=dict(zip(range(numEx), [x[1] for x in valIdx]))
            table.columnColor[j]=bgColor
            for i in range(numEx):
                # set sorting key to ranks
                pass
                #OWGUI.tableItem(table, i, j, str(data[i][key]), editType=QTableItem.Never, background=bgColor, sortingKey=self.sortingKey(idx2rankDict[i], numSpaces))
            # adjust the width of the table
            #table.showColumn(j)
            if numVarsMetas<=200:
                table.adjustColumn(j)
            #table.setColumnWidth(j, table.columnWidth(j)+22)
        #for j in range(numVarsMetas):
        #    table.adjustColumn(j)
        # add hidden column with consecutive numbers for restoring the original order of examples
        #hheader.setLabel(numVarsMetas, "")
        #table.setNumCols(numVarsMetas+1)
        table.show()
        #for i in range(numEx):
        #    OWGUI.tableItem(table, i, numVarsMetas, "", editType=QTableItem.Never, sortingKey=self.sortingKey(i, numSpaces))
        table.ranks[table.numCols()-1]=dict([(i,i) for i in range(numEx)])
        table.values[table.numCols()-1]=["" for i in range(numEx)]
        table.columnColor[table.numCols()-1]=QColor(0,0,0)
        table.disableUpdate=False
        table.hideColumn(numVarsMetas)

        # adjust vertical header
        table.verticalHeader().setClickEnabled(False)
        table.verticalHeader().setResizeEnabled(False)
        table.verticalHeader().setMovingEnabled(False)
        
        # manage sorting (not correct, does not handle real values)
        self.connect(hheader,SIGNAL("clicked(int)"),self.sort)
        self.sortby = 0
        #table.setColumnMovingEnabled(1)
        qApp.restoreOverrideCursor()
        table.setCurrentCell(-1,-1)
        table.clearCache()
        table.show()


    def sort(self, col):
        """Sorts the table by column col.
        """
        qApp.setOverrideCursor(QWidget.waitCursor)
        if col == self.sortby-1:
            self.sortby = - self.sortby
        else:
            self.sortby = col+1
        table = self.tabs.currentPage()
        table.sortColumn(col, self.sortby>=0, TRUE)
        table.horizontalHeader().setSortIndicator(col, self.sortby<0)
##        table.setSortIndicator(col, self.sortby>=0)
        qApp.restoreOverrideCursor()


    def sortingKey(self, val, len):
        """Returns a string with leading spaces followed by str(val), whose length is at least len.
        """
        s = "%0" + str(len) + "s"
        return s % str(val)


##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory
from sets import Set
class MyTable(QTable):    
    def __init__(self,*args):
        QTable.__init__(self, *args)
        self.disableUpdate=False
        self.disableColumnAdjust=False
        self.adjustedColumnCache=Set()
        self.sortingColumn=-1
        self.sortingAscending=True
        self.delayColumnAdjust=False
        self.columnColor={}
        #self.setWFlags(Qt.WRepaintNoErase | Qt.WNorthWestGravity)
        self.connect(self, SIGNAL("contentsMoving(int, int)"),self.update1)
        self.connect(self.horizontalScrollBar(), SIGNAL("sliderPressed()"), self.sliderPressed)
        self.connect(self.horizontalScrollBar(), SIGNAL("sliderReleased()"), self.sliderReleased)
        self.connect(self, SIGNAL("currentChanged(int, int)"), self.currentSelection)
        self.rectPen=QPen(Qt.black,1)
        self.selectedRectPen=QPen(Qt.black,2)
        self.currentSelected=(-1,-1)
        p=QPainter(self)
        self.setPainterFont(p)
        tm=p.fontMetrics()
        self.charWidth=tm.width("a")
        

    def setDelayColumnAdjust(self, bool):
        self.delayColumnAdjust=bool
        
    def clearCache(self):
        self.adjustedColumnCache=Set()
        
    def eventFilter(self, obj, event):
        if obj==self or obj==self.horizontalHeader:
            if event.type()==QEvent.Paint and self.delayColumnAdjust:
                self.adjustColumns()
                return True        
        return QTable.eventFilter(self, obj, event)
    
    def adjustColumns(self):
        if self.disableColumnAdjust:
            return
        #print "adjusting"
        cStart=self.columnAt(self.contentsX())+1
        cEnd=self.columnAt(self.contentsX()+self.visibleWidth())
        while cStart<min([self.columnAt(self.contentsX()+self.visibleWidth())+1, self.numCols()-1]):
            #for i in range(cStart, cEnd):
            if cStart not in self.adjustedColumnCache:
                self.adjustColumn(cStart)
                #self.setColumnWidth(cStart, self.columnWidth(cStart)+22)
                self.adjustedColumnCache.add(cStart)
            cStart+=1
    
    def setColumnWidth(self, i, w):
        #print i
        QTable.setColumnWidth(self, i, w+22)

    def adjustColumn(self, col):
        p=QPainter(self)
        self.setPainterFont(p)
        tm=p.fontMetrics()
        try:
            maxlen=max([len(t) for t in self.values[col]+[str(self.horizontalHeader().label(col))]])
            w=self.charWidth*maxlen
            self.setColumnWidth(col,w)
        except KeyError, err:
            pass
            print "Exception in adjustColumn ", col
                
    def sliderPressed(self):
        self.disableColumnAdjust=True

    def sliderReleased(self):
        self.disableColumnAdjust=False
        #self.update()
    
    def update1(self, i, j):
        self.update()

    def paintCell(self, painter, row, col, cr, selected):
        #print "Paint cell: ", row, col
        #from pywin import debugger
        #debugger.set_trace()
        if selected:
            painter.setPen(self.selectedRectPen)
            painter.drawRect(cr)
            painter.setPen(self.rectPen)
        else:
            p=QPoint(1,1)
            cr=QRect(cr.topLeft()-p, cr.bottomRight())
            painter.setPen(self.rectPen)
            painter.drawRect(cr)
        #try:
        if self.sortingAscending:
            text=self.values[col][self.ranks[self.sortingColumn][row]]
        else:
            numAll=self.numRows()
            text=self.values[col][self.ranks[self.sortingColumn][numAll-1-row]]
        painter.drawText(cr, Qt.AlignRight|Qt.AlignVCenter, text)
        #except:
        #    pass

    def clearCell(self, row, col):
        #print "Clear cell: ", row, col
        p=QPainter(self)
        p.fillRect(self.cellGeometry(row, col), QBrush(Qt.white))
        
    def updateCell(self, row, col):
        #print "Update cell: ",row, col
        if row!=-1 and col!=-1:
            pass
            #self.clearCell(row, col)
        QTable.updateCell(self, row, col)
        
    def drawContents(self, painter, cx=0, cy=0, cw=0, ch=0):
        #print "Draw contnents: ",cx,cy,cw,ch
        if self.sortingColumn not in self.ranks:
            self.sortingColumn=self.numCols()-1
        #self.paintEmptyArea(painter, cx, cy, cw, ch)
        self.setPainterFont(painter)
        xStart=max(self.columnAt(cx), 0)
        xEnd=min(self.columnAt(cx+cw)+1, self.numCols()) or self.numCols() #columnAt can return -1 if there is no cell at that position
        yStart=max(self.rowAt(cy),0)
        yEnd=min(self.rowAt(cy+ch)+1, self.numRows()) or self.numRows() # the same as above
        #print "X start:", xStart, "X end:", xEnd, "Y start:", yStart, "Y end:", yEnd
        for i in range(xStart, xEnd):
            painter.setBrush(QBrush(self.columnColor[i]))
            for j in range(yStart, yEnd):
                self.paintCell(painter, j, i, self.cellGeometry(j, i), self.isSelected(j, i))
                
    def paintEvent(self, paintEvent):
        QTable.paintEvent(self, paintEvent)
        #upper left corner gets painted like the 0,0 cell (why??) 
        painter=QPainter(self)
        painter.setBrush(QBrush(Qt.gray))
        painter.drawRect(1, 1, 32, 20)

    def paintEmptyArea(self, painter, cx, cy, cw, ch):
        painter.fillRect(cx, cy, cw, ch, QBrush(Qt.white))

    def sortColumn(self, col, ascending=True, wholeRows=False):
        self.sortingColumn=col
        self.sortingAscending=ascending
        self.repaintContents(self.contentsX(), self.contentsY(), self.visibleWidth(), self.visibleHeight())
        #print "sort by: ", col, ascending

    def currentSelection(self, row, col):
        if self.currentSelected!=(-1,-1):
            r,c=self.currentSelected
            self.currentSelected=(-1,-1)
            self.updateCell(r,c)
        self.currentSelected=(row,col)
        self.updateCell(row,col)

    def isSelected(self, row, col):
        return self.currentSelected==(row, col)
    
    def resizeData(self, i):
        return

    def setPainterFont(self, painter):
        font=QFont()
        font.setStyleHint(QFont.Courier)
        painter.setFont(font)

    """    
    def columnWidthChanged(self, col):
        pass
        #print col
        #QTable.columnWidthChanged(self, col)"""
        
if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDataTable()
    a.setMainWidget(ow)

##    d = orange.ExampleTable('wtclassed')
    d1 = orange.ExampleTable(r'..\..\doc\datasets\auto-mpg')
    d2 = orange.ExampleTable(r'..\..\doc\datasets\voting.tab')
    d3 = orange.ExampleTable(r'..\..\doc\datasets\sponge.tab')
    d4 = orange.ExampleTable(r'..\..\doc\datasets\wpbc.csv')
    d5 = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
    ow.show()
    ow.dataset(d1,"auto-mpg")
    ow.dataset(d2,"voting")
##    ow.dataset(None,"auto-mpg")
##    ow.dataset(d3,"sponge")
##    ow.dataset(None,"voting")
##    ow.dataset(None,"sponge")
    ow.dataset(d4,"wpbc")
    ow.dataset(d5,"adult_sample")
    a.exec_loop()
