# FIXME: Sortiranje nekaterih vrednosti v tabeli je se vedno narobe.
# FIXME: Zapomni si zadnji klik na vrstico v QListView. Ta vrstica naj potem ostala izbrana med prehodom iz multi-solo izbiro.

"""
<name>MeSH Term Browser</name>
<description>Browser over MeSH ontology</description>
<icon>icons/MeSH.png</icon>
<contact>Crt Gorup (crt.gorup@gmail.com)</contact> 
<priority>2016</priority>
"""

from OWWidget import *
from qttable import *
from qt import *
from orngMeSH import *
import OWGUI

class MyQTableItem(QTableItem):
    """ Our implementation of QTable item allowing numerical sorting.  """
    def __init__(self,table,text):
        QTableItem.__init__(self,table,QTableItem.Never,text)
        self.data = text
        
    def key(self):      # additional setting to correctly handle text and numerical sorting
        try:                    # sorting numerical column
            e = float(self.data)
            tdata = pval = "%.4g" % e
            l = len(self.data)
            pre = ""
            
            offset = 0          # additional parameter to hadle exponent '2e-14' float format
            if(tdata.count('e')>0 or tdata.count('E')>0):
                pre="*"
#               print e
#               print tdata
                offset = 1
                
            for i in range(0,40-l-offset):
                pre = pre + "0"
            return pre + tdata
        except ValueError:      # sorting text column
            return self.data

class ListViewToolTip(QToolTip):
    """brief A class to allow tooltips in a listview."""
    def __init__(self, view, column, data):
        """brief ListViewToolTip constructor.
        \param view          QListView instance
        \param column        Listview column
        \param truncatedOnly Only display the tooltip if the column data is truncated
        """
        QToolTip.__init__(self, view.viewport())
        self.__view = view
        self.__col  = column
        self.__data = data      # mapping from name -> description
        # self.setWakeUpDelay(400)

    def appendData(self,data):
        self.__data = data

    def maybeTip(self, pos):
        """brief Draw the tooltip.
        \param pos Tooltip position.
        """
        item = self.__view.itemAt(pos)
        if item is not None:
            if(self.__data.has_key(str(item.text(self.__col)))):
                tipString = self.__data[str(item.text(self.__col))]
                counter = 45
                newTipString = ""
                for i in tipString.split(" "):
                    if counter < 0:
                        newTipString = newTipString + i + "\n"
                        counter = 45
                    else:
                        newTipString = newTipString + i + " "
                        counter -= len(i)
                tipString = newTipString
                cr = self.__view.itemRect(item)
                headerPos = self.__view.header().sectionPos(self.__col)
                cr.setLeft(headerPos)
                cr.setRight(headerPos + self.__view.header().sectionSize(self.__col))
                self.tip(cr, tipString)
            else:
                print item.text(self.__col) + " not found in toDecs"

class OWMeSHBrowser(OWWidget):
    settingsList = ["multi", "maxPValue", "minExamplesInTerm"]
    def __init__(self,parent=None,signalManager=None):
        OWWidget.__init__(self,parent,signalManager,"MeshBrowser")
        self.inputs = [("Reference data", ExampleTable, self.getReferenceData),("Cluster data", ExampleTable, self.getClusterData) ]
        self.outputs = [("Selected examples", ExampleTable)]

        # widget variables
        self.loadedRef = 0
        self.loadedClu = 0
        self.maxPValue = 0.05
        self.minExamplesInTerm = 5
        self.multi = 1
        self.reference = None
        self.cluster = None
        self.loadSettings()        
        self.mesh = orngMeSH()          # main object is created
        self.dataLoaded = self.mesh.dataLoaded

        # left pane
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel("No reference data.", box)
        self.infob = QLabel("No cluster data.", box)
        self.ratio = QLabel("", box)
        self.ref_att = QLabel("", box)
        self.clu_att = QLabel("", box)
        self.resize(930,600)

        OWGUI.separator(self.controlArea)
        self.optionsBox = QVGroupBox("Options", self.controlArea)
        self.maxp = OWGUI.lineEdit(self.optionsBox, self, "maxPValue", label="threshold:", orientation="horizontal", labelWidth=120, valueType=float)
        self.minf = OWGUI.lineEdit(self.optionsBox, self, "minExamplesInTerm", label="min. frequency:", orientation="horizontal", labelWidth=120, valueType=int)        
        OWGUI.checkBox(self.optionsBox, self, 'multi', 'Multiple selection', callback= self.checkCLicked)
        OWGUI.button(self.optionsBox, self, "Refresh", callback=self.refresh)
        OWGUI.button(self.optionsBox, self, "Update MeSH ontology", callback=self.download)

        # right pane
        self.col_size = [300,84,84,100,120]
        self.sort_col = 0
        self.sort_dir = True
        self.columns = ['MeSH term', '# reference', '# cluster', 'p value','fold enrichment'] # both datasets
        layout=QVBoxLayout(self.mainArea)
        splitter = QSplitter(QSplitter.Vertical, self.mainArea)
        layout.add(splitter)

        # list view
        self.meshLV = QListView(splitter)
        self.meshLV.setMultiSelection(self.multi)
        self.meshLV.setAllColumnsShowFocus(1)
        self.meshLV.addColumn(self.columns[0])
        self.meshLV.setColumnWidth(0, self.col_size[0])
        self.meshLV.setColumnWidthMode(0, QListView.Manual)
        self.meshLV.setColumnAlignment(0, QListView.AlignLeft)
        self.meshLV.setSorting(-1)
        self.meshLV.setRootIsDecorated(1)
        i=1
        for title in self.columns[1:]:
            col = self.meshLV.addColumn(title)
            self.meshLV.setColumnWidth(col, self.col_size[i])
            self.meshLV.setColumnWidthMode(col, QListView.Manual)
            self.meshLV.setColumnAlignment(col, QListView.AlignCenter)
            i += 1
        self.connect(self.meshLV, SIGNAL("selectionChanged()"), self.viewSelectionChanged)
        self.tooltips = ListViewToolTip(self.meshLV,0, self.mesh.toDesc)        

        # table of significant mesh terms
        self.sigTermsTable = QTable(splitter)
        self.sigTermsTable.setNumCols(len(self.columns))
        self.sigTermsTable.setNumRows(4)
        ## hide the vertical header
        self.sigTermsTable.verticalHeader().hide()
        self.sigTermsTable.setLeftMargin(0)
        self.sigTermsTable.setSelectionMode(QTable.Multi)
        self.header = self.sigTermsTable.horizontalHeader() 
        for i in range(self.sigTermsTable.numCols()):
            self.sigTermsTable.setColumnWidth(i, self.col_size[i])
            self.header.setLabel(i, self.columns[i])

                   
        self.connect(self.sigTermsTable, SIGNAL("selectionChanged()"), self.tableSelectionChanged) 
        self.connect(self.sigTermsTable, SIGNAL("clicked(int,int,int,const QPoint&)"), self.tableClicked)  
        self.optionsBox.setDisabled(1)

    def download(self):
        self.progressBarInit()
        self.mesh.downloadOntology(callback=self.progressBarSet)
        self.progressBarFinished()

    def tableSelectionChanged(self):
        return True

    def tableClicked(self,row,col,button, point):
        if self.sort_col == col:
            self.sort_dir = not self.sort_dir
        else:
            self.sort_col = col
            self.sort_dir = True
            
        self.sigTermsTable.sortColumn(self.sort_col,self.sort_dir,True)
        #print "sortiram ", col, " ",row

    def checkCLicked(self):
        if self.multi == 0:
            self.meshLV.clearSelection()
        self.meshLV.setMultiSelection(self.multi)

    def viewSelectionChanged(self):
        items = list()
        self.progressBarInit()
        for i in self.lvItem2Mesh.iterkeys():
            if i.isSelected():
                items.append(self.lvItem2Mesh[i])
        
        if self.reference:
            data=self.mesh.findSubset(self.reference, items, callback=self.progressBarSet)
        else:
            data=self.mesh.findSubset(self.cluster, items, callback=self.progressBarSet)
        
        
        #print items
        self.send("Selected examples", data)
        self.progressBarFinished()

    def __updateData__(self):
        self.lvItem2Mesh = dict()
        
        if(self.reference and self.cluster):
            if(len(self.cluster) > len(self.reference)):
                self.optionsBox.setDisabled(1)
                QMessageBox.warning( None, "Invalid dataset length", "Cluster dataset is shorter than reference dataset. You should check if signals are connected to correct inputs." , QMessageBox.Ok)
                return False

            # everything is ok, now we can calculate enrichment and update labels, tree view and table data
            self.optionsBox.setDisabled(0)
            self.progressBarInit()
            self.treeInfo, self.results = self.mesh.findEnrichedTerms(self.reference,self.cluster,self.maxPValue, treeData= True, callback= self.progressBarSet)
            self.progressBarFinished()
            self.ratio.setText("ratio = %.4g" % self.mesh.ratio)
            self.clu_att.setText("cluster MeSH att: " + self.mesh.clu_att)
            self.ref_att.setText("reference MeSH att: " + self.mesh.ref_att)

            # table data update
            self.sigTermsTable.setNumRows(len(self.results))
            index = 0
            for i in self.results.iterkeys(): ## sorted by the p value
               # mTerm = i[0]
                mID = self.mesh.toName[i] + " (" + i + ")"
                rF = self.results[i][0]
                cF = self.results[i][1]
                pval = self.results[i][2]
                fold = self.results[i][3]
                pval = "%.4g" % pval
                fold = "%.4g" % fold
                vals = [mID ,rF,cF, pval, fold]
                for j in range(len(vals)):
                    self.sigTermsTable.setItem(index,j, MyQTableItem(self.sigTermsTable, str(vals[j])))
                index = index + 1

            # initial sorting - p value
            self.sort_col = 3
            self.sort_dir = True
            self.sigTermsTable.sortColumn(self.sort_col, self.sort_dir,True)

            # tree view update - The most beautiful part of this widget!
            starters = self.treeInfo["tops"]       # we get a list of possible top nodes            
            self.meshLV.clear()

            for e in starters:      # we manualy create top nodes
                f = QListViewItem(self.meshLV);
                f.setOpen(1)
                rfr = str(self.results[e][0])
                cfr = str(self.results[e][1])
                pval = "%.4g" % self.results[e][2]
                fold = "%.4g" % self.results[e][3]
                self.lvItem2Mesh[f] = self.mesh.toName[e]
                data = [self.mesh.toName[e], rfr, cfr, pval, fold]
                for t in range(len(data)):
                    f.setText(t,data[t])
                self.__treeViewMaker__(f, e, False)


        elif self.reference or self.cluster:
            if self.reference:
                current_data = self.reference
            else:
                current_data = self.cluster

            self.optionsBox.setDisabled(0)
            self.progressBarInit()
            self.treeInfo, self.results = self.mesh.findFrequentTerms(current_data,self.minExamplesInTerm, treeData= True, callback= self.progressBarSet)
            self.progressBarFinished()
            if self.reference:
                self.ref_att.setText("reference MeSH att: " + self.mesh.solo_att)
            else:
                self.clu_att.setText("cluster MeSH att: " + self.mesh.solo_att)

            # table data update
            self.sigTermsTable.setNumRows(len(self.results))
            index = 0
            for i in self.results.iterkeys(): 
                mID = self.mesh.toName[i] + " (" + i + ")"
                rF = self.results[i]
                vals = [mID ,rF]
                for j in range(len(vals)):
                    self.sigTermsTable.setItem(index,j, MyQTableItem(self.sigTermsTable, str(vals[j])))
                    #self.sigTermsTable.setText(index, j, str(vals[j]))
                index = index + 1

            # initial sorting - frequency
            self.sort_col = 1
            self.sort_dir = True
            self.sigTermsTable.sortColumn(self.sort_col, self.sort_dir,True)

            # tree view update - The most beautiful part of this widget!
            starters = self.treeInfo["tops"]       # we get a list of possible top nodes
            self.meshLV.clear()
            
            for e in starters:      # we manualy create top nodes
                f = QListViewItem(self.meshLV);
                f.setOpen(1)
                self.lvItem2Mesh[f] = self.mesh.toName[e]
                rfr = str(self.results[e])
                data = [self.mesh.toName[e], rfr]
                for t in range(len(data)):
                    f.setText(t,data[t])
                self.__treeViewMaker__(f, e, True)
            
    def __treeViewMaker__(self,parentLVI, parentID, soloMode):
        """ Function builds tree in treeListView. If soloMode = True function only display first two columns in tree view. (suitable when only one dataset is present) """
        for i in self.treeInfo[parentID]:   # for each succesor
            f = QListViewItem(parentLVI);
            f.setOpen(1)
            
            data = [self.mesh.toName[i]]
            if soloMode:
                rfr = str(self.results[i])
                data.append(rfr)
            else:           # when we have referece and cluster dataset we have to print additional info
                rfr = str(self.results[i][0])
                cfr = str(self.results[i][1])
                pval = "%.4g" % self.results[i][2]
                fold = "%.4g" % self.results[i][3]
                data.extend([rfr, cfr, pval, fold])

            self.lvItem2Mesh[f]=data[0]         # mapping   QListViewItem <-> mesh id

            for t in range(len(data)):
                f.setText(t,data[t])
            self.__treeViewMaker__(f,i, soloMode)            
        return True

    def refresh(self):
        if self.reference or self.cluster:        
            self.__switchGUI__() 
            self.__updateData__()

    def __clearGUI__(self):
        """ Function clears updates tree view, table and labels to theis default value """
        self.meshLV.clear()
        self.sigTermsTable.setNumRows(0)

    def __switchGUI__(self):
        """ Function updates GUI for one or two datasets. """
        if not self.reference and not self.cluster:
            self.optionsBox.setDisabled(1)
            return
        
        self.optionsBox.setDisabled(0)
        solo = True
        if self.reference and self.cluster:
            solo = False

        if solo:
            self.maxp.setDisabled(1)
            self.minf.setDisabled(0)
            for i in range(2,len(self.columns)):
                #self.meshLV.hideColumn(i)
                self.meshLV.setColumnWidth(i,0)                
                #self.sigTermsTable.hideColumn(i)
                self.sigTermsTable.setColumnWidth(i,0)
            
            self.header.setLabel(1, "frequency")
            self.meshLV.setColumnText(1,"frequency")
            self.ratio.setText("")
        else:
            self.maxp.setDisabled(0)
            self.minf.setDisabled(1)
            for i in range(2,len(self.columns)):
                self.meshLV.setColumnWidth(i,self.col_size[i])
                self.sigTermsTable.setColumnWidth(i, self.col_size[i])
            self.header.setLabel(1, self.columns[1])
            self.meshLV.setColumnText(1,self.columns[1])
            self.ratio.setText("ratio = %.4g" % self.mesh.ratio)

    def getReferenceData(self,data):
        if data:
            self.reference = data
            self.infoa.setText('%d reference instances' % len(data))
        else:
            self.reference = None
            self.infoa.setText('No reference data.')
            self.ref_att.setText('')

        if self.reference or self.cluster:        
            self.__switchGUI__() 
            self.__updateData__()
        else:
            self.__clearGUI__()

    def getClusterData(self,data):
        if data:
            self.cluster = data
            self.infob.setText('%d cluster instances' % len(data))
        else:
            self.cluster = None
            self.infob.setText('No cluster data.')
            self.clu_att.setText('')
        
        if self.reference or self.cluster:        
            self.__switchGUI__() 
            self.__updateData__()
        else:
            self.__clearGUI__()
