"""
<name>Save Naive Bayes</name>
<description>Saves Naive Bayesian classification model to an XML data file.</description>
<category>Classification</category>
<icon>icons/NaiveBayesSave.png</icon>
<priority>4100</priority>
"""

# to do:
# - defaults (get from data files)
# - author, date
# - panes, (Description, Pages, Variables)
# - how to kill/remove widget instead of hiding it
# - treatment of multi-class problems (iris does not load, since
#   has three classes). Change orngBayes so that it treats target class
#   accordingly

from qttable import *
from OWWidget import *
from OWGUI import *
import string, warnings
import orange, orngBayes

##############################################################################
#

class storeI:
    def __init__(self, i):
        self.i = i
    def __call__(self, i):
        self.i = i

class vcb:
    def __init__(self, widget, attribute, storeI, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        self.storeI = storeI
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        self.widget.info[self.storeI.i][self.attribute] = str(value)

class OWNaiveBayesSave(OWWidget):
#    settingsList = ["nbName", "nbDesc", "nbOutcome", "info"]
    settingsList = ["info", "fileName"]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "SaveNaiveBayes",
        "Saves Naive Bayesian classification model\n to an XML data file.")
        
        self.dataset = None
        self.tmpWidgets = []

        self.addInput("nbClassifier")
        self.callbackDeposit = []

        # current info set, determined based on classifier
        self.cInfo = 0
        self.storeI = storeI(self.cInfo)

        # Settings        
        self.nbName = ''; self.nbDesc = ''; self.nbOutcome = ''
        self.info = []
        self.fileName = "model.xml"
        self.info.append({'id':'', 'name':'', 'desc':'', 'outcome':''})
        self.loadSettings()

        # GUI
        self.saveBtn = QPushButton("&Save Model As ...", self.controlArea)
        self.connect(self.saveBtn, SIGNAL("clicked()"), self.saveAs)

        self.layout=QVBoxLayout(self.mainArea)
        (self.nameBox, self.nameW) = self.editBox(self.mainArea, '', 'name', 'Model Name', space=8)
        (self.descBox, self.descW) = self.editBox(self.mainArea, '', 'desc', 'Description', space=8)
        (self.outcBox, self.outW)  = self.editBox(self.mainArea, '', 'outcome', 'Outcome', space=8)
        self.layout.add(self.nameBox)
        self.layout.add(self.descBox)
        self.layout.add(self.outcBox)
        
        self.catAtt = QVGroupBox(self.mainArea)
        self.catAtt.setTitle('Categorical Attributes')
        self.catTable=QTable(self.catAtt)
        self.catTable.setSelectionMode(QTable.NoSelection)
        self.catTable.setLeftMargin(0)
        self.layout.add(self.catAtt)

        self.numAttG = QVGroupBox(self.mainArea)
        self.numAttG.setTitle('(Originally) Numerical Attributes')
        self.numTable=QTable(self.numAttG)
        self.numTable.setLeftMargin(0)
        self.numTable.setSelectionMode(QTable.NoSelection)
        self.layout.add(self.numAttG)
        
        #self.table.resize(self.mainArea.size())
        self.resize(760,200)

    def editBox(self, widget, text, var, boxText, space=None):
        nb = QVGroupBox(widget)
        nb.setTitle(boxText)
        if text:
            hb = QHBox(nb)
            QLabel(text, hb)
        wa = QLineEdit(nb)
        wa.setText(self.info[self.cInfo][var])
        self.connect(wa, SIGNAL("textChanged(const QString &)"), vcb(self, var, self.storeI, str))
        if space:
            QWidget(widget).setFixedSize(0, space)
        return (nb, wa)

    def setCInfo(self):
        # based on the data set determines what is the most appropriate
        # info slot, if none is found, creates a new one and copies
        # name, desc, and outcome info into the slot
        # assumes that classifier has been given
        bestMatches = 0
        self.cInfo = 0 # reset the domain index
        for i in range(len(self.info)):
            matches = 0.0
            id = self.info[i]['id']
            for a in self.model.domain.attributes:
                if find(id, a.name)>=0:
                    matches += 1.
            matches = matches / len(self.model.domain.attributes)
            if (matches > 0.8) and (matches > bestMatches):
                self.cInfo = i
                self.storeI(self.cInfo)
                bestMatches = matches
        if not (self.cInfo>0):
            # no appropriate record found, add a new one, copy current values of fixed records
            id = reduce(lambda a,b: a + ';' + b, [i.name for i in self.model.domain.attributes])
            self.info.append({'id':id, 'name':self.nbName, 'desc':self.nbDesc, 'outcome':self.nbOutcome})
            self.cInfo = len(self.info) - 1  # set the index to domain we have just added
            
            self.storeI(self.cInfo)
#            print 'new domain, new index', self.cInfo, len(self.info), self.info
            for a in ['name', 'desc', 'outcome']:
                self.info[self.cInfo][a] = self.info[0][a]
        else:
#            print 'domain matched %d (%5.3f)' % (self.cInfo, bestMatches)
            myinfo = self.info[self.cInfo]
            if myinfo.has_key("name"): self.nameW.setText(myinfo["name"])
            if myinfo.has_key("desc"): self.descW.setText(myinfo["desc"])
            if myinfo.has_key("outcome"): self.outW.setText(myinfo["outcome"])

    def setNumTable(self):
        self.numAtts = []
        for a in self.model.domain.attributes:
            if a.getValueFrom:
                self.numAtts.append(a)

        headLabel = ('Name', 'Description', 'Num?', 'Entry Type', 'Default')
        colWidths = (80,303,40,110,50)
        cols = len(headLabel); rows = len(self.numAtts)
        self.numTable.setNumCols(cols)
        self.numTable.setNumRows(rows)

        # set the header (attribute names)
        self.header=self.numTable.horizontalHeader()
        for i in range(len(headLabel)):
            self.header.setLabel(i, headLabel[i])
            self.numTable.setColumnWidth(i, colWidths[i])

        # set the entries
        self.numCngNumClbck = [None]*len(self.numAtts)
        self.numCngEntrClbck = [None]*len(self.numAtts)
        for j in range(len(self.numAtts)):
            myinfo = self.info[self.cInfo]
            a = self.numAtts[j].name
            self.numTable.setText(j, 0, myinfo.has_key('C:NAME:'+a) and myinfo['C:NAME:'+a] or a[2:])
            self.numTable.setText(j, 1, myinfo.has_key('C:DESC:'+a) and myinfo['C:DESC:'+a] or '')
            self.numTable.setText(j, 4, myinfo.has_key('C:DEFA:'+a) and str(myinfo['C:DEFA:'+a]) or '0')

            cb = QCheckBox('', None)
            self.numCngNumClbck[j] = lambda x, v=j, t='C:NUM:': self.bothCngNum(x, v, t)
            self.connect(cb,SIGNAL("toggled(bool)"), self.numCngNumClbck[j])
            if myinfo.has_key('C:NUM:'+a):
                cb.setChecked(myinfo['C:NUM:'+a])
            self.numTable.setCellWidget(j, 2, cb)
            self.tmpWidgets.append(cb)

            en = QComboBox(self, None)
            self.numCngEntrClbck[j] = lambda x, v=j, t='C:ENTR:': self.bothCngNum(x, v, t)
            self.connect(en, SIGNAL("activated(int)"), self.numCngEntrClbck[j])
            for m in ['Pulldown Menu', 'Radio Horizontal', 'Radio Vertical']:
                en.insertItem(m)
            en.setCurrentItem(myinfo.has_key('C:ENTR:'+a) and myinfo['C:ENTR:'+a])
            self.tmpWidgets.append(en)
                
            self.numTable.setCellWidget(j, 3, en)

        self.connect(self.numTable, SIGNAL("valueChanged(int, int)"), self.numCng)

    def setCatTable(self):
        self.catAtts = []
        for a in self.model.domain.attributes:
            if not a.getValueFrom:
                self.catAtts.append(a)

        headLabel = ('Name', 'Description', 'Entry Type', 'Default')
        colWidths = (80,245,110,150)
        cols = len(headLabel); rows = len(self.catAtts)
        self.catTable.setNumCols(cols)
        self.catTable.setNumRows(rows)

        # set the header (attribute names)
        self.header=self.catTable.horizontalHeader()
        for i in range(len(headLabel)):
            self.header.setLabel(i, headLabel[i])
            self.catTable.setColumnWidth(i, colWidths[i])

        # set the entries
        self.catCngEntrClbck = [None]*len(self.catAtts)
        self.catCngDefClbck = [None]*len(self.catAtts)
        for j in range(len(self.catAtts)):
            myinfo = self.info[self.cInfo]
            a = self.catAtts[j].name
            self.catTable.setText(j, 0, myinfo.has_key('D:NAME:'+a) and myinfo['D:NAME:'+a] or a)
            self.catTable.setText(j, 1, myinfo.has_key('D:DESC:'+a) and myinfo['D:DESC:'+a] or '')

            en = QComboBox(self, None)
            self.catCngEntrClbck[j] = lambda x, v=j, t='D:ENTR:': self.bothCngNum(x, v, t)
            self.connect(en, SIGNAL("activated(int)"), self.catCngEntrClbck[j])
            for m in ['Pulldown Menu', 'Radio Horizontal', 'Radio Vertical']:
                en.insertItem(m)
            en.setCurrentItem(myinfo.has_key('D:ENTR:'+a) and myinfo['D:ENTR:'+a])
            self.catTable.setCellWidget(j, 2, en)
            self.tmpWidgets.append(en)

            de = QComboBox(self, None)
            self.catCngDefClbck[j] = lambda x, v=j, t='D:DEFA:': self.bothCngNum(x, v, t)
            self.connect(de, SIGNAL("activated(int)"), self.catCngDefClbck[j])
            for k in range(len(self.catAtts[j].values)):
                de.insertItem(self.catAtts[j].values[k])
            de.setCurrentItem(myinfo.has_key('D:DEFA:'+a) and myinfo['D:DEFA:'+a])
            self.catTable.setCellWidget(j, 3, de)
            self.tmpWidgets.append(de)

        self.connect(self.catTable, SIGNAL("valueChanged(int, int)"), self.catCng)

    def numCng(self, row, col):
        a = self.numAtts[row].name
        if (col==0):
            self.info[self.cInfo]['C:NAME:'+a] = str(self.numTable.text(row, col))
        if (col==1):
            self.info[self.cInfo]['C:DESC:'+a] = str(self.numTable.text(row, col))
        elif (col==4):
            self.info[self.cInfo]['C:DEFA:'+a] = atof(str(self.numTable.text(row, col)))
            
    def catCng(self, row, col):
        a = self.catAtts[row].name
        if (col==0):
            self.info[self.cInfo]['D:NAME:'+a] = str(self.catTable.text(row, col))
        if (col==1):
            self.info[self.cInfo]['D:DESC:'+a] = str(self.catTable.text(row, col))
            
    def bothCngNum(self, value, row, id):
        if (id[0]=='C'):
            a = self.numAtts[row].name
        else:
            a = self.catAtts[row].name
        self.info[self.cInfo][id+a] = int(value)

##############################################################################
# Processing the input

    def nbClassifier(self, classifier):
        self.model = classifier
        self.setCInfo()
        for e in self.tmpWidgets:
            e.hide()
        self.tmpWidgets = []
        self.setNumTable()
        self.setCatTable()
    
##############################################################################
# Saving model into XML file

    def saveAs(self):
        qfileName = QFileDialog.getSaveFileName(self.fileName,"Decisions-at-Hand (.xml)", None, "Save to..")
        self.fileName = str(qfileName)
        if not self.fileName:
            return
        warnings.filterwarnings("ignore", ".*builtin attribute.*", orange.AttributeWarning)

        # print self.fileName
        #(fil,ext) = os.path.splitext(fileName)
        #ext = ext.replace(".","")
        #ext = ext.upper()
        #print fil, 'xxx', ext

        myinfo = self.info[self.cInfo]
        if myinfo.has_key('name'): self.model.xName = myinfo['name']
        if myinfo.has_key('desc'): self.model.xDescription = myinfo['desc']
        if myinfo.has_key('outcome'): self.model.xName = myinfo['outcome']

        for a in self.model.domain.attributes:
            if a.getValueFrom:
                prefix = 'C:'
            else:
                prefix = 'D:'

            if myinfo.has_key(prefix+'NAME:'+a.name): a.xName = myinfo[prefix+'NAME:'+a.name]
            if myinfo.has_key(prefix+'DESC:'+a.name): a.xDescription = myinfo[prefix+'DESC:'+a.name]
            if prefix == 'D:':
                if myinfo.has_key(prefix+'DEFA:'+a.name): a.xDefault = tuple(a.values)[myinfo[prefix+'DEFA:'+a.name]]
            else:
                if myinfo.has_key(prefix+'DEFA:'+a.name): a.xDefault = myinfo[prefix+'DEFA:'+a.name]
                if myinfo.has_key(prefix+'NUM:'+a.name): a.xInputAsNumber = myinfo[prefix+'NUM:'+a.name]
            if myinfo.has_key(prefix+'ENTR:'+a.name): a.xInputType = ['pulldown','radiohorizontal','radiovertical'][myinfo[prefix+'ENTR:'+a.name]]

        orngBayes.saveXML(self.fileName, self.model)
        
##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    import orange, orngDisc
    warnings.filterwarnings("ignore", ".*builtin attribute.*", orange.AttributeWarning)

    a=QApplication(sys.argv)
    ow=OWNaiveBayesSave()
    a.setMainWidget(ow)

    data = orange.ExampleTable(r'..\datasets\adult_sample')
    dd = orngDisc.entropyDiscretization(data)
    nb = orange.BayesLearner(dd)
    ow.nbClassifier(nb)

    data = orange.ExampleTable(r'..\datasets\iris')
    dd = orngDisc.entropyDiscretization(data)
    nb = orange.BayesLearner(dd)
    ow.nbClassifier(nb)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
