"""
<name>Rank</name>
<description>Rank is an Orange Widget that shows ranking of attributes 
by their relevance for particular classification.</description>
<icon>icons/Rank.png</icon>
<priority>200</priority>
"""
#
# OWRank.py
#
# Rank is an Orange Widget that
# shows ranking of attributes 
# by their relevance for particular classification
# 
# 
#

from qttable import *
from OWWidget import *

class MyTable(QTable):
    def __init__(self, parent, widget):
        QTable.__init__(self, widget)
        self.parent = parent
        
    def contentsMouseReleaseEvent(self, ev):
        self.parent.sendSelected()

class OWRank(OWWidget):
    settingsList=["Precision","ReliefK","ReliefN","DiscretizationMethod","DisplayReliefF","DisplayInfoGain","DisplayGainRatio","DisplayGini"]
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Rank")
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.data)]
        self.outputs = [("Selected Attributes", ExampleTableWithClass)] 
        
        #set default settings
        self.Precision=3
        self.ReliefK=11
        self.ReliefN=20
        self.DiscretizationMethod="equal-frequency intervals"
        self.DisplayReliefF=1
        self.DisplayInfoGain=1
        self.DisplayGainRatio=1
        self.DisplayGini=1
        #load settings
        self.loadSettings()
        
        self.data=None
        # add a settings dialog (create this dialog separately by inheriting from OWOptions
        # and adding your setting controls, like checkboxes, radiobuttons and sliders
        # and don't forget tooltips)
        self.options=OWRankOptions()
        self.activateLoadedSettings()
        
        #connect settingsbutton to show options
        #self.connect(self.settingsButton,SIGNAL("clicked()"),self.options.show),
               
        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.displayReliefF,SIGNAL("stateChanged(int)"),self.displayReliefF) 
        self.connect(self.options.displayInfoGain,SIGNAL("stateChanged(int)"),self.displayInfoGain) 
        self.connect(self.options.displayGainRatio,SIGNAL("stateChanged(int)"),self.displayGainRatio) 
        self.connect(self.options.displayGini,SIGNAL("stateChanged(int)"),self.displayGini) 
        self.connect(self.options.kSlider,SIGNAL("valueChanged(int)"),self.setK)
        self.connect(self.options.nSlider,SIGNAL("valueChanged(int)"),self.setN)
        self.connect(self.options.discretization,SIGNAL("activated(int)"),self.setDM)
        self.connect(self.options.precisionSlider,SIGNAL("valueChanged(int)"),self.setPrecision)
        
        #GUI
        
        #give mainArea a layout
        self.layout=QVBoxLayout(self.mainArea)
        #add your components here
        self.table = MyTable(self, self.mainArea)
        #self.table.setSelectionMode(QTable.NoSelection) #hell, if i set this, swapRows() doesn't work - weird. and if i do not set this, the users can edit the fields
        self.table.setSelectionMode(QTable.Multi)
        self.layout.add(self.table)
        self.table.setNumCols(7)
        self.table.setNumRows(0)
        self.est_names = ["ReliefF", "InfoGain", "GainRatio", "Gini"]
        self.topheader=self.table.horizontalHeader()
        self.topheader.setLabel(0,"Attribute")
        self.table.adjustColumn(0)
        self.topheader.setLabel(1,"C/D")
        self.table.adjustColumn(1)
        self.topheader.setLabel(2,"#")
        self.table.adjustColumn(2)
        self.connect(self.topheader,SIGNAL("pressed(int)"),self.sort)
        self.sortby=-1
        for i in range(4):
            self.topheader.setLabel(3+i,self.est_names[i])  
            self.table.adjustColumn(i)     
        self.resize(600,200)

    def sendSelected(self):
        attrs = []
        if not self.data: return
        for i in range(self.table.numSelections()):
            selection = self.table.selection(i)
            top = selection.topRow(); bottom = selection.bottomRow()
            for c in range(top, bottom+1):
                attr = str(self.table.text(c, 0))
                if attr not in attrs: attrs.append(attr)
        if attrs != []: self.send("Selected Attributes", self.data.select(attrs + [self.data.domain.classVar.name]))
        
    def displayReliefF(self,checked):
        self.DisplayReliefF=checked
        if checked:
            self.table.showColumn(3)
        else:
            self.table.hideColumn(3)
    
    def displayInfoGain(self,checked):
        self.DisplayInfoGain=checked
        if checked:
            self.table.showColumn(4)
        else:
            self.table.hideColumn(4)
    
    def displayGainRatio(self,checked):
        self.DisplayGainRatio=checked
        if checked:
            self.table.showColumn(5)
        else:
            self.table.hideColumn(5)
    
    def displayGini(self,checked):
        self.DisplayGini=checked
        if checked:
            self.table.showColumn(6)
        else:
            self.table.hideColumn(6)
            
    def setPrecision(self,num):
        self.Precision=num
        self.recalculate()            
            
    def setK(self,k):
        self.ReliefK=k
        self.recalculate()
    
    def setN(self,n):
        self.ReliefN=n
        self.recalculate()
        
    def setDM(self,dm):
        self.DiscretizationMethod=self.options.discretizationStrings[dm]
        self.recalculate()

    def activateLoadedSettings(self):
        self.options.precisionSlider.setValue(self.Precision)
        self.options.kSlider.setValue(self.ReliefK)
        self.options.nSlider.setValue(self.ReliefN)
        self.options.discretization.setCurrentItem(self.options.discretizationStrings.index(self.DiscretizationMethod))
        self.options.displayReliefF.setChecked(self.DisplayReliefF)
        self.options.displayInfoGain.setChecked(self.DisplayInfoGain)
        self.options.displayGainRatio.setChecked(self.DisplayGainRatio)
        self.options.displayGini.setChecked(self.DisplayGini)
        self.recalculate()

    def sort(self,col):
        if not col==self.sortby:
            if col<2: 
                #table.sortColumn sort in alphabetical order, so it can only by used for the attribute and C/D columns
                self.table.sortColumn(col,TRUE,TRUE)
            else:            
            #we must do our own sorting
            #should take advantage of QTable's swapRows() method
            #so how about a simple bubblesort?
                for i in range(0,self.table.numRows()-1):
                    x=float(str(self.table.text(i,col)))
                    for j in range(i+1,self.table.numRows()):
                        y=float(str(self.table.text(j,col)))
                        if x>y:
                            self.swapRows(i,j)
                            x=y
        else:   #reversing selection is faster
            for i in range(0,self.table.numRows()/2):
                self.swapRows(i,self.table.numRows()-i-1)
        self.sortby=col
        self.repaint()
        
    def swapRows(self,i,j):
        for k in range (0, self.table.numCols()):
#            print "swapping ", self.table.text(i,k), "with ", self.table.text(j,k)
            temp=self.table.text(i,k)
            self.table.setText(i,k,self.table.text(j,k))
            self.table.setText(j,k,temp)
#            self.table.swapCells(i,k,j,k)  #why in heavens name doesn't this work for any other than the current column? 
            #Maybe sorting must be enabled? But how is one supposed to do his own sort?
 
    def data(self,data):
        if data==None:
            self.data=None
            self.table.setNumRows(0)
        else:
            self.data=data
        self.recalculate()
    
    def recalculate(self):
        if self.data==None:
            
            return
        self.table.setNumRows(len(self.data.domain.attributes))
        
        ###############################################################
        # construct a data set with only continuous attributes that
        # are discretized
        
        # there will be three types of discretization that user
        # can choose
        
        disMet=self.DiscretizationMethod
        # equal-frequency intervals
        if disMet==self.options.discretizationStrings[0]:
            discretizer = orange.EquiNDiscretization(numberOfIntervals=5)
        #entropy-based discretization
        elif disMet==self.options.discretizationStrings[1]:
            discretizer = orange.EntropyDiscretization()
        # equal-width intervals
        elif disMet==self.options.discretizationStrings[2]:
            discretizer = orange.EquiDistDiscretization(numberOfIntervals=5)      
        
        at = []
        for i in self.data.domain.attributes:
            if i.varType == 2: # a continuous variable?
                d=discretizer(i, self.data)
                at.append(d)
        at.append(self.data.domain.classVar)
        
        # construct a new, discretized, data set
        disc_data = self.data.select(orange.Domain(at))
        
        ###############################################################
        # estimate the relevancy of parameters
        # print them out together with attribute names,
        # attribute type and cardinality
        
        # notice that Relief is the only measure that can estimate
        # relevancy for both discrete and continuous variables
        # for other measure, we have to use the discretized data set
        # to estimate the relevancy of continuous attributes
        
        # for Relief
        # number of reference examples (m)
        # number of closest neighbors that are observed (k)
        
        relieff = orange.MeasureAttribute_relief(k=self.ReliefK, m=self.ReliefN)
        entr = orange.MeasureAttribute_info()
        gainr = orange.MeasureAttribute_gainRatio()
        gini = orange.MeasureAttribute_gini()
        
        prec="%%.%df" % self.Precision
        
        estimators = (relieff, entr, gainr, gini)
        handle_cont = (1,0,0,0) # does estimator handle cont values?
        
        fmt='%-20s '+'%20s '*len(self.est_names)
        fm=' '*13+'%7.4f'
#        print fmt % tuple(['Attribute']+self.est_names)
        cont = 0
        j=-1
        for i in self.data.domain.attributes:
            j+=1
#            cell=QTableItem(self.table,QTableItem.Never,i.name)
#            cell.setText(i.name)
#            self.table.setItem(j,0,cell)
            self.table.setText(j,0,i.name)
            self.table.setText(j,1,('D','C')[i.varType==2])
#            print '%-15s %1s' % (i.name, ('D','C')[i.varType==2]),
            if i.varType==1:
#                print '%2d' % (len(i.values)),
                self.table.setText(j,2,prec % (len(i.values)))
                k=-1
                for e in estimators:
                    k+=1
                    self.table.setText(j,3+k,prec % (e(i,self.data)))
#                    print fm % e(i,data)[0],
            else:
                v = disc_data.domain.attributes[cont]
#                print '%2d' % (len(v.values)),
                self.table.setText(j,2,prec % (len(v.values)))
                k=-1
                for ii in range(len(estimators)):
                    k+=1
                    e = estimators[ii]
                    if handle_cont[ii]:
                        self.table.setText(j,3+k,prec % (e(i,self.data)))
#                        print fm % e(i,data)[0],
                    else:
                        self.table.setText(j,3+k,prec % (e(v,disc_data)))
#                        print fm % e(v,disc_data)[0],
                cont = cont + 1
#            print
        for k in range(2,2+len(estimators)):
            self.table.adjustColumn(k)


from OWOptions import *
from OWTools import *

class OWRankOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self,"Rank Options","OrangeWidgetsIcon.png",parent,name)  

        #precision
        precisionBox=QHGroupBox("Precision",self.top)
        QToolTip.add(precisionBox,"Number of decimal places in output")
        self.precisionSlider=QSlider(0,9,1,3,QSlider.Horizontal,precisionBox)
        self.precisionSlider.setTickmarks(QSlider.Below)
        self.precisionLCD=QLCDNumber(1,precisionBox)
        self.precisionLCD.display(3)
        self.connect(self.precisionSlider,SIGNAL("valueChanged(int)"),self.precisionLCD,SLOT("display(int)"))

        #relief options
        reliefBox=QVGroupBox("Relief options",self.top)
        reliefKBox=QHGroupBox("k",reliefBox)
        self.kSlider=QSlider(1,50,5,11,QSlider.Horizontal,reliefKBox)
        self.kSlider.setTickmarks(QSlider.Below)
        self.kLCD=QLCDNumber(2,reliefKBox)
        self.kLCD.display(11)
        self.connect(self.kSlider,SIGNAL("valueChanged(int)"),self.kLCD,SLOT("display(int)"))
        QToolTip.add(reliefKBox,"k")
        reliefNBox=QHGroupBox("n",reliefBox)
        QToolTip.add(reliefNBox,"n")
        self.nSlider=QSlider(1,50,5,20,QSlider.Horizontal,reliefNBox)
        self.nSlider.setTickmarks(QSlider.Below)
        self.nLCD=QLCDNumber(2,reliefNBox)
        self.nLCD.display(20)
        self.connect(self.nSlider,SIGNAL("valueChanged(int)"),self.nLCD,SLOT("display(int)"))

        #discretization options
        discretizationBox=QHGroupBox("Discretization Method",self.top)
        QToolTip.add(discretizationBox,"Method of discretization")
        self.discretizationStrings=["equal-frequency intervals","entropy-based discretization","equal-width intervals"]
        self.discretization=QComboBox(discretizationBox)
        self.discretization.insertItem(self.discretizationStrings[0])
        self.discretization.insertItem(self.discretizationStrings[1])
        self.discretization.insertItem(self.discretizationStrings[2])

        #display selection
        displayBox=QVGroupBox("Measures used",self.top)
        QToolTip.add(displayBox,"Select which measure to display")
        self.displayReliefF=QCheckBox("ReliefF",displayBox)
        self.displayReliefF.setChecked(TRUE)
        self.displayInfoGain=QCheckBox("InfoGain",displayBox)
        self.displayInfoGain.setChecked(TRUE)
        self.displayGainRatio=QCheckBox("GainRation",displayBox)
        self.displayGainRatio.setChecked(TRUE)
        self.displayGini=QCheckBox("Gini",displayBox)
        self.displayGini.setChecked(TRUE)
        

#test widget appearance        
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRank()
    a.setMainWidget(ow)
#here you can test setting some stuff
    ow.show()
    a.exec_loop()
    
    #save settings 
    ow.saveSettings()

