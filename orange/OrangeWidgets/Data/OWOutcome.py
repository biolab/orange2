"""
<name>Outcome</name>
<description>Enables selecting the target outcome from all the possible outcomes in the data.
It also provides some basic data statistics.</description>
<category>Data</category>
<icon>icons/Outcome.png</icon>
<priority>20</priority>
"""
#
# OWOutcome.py
# Outcome Widget
# The outcome selection widget
#

from OWWidget import *
from OData import *

class OWOutcome(OWWidget):
    settingsList = []
    
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        'O&utcome Widget',
"""The Outcome Widget is an Orange Widget
for selecting the target outcome
from all the possible outcomes in the data.
It also provides some basic data statistics.""",
        FALSE
        )

        self.inputs = [("Examples", ExampleTable, self.data, 1)]
        self.outputs = [("Classified Examples", ExampleTableWithClass), ("target", int)]
        
        #GUI       
        self.gridr=QGridLayout(self.mainArea,2,2)
        self.selout=QVGroupBox(self.controlArea)
        self.selout.setTitle('Observed outcome')
#        self.lo=QLabel(self.selout)
#        self.lo.setText('Outcome')
        self.outcome=QComboBox(self.selout)
        self.lt=QVGroupBox(self.controlArea)
        self.lt.setTitle('Target Value')
        self.tar=QComboBox(self.lt)
        self.seltar=QVGroupBox(self.mainArea)
        self.seltar.setTitle('Basic Statistics')
        self.stats=QLabel(self.seltar)
        self.stats.setText("N/A")
        
        self.space2=QWidget(self.mainArea)
        
        self.gridr.addWidget(self.seltar,0,0)       
        self.gridr.addWidget(self.space2,1,0)
        self.gridr.addWidget(self.space2,0,1)
               
        self.gridr.setRowStretch(1,10)
        self.gridr.setColStretch(1,10)
        
        #the data                
        self.data=None
        
        #GUI Connections
        self.connect(self.outcome,SIGNAL('activated ( const QString & )'), self.setOutcome)
        self.connect(self.tar,SIGNAL('activated ( const QString & )'),self.setTargets)

        self.resize(100,100)


    def data(self,data):
        """
        Sets the oData and extracts all needed info
        """
        self.data=data
        if data==None:
            self.outcome.clear()
            self.tar.clear()
            self.stats.setText("")
            self.send("Classified Examples",None)
            return
        
        #set outcomes
        self.outcome.clear()
        for attr in data.domain:
                if attr.varType == orange.VarTypes.Discrete: self.outcome.insertItem(attr.name)
        self.outcome.setCurrentItem(self.outcome.count()-1)
        self.setOutcome(str(self.outcome.currentText()))
    
    def setOutcome(self,variable):
        variable=str(variable) #if variable was QString
        self.tar.clear()
        for val in self.data.domain[variable].values:
                self.tar.insertItem(val)

        self.paintStatistics(variable)
        
        self.send("Classified Examples",self.data)
        self.send("target", 0)   # cross fingers, this part should be written clearly
        
    def setTargets(self, value):
        self.send("target", self.data.domain.classVar.values.index(str(value)))
     
    def paintStatistics(self, out):
        """
        TO DO: repair this code not using OData
        out = str(out) #QString to string
        self.oData.setOutcomeByName(out)
        instances = self.oData.getInstances()
        s = "%d"%self.oData.getOriginalNumInstances()+' instances in the oData set\n\n'
        perc = float(len(instances))/self.oData.getOriginalNumInstances()*100
        nOut = len(instances)
        s += '%d' % nOut+' ('+'%1.1f'%perc+'%) instances with defined class\n'
        
        outcomes = self.oData.getOutcomeValues()
        dis = [0]*len(outcomes)
        for i in instances:
          x = int(i.getclass())
          dis[x] = dis[x] + 1
    
        for i in range(0,len(outcomes)):
          ss = '%1d' % dis[i] + ' (' + '%1.1f' % (float(dis[i])/nOut*100.) + '%) '
          s += '\n'+(ss+'outcomes with label '+outcomes[i])

        self.stats.setText(s)
        """

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owo=OWOutcome()
    a.setMainWidget(owo)
    owo.show()
    a.exec_loop()
