#
# OWSmartVisualizationOptions.py
#

from OWOptions import *
from OWTools import *

class OWSieveMultigramOptions(OWOptions):
    kvocList = ['1.5','2','3','5','10','20']
    kvocNums = [1.5,   2,  3,  5,  10,  20]
    pearsonMaxList = ['4','6','8','10','12']
    pearsonMaxNums = [ 4,  6,  8,  10,  12]
    
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self, "Sieve multigram options", "OrangeWidgetsIcon.png", parent, name)

        self.lineGroup = QVGroupBox(self.top)
        self.lineGroup.setTitle("Max line width")
        self.lineCombo = QComboBox(self.lineGroup)

        self.independenceGroup = QVGroupBox(self.top)
        self.independenceGroup.setTitle("Attribute Independence")
        self.hbox = QHBox(self.independenceGroup, "kvocient")
        self.kvocLabel = QLabel('Max deviation quotient', self.hbox)
        self.kvocCombo = QComboBox(self.hbox)
        QToolTip.add(self.hbox, "What is maximum expected ratio between p(x,y) and p(x)*p(y). Greater the ratio, brighter the colors.")

        self.pearsonGroup = QVGroupBox(self.top)
        self.pearsonGroup.setTitle("Attribute independence (Pearson residuals)")

        self.hbox2 = QHBox(self.pearsonGroup, "residual")
        self.residualLabel = QLabel('Max residual', self.hbox2)
        self.pearsonMaxResCombo = QComboBox(self.hbox2)
        QToolTip.add(self.hbox2, "What is maximum expected Pearson standardized residual. Greater the maximum, brighter the colors.")

        self.hbox3 = QHBox(self.pearsonGroup, "minimum")
        self.residualLabel2 = QLabel('Min residual   ', self.hbox3)
        self.minResidualEdit = QLineEdit(self.hbox3)
        QToolTip.add(self.hbox3, "What is minimal absolute residual value that will be shown in graph.")

        self.applyButton = QPushButton("Apply changes", self.top)

        self.initSettings()        

    def initSettings(self):
        # quotien combo box values        
        for item in self.kvocList: self.kvocCombo.insertItem(item)

        # line width combo values
        for i in range(1,10): self.lineCombo.insertItem(str(i))

        # max residual combo values
        for item in self.pearsonMaxList: self.pearsonMaxResCombo.insertItem(item)     

        
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=OWSieveMultigramOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()

    
