#
# VisualTreeOptions.py
#
# options dialog for VisualTree
#

from OWOptions import *
from OWTools import *

class VisualTreeOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        global classesNames

        OWOptions.__init__(self,"VisualTree Options","OrangeWidgetsIcon.png",parent,name)

	self.tabs = QTabWidget(self.top, 'tabWidget')

	# === GENERAL TAB ===
	GeneralTab = QVGroupBox(self)

    	self.AutorefreshBox = QVButtonGroup("Auto refresh", GeneralTab)
	self.ZoomAutoRefresh = QCheckBox("Zooming auto-refresh", self.AutorefreshBox)
	QToolTip.add(self.ZoomAutoRefresh, "Should moving the zoom slider be applied at once?")
	self.OptionsAutoRefresh = QCheckBox("Options auto-refresh", self.AutorefreshBox)
	QToolTip.add(self.OptionsAutoRefresh, "Should any changes made in options (this window) be applied at once?")
	
	self.tabs.insertTab(GeneralTab, "General")
	# === END GENERAL TAB === 

	# === TREE TAB ===
	TreeTab = QVGroupBox(self)

	# Tree depth
	TreeDepthGroup = QHGroupBox("Tree depth", TreeTab)
        self.TreeDepthBox = QLineEdit("", TreeDepthGroup)
	QToolTip.add(self.TreeDepthBox, "Define how many levels will be displayed")

        # Drawing algorithm
    	self.AlgorithmBox = QVButtonGroup("Drawing algorithm", TreeTab)
    	
    	self.AlgorithmBox.setRadioButtonExclusive(True)
    	self.StandardAlgorithm = QRadioButton(self.AlgorithmBox)
    	self.StandardAlgorithm.setText('Standard')
    	self.AdvancedAlgorithm = QRadioButton(self.AlgorithmBox)
    	self.AdvancedAlgorithm.setText('Advanced')
    	self.Advanced2Algorithm = QRadioButton(self.AlgorithmBox)
    	self.Advanced2Algorithm.setText('Advanced2')
    	self.BottomUpAlgorithm = QRadioButton(self.AlgorithmBox)
    	self.BottomUpAlgorithm.setText('Enhanced BottomUp')
    	self.BottomUpMode = QCheckBox(self.AlgorithmBox)
    	self.BottomUpMode.setText('Allow Nodes Collision')
    	QToolTip.add(self.BottomUpMode, "Only applies to Enhanced bottom-up algorithm")
	QToolTip.add(self.AlgorithmBox, "Which drawing algorithm should be used")

        # Maximum line width
        LineWidthBox = QHGroupBox("Max line width", TreeTab)
        QToolTip.add(LineWidthBox,"Define maximum line width which is then reduced")
        self.LineWidthSelectBox = QLineEdit("", LineWidthBox)        

        # Line width selection
    	self.LinesBox = QVButtonGroup("Line width relative to:", TreeTab)
    	self.LinesBox.setRadioButtonExclusive(True)
    	self.LineOnRoot = QRadioButton(self.LinesBox)
    	self.LineOnRoot.setText('root')
    	self.LineOnNode = QRadioButton(self.LinesBox)
    	self.LineOnNode.setText('node')        
	self.LineEqual = QRadioButton(self.LinesBox)
    	self.LineEqual.setText('all lines equal')        
        QToolTip.add(self.LineOnNode, "Line width is relative to each node")
        QToolTip.add(self.LineOnRoot, "Line width is relative to root node")
	QToolTip.add(self.LineEqual, "Line width is equal on all nodes")

	self.tabs.insertTab(TreeTab, "Tree")
	# === END NODE TAB ===

	# === NODE TAB ===
	NodeTab = QVGroupBox(self)
	
	# Node size options
	NodeSizeBox=QVButtonGroup("Node size", NodeTab)
	NodeSizeBox.setRadioButtonExclusive(True)
	self.NodeSizeBig = QRadioButton(NodeSizeBox)
	self.NodeSizeBig.setText('Big')
    	self.NodeSizeMedium = QRadioButton(NodeSizeBox)
    	self.NodeSizeMedium.setText('Medium')
    	self.NodeSizeSmall = QRadioButton(NodeSizeBox)
    	self.NodeSizeSmall.setText('Small')	
		
	#izbira dveh "podatkov" za izris v body-ju
	SelectTextBox = QVButtonGroup("Select text in body",NodeTab)
	self.MajorityClass = QCheckBox ("Majority class",SelectTextBox)	
	self.TargetClass = QCheckBox ("Target class",SelectTextBox)
	self.TargetClassProbbability = QCheckBox ("Probability of target class",SelectTextBox)
	self.NumOfInstances = QCheckBox ("Number of instances",SelectTextBox)
	
        # Node body color options
        NodeBodyColorBox=QVButtonGroup("Node body color", NodeTab)
    	NodeBodyColorBox.setRadioButtonExclusive(True)
    	self.NodeBodyColorDefault = QRadioButton(NodeBodyColorBox)
    	self.NodeBodyColorDefault.setText('Default')
    	self.NodeBodyColorCases = QRadioButton(NodeBodyColorBox)
    	self.NodeBodyColorCases.setText('Cases')
    	self.NodeBodyColorMajorityClass = QRadioButton(NodeBodyColorBox)
    	self.NodeBodyColorMajorityClass.setText('Majority class')	
	self.NodeBodyColorTargetClass = QRadioButton(NodeBodyColorBox)
	self.NodeBodyColorTargetClass.setText('Target class - in node')	
	self.NodeBodyColorTargetClassRelative = QRadioButton(NodeBodyColorBox)
	self.NodeBodyColorTargetClassRelative.setText('Target class - global')	


	#prikazi/skrij pite
	PieShowBox = QVButtonGroup("Pies",NodeTab)
	self.PieShowCheckBox = QCheckBox ("Show pies",PieShowBox)


        QToolTip.add(self.NodeBodyColorDefault, "Set the color of the node body to default")
        QToolTip.add(self.NodeBodyColorCases, "Set the color of the body relative to the number of instances")
        QToolTip.add(self.NodeBodyColorMajorityClass, "Set the color of the body relative to the majority class")
	QToolTip.add(self.NodeBodyColorTargetClass, "Set the color of the body relative to the target class probability in node")
	QToolTip.add(self.NodeBodyColorTargetClassRelative, "Set the color of the body relative to the target class probability relative to all the target class instances")
	QToolTip.add(self.PieShowCheckBox, "Show probability pies in node")


	self.tabs.insertTab(NodeTab, "Node")	
	
	# === END NODE TAB ===
	


if __name__=="__main__":
    a=QApplication(sys.argv)
    w=VisualTreeOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()


