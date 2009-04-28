from OWWidget import *
import OWGUI, OWGUIEx, string

class OWLineEditFilter(OWWidget):
    settingsList = []
    def __init__(self, parent=None):
		OWWidget.__init__(self, parent, title='Line Edit as Filter')
		
		self.filter = ""
		self.listboxValue = ""
		lineEdit = OWGUIEx.lineEditFilter(self.controlArea, self, "filter", "Filter:", useRE = 1, emptyText = "filter...")
		    
		lineEdit.setListBox(OWGUI.listBox(self.controlArea, self, "listboxValue"))
		names = []
		for i in range(10000):
		    names.append("".join([string.ascii_lowercase[random.randint(0, len(string.ascii_lowercase)-1)] for c in range(10)]))
		lineEdit.listbox.addItems(names)
		lineEdit.setAllListItems(names)

    
##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWLineEditFilter()
    ow.show()
    appl.exec_()
