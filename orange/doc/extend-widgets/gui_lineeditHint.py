from OWWidget import *
import OWGUI, OWGUIEx, string
import orngRegistry, orngEnviron
   
def getFullWidgetIconName(category, widgetInfo):
	import os
	iconName = widgetInfo.icon
	names = []
	name, ext = os.path.splitext(iconName)
	for num in [16, 32, 42, 60]:
	    names.append("%s_%d%s" % (name, num, ext))
	    
	widgetDir = str(category.directory)  
	fullPaths = []
	dirs = orngEnviron.directoryNames
	for paths in [(dirs["picsDir"],), (dirs["widgetDir"],), (dirs["widgetDir"], "icons")]:
	    for name in names + [iconName]:
	        fname = os.path.join(*paths + (name,))
	        if os.path.exists(fname):
	            fullPaths.append(fname)
	    if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
	        fullPaths.pop()     # if we have the new icons we can remove the default icon
	    if fullPaths != []:
	        return fullPaths    
	return ""  


	
class OWLineEditHint(OWWidget):
    settingsList = []
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Line Edit as Filter', wantMainArea = 0)
        
        self.text = ""
        s = OWGUIEx.lineEditHint(self.controlArea, self, "text", useRE = 0, caseSensitive = 0, matchAnywhere = 0)
        s.listWidget.setSpacing(2)
        s.setStyleSheet(""" QLineEdit { background: #fffff0; border: 1px solid blue} """)
        s.listWidget.setStyleSheet(""" QListView { background: #fffff0; } QListView::item {padding: 3px 0px 3px 0px} QListView::item:selected, QListView::item:hover { color: white; background: blue;} """)
        
        cats = orngRegistry.readCategories()
        items = []
        for cat in cats.values():
            for widget in cat.values():
                iconNames = getFullWidgetIconName(cat, widget)
                icon = QIcon()
                for name in iconNames:
                    icon.addPixmap(QPixmap(name))
                item = QListWidgetItem(icon, widget.name)
                #item.setSizeHint(QSize(100, 32))
                #
                items.append(item)
        s.setItems(items)

    

# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWLineEditHint()
    ow.show()
    appl.exec_()
