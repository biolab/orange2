"""
<name>Load Object</name>
<description>Load (unpickle) any python object</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
<tags>load,unpickle,pickle</tags>

"""

from OWWidget import *
import OWGUI


class OWUnpickle(OWWidget):
    settingsList = ["filename_history", "selected_file_index", "last_file"]
    
    def __init__(self, parent=None, signalManager=None, name="Load Object"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)
        
        self.outputs = [("Object", object, Dynamic)]
        
        self.filename_history = []
        self.selected_file_index = 0
        self.last_file = os.path.expanduser("~/orange_object.pck")
        
        self.loadSettings()
        
        self.filename_history= filter(os.path.exists, self.filename_history)
        
        #####
        # GUI
        #####
        
        box = OWGUI.widgetBox(self.controlArea, "File", orientation="horizontal", addSpace=True)
        self.files_combo = OWGUI.comboBox(box, self, "selected_file_index", 
                                         items = [os.path.basename(p) for p in self.filename_history],
                                         tooltip="Select a recent file", 
                                         callback=self.on_recent_selection)
        
        self.browseButton = OWGUI.button(box, self, "...", callback=self.browse,
                                         tooltip = "Browse file system")

        self.browseButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.browseButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        
        OWGUI.rubber(self.controlArea)
        
        self.resize(200, 50)
        
        if self.filename_history:
            self.load_and_send()
        
        
    def on_recent_selection(self):
        filename = self.filename_history[self.selected_file_index]
        self.filename_history.pop(self.selected_file_index)
        self.filename_history.insert(0, filename)
        self.files_combo.removeItem(self.selected_file_index)
        self.files_combo.insertItem(0, os.path.basename(filename))
        self.selected_file_index = 0
        
        self.load_and_send()
        
    def browse(self):
        filename = QFileDialog.getOpenFileName(self, "Load Object From File",
                        self.last_file, "Pickle files (*.pickle *.pck)\nAll files (*.*)")
        filename = str(filename)
        if filename:
            if filename in self.filename_history:
                self.selected_file_index = self.filename_history.index(filename)
                self.on_recent_selection()
                return
            self.last_file = filename
            self.filename_history.insert(0, filename)
            self.files_combo.insertItem(0, os.path.basename(filename))
            self.files_combo.setCurrentIndex(0)
            self.selected_file_index = 0
            self.load_and_send()
            
    def load_and_send(self):
        filename = self.filename_history[self.selected_file_index]
        import cPickle as pickle
        self.error([0, 1])
        try:
            obj = pickle.load(open(filename, "rb"))
        except Exception, ex:
            self.error(0, "Could not load object! %s" % str(ex))
            return
        
        if not isinstance(obj, object):
            # Old style instance objects
            self.error(1, "'%s' is not an instance of %r" % (os.path.basename(filename),
                                                             object))
            return 
        
        self.send("Object", obj)