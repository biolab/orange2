"""<name>Save Association Rules</name>
"""

from OWWidget import *
import OWGUI

import Orange
from Orange import associate

class OWSaveAssocRules(OWWidget):
    settingsList = ["last_save_file", "filename_history"]
    def __init__(self, parent=None, signalManager=None, name="Save Association Rules"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)
        
        self.inputs = [("Association Rules", associate.AssociationRules, self.set_rules)]
        
        self.last_save_file = os.path.expanduser("~/orange_assoc_rules.pck")
        self.filename_history = []
        self.selected_file_index = 0
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        box = OWGUI.widgetBox(self.controlArea, "File",
                              orientation="horizontal",
                              addSpace=True)
        
        self.files_combo = OWGUI.comboBox(box, self, "selected_file_index",
                                         items=[os.path.basename(f) for f in self.filename_history],
                                         tooltip="Select a recently saved file",
                                         callback=self.on_recent_selection)
        
        self.browse_button = OWGUI.button(box, self, "...",
                                          tooltip="Browse local file system",
                                          callback=self.browse)
        
        self.browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
         
        box = OWGUI.widgetBox(self.controlArea, "Save")
        self.save_button = OWGUI.button(box, self, "Save current rules",
                                       callback=self.save_rules,
                                       autoDefault=True)
        
        self.save_button.setEnabled(False)
        
        OWGUI.rubber(self.controlArea)
        
        self.resize(200, 100)
        
        self.rules = None
        
    def on_recent_selection(self):
        filename = self.filename_history[self.selected_file_index]
        self.filename_history.pop(self.selected_file_index)
        self.filename_history.insert(0, filename)
        self.files_combo.removeItem(self.selected_file_index)
        self.files_combo.insertItem(0, os.path.basename(filename))
        self.selected_file_index = 0
    
    def browse(self):
        filename = QFileDialog.getSaveFileName(self, "Save Rules As ...",
                    self.last_save_file, "Pickle files (*.pickle *.pck);; All files (*.*)")
        filename = str(filename)
        if filename:
            if filename in self.filename_history:
                self.selected_file_index = self.filename_history.index(filename)
                self.on_recent_selection()
                return
            
            self.last_save_file = filename
            self.filename_history.insert(0, filename)
            self.files_combo.insertItem(0, os.path.basename(filename))
            self.files_combo.setCurrentIndex(0)
            self.save_button.setEnabled(self.rules is not None and bool(self.filename_history))
    
    def save_rules(self):
        if self.rules is not None:
            filename = self.filename_history[self.selected_file_index]
            import cPickle
            self.error(0)
            try:
                cPickle.dump(self.rules, open(filename, "wb"))
            except Exception, ex:
                self.error(0, "Could not save rules! %s" % str(ex))
            
            
    def set_rules(self, rules=None):
        self.rules = rules
        self.save_button.setEnabled(rules is not None and bool(self.filename_history))