"""
<name>CSV File import</name>
<description>Import .csv file</description>

"""
import os
import csv
from StringIO import StringIO

from PyQt4 import QtCore, QtGui

from OWWidget import *
import OWGUI
import Orange

#from OWFile import FileNameContextHandler as PathContextHandler
from OWDataTable import ExampleTableModel

DEFAULT_OPTIONS = {"delimiter":",", 
                   "quote":"'", 
                   "has_header": True,
                   "has_orange_definitions": False,
                   }

Slot = pyqtSlot

class standard_icons(object):
    def __init__(self, qwidget=None, style=None):
        self.qwidget = qwidget
        if qwidget is None:
            self.style = QApplication.instance().style()
        else:
            self.style = qwidget.style()
    
    @property
    def dir_open_icon(self):
        return self.style.standardIcon(QStyle.SP_DirOpenIcon)
    
    @property
    def reload_icon(self):
        return self.style.standardIcon(QStyle.SP_BrowserReload)
    
    
            
            
class OWCSVFileImport(OWWidget):
#    contextHandlers = {"": PathContextHandler("")}
    settingsList = ["recent_files"]
    
    DELIMITERS = [("Tab", "\t"),
                  ("Comma", ","),
                  ("Semicolon", ";"),
                  ("Space", " "),
                  ("Others", None),
                  ]
    def __init__(self, parent=None, signalManager=None, 
                 title="CSV File Import"):
        
        OWWidget.__init__(self, parent, signalManager, title, 
                          wantMainArea=False)
        
        self.inputs = []
        self.outputs = [("Data", Orange.data.Table)]
        
        # Settings
        self.delimiter = ","
        self.quote = '"'
        
        self.has_header = True
        self.has_orange_header = True
        
        self.recent_files = []
        self.hints = {}
        
        self.loadSettings()
        
        self.recent_file = filter(os.path.exists, self.recent_files)
        
        self.csv_options = dict(DEFAULT_OPTIONS)
        
        layout = QHBoxLayout()
        box = OWGUI.widgetBox(self.controlArea, "File", orientation=layout)
        self.style().standardIcon(QStyle.SP_DirOpenIcon)
        icons = standard_icons(self)
        
        self.recent_combo = QComboBox(self, objectName="recent_combo",
                                      toolTip="Recent files.",
                                      activated=self.on_select_recent)
        self.recent_combo.addItems([os.path.basename(p) for p in self.recent_files])
        
        self.browse_button = QPushButton("...", icon=icons.dir_open_icon,
                                         toolTip="Browse filesystem",
                                         clicked=self.on_open_dialog)
        
#        self.reload_button = QPushButton("Reload", icon=icons.reload_icon,
#                                         clicked=self.reload_file)
        
        layout.addWidget(self.recent_combo, 2)
        layout.addWidget(self.browse_button)
#        layout.addWidget(self.reload_button)

        form_left = QFormLayout()
        form_right = QFormLayout()
        h_layout = QHBoxLayout()
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(4)
        grid_layout.setHorizontalSpacing(4)
        box = OWGUI.widgetBox(self.controlArea, "Cell Separator",
                              orientation=grid_layout)
        
        button_group = QButtonGroup(box)
        QObject.connect(button_group, 
                        SIGNAL("buttonPressed(int)"),
                        self.delimiter_changed)
#        button_group.buttonPressed.connect(self.delimiter_changed)
        
        for i, (name, char) in  enumerate(self.DELIMITERS[:-1]):
            button = QRadioButton(name, box,
                                  toolTip="Use %r as cell separator" % char)
            
            button_group.addButton(button, i)
            grid_layout.addWidget(button, i / 3, i % 3)
            
        button = QRadioButton("Other", box,
                              toolTip="Use other character")
        
        button_group.addButton(button, i + 1)
        grid_layout.addWidget(button, i / 3 + 1, 0)
        self.delimiter_button_group = button_group
            
        self.delimiter_edit = QLineEdit(objectName="delimiter_edit",
                                        text=self.delimiter,
                                        editingFinished=self.delimiter_changed,
                                        toolTip="Cell delimiter character.")
        grid_layout.addWidget(self.delimiter_edit, i / 3 + 1, 1, -1, -1)
        
        self.quote_edit = QLineEdit(objectName="quote_edit",
                                    text=self.quote,
                                    editingFinished=self.quote_changed,
                                    toolTip="Text quote character.")
        
        form = QFormLayout()
        box = OWGUI.widgetBox(self.controlArea, "Other Options",
                              orientation=form)
        
#        form_left.addRow("Delimiter", self.delimiter_edit)
        form.addRow("Quote", self.quote_edit)
        
        self.has_header_check = \
                QCheckBox(objectName="has_header_check",
                          checked=self.has_header,
                          text="Header line",
                          toolTip="Use the first line as a header",
                          toggled=self.has_header_changed
                          )
        
        form.addRow(self.has_header_check)
        
        self.has_orange_header_check = \
                QCheckBox(objectName="has_orange_header_check",
                          checked=self.has_orange_header,
                          text="Has orange variable type definitions",
                          toolTip="Use second and third line as a orange style '.tab' format feature definitions.",
                          toggled=self.has_orange_header_changed
                          )
                
        form.addRow(self.has_orange_header_check)
        
        box = OWGUI.widgetBox(self.controlArea, "Preview")
        self.preview_view = QTableView()
        box.layout().addWidget(self.preview_view)
        
        OWGUI.button(self.controlArea, self, "Send", callback=self.send_data)
        
        self.selected_file = None
        self.data = None
        
        self.resize(400, 350)
        
    def on_select_recent(self, recent):
        if isinstance(recent, int):
            recent = self.recent_files[recent]
        
        self.set_selected_file(recent)
    
    def on_open_dialog(self): 
        last = os.path.expanduser("~/Documents")
        path = QFileDialog.getOpenFileName(self, "Open File", last)
        path = unicode(path)
        if path:
            basedir, name = os.path.split(path)
            self.recent_combo.insertItem(0, name)
            self.recent_files.insert(0, path)
            self.set_selected_file(path)
    
    @Slot(int)
    def delimiter_changed(self, index):
        self.delimiter = self.DELIMITERS[index][1]
        if self.delimiter is None:
            self.other_delimiter = str(self.delimiter_edit.text())
        self.update_preview()
    
    def quote_changed(self):
        if self.quote_edit.text():
            self.quote = str(self.quote_edit.text())
            self.update_preview()
            
    def has_header_changed(self):
        self.has_header = self.has_header_check.isChecked()
        self.update_preview()
        
    def has_orange_header_changed(self):
        self.has_orange_header = self.has_orange_header_check.isChecked()
        self.update_preview()
            
    def set_selected_file(self, filename):
        if filename in self.hints:
            hints = self.hints[filename]
        else:
            hints = sniff_csv(filename)
            self.hints[filename] = hints
        
        delimiter = hints["delimiter"]
        preset = [d[1] for d in self.DELIMITERS[:-1]]
        if delimiter not in preset:
            self.delimiter = None
            self.other_delimiter = delimiter
            self.delimiter_edit.setText(self.other_delimiter)
            self.delimiter_edit.setEnabled(True)
        else:
            index = preset.index(delimiter)
            button = self.delimiter_button_group.button(index)
            button.setChecked(True)
            self.delimiter_edit.setEnabled(False)
        
        self.quote = hints["quotechar"]
        self.quote_edit.setText(self.quote)
        
        self.has_header = hints["has_header"]
        self.has_header_check.setChecked(self.has_header)
        
        self.has_orange_header = hints["has_orange_header"]
        self.has_orange_header_check.setChecked(self.has_orange_header)
        
        self.selected_file = filename
        self.selected_file_head = []
        with open(self.selected_file, "rb") as f:
            for i, line in zip(range(30), f):
                self.selected_file_head.append(line)
        
        self.update_preview()
        
    def update_preview(self):
        if self.selected_file:
            head = StringIO("".join(self.selected_file_head))
            hints = self.hints[self.selected_file]
            
            hints["quote"] = self.quote
            hints["delimiter"] = self.delimiter or self.other_delimiter
            hints["has_header"] = self.has_header
            hints["has_orange_header"] = self.has_orange_header 
            
            try:
                data = Orange.data.io.load_csv(head, delimiter=self.delimiter,
                                               quotechar=self.quote,
                                               has_header=self.has_header,
                                               has_types=self.has_orange_header,
                                               has_annotations=self.has_orange_header,
                                               )
            except Exception, ex:
                self.error(0, "Cannot parse (%r)" % ex)
                self.data = None
                return
            model = ExampleTableModel(data, None, self)
            self.preview_view.setModel(model)
            self.data = data
            
    def send_data(self):
        self.send("Data", self.data)
        
        
def sniff_csv(file):
    snifer = csv.Sniffer()
    if isinstance(file, basestring):
        file = open(file, "rb")
    
    sample = file.read(2 ** 20) # max 1MB sample
    dialect = snifer.sniff(sample)
    has_header = snifer.has_header(sample)
    
    return {"delimiter": dialect.delimiter,
            "doublequote": dialect.doublequote,
            "escapechar": dialect.escapechar,
            "quotechar": dialect.quotechar,
            "quoting": dialect.quoting,
            "skipinitialspace": dialect.skipinitialspace,
            "has_header": has_header,
            "has_orange_header": False}
    
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWCSVFileImport()
    w.show()
    app.exec_()
    w.saveSettings()
        
        
        
        