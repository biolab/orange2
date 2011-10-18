"""<name>Add Row Ids</name>
<description>Add unique row ids to the data table</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
"""

import sys
import uuid

from OWWidget import *
import OWGUI
import Orange

from threading import Lock
global_curent_id_lock = Lock()
global_curent_id = 0

class OWAddRowIds(OWWidget):
    meta_id = Orange.data.new_meta_id()
    id_var = Orange.data.variable.String("Row Id")
    
    def __init__(self, parent=None, signalManager=None, title="Add row ids"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Input Table", Orange.data.Table, self.set_table)]
        self.outputs = [("Output Table", Orange.data.Table)]
        
        self.use_guid = False
        
        OWGUI.checkBox(self.controlArea, self, "use_guid", 
                       label="Use unique global identifiers", 
                       tooltip="Use unique global identifiers. Identifiers will\
be unique across all widget istances.", 
                       callback=self.commit
                       )
        self.table = None
        
    def set_table(self, table=None):
        self.table = table
        self.commit()
        
    def commit(self):
        if self.table is not None:
            table = Orange.data.Table(self.table)
            table.domain.add_meta(self.meta_id, self.id_var)
            global global_curent_id
            for ex in table:
                if self.use_guid:
                    ex[self.id_var] = str(uuid.uuid4())
                else:
                    with global_curent_id_lock:
                        ex[self.id_var] = str(global_curent_id)
                        global_curent_id += 1 
        else:
            table = None
            
        self.send("Output Table", table)
            
            
                
            
        
    