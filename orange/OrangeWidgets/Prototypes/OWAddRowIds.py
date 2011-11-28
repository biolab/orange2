"""<name>Add Row Ids</name>
<description>Add unique row ids to the data table</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
"""

import sys
import uuid

from OWWidget import *
import OWGUI
import Orange

class OWAddRowIds(OWWidget):    
    def __init__(self, parent=None, signalManager=None, title="Add row ids"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Input Table", Orange.data.Table, self.set_table)]
        self.outputs = [("Output Table", Orange.data.Table)]
        
        self.use_guid = False
        
        OWGUI.checkBox(self.controlArea, self, "use_guid", 
                       label="Use unique global identifiers", 
                       tooltip="Use unique global identifiers. Identifiers will\
be unique across all widget istances and orange sessions.", 
                       callback=self.commit
                       )
        self.table = None
        
    def set_table(self, table=None):
        self.table = table
        self.commit()
        
    def commit(self):
        from Orange.data import utils
        if self.table is not None:
            table = Orange.data.Table(self.table)
            if self.use_guid:
                utils.add_row_id(table, utils.uuid_generator())
            else:
                utils.add_row_id(table, utils.range_generator())
        else:
            table = None
            
        self.send("Output Table", table)
            
            
                
            
        
    