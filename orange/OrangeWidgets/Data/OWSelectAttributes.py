# when change in attributes, do check if groups are still valid, place
# a block on update attribute change, but do change the selection of groups
# remove the block.

"""
<name>Select Attributes</name>
<description>Manual attribute selection and attribute grouping</description>
<icon>icons/SelectAttributes.png</icon>
<priority>1090</priority>
"""
# SelectAttributes.py
#
# Used for manual attribute selection, attribute grouping
# 

from OWWidget import *
from OWGUI import *

#############################################################################
#
class OWSelectAttributes(OWWidget):
    settingsList = ['groupID', 'groups', 'attsSel', 'groupsSel', 'domainID']
    #settingsList = []
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Select Attributes")

        # seet channells
        self.inputs = [("Examples", ExampleTable, self.cdata)]
        self.outputs = [("Examples", ExampleTable)]

        # set default settings
        self.data = None   # data set
        self.attsSel = []  # indices of selected attributes
        self.groupID = 0   # number of groups defined
        self.groups = {}   # group info
        self.domainID = [] # id of the data domain (list with att names)

        self.grpSelChangedBlocked = 0
        self.refreshOutputBlocked = 0

        # load settings
        self.loadSettings()

        # GUI
        self.grid.addMultiCellWidget(self.space,0,1,0,1)
        self.hbox = QHBox(self.space)
        self.hbox.setSpacing(10)

        # attributes        
        self.attsGroup = QVGroupBox(self.hbox)

        self.attsGroup.setTitle("Attributes")
        self.attsLB = QListBox(self.attsGroup)
        self.attsLB.setSelectionMode(QListBox.Multi)
        self.connect(self.attsLB, SIGNAL('selectionChanged()'), self.attSelChanged)
         
        self.invButton = button(self.attsGroup, self, "Invert Selection", self.attInvSelection)

        # attribute groups
        self.groupsGroup = QVGroupBox(self.hbox)
        self.groupsGroup.setTitle("Groups")
        self.groupsLB = QListBox(self.groupsGroup)
        self.groupsLB.setSelectionMode(QListBox.Multi)
        self.connect(self.groupsLB, SIGNAL('selectionChanged()'), self.grpSelChanged)

        self.grpAddButton = button(self.groupsGroup, self, "Add", self.grpAdd, disabled=1)
        self.grpRenameButton = button(self.groupsGroup, self, "Rename", self.grpRename, disabled=1)
        self.grpUpdateButton = button(self.groupsGroup, self, "Update", self.grpUpdate, disabled=1)
        self.grpRemoveButton = button(self.groupsGroup, self, "Remove", self.grpRemove, disabled=1)

        self.resize(320,480)

    #################################################################################################
    # signal handling (output)

    def refreshOutput(self):
        if self.refreshOutputBlocked: return
        # delivers output signals
        domain = self.data.domain
        atts = [self.data.domain[x] for x in self.attsSel]
        if self.data.domain.classVar:
            atts.append(self.data.domain.classVar)
            #atts = atts + [self.data.domain.classVar]
        newdomain = orange.Domain(atts)
        newdata = self.data.select(newdomain)
        self.send("Examples", newdata)
        # replace xxx with newdata

    #################################################################################################
    # handling of attributes        

    def setAttsLB(self, data):
        self.attsLB.clear()

        if self.data == None: return

        for a in self.data.domain.attributes:
            self.attsLB.insertItem(a.name)
        for i in self.attsSel:
            self.attsLB.setSelected(i, 1)

    def attSelected(self, i):
        pass

    def attSelChanged(self):
        # if nothing is selected, disable add and update group
        self.attsSel = filter(lambda i: self.attsLB.isSelected(i), range(self.attsLB.count()))
        # print '-> attSelChanged', self.attsSel
        if len(self.attsSel) == 0:
            self.grpAddButton.setDisabled(1)
            self.grpUpdateButton.setDisabled(1)
        else:
            self.grpAddButton.setEnabled(1)
            grps = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))
            if len(grps) == 1:
                self.grpUpdateButton.setEnabled(1)
        # check if this change affects the selection of groups
        # well, this creates a lot of mess and i have removed it
##        self.grpSelChangedBlocked = 1
##        for i in range(self.groupsLB.count()):
##            key = str(self.groupsLB.text(i))
##            atts = self.groups[key][1]
##            sel = reduce(lambda x, y: x * y, [x in self.attsSel for x in atts])
##            self.groupsLB.setSelected(i, sel)
        self.grpSelChangedBlocked = 0
        self.refreshOutput()

    def attInvSelection(self):
        self.attsLB.invertSelection()

    #################################################################################################
    # handling of attribute groups

    def grpSelChanged(self):
        self.refreshOutputBlocked = 1
        if self.grpSelChangedBlocked: return  # selection changed through attribute selection
        #SHOULD CHANGE THIS TO A SINGLE CLICK???
        #OR BETTER: SHOULD JUST RETURN WHEN SETTING THE ATTRIBUTE LIST
        grps = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))

        # toggle the buttons (enable / disable)        
        if len(grps) == 1:
            self.grpRenameButton.setEnabled(1)
            sel = filter(lambda i: self.attsLB.isSelected(i), range(self.attsLB.count()))
            if len(sel)>0:
                self.grpUpdateButton.setEnabled(1)
        else:
            self.grpRenameButton.setDisabled(1)
            self.grpUpdateButton.setDisabled(1)

        if len(grps) > 0:
            self.grpRemoveButton.setEnabled(1)
        else:
            self.grpRemoveButton.setDisabled(1)

        # toggle the attribute selection
        sel = [0]*len(self.data.domain.attributes)
        for i in grps:
            id, atts = self.groups[str(self.groupsLB.text(i))]
            for a in atts:
                sel[a] = 1
        for i in range(len(self.data.domain.attributes)):
            if self.attsLB.isSelected(i) <> sel[i]:
                self.attsLB.setSelected(i, sel[i])

        self.groupsSel = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))
        self.refreshOutputBlocked = 0
        self.refreshOutput()

    def grpAdd(self):
        (text, ok) = QInputDialog.getText('Qt '+'Set Group Name', 'Group Name:', 'Group'+str(self.groupID+1))
        if not ok:
            return
        text = str(text)

        # check if the name is unique
        if self.groups.has_key(text):
            QMessageBox.information(self, 'QT '+'Attribute Group Name Error',
                                    'Attribute group with the same name already\n'+
                                    'exists. Use some other name.');
            return

        # add attribute group
        self.groupID += 1
        sel = filter(lambda i: self.attsLB.isSelected(i), range(self.attsLB.count()))
        self.groups[text] = [self.groupID, sel]
        self.groupsLB.insertItem(text)
        self.groupsLB.setSelected(self.groupsLB.numRows()-1, 1)

    def grpRename(self):
        grp = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))[0]
        key = str(self.groupsLB.text(grp))
        (text, ok) = QInputDialog.getText('Qt '+'Set Group Name', 'Group Name:', key)
        if not ok:
            return
        
        self.groups[str(text)] = self.groups[key]
        del self.groups[key]
        self.groupsLB.changeItem(str(text), grp)
        self.groupsLB.setSelected(grp, 1)

    def grpUpdate(self):
        # assumes only one group was selected
        grp = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))[0]
        sel = filter(lambda i: self.attsLB.isSelected(i), range(self.attsLB.count()))
        #self.groups[str(self.groupsLB.text(grp))] = [12,[1,2]]
        self.groups[str(self.groupsLB.text(grp))][1] = sel

    def grpRemove(self):
        self.grpSelChangedBlocked = 1
        grps = filter(lambda i: self.groupsLB.isSelected(i), range(self.groupsLB.count()))
        grps.reverse()
        for i in grps:
            del self.groups[str(self.groupsLB.text(i))]
            self.groupsLB.removeItem(i)
        print self.groupsLB.count()
        if self.groupsLB.count() == 0:
            self.groupID = 0
        self.grpSelChangedBlocked = 0

    def grpRepaint(self):
        self.groupsLB.clear()
        if self.groups:
            #grps(id, name)
            grps = [(self.groups[x][0], x) for x in self.groups.keys()]
            grps.sort(lambda x, y: cmp(x[0], y[0]))
            for g in grps:
                self.groupsLB.insertItem(g[1])
            for s in self.groupsSel:
                self.groupsLB.setSelected(s, 1)

    #################################################################################################
    # signal handling (input)

    def cdata(self, data):
        self.refreshOutputBlocked = 1
        previous = self.data
        self.data = data
        # here we should compare to past data sets and set
        # the selection and groups accordingly
        # for now, we just check the attribute names, if equal, we keep with settings
        equal = 0
        if len(self.domainID) == len(self.data.domain.attributes):
            equal = 1
            for i in range(len(data.domain.attributes)):
                if data.domain.attributes[i].name <> self.domainID[i]:
                    equal = 0

        if equal and not previous: # equal
            # if data has arrived for the first time, then update attribute and group list boxes
            self.attsLB.clear()
            self.setAttsLB(self.data)
            self.grpRepaint()
        else: # different
            self.domainID = [x.name for x in self.data.domain.attributes]
            self.attsLB.clear()
            self.attsSel = range(len(self.data.domain.attributes)) # all attributes selected
#            attsSel
            self.setAttsLB(self.data)
            self.groupsLB.clear()
            self.groups = {}
            self.groupsSel = []
            self.groupID = 0
        self.refreshOutputBlocked = 0
        self.refreshOutput()

#test widget appearance
if __name__=="__main__":
    data = orange.ExampleTable('adult_sample.tab')

    a=QApplication(sys.argv)
    ow=OWSelectAttributes()
    a.setMainWidget(ow)
    ow.show()
    ow.cdata(data)
    a.exec_loop()

    #save settings 
    ow.saveSettings()
