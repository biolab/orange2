# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	dialogs 

from qt import *
from qtcanvas import *
from copy import *
from string import strip
import os
import sys
TRUE  = 1
FALSE = 0


# #######################################
# # Signal dialog - let the user select active signals between two widgets
# #######################################
class SignalDialog(QDialog):
    def __init__(self, *args):
        apply(QDialog.__init__,(self,) + args)
        self.topLayout = QVBoxLayout( self, 10 )
        self.signals = []
        self.symbSignals = []
    
    # add checkboxes for all signals that are in inSignals and outSignals
    def addSignals(self, inSignals, outSignals, canvasDlg):
        for signal in inSignals:
            if signal in outSignals:
                self.symbSignals.append(signal)
                self.signals.append(canvasDlg.getChannelName(signal))

        self.buttons = range(len(self.signals))

        for i in range(len(self.signals)):
            self.buttons[i] = QCheckBox(self)
            self.buttons[i].setText(self.signals[i])
            self.topLayout.addWidget(self.buttons[i])
            self.buttons[i].setChecked(TRUE)
            self.buttons[i].show()

        okButton = QPushButton("OK",self)
        #cancelButton = QPushButton("Cancel",self)
        self.topLayout.addWidget(okButton)
        #self.topLayout.addWidget(cancelButton)
        okButton.show()
        #cancelButton.show()
        self.connect(okButton, SIGNAL("clicked()"),self.okclicked)
        #self.connect(cancelButton, SIGNAL("clicked()"),self.reject)
        self.topLayout.activate()        

    def okclicked(self):
        selected = FALSE
        for i in range(len(self.signals)):
            if self.buttons[i].isChecked():
                selected = TRUE

        if not selected:
            res = QMessageBox.information(self,'Qrange Canvas','The link will be removed since no signal is checked. Continue?','Yes','No', QString.null,0,1)
            if res == 0:
                self.reject()
                return
            else:
                return
        self.accept()
        
# #######################################
# # Preferences dialog - preferences for signals
# #######################################
class PreferencesDlg(QDialog):
    def __init__(self, canvasDlg, *args):
        apply(QDialog.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        self.topLayout = QVBoxLayout( self, 10 )
        self.grid = QGridLayout( 5, 3 )
        self.topLayout.addLayout( self.grid, 10 )
        
        groupBox  = QGroupBox(self, "Channel_settings")
        groupBox.setTitle("Channel settings")
        self.grid.addWidget(groupBox, 1,1)
        topLayout2 = QVBoxLayout(groupBox, 10 )
        propGrid = QGridLayout(groupBox, 4, 2 )
        topLayout2.addLayout(propGrid, 10)

        cap0 = QLabel("Symbolic channel names:", self)
        cap1 = QLabel("Full name:", groupBox)
        cap2 = QLabel("Priority:", groupBox)
        cap3 = QLabel("Color:", groupBox)
        self.editFullName = QLineEdit(groupBox)
        self.editPriority = QComboBox( FALSE, groupBox, "priority" ) 
        self.editColor    = QComboBox( FALSE, groupBox, "color" )
        #self.connect( self.editPriority, SIGNAL("activated(int)"), self.comboValueChanged )
        #self.connect( self.editColor, SIGNAL("activated(int)"), self.comboValueChanged ) 

        propGrid.addWidget(cap1, 0,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(cap2, 1,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(cap3, 2,0, Qt.AlignVCenter+Qt.AlignHCenter)
        propGrid.addWidget(self.editFullName, 0,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editPriority, 1,1, Qt.AlignVCenter)
        propGrid.addWidget(self.editColor, 2,1, Qt.AlignVCenter)

        groupBox.setMinimumSize(180,150)
        groupBox.setMaximumSize(180,150)
        
        saveButton = QPushButton("Save changes", groupBox)
        addButton = QPushButton("Add new channel name", self)
        removeButton = QPushButton("Remove selected name", self)
        closeButton = QPushButton("Close",self)
        self.channelList = QListBox( self, "channels" )
        self.channelList.setMinimumHeight(200)
        self.connect( self.channelList, SIGNAL("highlighted(int)"), self.listItemChanged ) 

        self.grid.addWidget(cap0,0,0, Qt.AlignLeft+Qt.AlignBottom)
        self.grid.addWidget(addButton, 2,1)
        self.grid.addWidget(removeButton, 3,1)
        self.grid.addMultiCellWidget(self.channelList, 1,5,0,0)
        self.grid.addWidget(closeButton, 4,1)
        propGrid.addMultiCellWidget(saveButton, 3,3,0,1)

        saveButton.show()
        addButton.show()
        removeButton.show()
        self.channelList.show()
        closeButton.show()
        self.connect(saveButton, SIGNAL("clicked()"),self.saveChanges)
        self.connect(addButton , SIGNAL("clicked()"),self.addNewSignal)
        self.connect(removeButton, SIGNAL("clicked()"),self.removeSignal)
        self.connect(closeButton, SIGNAL("clicked()"),self.closeClicked)
        self.topLayout.activate()

        self.editColor.insertItem( "black" )
        self.editColor.insertItem( "darkGray" )
        self.editColor.insertItem( "gray" )
        self.editColor.insertItem( "lightGray" )
        self.editColor.insertItem( "red" )
        self.editColor.insertItem( "green" )
        self.editColor.insertItem( "blue" )
        self.editColor.insertItem( "cyan" )
        self.editColor.insertItem( "magenta" )
        self.editColor.insertItem( "yellow" )
        self.editColor.insertItem( "darkRed" )
        self.editColor.insertItem( "darkGreen" )
        self.editColor.insertItem( "darkBlue" )
        self.editColor.insertItem( "darkCyan" )
        self.editColor.insertItem( "darkMagenta" )
        self.editColor.insertItem( "darkYellow" )

        for i in range(20):
            self.editPriority.insertItem(str(i+1))

        self.channels = {}
        if self.canvasDlg.settings.has_key("Channels"):
            self.channels = self.canvasDlg.settings["Channels"]

        self.reloadList()

    def listItemChanged(self, index):
        name = str(self.channelList.text(index))
        value = self.channels[name]
        items = value.split("::")
        self.editFullName.setText(items[0])

        for i in range(self.editPriority.count()):
            if (str(self.editPriority.text(i)) == items[1]):
                self.editPriority.setCurrentItem(i)

        for i in range(self.editColor.count()):
            if (str(self.editColor.text(i)) == items[2]):
                self.editColor.setCurrentItem(i)

    def reloadList(self):
        self.channelList.clear()
        for (key,value) in self.channels.items():
            self.channelList.insertItem(key)

    def saveChanges(self):
        index = self.channelList.currentItem()
        if index != -1:
            name = str(self.channelList.text(index))
            self.channels[name] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())

    def addNewSignal(self):
        (Qstring,ok) = QInputDialog.getText("Add New Channel Name", "Enter new symbolic channel name")
        string = str(Qstring)
        if ok:
            self.editColor.setCurrentItem(0)
            self.editPriority.setCurrentItem(0)
            self.editFullName.setText(string)
            self.channels[string] = str(self.editFullName.text()) + "::" + str(self.editPriority.currentText()) + "::" + str(self.editColor.currentText())
            self.reloadList()
            self.selectItem(string)

    def selectItem(self, string):
        for i in range(self.channelList.count()):
            temp = str(self.channelList.text(i))
            if temp == string:
                self.channelList.setCurrentItem(i)
                return
            
    def removeSignal(self):
        index = self.channelList.currentItem()
        if index != -1:
            tempDict = {}
            symbName = str(self.channelList.text(index))
            
            for key in self.channels.keys():
                if key != symbName:
                    tempDict[key] = self.channels[key]
            self.channels = copy(tempDict)        
            
        self.reloadList()

    def closeClicked(self):
        self.canvasDlg.settings["Channels"] = self.channels
        self.accept()
        return

if __name__=="__main__":
    app = QApplication(sys.argv) 
    dlg = PreferencesDlg(app)
    app.setMainWidget(dlg)
    dlg.show()
    #dlg.addSignals(["data", "cdata", "ddata"], ["test", "ddata", "cdata"])
    app.exec_loop() 

