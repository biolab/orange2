#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

import sys
import ConfigParser,os
from string import *
from OWTools import *
from OWAboutX import *

class OWWidget(QWidget):
    def __init__(
    self,
    parent=None,
    title="O&range Widget",
    description="This a general Orange Widget\n from which all the other Orange Widgets are derived.",
    wantSettings=TRUE,
    wantGraph=FALSE,
    icon="OrangeWidgetsIcon.png",
    logo="OrangeLogo.png",    
    ):
        """
        Initialization
        Parameters: 
            title - The title of the widget, including a "&" (for shortcut in about box)
            description - self explanatory
            wantSettings - display settings button or not
            wantGraph - displays a save graph button or not
            icon - the icon file
            logo - the logo file
            parent - parent of the widget if needed
        """
        self.title=title.replace("&","")
        QWidget.__init__(self,parent)
        #list of all active connections
        self.connections=[]
        #list of inputs - should list all the channels that can be received
        self.inputs=[]
        #list of outputs - should list all the channels that can be emited
        self.outputs=[]
        #the map with settings
        self.settings={} # OBSOLETE
        self.settingsList = []
        #is widget enabled?
        self.enabled=TRUE
        #the title
        self.setCaption(self.title)
        self.setIcon(QPixmap(icon))
        #about box
        self.about=OWAboutX(title,description,icon,logo)
        self.buttonBackground=QVBox(self)
        self.aboutButton=QPushButton("&About",self.buttonBackground)
        if wantSettings:
            self.settingsButton=QPushButton("&Settings",self.buttonBackground)
        if wantGraph:
            self.graphButton=QPushButton("&Save Graph",self.buttonBackground)
        if parent==None:
            self.closeButton=QPushButton("&Close",self.buttonBackground)
        self.connect(self.aboutButton,SIGNAL("clicked()"),self.about.show)
        if parent==None:
            self.connect(self.closeButton,SIGNAL("clicked()"),self.close)
        self.mainArea=QWidget(self)
#        self.mainArea.setBackgroundColor(Qt.white)
        self.controlArea=QVBox(self)
        self.space=QVBox(self)
#        self.controlArea.setBackgroundColor(Qt.white)
        self.grid=QGridLayout(self,3,2,5)
        self.grid.addWidget(self.buttonBackground,2,0)
        self.grid.addWidget(self.space,1,0)
        self.grid.setRowStretch(1,10)
        self.grid.setColStretch(1,10)
        self.grid.addWidget(self.controlArea,0,0)
        self.grid.addMultiCellWidget(self.mainArea,0,3,1,1)
        self.resize(640,480)
#        self.connect(self,SIGNAL("exit"),self.saveSettings)
        
    def link(self,source,channel):
        """
        Link a widget to this widget
        Parameters:
            source - source widget
            cnannel - name of the channel and the function it connects to
        """
        #what to do with those tree:classifier signals?!
        
        #check if this channel exist
        if channel not in self.inputs:
            return
        if channel not in self.connections:
            self.connections.append([channel,source])
            #there should be a function with the same name as the signal 
            #in the destination class, otherwise this won't work!
            #I don't think there is a less critical way of doing this :(
            if self.enabled:                                
#                print "self."+channel
                self.connect(source,PYSIGNAL(channel),eval("self."+channel))
        
    def unlink(self,source,channel):
        """
        Destroy link from a widget to this widget
        Parameters:
            source - source widget
            cnannel - name of the channel and the function it connects to
        """
        if channel in self.connections:
            self.connections.remove(channel)
            self.disconnect(source,PYSIGNAL(channel),eval("self."+channel))
            
    def send(self,channel,data):
        """
        Send data over a channel
            channel - the channel name
            data - the data to be sent, must be a tuple!
        Saves some typing compared to QObject.emit.
        """
#        print "sending", data, "over channel", channel
        self.emit(PYSIGNAL(channel),(data,))
    
    def getConnections(self):
        """
        Gets all connections
        Returns the list with all connections
        """
        return self.connections
        
    def setEnabled(self,enable):
        """
        Set this widget in enabled or disabled state
        enable: TRUE - widget is enabled 
                FALSE - widget is disabled
        """
        #setEnabled(FALSE) removes all links in connections
        #setEnabled(TRUE) restores all this links
        #we'll see if this works
        self.enabled=enable
        for connection in self.connections:
            if enabled:
                self.connect(connection[0],PYSIGNAL(connection[1]),eval("self."+connection[1]))
            else:
                self.disconnect(connection[0],PYSIGNAL(connection[1]),eval("self."+connection[1]))
        
    def setSetting(self,name,value):
        """
        Set a setting
        name - the name that uniquely identifies the setting
        value - the value of the setting
        """
        if not getattr(self, "settingsList", None):
            self.settingsList = [name]
        if not name in self.settingsList:
            self.settingsList.append(name)
        setattr(self, name, value)
        
    def getSetting(self,name):
        """
        Get a setting 
        name - the name that uniquely identifies the setting
        returns the setting
        """
        return getattr(self, name, None)
    
    def setSettings(self,settings):
        """
        Set all settings
        settings - the map with the settings
        """
        self.__dict__.update(settings)

    def getSettings(self):
        """
        Get all settings
        returns map with all settings
        """
        return dict([(x, getattr(self, x, None)) for x in settingsList])
   
    def loadSettings(self):
        """
        Loads settings from the widget's .ini file
        """
        filename=self.title+".ini"
        if os.path.exists(filename):
            file=open(filename)
            lines=file.readlines()
            for line in lines:
                line1=line.strip()
                if line1=="":
                    continue
                (key,value)=split(line1,"=")
                try:
                    evaledvalue=eval(value)
#                    print "evaled", evaledvalue
                except Exception:
                    evaledvalue=value
#                    print "not evaled", evaledvalue
                setattr(self, key, evaledvalue)
            file.close()
        
    def saveSettings(self):
        """
        Saves settings to this widget's .ini file
        """
        print "SS"
        file=open(self.title+".ini","w")
        for key in self.settingsList:
            file.write(key+"="+repr(getattr(self, key))+"\n")
            print key+"="+repr(getattr(self, key))+"\n",
        file.close()
        
    def addInput(self,inpu):
        """
        Adds an input.
        Should be called for all functions that can be inputs (link destinations).
        """
        self.inputs.append(inpu)
    
    def addOutput(self,output):
        """
        Adds an output.
        Should be called for all the outputs (link sources) that can be generated.
        """
        self.outputs.append(output)
        
    def getInputs(self):
        """
        Get the names of all inputs
        """
        return self.inputs
        
    def getOutputs(self):
        """
        Get the names of all outputs
        """
        return self.outputs
        
if __name__ == "__main__":  
    a=QApplication(sys.argv)
    oww=OWWidget()
#    oww.setSetting("gk",[43,23])
#    oww.setSettings(eval('{"one":1,"two":2}')) # this is good.. can read a line and set it easily
#    print oww.getSettings()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()
