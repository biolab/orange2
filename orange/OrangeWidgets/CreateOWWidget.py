#
# CreateOWWidget.py
# A handy widget fom creating new widgets inherited from OWWidget.
# The python code inside triple qoute strings that is output to the new widget file
# makes this code a bit less readable, but there's not much that can be done about it
# 

import sys
from qt import *
from OWTools import *

class OWCreator(QVBox):
    def __init__(self):
        "Constructor"
        QVBox.__init__(self)
        self.resize(200,70)
        self.fileName="OW.py"
        self.name=""
        self.description="OW is an Orange Widget\nline 2"
        self.iconFile="OrangeWidgetsIcon.png"
        self.logoFile="OrangeWidgetsLogo.png"
        self.nameInputBox=QVGroupBox("Widget Name",self)
        self.nameInputLine=QLineEdit(self.name,self.nameInputBox)
        self.iconFileBox=QVGroupBox("Widget Icon File:",self)
        self.iconFileLine=QLineEdit(self.iconFile,self.iconFileBox)
        self.logoFileBox=QVGroupBox("Widget Logo File:",self)
        self.logoFileLine=QLineEdit(self.logoFile,self.logoFileBox)
        self.descriptionBox=QVGroupBox("Widget Description:",self)
        self.descriptionLine=QMultiLineEdit(self.descriptionBox)
        self.descriptionLine.setText(self.description)
        self.flagBox=QVGroupBox("Options",self)
        self.hasSettings=QCheckBox("Has Settings Dialog",self.flagBox)
        self.hasSettings.setChecked(TRUE)
        self.hasGraph=QCheckBox("Has a Graph",self.flagBox)
        self.createButton=QPushButton("Create",self)
        self.cancelButton=QPushButton("Cancel",self)
        self.connect(self.createButton,SIGNAL("clicked()"),self.createWidget)
        self.connect(self.cancelButton,SIGNAL("clicked()"),a,SLOT("quit()"))
        self.resize(200,400)
        self.setCaption("Qt Orange Widget Creator")
    

    def createWidget(self):
        "Creates an Orange Widget"
        self.name=str(self.nameInputLine.text())
        self.className="OW"+self.name.replace(" ","").replace("&","")       
        self.fileName=self.className+".py"
        self.description=str(self.descriptionLine.text())
        self.iconFile=str(self.iconFileLine.text())
        self.logoFile=str(self.logoFileLine.text())
        self.needSettings=self.hasSettings.isChecked()
        self.needGraph=self.hasGraph.isChecked()
        commentedDescription=self.description.replace("\n","\n# ")
        if self.needSettings:
            settingscode="""
        #set settings list
        #self.settingsList=["NameOfSetting1","NameOfSetting2",...]
        #set default settings
        #self.NameOfSetting=value       
        #load settings
        self.loadSettings()
              
        #add a settings dialog (modify the automatically generated """+self.className+"""Options dialog
        #by adding your setting controls, like checkboxes, radiobuttons and sliders
        #and don't forget tooltips)
        self.options="""+self.className+"""Options()
        #set initial values in the options dialog
        #self.setOptions()

        #add your own initialization here
        
        #connect settingsbutton to show options
        self.connect(self.settingsButton,SIGNAL("clicked()"),self.options.show),
        
        #connect GUI controls of options in options dialog to settings
        #self.connect(self.options.control,SIGNAL("done()"),self.setControl) 
        #setControl is a function that calls setSetting() and applies the setting"""   
            savesettingscode="""
    #save settings 
    ow.saveSettings()""" 
            importsettingscode="from "+ self.className+"Options import *"
        else:
            settingscode=""
            savesettingscode=""
            importsettingscode=""
        
        if self.needGraph:
            mainareacode="""
        #add a graph widget
        #the graph widget needs to be created separately, preferably by inheriting from OWGraph
        #self.box=QVBoxLayout(self.mainArea)
        #self.graph=OW"""+self.className+"""Graph(self.mainArea)
        #self.box.addWidget(self.graph)
        #connect graph saving button to graph
        #self.connect(self.graphButton,SIGNAL("clicked()"),self.graph.saveToFile)"""
        else:
            mainareacode="""
        #give mainArea a layout (it hasn't got one so it as flexible as possible)
        #self.layout=QLayout(self.mainArea)
        #add your components here
        #self.x=QX(self.layout)"""
            
        fil=file(self.fileName,"w")
        print >>fil, """#
# %s
#
# %s
#

from OWWidget import *
%s
# from %sGraph import * #if using a graph

class %s(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "%s",
        \"\"\"%s\"\"\",
        %d,
        %d,
        "%s",
        "%s")        
        %s              
        
        #add all inputs
        #self.addInput("bla")
        #and inputs from multiple sources
        #self.addInput("bla",FALSE)
        #add outputs
        #self.addOutput("bla")
        
        #GUI
        %s
        
        #add controls to self.controlArea widget 
        #(it's a VBox, so just add them, they will appear in top to bottom order)
        #self.button=QPushButton("test",self.control)
        #connect controls to appropriate functions
       
    #define functions for all input channels
    #def data(self,data):
    #define functions for all multi input channels
    #def data(self,data,sourceID):

    
#test widget appearance        
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=%s()
    a.setMainWidget(ow)
#here you can test setting some stuff
    ow.show()
    a.exec_loop()
    %s
""" % (
        self.fileName, 
        commentedDescription,
        importsettingscode,
        self.className,
        self.className, 
        self.name,
        self.description,
        self.needSettings,
        self.needGraph,
        self.iconFile,
        self.logoFile,
        settingscode,
        mainareacode,
        self.className,        
        savesettingscode
        )

        fil.close()
        
        if self.needSettings:
            optionsFileName=self.fileName.replace(".","Options.")
            fil=file(optionsFileName,"w")
            print >>fil, """#
# %sOptions.py
#
# options dialog for distributions graph
#

from OWOptions import *

class %sOptions(OWOptions):
    def __init__(self,parent=None,name=None):
        OWOptions.__init__(self,"%s Options","%s",parent,name)  
        #add your controls here      
               
if __name__=="__main__":
    a=QApplication(sys.argv)
    w=%sOptions()
    a.setMainWidget(w)
    w.show()
    a.exec_loop()
""" % (self.className,self.className,self.name,self.iconFile,self.className)
    
        fil.close()
        self.close()
        
        
        
a=QApplication(sys.argv)
owc=OWCreator()
a.setMainWidget(owc)
owc.show()
a.exec_loop()