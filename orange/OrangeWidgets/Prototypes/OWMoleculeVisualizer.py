"""<name>Molecule Visualizer</name>
<description>Rendering of 2D structure of molecules based on their SMILES description.</description>
<icon>icons/MoleculeVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>2015</priority>
"""


import orange
#import orngChem_Old as orngChem
from OWWidget import *
from qt import *
import OWGUI
import sys, os
#import vis

localOpenEye = True
try:
    from openeye.oechem import *
    from openeye.oedepict import *
except ImportError:
    localOpenEye = False

class DrawContext(object):
    def __init__(self, molecule="", fragment="", size=200, imageprefix="", imagename="", title="", grayedBackground=False, useCached=False):
        self.molecule=molecule
        self.fragment=fragment
        self.size=size
        self.imageprefix=imageprefix
        self.imagename=imagename
        self.title=title
        self.grayedBackground=grayedBackground
        self.useCached=useCached

    def __hash__(self):
        return (self.molecule+"%%"+self.fragment+"%%"+str(self.size)+"%%"+self.title+str(self.grayedBackground)).__hash__()

    def __eq__(self, other):
        return self.molecule==other.molecule and self.fragment==other.fragment and \
               self.size==other.size and self.title==other.title and self.grayedBackground==other.grayedBackground

    def __ne__(self, other):
        return not self==other
        
class ImageCacheManager(object):
    def __init__(self, maxSize=2000, imageprefix="image", ext=".png"):
        self.maxSize = maxSize
        self.imageprefix = imageprefix
        self.ext = ext
        self.usedIndices = set()
        self.unusedIndices = set(range(maxSize))
        self.currentImages = {}
        self.cachedImages = {}
        self.epoch = 0

    def setMaxSize(self, maxSize):
        self.maxSize = maxSize

    def getUnusedIndex(self):
        return (set(range(max(len(self.cachedImages)+len(self.currentImages)+1, self.maxSize)))-self.usedIndices).pop()

    def getNewImageName(self, key):
        index = self.getUnusedIndex()
        self.usedIndices.add(index)
        name = self.imageprefix+str(index)+self.ext
        self.currentImages[key] = (index, name)
        return name

    def __contains__(self, key):
        return key in self.cachedImages or key in self.currentImages

    def __getitem__(self, key):
        if key in self.cachedImages:
            self.currentImages[key] = self.cachedImages[key]
            return self.cachedImages[key][1]
        if key in self.currentImages:
            return self.currentImages[key][1]
        raise KeyError(key)

    def __delitem__(self, key):
        if key in self.cachedImages:
            del self.cachedImages[key]
        elif key in self.currentImages:
            del self.currentImages[key]
        else:
            raise KeyError(key)
        
    def newEpoch(self):
        self.cachedImages = dict([(key, value) for key, value in self.cachedImages.items()[-self.maxSize:]])
        self.cachedImages.update(self.currentImages)
        self.usedIndices = set([v[0] for v in self.cachedImages.values()])
        self.currentImages = {}
        
class BigImage(QDialog):
    def __init__(self, context, *args):
        apply(QDialog.__init__, (self,)+args)
        self.context=context
        self.label=QLabel(self)
        self.imagename=context.imagename or context.imageprefix+"_big.bmp"
        self.imageSize=400
        self.renderImage()

    def renderImage(self):
        try:
            if self.context.fragment:
                moleculeFragment2BMP(self.context.molecule, self.context.fragment, self.imagename, self.imageSize, self.context.title, self.context.grayedBackground)
            else:
                molecule2BMP(self.context.molecule, self.imagename, self.imageSize, self.context.title, self.context.grayedBackground)
        except IOError, er:
            pix=QPixmap(self.imageSize, self.imageSize)
            pix.fill()
            painter=QPainter(pix)
            painter.drawText(6, self.height()/2-4, "Unable to load")
            painter.end()
            self.label.setPixmap(pix)
            self.label.resize(pix.width(), pix.height())
            return
        pix=QPixmap()
        pix.load(self.imagename)
        self.label.setPixmap(pix)
        self.label.resize(pix.width(), pix.height())
            
    def resizeEvent(self, event):
        apply(QDialog.resizeEvent, (self, event))
        self.imageSize=min(event.size().width(), event.size().height())
        self.renderImage()
        
class MolWidget(QVBox):
    def __init__(self, master, parent, context):
        apply(QVBox.__init__, (self, parent))
        self.master=master
        self.context=context
        self.label=QLabel(self)
        self.selected=False
        self.image=MolImage(master, self, context)
        self.show()
        self.label.setText(context.title)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.label.setMaximumWidth(context.size)

    def repaint(self):
        apply(QVBox.repaint,(self, ))
        self.label.repaint()
        self.image.repaint()
        
class MolImage(QLabel):
    def __init__(self, master, parent, context):
        apply(QLabel.__init__,(self, parent))
        self.context=context
        self.master=master
        #imagename=context.imagename or context.imageprefix+".bmp"
        redraw=False
        if context not in master.imageCache:
            imagename = master.imageCache.getNewImageName(context)
            self.context.imagename = imagename
            try:
                if master.failedCount>5 :
                    raise IOError
                if context.fragment:
                    moleculeFragment2BMP(context.molecule, context.fragment, imagename, context.size, grayedBackground=context.grayedBackground)
                else:
                    molecule2BMP(context.molecule, imagename, context.size, grayedBackground=context.grayedBackground)
            except IOError, err:
                master.failedCount+=1
                self.load("unknown")
                self.selected=False
                self.setFrameStyle(QFrame.Panel | QFrame.Raised)
                self.show()
                del master.imageCache[context]
                #print err , "MI"
                #raise err
            else:
                #master.failedCount-=1
                self.load(imagename)
                self.selected=False
                self.setFrameStyle(QFrame.Panel | QFrame.Raised)
                self.show()
                
        else:
            imagename=master.imageCache[context]
            context.imagename=imagename
            self.load(imagename)
            self.selected=False
            self.setFrameStyle(QFrame.Panel | QFrame.Raised)
            self.show()
        
    def load(self, filename):
        self.pix=QPixmap()
        if not self.pix.load(filename):
            self.pix=QPixmap(self.context.size, self.context.size)
            self.pix.fill()
            painter=QPainter(self.pix)
            #painter.begin(self.pix)
            painter.drawText(6, self.context.size/2-4, "Unable to load")
            painter.end()
            #print "Failed to load "+filename
        self.resize(self.pix.width(), self.pix.height())
        self.setPixmap(self.pix)

    def paintEvent(self, event):
        apply(QLabel.paintEvent,(self, event))
        if self.parent().selected:
            painter=QPainter(self)
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(2, 2, self.width()-3, self.height()-3)

    def mousePressEvent(self, event):
        self.master.mouseAction(self.parent(), event)
##        print self.x(), self.y(), event.pos().x(), event.pos().y()
        self.update()

    def mouseDoubleClickEvent(self, event):
        context = DrawContext(self.context.molecule, self.context.fragment, self.context.size, self.master.imageprefix, "", self.context.title, self.context.grayedBackground)
        d=BigImage(context, self)
        d.show()
        
class ScrollView(QScrollView):
    def __init__(self, master, *args):
        apply(QScrollView.__init__, (self,)+args)
        self.master=master
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        
    def resizeEvent(self, event):
        apply(QScrollView.resizeEvent, (self, event))
        size=event.size()
        w,h=self.width(), self.height()
        oldNumColumns=self.master.numColumns
        numColumns=w/(self.master.imageSize+4) or 1
        if numColumns!=oldNumColumns:
            self.master.numColumns=numColumns
            self.master.redrawImages(useCached=not self.master.overRideCache)
##        print self.maximumSize().height(), self.viewport().maximumSize().height()
        

class OWMoleculeVisualizer(OWWidget):
    settingsList=["colorFragmets","showFragments", "serverPassword"]
    contextHandlers={"":DomainContextHandler("", [ContextField("moleculeTitleAttributeList",
                                                    DomainContextHandler.List + DomainContextHandler.SelectedRequired + DomainContextHandler.IncludeMetaAttributes,
                                                    selected="selectedMoleculeTitleAttrs"),
                                                  ContextField("moleculeSmilesAttr", DomainContextHandler.Required + DomainContextHandler.IncludeMetaAttributes)], maxAttributesToPickle=10000)} ##maxAttributesToPickle=10000 some bug in Context handler 
    serverPwd = ""
    uniqueWidgetImageId = 0
    def __init__(self, parent=None, signalManager=None, name="Molecule visualizer"):
        apply(OWWidget.__init__,(self, parent, signalManager, name))
        self.inputs=[("Molecules", ExampleTable, self.setMoleculeTable), ("Molecule subset", ExampleTable, self.setMoleculeSubset), ("Fragments", ExampleTable, self.setFragmentTable)]
        self.outputs=[("Selected Molecules", ExampleTable)]
        self.colorFragments=1
        self.showFragments=0
        self.selectedFragment=""
        self.moleculeSmiles=[]
        self.fragmentSmiles=[]
        self.defFragmentSmiles=[]
        self.moleculeSmilesAttr=0
        self.moleculeTitleAttr=0
        self.moleculeTitleAttributeList=[]
        self.selectedMoleculeTitleAttrs=[]
        self.fragmentSmilesAttr=0
        self.imageSize=200
        self.numColumns=4
        self.commitOnChange=0
        self.serverPassword=OWMoleculeVisualizer.serverPwd
        self.overRideCache=True
        ##GUI
        box=OWGUI.widgetBox(self.controlArea, "Info", addSpace = True)
        self.infoLabel=OWGUI.label(box, self, "Chemicals: ")
##        if not localOpenEye:
##            OWGUI.label(box, self, "OpenEye not installed, access to server required.")
##            self.serverInfo = OWGUI.label(box, self, "")
        box=OWGUI.radioButtonsInBox(self.controlArea, self, "showFragments", ["Show molecules", "Show fragments"], "Show", callback=self.showImages)
        self.showFragmentsRadioButton=box.buttons[-1]
        self.markFragmentsCheckBox=OWGUI.checkBox(box, self, "colorFragments", "Mark fragments", callback=self.redrawImages)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        OWGUI.separator(self.controlArea)
        self.moleculeSmilesCombo=OWGUI.comboBox(self.controlArea, self, "moleculeSmilesAttr", "Molecule SMILES attribute",callback=self.showImages)
        self.moleculeSmilesCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        OWGUI.separator(self.controlArea)
##        self.moleculeTitleCombo=OWGUI.comboBox(self.controlArea, self, "moleculeTitleAttr", "Molecule title attribute", callback=self.redrawImages)
        box=OWGUI.widgetBox(self.controlArea, "Molecule title attributes", addSpace = True)
##        self.moleculeTitleListBox=QListBox(box)
##        self.moleculeTitleListBox.setSelectionMode(QListBox.Extended)
##        self.moleculeTitleListBox.setMinimumHeight(100)
##        self.connect(self.moleculeTitleListBox, SIGNAL("selectionChanged()"), self.updateTitles)
        self.moleculeTitleListBox=OWGUI.listBox(box, self, "selectedMoleculeTitleAttrs", "moleculeTitleAttributeList", selectionMode = QListBox.Extended, callback=self.updateTitles)
        self.moleculeTitleListBox.setMinimumHeight(100)
##        OWGUI.separator(self.controlArea)
##        self.moleculeTitleCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        OWGUI.separator(self.controlArea)
        self.fragmentSmilesCombo=OWGUI.comboBox(self.controlArea, self, "fragmentSmilesAttr", "Fragment SMILES attribute", callback=self.updateFragmentsListBox)
        self.fragmentSmilesCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        OWGUI.separator(self.controlArea)
        box=OWGUI.spin(self.controlArea, self, "imageSize", 50, 500, 50, box="Image size", callback=self.redrawImages, callbackOnReturn = True)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        OWGUI.separator(self.controlArea)
        box=OWGUI.widgetBox(self.controlArea,"Selection", addSpace = True)
        OWGUI.checkBox(box, self, "commitOnChange", "Commit on change")
        self.selectMarkedMoleculesButton=OWGUI.button(box, self, "Select &matched molecules", self.selectMarked)
        OWGUI.button(box, self, "&Commit", callback=self.commit)
        box=OWGUI.widgetBox(self.controlArea,"Server Password", addSpace=True)
        OWGUI.lineEdit(box, self, "serverPassword", callback=self.setServPwd)
        OWGUI.separator(self.controlArea)
        OWGUI.button(self.controlArea, self, "&Save to HTML", self.saveToHTML)
        OWGUI.rubber(self.controlArea)
        
        self.mainAreaLayout=QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom)
        spliter=QSplitter(Qt.Vertical, self.mainArea)
        self.scrollView=ScrollView(self, spliter)
        self.scrollView.setHScrollBarMode(QScrollView.Auto)
        self.scrollView.setVScrollBarMode(QScrollView.Auto)
        self.molWidget=QWidget(self.scrollView.viewport())
        self.scrollView.addChild(self.molWidget)
        self.mainAreaLayout.addWidget(spliter)
        self.gridLayout=QGridLayout(self.molWidget,100,100,2,2)
        self.gridLayout.setAutoAdd(False)
        self.listBox=QListBox(spliter)
        self.connect(self.listBox, SIGNAL("highlighted(int)"), self.fragmentSelection)
        self.scrollView.setFocusPolicy(QWidget.StrongFocus)
        self.listBox.setFocusPolicy(QWidget.NoFocus)

        self.imageprefix=os.path.split(__file__)[0]
        if "molimages" not in os.listdir(self.imageprefix):
            try:
                os.mkdir(self.imageprefix and self.imageprefix+"/molimages" or "molimages")
            except:
                pass
        if self.imageprefix:
            self.imageprefix+="\molimages\imageW"+str(self.uniqueWidgetImageId)+"_"
        else:
            self.imageprefix="molimages\imageW"+str(self.uniqueWidgetImageId)+"_"
        OWMoleculeVisualizer.uniqueWidgetImageId+=1
        self.imageWidgets=[]
        self.candidateMolSmilesAttr=[]
        self.candidateMolTitleAttr=[None]
        self.candidateFragSmilesAttr=[None]
        self.molData=None
        self.molSubset=[]
        self.fragData=None
        self.ctrlPressed=FALSE
        self.resize(600,600)
        self.listBox.setMaximumHeight(150)
        self.fragmentSmilesCombo.setDisabled(True)
        self.selectMarkedMoleculesButton.setDisabled(True)
        self.markFragmentsCheckBox.setDisabled(True)
        self.showFragmentsRadioButton.setDisabled(True)
        self.loadSettings()
        OWMoleculeVisualizer.serverPwd=self.serverPassword
        self.imageCache=ImageCacheManager(1000, self.imageprefix, ".png")
        self.failedCount=0
        
    def setMoleculeTable(self, data):
        self.closeContext()
        self.molData=data
        if data:
            self.setMoleculeSmilesCombo()
            self.setMoleculeTitleListBox()
            self.setFragmentSmilesCombo()
            self.updateFragmentsListBox()
            if self.molSubset:
                try:
                    self.molSubset=self.molSubset.select(self.molData.domain)
                except:
                    self.molSubset=[]
##            tmp=self.moleculeTitleAttributeList
            self.openContext("",data)
##            if len(tmp)>=100 and not self.moleculeTitleAttributeList:
##                self.moleculeTitleAttributeList=tmp
            self.showImages()
        else:
            self.moleculeSmilesCombo.clear()
            self.moleculeTitleListBox.clear()
            self.moleculeSmilesAttr=0
            self.moleculeTitleAttributeList=[]
            self.selectedMoleculeTitleAttrs=[]
            self.defFragmentSmiles=[]
            if not self.fragmentSmilesAttr:
                self.listBox.clear()
            self.destroyImageWidgets()
            self.openContext("",data)
            self.send("Selected Molecules", None)
        

    def setMoleculeSubset(self, data):
        self.molSubset=data
        try:
            self.molSubset=self.molSubset.select(self.molData.domain)
        except:
            self.molSubset=[]
        self.showImages()
            
    def setFragmentTable(self, data):
        self.fragData=data
        if data:
            self.setFragmentSmilesCombo()
            self.updateFragmentsListBox()
            self.selectedFragment=""
            self.showImages()
        else:
            self.setFragmentSmilesCombo()
            #self.fragmentSmilesAttr=0
            self.updateFragmentsListBox()
            if self.showFragments:
                self.destroyImageWidgets()
        self.fragmentSmilesCombo.setDisabled(bool(data))

    def setServPwd(self):
        if self.serverPassword:
            OWMoleculeVisualizer.serverPwd=self.serverPassword
            self.showImages()

    def filterSmilesVariables(self, data):
        candidates=data.domain.variables+data.domain.getmetas().values()
        candidates=filter(lambda v:v.varType==orange.VarTypes.Discrete or v.varType==orange.VarTypes.String, candidates)
        if len(data)>20:
            data=data.select(orange.MakeRandomIndices2(data, 20))
        vars=[]
        if localOpenEye:
            isValidSmiles = lambda s: OEParseSmiles(OEGraphMol(), s)
        else:
            from openbabel import OBConversion, OBMol
            loader = OBConversion()
            loader.SetInAndOutFormats("smi", "smi")
            isValidSmiles = lambda s: loader.ReadString(OBMol(), s)
            
        import os
        tmpFd1=os.dup(1)
        tmpFd2=os.dup(2)
        fd=os.open(os.devnull, os.O_APPEND)
##        os.close(1)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
##        os.close(fd)
        for var in candidates:
            count=0
            for e in data:
                if isValidSmiles(str(e[var])):
                    count+=1
            vars.append((count,var))
        names=[v.name for v in data.domain.variables+data.domain.getmetas().values()]
        names=filter(isValidSmiles, names)
##        os.close(1)
        os.dup2(tmpFd1, 1)
        os.dup2(tmpFd2, 2)
##        os.close(tmpFd)
        return vars, names
        
    def setMoleculeSmilesCombo(self):
        candidates, self.defFragmentSmiles=self.filterSmilesVariables(self.molData)
        self.candidateMolSmilesAttr=[c[1] for c in candidates]
        best = reduce(lambda best,current:best[0]<current[0] and current or best, candidates)
        self.moleculeSmilesCombo.clear()
        self.moleculeSmilesCombo.insertStrList([v.name for v in self.candidateMolSmilesAttr])
        self.moleculeSmilesAttr=candidates.index(best)

    def setMoleculeTitleListBox(self):
        self.icons=self.createAttributeIconDict()
        vars=self.molData.domain.variables+self.molData.domain.getmetas().values()
        self.moleculeTitleAttributeList=[attr.name for attr in vars]
        self.selectedMoleculeTitleAttrs=[]
        """vars=self.molData.domain.variables+self.molData.domain.getmetas().values()
##        self.moleculeTitleCombo.clear()
##        self.moleculeTitleCombo.insertStrList(["None"]+[v.name for v in vars])
        self.moleculeTitleListBox.clear()
        self.moleculeTitleListBox.insertStrList(["None"]+[v.name for v in vars])
        if self.moleculeTitleAttr>len(vars):
            self.moleculeTitleAttr=0
        self.candidateMolTitleAttr=[None]+[v for v in vars]"""

    def updateTitles(self):
        if not self.molData:
            return
        #selected=filter(lambda (i,attr):self.moleculeTitleListBox.isSelected(i), list(enumerate(self.candidateMolTitleAttr))[1:])
        smilesAttr=self.candidateMolSmilesAttr[min(self.moleculeSmilesAttr, len(self.candidateMolSmilesAttr)-1)]
        attrs = self.molData.domain.variables+self.molData.domain.getmetas().values()
        selected=[attrs[i] for i in self.selectedMoleculeTitleAttrs]
        for widget, example in zip(self.imageWidgets, filter(lambda e:not e[smilesAttr].isSpecial(),self.molData)):
            text=" / ".join(map(str, [example[attr] for attr in selected]))
            widget.label.setText(text)

    def setFragmentSmilesCombo(self):
        if self.fragData:
            candidates, names=self.filterSmilesVariables(self.fragData)
        else:
            candidates=[]
        self.candidateFragSmilesAttr=[None]+candidates
        self.fragmentSmilesCombo.clear()
        self.fragmentSmilesCombo.insertStrList(["Default"]+[v.name for v in candidates])
        if self.fragmentSmilesAttr>len(candidates):
            self.fragmentSmilesAttr=0

    def updateFragmentsListBox(self):
        fAttr=self.candidateFragSmilesAttr[self.fragmentSmilesAttr]
        if fAttr:
            self.fragmentSmiles=[""]+[str(e[fAttr]) for e in self.fragData if not e[fAttr].isSpecial()]
        else:
            self.fragmentSmiles=[""]+self.defFragmentSmiles
        self.listBox.clear()
        self.listBox.insertStrList(self.fragmentSmiles)
        self.showFragmentsRadioButton.setDisabled(len(self.fragmentSmiles)==1)
        self.markFragmentsCheckBox.setDisabled(len(self.fragmentSmiles)==1)
        self.selectMarkedMoleculesButton.setDisabled(True)
        
    def fragmentSelection(self, index):
        self.selectedFragment=self.fragmentSmiles[index]
        self.selectMarkedMoleculesButton.setEnabled(bool(self.selectedFragment))
        self.markFragmentsCheckBox.setEnabled(bool(self.selectedFragment))
        if not self.showFragments and self.colorFragments:
            self.redrawImages()
        
    def renderImages(self,useCached=False):
        def fixNumColumns(numItems, numColumns):
            if (self.imageSize+4)*(numItems/numColumns+1)>30000:
                return numItems/(30000/(self.imageSize+4))
            else:
                return numColumns
        
        self.numColumns=self.scrollView.width()/(self.imageSize+4) or 1
        self.imageWidgets=[]
        self.imageCache.newEpoch()
        self.failedCount=0
        if self.showFragments and self.fragmentSmiles:
            correctedNumColumns=fixNumColumns(len(self.fragmentSmiles[1:]), self.numColumns)
            self.progressBarInit()
            for i,fragment in enumerate(self.fragmentSmiles[1:]):
                #imagename=self.imageprefix+str(i)+".bmp"
                #vis.molecule2BMP(fragment, imagename, self.imageSize)
##                image=MolImage(self,  self.molWidget, DrawContext(molecule=fragment, imagename=imagename, size=self.imageSize))
                image=MolWidget(self, self.molWidget, DrawContext(molecule=fragment, size=self.imageSize, useCached=useCached))
                self.gridLayout.addWidget(image, i/correctedNumColumns, i%correctedNumColumns)
                self.imageWidgets.append(image)
                self.progressBarSet(i*100/len(self.fragmentSmiles))
            self.progressBarFinished()
        elif self.molData and self.candidateMolSmilesAttr:
            sAttr=self.candidateMolSmilesAttr[min(self.moleculeSmilesAttr, len(self.candidateMolSmilesAttr)-1)]
            tAttr=self.candidateMolTitleAttr[min(self.moleculeTitleAttr, len(self.candidateMolTitleAttr)-1)]
            if self.moleculeTitleAttr:
                titleList=[str(e[tAttr]) for e in self.molData if not e[sAttr].isSpecial()]
            else:
                titleList=[]
                if not sAttr:
                    return
            molSmiles=[(str(e[sAttr]), e) for e in self.molData if not e[sAttr].isSpecial()]
            correctedNumColumns=fixNumColumns(len(molSmiles), self.numColumns)
            self.progressBarInit()
            if self.colorFragments and self.selectedFragment:
                fMap=map_fragments([self.selectedFragment], [t[0] for t  in molSmiles])
            for i,((molecule, example), title) in enumerate(zip(molSmiles, titleList or [""]*len(molSmiles))):
                #imagename=self.imageprefix+str(i)+".bmp"
                if self.colorFragments and self.selectedFragment and fMap[molecule][self.selectedFragment]:
                    context=DrawContext(molecule=molecule, fragment=self.selectedFragment, size=self.imageSize, title=title, grayedBackground=example in self.molSubset, useCached=useCached)
                    #vis.moleculeFragment2BMP(molecule, self.selectedFragment, imagename, self.imageSize)
                else:
                    context=DrawContext(molecule=molecule, size=self.imageSize, title=title, grayedBackground=example in self.molSubset, useCached=useCached)
                    #vis.molecule2BMP(molecule, imagename, self.imageSize)
##                image=MolImage(self, self.molWidget, context)
                image=MolWidget(self, self.molWidget, context)
                self.gridLayout.addWidget(image, i/correctedNumColumns, i%correctedNumColumns)
                self.imageWidgets.append(image)
                self.progressBarSet(i*100/len(molSmiles))
            self.progressBarFinished()
            self.updateTitles()
        #print "done drawing"
        self.overRideCache=False
##        if self.imageWidgets:
##            self.scrollView.viewport().setMaximumHeight(self.imageSize*(len(self.imageWidgets)/self.numColumns+1))
##            print self.imageWidgets[-1].y()+self.imageWidgets[-1].height(), viewportHeight

    def destroyImageWidgets(self):
        for w in self.imageWidgets:
            self.molWidget.removeChild(w)
        self.imageWidgets=[]
            
    def showImages(self, useCached=False):
        self.destroyImageWidgets()
        #print "destroyed"
        self.warning()
        
        self.renderImages(useCached)
        if not localOpenEye:
            if self.failedCount>0:
                self.infoLabel.setText("Chemicals %i\nFailed to retrieve images from server" % len(self.imageWidgets))
                self.warning("Failed to retrieve images from server")
            else:
                self.infoLabel.setText("Chemicals %i\nAll images retrieved from server" % len(self.imageWidgets))
        else:
            self.infoLabel.setText("Chemicals %i" % len(self.imageWidgets))

    def redrawImages(self, useCached=False):
        selected=map(lambda i:self.imageWidgets.index(i), filter(lambda i:i.selected, self.imageWidgets))
        self.showImages(useCached=useCached)
        for i in selected:
            self.imageWidgets[i].selected=True
            self.imageWidgets[i].repaint()
            
    def mouseAction(self, image, event):
        if self.ctrlPressed:
            image.selected=not image.selected
        else:
            for i in self.imageWidgets:
                i.selected=False
                i.repaint()
            image.selected=True
        image.repaint()
        if self.commitOnChange:
            self.commit()

    def selectMarked(self):
        if not self.showFragments:
            molecules=[i.context.molecule for i in self.imageWidgets]
            fMap=map_fragments([self.selectedFragment], molecules)
            for i in self.imageWidgets:
                if fMap[i.context.molecule][self.selectedFragment]:
                    i.selected=True
                else:
                    i.selected=False
                i.repaint()
        if self.commitOnChange:
            self.commit()
    
    def commit(self):
        if self.showFragments:
            sAttr=self.candidateMolSmilesAttr[self.moleculeSmilesAttr]
            molecules=map(str, [e[sAttr] for e in self.molData])
            fragments=[i.context.molecule for i in self.imageWidgets if i.selected]
            fragmap=map_fragments(fragments, molecules)
            match=filter(lambda m:max(fragmap[m].values()), molecules)
            examples=[e for e in self.molData if str(e[sAttr]) in match]
            if examples:
                table=orange.ExampleTable(examples)
                self.send("Selected Molecules", table)
            else:
                self.send("Selected Molecules", None)                
        else:
            mols=[i.context.molecule for i in self.imageWidgets if i.selected]
            sAttr=self.candidateMolSmilesAttr[self.moleculeSmilesAttr]
            examples=[e for e in self.molData if str(e[sAttr]) in mols]
            if examples:
                table=orange.ExampleTable(examples)
                self.send("Selected Molecules", table)
            else:
                self.send("Selected Molecules", None)

    def keyPressEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=TRUE
        else:
            OWWidget.keyPressEvent(self, key)

    def keyReleaseEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=FALSE
        else:
            OWWidget.keyReleaseEvent(self, key)

    def saveToHTML(self):
        fileName=str(QFileDialog.getSaveFileName("index.html","HTML (.html)", None, "Save to.."))
        if not fileName:
            return
        else:
            file=open(fileName, "w")
        import os
        path, _ =os.path.split(fileName)
        if "molimages" not in os.listdir(path):
            os.mkdir(path+"/molimages")
        title="Molekule"
        file.write("<html><title>"+title+"</title>\n")
        file.write("<body> <table border=\"1\">\n")
        i=0
        try:
            import Image
        except:
            pass
        for row in range(len(self.imageWidgets)/self.numColumns+1):
            file.write("<tr>\n")
            for col in range(self.numColumns):
                if i>=len(self.imageWidgets):
                    break
                try:
                    im=Image.open(self.imageWidgets[i].context.imagename)
                    if im.mode!="RGB":
                        im=im.convert("RGB")
                    im.save(path+"/molimages/image"+str(i)+".gif", "GIF")
                    file.write("<td><p>"+str(self.imageWidgets[i].label.text())+"</p><img src=\"./molimages/image"+str(i)+".gif\"></td>\n")
                except:
                    from shutil import copy
                    copy(self.imageWidgets[i].context.imagename, path+"/molimages/")
                    file.write("<td><p>"+str(self.imageWidgets[i].label.text())+"</p><img src=\"./molimages/image"+str(i)+".bmp\"></td>\n")
                i+=1
            file.write("</tr>\n")
        file.write("</table></body></html>")
        file.close()
        
def moleculeFragment2BMP(molSmiles, fragSmiles, filename, size=200, title="", grayedBackground=False):
    """given smiles codes of molecle and a fragment will draw the molecule and save it
    to a file"""
    mol=OEGraphMol()
    OEParseSmiles(mol, molSmiles)
    depict(mol)
    mol.SetTitle(title)
    match=subsetSearch(mol, fragSmiles)
    view=createMolView(mol, size)
    colorSubset(view, mol, match)
    if grayedBackground:
        view.SetBackColor(245,245,245)
    renderImage(view, filename)

def molecule2BMP(molSmiles, filename, size=200, title="", grayedBackground=False):
    """given smiles code of a molecule will draw the molecule and save it
    to a file"""
    mol=OEGraphMol()
    OEParseSmiles(mol, molSmiles)
    mol.SetTitle(title)
    depict(mol)
    view=createMolView(mol, size)
    if grayedBackground:
        view.SetBackColor(240,240,240)
    renderImage(view, filename)

def depict(mol):
    """depict a molecule - i.e assign 2D coordinates to atoms"""
    if mol.GetDimension()==3:
        OEPerceiveChiral(mol)
        OE3DToBondStereo(mol)
        OE3DToAtomStereo(mol)
    OEAddDepictionHydrogens(mol)
    OEDepictCoordinates(mol)
    OEMDLPerceiveBondStereo(mol)
    OEAssignAromaticFlags(mol)

def subsetSearch(mol, pattern):
    """finds the matches of pattern in mol"""
    pat=OESubSearch()
    pat.Init(pattern)
    return pat.Match(mol,1)

def createMolView(mol, size=200, title=""):
    """creates a view for the molecule mol"""
    view=OEDepictView()
    view.SetMolecule(mol)
    view.SetLogo(False)
    view.SetTitleSize(12)
    view.AdjustView(size, size)
    return view

def colorSubset(view, mol, match):
    """assigns a differnet color to atoms and bonds of mol in view that are present in match"""
    for matchbase in match:
        for mpair in matchbase.GetAtoms():
            style=view.AStyle(mpair.target.GetIdx())
            #set style
            style.r=255
            style.g=0
            style.b=0

    for matchbasem in match:
        for mpair in matchbase.GetBonds():
            style=view.BStyle(mpair.target.GetIdx())
            #set style
            style.r=255
            style.g=0
            style.b=0

def renderImage(view, filename):
    """renders the view to a filename"""
    img=OE8BitImage(view.XRange(), view.YRange())
    view.RenderImage(img)
    ofs=oeofstream(filename)
    OEWriteBMP(ofs, img)

def render2OE8BitImage(view):
    """renders the view to a OE8BitImage"""
    img=OE8BitImage(view.XRange(), view.YRange())
    view.RenderImage(img)
    return view

def remoteMoleculeFragment2BMP(molSmiles, fragSmiles, filename, size=200, title="", grayedBackground=False):
    import cStringIO
    import urllib
    import Image
    
    params = urllib.urlencode({'password': OWMoleculeVisualizer.serverPwd, 'molSmiles': molSmiles, 'fragSmiles': fragSmiles, 'size': size, 'title': title, 'grayBack': grayedBackground})
    try:
        f = urllib.urlopen("http://212.235.189.53/openEye/drawMol.py", params)
        imgstr = f.read()
    except IOError, er:
        #print er
        raise er
    im = cStringIO.StringIO(imgstr)
    img = Image.open(im)
    #print img.format, img.size, img.mode
    img.save(filename)
    #del img

def remoteMolecule2BMP(molSmiles, filename, size=200, title="", grayedBackground=False):
    import cStringIO
    import urllib
    import Image
    
    params = urllib.urlencode({'password': OWMoleculeVisualizer.serverPwd, 'molSmiles': molSmiles, 'size': size, 'title': title, 'grayBack': grayedBackground})
    try:
        f = urllib.urlopen("http://212.235.189.53/openEye/drawMol.py", params)
        imgstr = f.read()
    except IOError, er:
        #print er, "rm2BMP"
        raise er
    im = cStringIO.StringIO(imgstr)
    img = Image.open(im)
    #print img.format, img.size, img.mode
    img.save(filename)
    #del img        

def map_fragments(fragments, smiles, binary=True):
    def molGraph(smiles):
        mol=OEGraphMol()
        OEParseSmiles(mol, smiles)
        OEAssignAromaticFlags(mol)
        return mol
    ret={}
    pat=OESubSearch()
    for s in smiles:
        c=molGraph(s)
        d={}
        for f in fragments:
            pat.Init(f)
            count=0
            for m in pat.Match(c,1):
                count+=1
            if binary:
                d[f]=count!=0 and 1 or 0
            else:
                d[f]=count
        ret[s]=d
    return ret

def map_fragments_OB(fragments, smiles, binary=True):
    from openbabel import OBMol, OBSmartsPattern, OBConversion
    ret = {}
    pat = OBSmartsPattern()
    loader = OBConversion()
    loader.SetInAndOutFormats("smi","smi")
    for s in smiles:
        mol = OBMol()
        loader.ReadString(mol, s)
        d={}
        for f in fragments:
            pat.Init(f)
            count=0
            if pat.Match(mol, True):
                count+=1
            if binary:
                d[f]=count!=0 and 1 or 0
            else:
                d[f]=count
        ret[s]=d
    return ret  

if not localOpenEye:
    moleculeFragment2BMP = remoteMoleculeFragment2BMP
    molecule2BMP = remoteMolecule2BMP
    map_fragments = map_fragments_OB

if __name__=="__main__":
    app=QApplication(sys.argv)
    from pywin.debugger import set_trace
##    set_trace()
    w=OWMoleculeVisualizer()
    app.setMainWidget(w)
    w.show()
    data=orange.ExampleTable("E://chem/chemdata/BCMData_growth_frag.tab")
    
    w.setMoleculeTable(data)
##    data=orange.ExampleTable("E://chem//new//sf.tab")
##    w.setFragmentTable(data)
    app.exec_loop()
    w.saveSettings()
