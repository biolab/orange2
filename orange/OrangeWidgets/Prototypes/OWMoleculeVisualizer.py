"""<name>Molecule visualizer</name>"""

import orange
import orngChem
from OWWidget import *
from qt import *
import OWGUI
import sys, os
import vis
from openeye.oechem import *

class DrawContext:
    def __init__(self, molecule="", fragment="", size=200, imageprefix="", imagename="", title=""):
        self.molecule=molecule
        self.fragment=fragment
        self.size=size
        self.imageprefix=imageprefix
        self.imagename=imagename
        self.title=title

class BigImage(QDialog):
    def __init__(self, context, *args):
        apply(QDialog.__init__, (self,)+args)
        self.context=context
        self.label=QLabel(self)
        self.imagename=context.imagename or context.imageprefix+"_big.bmp"
        self.imageSize=400
        self.renderImage()

    def renderImage(self):
        if self.context.fragment:
            vis.moleculeFragment2BMP(self.context.molecule, self.context.fragment, self.imagename, self.imageSize, self.context.title)
        else:
            vis.molecule2BMP(self.context.molecule, self.imagename, self.imageSize, self.context.title)
        pix=QPixmap()
        pix.load(self.imagename)
        self.label.setPixmap(pix)
        self.label.resize(pix.width(), pix.height())
            
    def resizeEvent(self, event):
        apply(QDialog.resizeEvent, (self, event))
        self.imageSize=min(event.size().width(), event.size().height())
        self.renderImage()
        
    
class MolImage(QLabel):
    def __init__(self, master, parent, context):
        apply(QLabel.__init__,(self, parent))
        #print filename
        self.context=context
        self.master=master
        imagename=context.imagename or context.imageprefix+".bmp"
        if context.fragment:
            vis.moleculeFragment2BMP(context.molecule, context.fragment, imagename, context.size, context.title)
        else:
            vis.molecule2BMP(context.molecule, imagename, context.size, context.title)
        self.load(imagename)
        self.selected=False
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        
    def load(self, filename):
        self.pix=QPixmap()
        if not self.pix.load(filename):
            print "Failed to load "+filename
            return
        self.resize(self.pix.width(), self.pix.height())
        self.setPixmap(self.pix)

    def paintEvent(self, event):
        apply(QLabel.paintEvent,(self, event))
        if self.selected:
            painter=QPainter(self)
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(2, 2, self.width()-3, self.height()-3)

    def mousePressEvent(self, event):
        self.master.mouseAction(self, event)

    def mouseDoubleClickEvent(self, event):
        d=BigImage(self.context, self)
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
        w,h=size.width(), size.height()
        oldNumColumns=self.master.numColumns
        numColumns=min(w/self.master.imageSize or 1, 100)
        if numColumns!=oldNumColumns:
            self.master.numColumns=numColumns
            self.master.redrawImages()
        

class OWMoleculeVisualizer(OWWidget):
    settingsList=["colorFragmets","showFragments"]
    def __init__(self, parent=None, signalManager=None, name="Molecule visualizer"):
        apply(OWWidget.__init__,(self, parent, signalManager, name))
        self.inputs=[("Molecules", ExampleTable, self.setMoleculeTable), ("Fragments", ExampleTable, self.setFragmentTable)]
        self.outputs=[("Selected Molecules", ExampleTable)]
        self.colorFragments=1
        self.showFragments=0
        self.selectedFragment=""
        self.moleculeSmiles=[]
        self.fragmentSmiles=[]
        self.defFragmentSmiles=[]
        self.moleculeSmilesAttr=0
        self.moleculeTitleAttr=0
        self.fragmentSmilesAttr=0
        self.imageSize=200
        self.numColumns=4
        self.commitOnChange=0
        ##GUI
        box=OWGUI.radioButtonsInBox(self.controlArea, self, "showFragments", ["Show molecules", "Show fragments"], "Show", callback=self.showImages)
        OWGUI.checkBox(box, self, "colorFragments", "Mark fragments", callback=self.redrawImages)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        self.moleculeSmilesCombo=OWGUI.comboBox(self.controlArea, self, "moleculeSmilesAttr", "Molecule SMILES attribute",callback=self.showImages)
        self.moleculeSmilesCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        self.moleculeTitleCombo=OWGUI.comboBox(self.controlArea, self, "moleculeTitleAttr", "Molecule title attribute", callback=self.redrawImages)
        self.moleculeTitleCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        self.fragmentSmilesCombo=OWGUI.comboBox(self.controlArea, self, "fragmentSmilesAttr", "Fragment SMILES attribute", callback=self.updateFragmentsListBox)
        self.fragmentSmilesCombo.box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        box=OWGUI.spin(self.controlArea, self, "imageSize", 50, 500, 50, box="Image size", callback=self.redrawImages)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        box=OWGUI.widgetBox(self.controlArea,"Selection")
        OWGUI.checkBox(box, self, "commitOnChange", "Commit on change")
        OWGUI.button(box, self, "&Select marked molecules", self.selectMarked)
        OWGUI.button(box, self, "&Commit", callback=self.commit)
        OWGUI.rubber(self.controlArea)
        
        self.mainAreaLayout=QVBoxLayout(self.mainArea, QVBoxLayout.TopToBottom)
        spliter=QSplitter(Qt.Vertical, self.mainArea)
        self.scrollView=ScrollView(self, spliter)
        self.scrollView.setHScrollBarMode(QScrollView.Auto)
        self.scrollView.setVScrollBarMode(QScrollView.Auto)
        self.molWidget=QWidget(self.scrollView.viewport())
        self.scrollView.addChild(self.molWidget)
        self.mainAreaLayout.addWidget(spliter)
        self.gridLayout=QGridLayout(self.molWidget,10,100,2,2)
        self.gridLayout.setAutoAdd(False)
        self.listBox=QListBox(spliter)
        self.connect(self.listBox, SIGNAL("highlighted(int)"), self.fragmentSelection)

        self.imageprefix=os.path.split(__file__)[0]
        if self.imageprefix:
            self.imageprefix+="\molimages\image"
        else:
            self.imageprefix="molimages\image"
        self.imageWidgets=[]
        self.candidateMolSmilesAttr=[]
        self.candidateMolTitleAttr=[None]
        self.candidateFragSmilesAttr=[None]
        self.molData=None
        self.fragData=None
        self.ctrlPressed=FALSE
        self.resize(600,600)
        self.listBox.setMaximumHeight(150)
        
    def setMoleculeTable(self, data):
        self.molData=data
        if data:
            self.setMoleculeSmilesCombo()
            self.setMoleculeTitleCombo()
            self.setFragmentSmilesCombo()
            self.updateFragmentsListBox()
            self.showImages()
        else:
            self.moleculeSmilesCombo.clear()
            self.moleculeTitleCombo.clear()
            self.defFragmentSmiles=[]
            if not self.fragmentSmilesAttr:
                self.listBox.clear()
            self.destroyImageWidgets()
            self.send("Selected Molecules", None)
            
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

    def filterSmilesVariables(self, data):
        candidates=data.domain.variables+data.domain.getmetas().values()
        candidates=filter(lambda v:v.varType==orange.VarTypes.Discrete or v.varType==orange.VarTypes.String, candidates)        
        vars=[]
        for var in candidates:
            count=0
            for e in data:
                if OEParseSmiles(OEGraphMol(), str(e[var])):
                    count+=1
            if float(count)/float(len(data))>0.5:
                vars.append(var)
        names=[v.name for v in data.domain.variables+data.domain.getmetas().values()]
        names=filter(lambda n:OEParseSmiles(OEGraphMol(), n), names)
        return vars, names
        
    def setMoleculeSmilesCombo(self):
        candidates, self.defFragmentSmiles=self.filterSmilesVariables(self.molData)
        self.candidateMolSmilesAttr=candidates
        self.moleculeSmilesCombo.clear()
        self.moleculeSmilesCombo.insertStrList([v.name for v in candidates])
        if self.moleculeSmilesAttr>=len(candidates):
            self.moleculeSmilesAttr=0

    def setMoleculeTitleCombo(self):
        vars=self.molData.domain.variables+self.molData.domain.getmetas().values()
        self.moleculeTitleCombo.clear()
        self.moleculeTitleCombo.insertStrList(["None"]+[v.name for v in vars])
        if self.moleculeTitleAttr>len(vars):
            self.moleculeTitleAttr=0
        self.candidateMolTitleAttr=[None]+[v for v in vars]

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
        
    def fragmentSelection(self, index):
        self.selectedFragment=self.fragmentSmiles[index]
        if not self.showFragments and self.colorFragments:
            self.redrawImages()
        
    def renderImages(self):
        self.imageWidgets=[]
        if self.showFragments and self.fragmentSmiles:
            for i,fragment in enumerate(self.fragmentSmiles[1:]):
                imagename=self.imageprefix+str(i)+".bmp"
                #vis.molecule2BMP(fragment, imagename, self.imageSize)
                image=MolImage(self, self.molWidget, DrawContext(molecule=fragment, imagename=imagename, size=self.imageSize))
                self.gridLayout.addWidget(image, i/self.numColumns, i%self.numColumns)
                self.imageWidgets.append(image)
        elif self.molData:
            sAttr=self.candidateMolSmilesAttr[min(self.moleculeSmilesAttr, len(self.candidateMolSmilesAttr)-1)]
            tAttr=self.candidateMolTitleAttr[min(self.moleculeTitleAttr, len(self.candidateMolTitleAttr)-1)]
            if self.moleculeTitleAttr:
                titleList=[str(e[tAttr]) for e in self.molData if not e[sAttr].isSpecial()]
            else:
                titleList=[]
                if not sAttr:
                    return
            molSmiles=[str(e[sAttr]) for e in self.molData if not e[sAttr].isSpecial()]
            for i,(molecule, title) in enumerate(zip(molSmiles, titleList or [""]*len(molSmiles))):
                imagename=self.imageprefix+str(i)+".bmp"
                if self.colorFragments:
                    context=DrawContext(molecule=molecule, fragment=self.selectedFragment, imagename=imagename, size=self.imageSize, title=title)
                    #vis.moleculeFragment2BMP(molecule, self.selectedFragment, imagename, self.imageSize)
                else:
                    context=DrawContext(molecule=molecule, imagename=imagename, size=self.imageSize, title=title)
                    #vis.molecule2BMP(molecule, imagename, self.imageSize)
                image=MolImage(self, self.molWidget, context)
                self.gridLayout.addWidget(image, i/self.numColumns, i%self.numColumns)
                self.imageWidgets.append(image)
        #print "done drawing"
        for w in self.imageWidgets:
            w.show()

    def destroyImageWidgets(self):
        for w in self.imageWidgets:
            self.molWidget.removeChild(w)
        self.imageWidgets=[]
            
    def showImages(self):
        self.destroyImageWidgets()
        #print "destroyed"
        self.renderImages()

    def redrawImages(self):
        selected=map(lambda i:self.imageWidgets.index(i), filter(lambda i:i.selected, self.imageWidgets))
        self.showImages()
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
            fMap=orngChem.map_fragments([self.selectedFragment], molecules)
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
            fragmap=orngChem.map_fragments(fragments, molecules)
            match=filter(lambda m:max(fragmap[m].values()), molecules)
            examples=[e for e in self.molData if str(e[sAttr]) in match]
            table=orange.ExampleTable(examples)
            self.send("Selected Molecules", table)
        else:
            mols=[i.context.molecule for i in self.imageWidgets if i.selected]
            sAttr=self.candidateMolSmilesAttr[self.moleculeSmilesAttr]
            examples=[e for e in self.molData if str(e[sAttr]) in mols]
            table=orange.ExampleTable(examples)
            self.send("Selected Molecules", table)

    def keyPressEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=TRUE
        else:
            key.ignore()

    def keyReleaseEvent(self, key):
        if key.key()==Qt.Key_Control:
            self.ctrlPressed=FALSE
        else:
            key.ignore()        

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWMoleculeVisualizer()
    app.setMainWidget(w)
    w.show()
    data=orange.ExampleTable("E://fragG.tab")
    w.setMoleculeTable(data)
    data=orange.ExampleTable("E://chem//new//sf.tab")
    w.setFragmentTable(data)
    app.exec_loop()
