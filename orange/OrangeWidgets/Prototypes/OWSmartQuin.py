"""
<name>Smart Quin</name>
<description>Computes local partial derivatives</description>
<icon>icons/Smart Quin.png</icon>
<priority>3500</priority>
"""

import orange
from OWWidget import *
import OWGUI
import os, string, math, profile
import numpy

from orangeom import star, dist

#pathQHULL = r"c:\qhull"
pathQHULL = r"c:\D\ai\Orange\test\squin\qhull"

class OWSmartQuin(OWWidget):

    settingsList = ["output", "outputAttr", "derivativeAsMeta", "savedDerivativeAsMeta", "differencesAsMeta", "enableThreshold", "threshold"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", DomainContextHandler.SelectedRequiredList, selected="dimensions")])}

    def __init__(self, parent = None, signalManager = None, name = "Smart Quin"):
        OWWidget.__init__(self, parent, signalManager, name)  #initialize base class
        self.inputs = [("Examples", ExampleTableWithClass, self.onDataInput)]
        self.outputs = [("Examples", ExampleTableWithClass)]

        self.attributes = []
        self.dimensions = []
        self.output = 0
        self.outputAttr = 0
        self.derivativeAsMeta = 0
        self.savedDerivativeAsMeta = 0
        self.differencesAsMeta = 1
        self.enableThreshold = 0
        self.threshold = 0.0
        
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Attributes", addSpace = True)
        lb = OWGUI.listBox(box, self, "dimensions", "attributes", selectionMode=QListBox.Multi, callback=self.dimensionsChanged)
        hbox = OWGUI.widgetBox(box, orientation=0)
        OWGUI.button(hbox, self, "All", callback=self.onAllAttributes)
        OWGUI.button(hbox, self, "None", callback=self.onNoAttributes)
        lb.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        lb.setMinimumSize(200, 200)

        box = OWGUI.widgetBox(self.controlArea, "Threshold", orientation=0, addSpace = True)
        threshCB = OWGUI.checkBox(box, self, "enableThreshold", "Ignore differences below ")
        ledit = OWGUI.lineEdit(box, self, "threshold", valueType=float, validator=QDoubleValidator(0, 1e30, 0, self, ""))
        threshCB.disables.append(ledit)
        threshCB.makeConsistent()
        
        box = OWGUI.radioButtonsInBox(self.controlArea, self, "output", ["Qualitative derivative", "Quantitative differences"], box="Output", addSpace = True, callback=self.dimensionsChanged)
        self.outputLB = OWGUI.comboBox(OWGUI.indentedBox(box), self, "outputAttr", callback=self.outputDiffChanged)
        OWGUI.separator(box)
        self.metaCB = OWGUI.checkBox(box, self, "derivativeAsMeta", label="Store q-derivative as meta attribute")
        OWGUI.checkBox(box, self, "differencesAsMeta", label="Store differences as meta attributes")
        
        self.applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.apply)

        self.adjustSize()
        self.activateLoadedSettings()
        
        self.setFixedWidth(self.sizeHint().width())


    def triangulate(self, points):
        num_points = points[0]
        pts1 = points[1:]
        #print [string.join([str(x) for x in pts1[i][:-1]],'\t')+'\n' for i in xrange(num_points)]
        f = file('input4qdelaunay.tab','w')
        f.write(reduce(lambda x, y: x+y, [str(self.dimension)+"\n"+str(len(pts1))+"\n"]+ [string.join([str(x) for x in pts1[i][:-1]],'\t')+'\n' for i in xrange(num_points)] )) # [str(pts1[i][0])+"\t"+str(pts1[i][1])+"\n" for i in xrange(num_points)]
        f.close()
        os.system(pathQHULL + r"\qdelaunay s i Qt TO 'outputFromQdelaunay.tab' < input4qdelaunay.tab")
        f = file('outputFromQdelaunay.tab','r')
        vhod = f.read()
        f.close()
        k = string.find(vhod,'\n')
        num_of_triangles = int(vhod[0:k])
        #print "num_of_triangles:",num_of_triangles
        vhod = vhod[k+1:]
        # sparsamo vhod, da lahko dolocimo robne tocke, ki jih hocemo povezat s tocko X
        #print "vhod",vhod
        l = string.split(vhod,' \n')
        # indeksom pristejemo 1, da zacnemo steti z 1 in ne z 0
        return [map(lambda x: int(x)+1, string.split(l[i],' ')) for i in xrange(num_of_triangles+1) if l[i]!='']
        #return tri

    def change(self, i,j,n):
        if i==j:
            return n+[1]
        return self.points[j][:-1]+[1]

    def inside(self, vertex,simplex):
        # tole se da se optimizirat: sestavit je treba seznam determinant n x n matrik, v katerih je potrebno vedno
        # nadomestiti i-to vrstico, i=1..n
        return [numpy.array([self.change(i,j,vertex) for j in simplex]) for i in simplex]

    def simplex_with_xn(self, xn,Star):
    #    print "->"
        for simplex in Star:
            #print inside(xn,simplex)
            bl = [numpy.linalg.det(a) for a in self.inside(xn,simplex)]
            #,[points[i] for i in simplex]
    #        print simplex, "bl = ", bl, reduce(lambda x,y: x and y, [i<0 for i in bl]) or reduce(lambda x,y: x and y, [i>0 for i in bl])
            if reduce(lambda x,y: x and y, [i<0 for i in bl]) or reduce(lambda x,y: x and y, [i>0 for i in bl]):
                return simplex
        return None

    def D(self, x):
        points = self.points
        S = star(x, self.tri)
        xp = points[x]
        # dt bi morda se malo popravili, da bi tocka gotovo lezala znotraj trikotnika
        dt = min([ min([ dist(points[x][:-1],points[v][:-1]) for v in simplex if v!=x]) for simplex in S])*.1
        odvodi = ''
        deltas = []
        for d in self.dimensions:
            obrni = False
            xn = xp[:-1]
            O = numpy.array(xp[:-1])
            xn[d] += dt
            swx = self.simplex_with_xn(xn,S)
            if swx==None:
                xn[d] = xp[d]-dt
                swx = self.simplex_with_xn(xn,S) # pazi: obrnit je treba predznake
                obrni = True
            # ce v obeh smereh ni simpleksa, ki bi ga poltrak sekal, pogledamo najblizjo tocko v zvezdi in skopiramo njene lastnosti
            if swx==None: # popravi, da bo kot pise v zgornjem komentarju!
                deltas.append(0)
                odvodi += 'X'
                # MOZNA RESITEV ZA TE PRIMERE: zapomni si jih in jim na koncu priredi vecinski razred sosedov iz zvezde
                # Tule to se ni mozno, ker se lahko zgodi, da se niso izracunani!
                continue
            vecs = numpy.array([numpy.array(points[p][:-1])-O for p in swx if p!=x])
            vecs = vecs.transpose()
            XN = numpy.array(xn)-O
            coef = numpy.linalg.solve(vecs,XN)
            xnz = sum(coef*[numpy.array(points[p][-1]-xp[-1])for p in swx if p!=x])+xp[-1]
            delta = xnz-xp[-1]
            deltas.append(delta)
            if delta > self.actThreshold:
                odvodi += "+-"[obrni]
            elif delta < -self.actThreshold:
                odvodi += "-+"[obrni]
            else:
                odvodi += 'o'
        return (odvodi,deltas)
            

    def onAllAttributes(self):
        self.dimensions = range(len(self.attributes))
        self.dimensionsChanged()

    def onNoAttributes(self):
        if not self.output:
            self.dimensions = []
        else:
            self.dimensions = [self.outputAttr]
        self.dimensionsChanged()

    def outputDiffChanged(self):
        if not self.output:
            self.output = 1
        self.dimensionsChanged()
        
    def dimensionsChanged(self):
        if self.output:
            if self.outputAttr not in self.dimensions:
                self.dimensions.append(self.outputAttr)
                self.dimensions.sort()
            if not self.metaCB.isEnabled():
                self.derivativeAsMeta = self.savedDerivativeAsMeta
                self.metaCB.setEnabled(True)
        else:
            if self.metaCB.isEnabled():
                self.savedDerivativeAsMeta = self.derivativeAsMeta
                self.derivativeAsMeta = 0
                self.metaCB.setEnabled(False)

        self.applyButton.setEnabled(bool(self.dimensions))

    def onDataInput(self, data):
        self.closeContext()
        self.data = data
        if data:
            contAttributes = [attr for attr in self.data.domain.attributes if attr.varType == orange.VarTypes.Continuous]
            self.attributes = [(attr.name, attr.varType) for attr in contAttributes]
            self.dimensions = range(len(self.attributes))
            self.dimension = len(self.dimensions)

            icons = OWGUI.getAttributeIcons()
            self.outputLB.clear()
            for attr in contAttributes:
                self.outputLB.insertItem(icons[attr.varType], attr.name)

            self.points = [len(data)] + orange.ExampleTable(orange.Domain(contAttributes, self.data.domain.classVar), data).native(0)
            self.tri = self.triangulate(self.points)
        else:
            self.attributes = []
            self.dimensions = []

        self.openContext("", data)
            
        self.dimensionsChanged()

    def apply(self):
        import orngMisc
        data = self.data
        if not data:
            self.send("Examples", None)
            return

        self.dimension = len(self.dimensions)
        self.actThreshold = self.enableThreshold and abs(self.threshold)
            
        if self.output:
            classVar = orange.FloatVariable("d"+self.attributes[self.outputAttr][0])
        else:
            classVar = orange.EnumVariable("Q", values = ["".join(["+-oX"[x] for x in v]) for v in orngMisc.LimitedCounter([4]*self.dimension)])
            
        dom = orange.Domain(data.domain.attributes, classVar)

        if self.derivativeAsMeta:
            derivativeID = orange.newmetaid()
            dom.addmeta(derivativeID, orange.EnumVariable("Q", values = ["".join(["+-oX"[x] for x in v]) for v in orngMisc.LimitedCounter([4]*self.dimension)]))
            
        metaIDs = []        
        if self.differencesAsMeta:
            for dim in self.dimensions:
                metaVar = orange.FloatVariable("d"+self.attributes[dim][0])
                metaID = orange.newmetaid()
                dom.addmeta(metaID, metaVar)
                metaIDs.append(metaID)
                
        quined = orange.ExampleTable(dom, data)

        self.progressBarInit()
        nPoints = 100.0/(len(self.points)+1)
        for x in xrange(1, len(self.points)):
            D = self.D(x)
            if self.output:
                quined[x-1].setclass(D[1][self.outputAttr])
            else:
                quined[x-1].setclass(D[0])

            if self.derivativeAsMeta:
                quined[x-1].setmeta(derivativeID, D[0])

            if self.differencesAsMeta:
                for a in zip(metaIDs, D[1]):
                    quined[x-1].setmeta(*a)
                    
            self.progressBarSet(x*nPoints)
        self.progressBarFinished()
        self.send("Examples", quined)

                            
       
if __name__=="__main__":
    import sys

    a=QApplication(sys.argv)
    ow=OWSmartQuin()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(orange.ExampleTable(r"c:\D\ai\Orange\test\squin\test1-t"))
#    ow.onDataInput(orange.ExampleTable(r"c:\delo\qing\smartquin\x2y2.txt"))
    a.exec_loop()
    
    ow.saveSettings()
