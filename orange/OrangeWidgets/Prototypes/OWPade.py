"""
<name>Pade</name>
<description>Computes local partial derivatives</description>
<icon>icons/Pade.png</icon>
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

class OWPade(OWWidget):

    settingsList = ["output", "outputAttr", "derivativeAsMeta", "savedDerivativeAsMeta", "differencesAsMeta", "enableThreshold", "threshold"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", DomainContextHandler.SelectedRequiredList, selected="dimensions")])}

    def __init__(self, parent = None, signalManager = None, name = "Pade"):
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


    ## Triangulates the data

    def triangulate(self, points):
        num_points = len(points)
        pts1 = points
        f = file('input4qdelaunay.tab','w')
        f.write(reduce(lambda x, y: x+y, [str(self.dimension)+"\n"+str(len(pts1))+"\n"]+ [string.join([str(x) for x in pts1[i][:-1]],'\t')+'\n' for i in xrange(num_points)] )) # [str(pts1[i][0])+"\t"+str(pts1[i][1])+"\n" for i in xrange(num_points)]
        f.close()
        os.system(pathQHULL + r"\qdelaunay s i Qt TO 'outputFromQdelaunay.tab' < input4qdelaunay.tab")
        f = file('outputFromQdelaunay.tab','r')
        vhod = f.read()
        f.close()
        k = string.find(vhod,'\n')
        num_of_triangles = int(vhod[0:k])
        vhod = vhod[k+1:]
        l = string.split(vhod,' \n')
        return [map(int, string.split(l[i],' ')) for i in xrange(num_of_triangles+1) if l[i]!='']


    ## Support functions
    
    # Returns the simplex to which the point xn belongs
    def simplex_with_xn(self, xn,Star):
        for simplex in Star:
            bl = [numpy.linalg.det(a) for a in self.inside(xn,simplex)]
            if reduce(lambda x,y: x and y, [i<0 for i in bl]) or reduce(lambda x,y: x and y, [i>0 for i in bl]):
                return simplex
        return None

    # Replaces one matrix column (used by simplex_with_xn)
    def change(self, i,j,n):
        if i==j:
            return n+[1]
        return self.points[j][:-1]+[1]

    # Prepares matrices for simplex_with_xn
    def inside(self, vertex,simplex):
        return [numpy.array([self.change(i,j,vertex) for j in simplex]) for i in simplex]


    ## Computes the derivatives in required dimensions (self.dimensions),
    ##   except for those which have been already computed and cached
    
    def D(self):
        dimensions = [d for d in self.dimensions if not self.deltas[0][d]]
        if not dimensions:
            return
        
        points = self.points
        nPoints = 100.0/len(points)

        self.progressBarInit()
        for x, (S, xp, dt, deltas) in enumerate(zip(self.stars, points, self.dts, self.deltas)):
            for d in dimensions:

                # find the simplex in the given direction             
                xn = xp[:-1]
                O = numpy.array(xp[:-1])

                xn[d] += dt
                swx = self.simplex_with_xn(xn, S)
                if swx:                
                    obrni = 1
                else:
                    xn[d] = xp[d]-dt
                    swx = self.simplex_with_xn(xn, S)
                    if swx:
                        obrni = -1
                    else:
                        deltas[d] = "?"
                        continue

                # Interpolate the function value at the point on the simplex
                vecs = numpy.array([numpy.array(points[p][:-1])-O for p in swx if p!=x])
                vecs = vecs.transpose()
                XN = numpy.array(xn)-O
                coef = numpy.linalg.solve(vecs,XN)
                xnz = sum(coef*[numpy.array(points[p][-1]-xp[-1])for p in swx if p!=x])+xp[-1]

                # Store the derivative                
                deltas[d] = obrni * (xnz-xp[-1])

            self.progressBarSet(x*nPoints)
            
        self.progressBarFinished()
            

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

            points = self.points = orange.ExampleTable(orange.Domain(contAttributes, self.data.domain.classVar), data).native(0)
            npoints = len(points)
            tri = self.tri = self.triangulate(self.points)
            S = self.stars = [star(x, tri) for x in xrange(npoints)]
            self.dts = [min([ min([ dist(points[x][:-1],points[v][:-1]) for v in simplex if v!=x]) for simplex in S[x]])*.1 for x in xrange(npoints)]
            
            self.deltas = [[None] * self.dimension for x in xrange(npoints)]
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
        self.D()            

        self.actThreshold = self.enableThreshold and abs(self.threshold)

        mpart = "(" + ",".join([self.attributes[i][0] for i in self.dimensions]) + ")"
        
        if self.output:
            classVar = orange.FloatVariable("d"+self.attributes[self.outputAttr][0])
        else:
#            classVar = orange.EnumVariable("Q", values = [", ".join([self.attributes[i][0]+"+-oX"[x] for i, x in enumerate(v)]) for v in orngMisc.LimitedCounter([4]*self.dimension)])
            classVar = orange.EnumVariable("Q", values = ["M"+ "".join(["+-oX"[x] for x in v]) + mpart for v in orngMisc.LimitedCounter([4]*self.dimension)])
#            classVar = orange.EnumVariable("Q", values = ["M" + "".join(["+-oX"[x] for x in v]) + mpart for v in orngMisc.LimitedCounter([4]*self.dimension)])
            
        dom = orange.Domain(data.domain.attributes, classVar)

        if self.derivativeAsMeta:
            derivativeID = orange.newmetaid()
#            dom.addmeta(derivativeID, orange.EnumVariable("Q", values = [", ".join([self.attributes[i][0]+"+-oX"[x] for i, x in enumerate(v)]) for v in orngMisc.LimitedCounter([4]*self.dimension)]))
            dom.addmeta(derivativeID, orange.EnumVariable("Q", values = ["M" + "".join(["+-oX"[x] for x in v]) + mpart for v in orngMisc.LimitedCounter([4]*self.dimension)]))
#            dom.addmeta(derivativeID, orange.EnumVariable("Q", values = ["M" + "".join(["+-oX"[x] for x in v]) + mpart for v in orngMisc.LimitedCounter([4]*self.dimension)]))
            
        metaIDs = []        
        if self.differencesAsMeta:
            for dim in self.dimensions:
                metaVar = orange.FloatVariable("d"+self.attributes[dim][0])
                metaID = orange.newmetaid()
                dom.addmeta(metaID, metaVar)
                metaIDs.append(metaID)
                
        paded = orange.ExampleTable(dom, data)

        for pad, alldeltas in zip(paded, self.deltas):
            deltas = [alldeltas[d] for d in self.dimensions]
            if self.output:
                pad.setclass(alldeltas[self.outputAttr])
            else:
#                pad.setclass("M" + "".join([((delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o") for delta in deltas]) + mpart)
                pad.setclass("M" + "".join([((delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o") for delta in deltas]) + mpart)
#                pad.setclass(", ".join([self.attributes[i][0]+((delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o") for i, delta in enumerate(deltas)]))

            if self.derivativeAsMeta:
#                pad.setmeta(derivativeID, "".join([(delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o" for delta in deltas]))
                pad.setmeta(derivativeID, "M" + "".join([(delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o" for delta in deltas]) + mpart)
#                pad.setmeta(derivativeID, "M"+ "".join([(delta > self.threshold and "+") or (delta < -self.threshold and "-") or (delta == "?" and delta) or "o" for delta in deltas]) + mpart)

            if self.differencesAsMeta:
                for a in zip(metaIDs, deltas):
                    pad.setmeta(*a)
                    
        self.send("Examples", paded)

                            
       
if __name__=="__main__":
    import sys

    a=QApplication(sys.argv)
    ow=OWPade()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(orange.ExampleTable(r"c:\D\ai\Orange\test\squin\xyz-t"))
#    ow.onDataInput(orange.ExampleTable(r"c:\delo\qing\smartquin\x2y2.txt"))
    a.exec_loop()
    
    ow.saveSettings()
