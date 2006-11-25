"""
<name>Pade</name>
<description>Computes local partial derivatives</description>
<icon>icons/Pade.png</icon>
<priority>3500</priority>
"""

import orange, statc
from OWWidget import *
import OWGUI
import os, string, math, profile
import numpy, math

from orangeom import star, dist

#pathQHULL = r"c:\qhull"
pathQHULL = r"c:\D\ai\Orange\test\squin\qhull"

class OWPade(OWWidget):

    settingsList = ["output", "outputAttr", "method", "derivativeAsMeta", "originalAsMeta", "savedDerivativeAsMeta", "differencesAsMeta", "enableThreshold", "threshold"]
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
        self.originalAsMeta = 1
        self.enableThreshold = 0
        self.threshold = 0.0
        self.method = 2
        
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Attributes", addSpace = True)
        lb = OWGUI.listBox(box, self, "dimensions", "attributes", selectionMode=QListBox.Multi, callback=self.dimensionsChanged)
        hbox = OWGUI.widgetBox(box, orientation=0)
        OWGUI.button(hbox, self, "All", callback=self.onAllAttributes)
        OWGUI.button(hbox, self, "None", callback=self.onNoAttributes)
        lb.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        lb.setMinimumSize(200, 200)

        box = OWGUI.widgetBox(self.controlArea, "Method", addSpace = True)
        OWGUI.comboBox(box, self, "method", callback = self.methodChanged, items = ["First Triangle", "Star Regression", "Tube Regression"])
                       
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
        OWGUI.checkBox(box, self, "originalAsMeta", label="Store original class as meta attribute")
        
        self.applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.apply)

        self.adjustSize()
        self.activateLoadedSettings()
        
        self.setFixedWidth(self.sizeHint().width())


    ## Triangulates the data

    def triangulate(self, points):
        num_points = len(points)
        pts1 = points
        f = file('input4qdelaunay.tab','w')
        f.write(reduce(lambda x, y: x+y, [str(len(self.contAttributes))+"\n"+str(len(pts1))+"\n"]+ [string.join([str(x) for x in pts1[i][:-1]],'\t')+'\n' for i in xrange(num_points)] )) # [str(pts1[i][0])+"\t"+str(pts1[i][1])+"\n" for i in xrange(num_points)]
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
        if not self.deltas:
            self.deltas = [[None] * len(self.contAttributes) for x in xrange(len(self.data))]

        dimensions = [d for d in self.dimensions if not self.deltas[0][d]]
        if not dimensions:
            return

        if not self.points:        
            self.points = orange.ExampleTable(orange.Domain(self.contAttributes, self.data.domain.classVar), self.data).native(0)
        points = self.points
        npoints = len(points)

        if not self.tri:
            print self.dimension
            self.tri = self.triangulate(points)
        tri = self.tri
            
        if not self.stars:
            self.stars = [star(x, tri) for x in xrange(npoints)]
        S = self.stars

        if not self.dts:        
            self.dts = [min([ min([ dist(points[x][:-1],points[v][:-1]) for v in simplex if v!=x]) for simplex in S[x]])*.1 for x in xrange(npoints)]

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
            

    def tubedRegression(self):
        if not self.deltas:
            self.deltas = [[None] * len(self.contAttributes) for x in xrange(len(self.data))]

        dimensions = [d for d in self.dimensions if not self.deltas[0][d]]
        if not dimensions:
            return

        if not self.findNearest:
            self.findNearest = orange.FindNearestConstructor_BruteForce(self.data, distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean())
            
        if not self.attrStat:
            self.attrStat = orange.DomainBasicAttrStat(self.data)

        self.progressBarInit()
        nExamples = len(self.data)
        nPoints = 100.0/nExamples/self.dimension

        normalizers = self.findNearest.distance.normalizers
        
        for di, d in enumerate(dimensions):
            contIdx = self.contIndices[d]

            minV, maxV = self.attrStat[contIdx].min, self.attrStat[contIdx].max
            if minV == maxV:
                continue
            span = (maxV - minV) /2 / math.log(.001)
            
            oldNormalizer = normalizers[self.contIndices[d]]
            normalizers[self.contIndices[d]] = 0

            for exi, ref_example in enumerate(self.data):
                if ref_example[contIdx].isSpecial():
                    self.deltas[exi][d] = "?"
                    continue

                ref_x = float(ref_example[contIdx])

                Sx = Sy = Sxx = Syy = Sxy = n = 0.0
                
                for ex in self.findNearest(ref_example, 0, True):
                    if ex[contIdx].isSpecial():
                        continue
                    ex_x = float(ex[contIdx])
                    ex_y = float(ex.getclass())
                    w = math.exp(-((ex_x-ref_x)/span)**2)
                    Sx += w * ex_x
                    Sy += w * ex_y
                    Sxx += w * ex_x**2
                    Syy += w * ex_y**2
                    Sxy += w * ex_x * ex_y
                    n += w

                div = n*Sxx-Sx**2
                if div and n>=3:# and i<40:
                    b = (Sxy*n - Sx*Sy) / div
##                    a = (Sy - b*Sx)/n
##                    err = (n * a**2 + b**2 * Sxx + Syy + 2*a*b*Sx - 2*a*Sy - 2*b*Sxy)
##                    tot = Syy - Sy**2/n
##                    mod = tot - err
##                    merr = err/(n-2)
##                    F = mod/merr
##                    Fprob = statc.fprob(F, 1, int(n-2))
#                        print "%.4f" % Fprob,
                    #print ("%.3f\t" + "%.0f\t"*6 + "%f\t%f") % (w, ref_x, ex_x, n, a, b, merr, F, Fprob)
                    self.deltas[exi][d] = b
                else:
                    self.deltas[exi][d] = "?"

                self.progressBarSet((nExamples*di+exi)*nPoints)
            
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


    def methodChanged(self):
        self.deltas = None
        
    def onDataInput(self, data):
        self.closeContext()
        self.data = data
        if data:
            self.npoints = len(data)

            attributes = self.data.domain.attributes            
            self.contIndices = [i for i, attr in enumerate(attributes) if attr.varType == orange.VarTypes.Continuous]
            self.contAttributes = [attributes[i] for i in self.contIndices]
            self.attributes = [(attr.name, attr.varType) for attr in self.contAttributes]
            self.dimensions = range(len(self.attributes))
            self.dimension = len(self.dimensions)

            icons = OWGUI.getAttributeIcons()
            self.outputLB.clear()
            for attr in self.contAttributes:
                self.outputLB.insertItem(icons[attr.varType], attr.name)
           
        else:
            self.attributes = []
            self.contAttributes = []
            self.dimensions = []
            self.dimension = 0
            self.npoints = 0

        self.points = self.tri = self.stars = self.dts = self.deltas = None
        self.findNearest = self.attrStat = None
        
        self.openContext("", data)
            
        self.dimensionsChanged()


    def apply(self):
        import orngMisc
        data = self.data
        if not data:
            self.send("Examples", None)
            return

        self.dimension = len(self.dimensions)
        [self.D, None, self.tubedRegression][self.method]()

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

        if self.originalAsMeta:
            originalID = orange.newmetaid()
            dom.addmeta(originalID, self.data.domain.classVar)
                
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
