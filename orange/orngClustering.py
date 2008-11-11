import orange
import math, sys

def orderLeafs(tree, matrix):
    """Order the leafs in the clustering tree (based on 'Fast optimal leaf ordering for herarchical clustering. Ziv Bar-Joseph et al.')
    Arguments:
        tree   --binary hierarchical clustering tree of type orange.HierarchicalCluster
        matrix --orange.SymMatrix that was used to compute the clustering
    """
    objects = getattr(tree.mapping, "objects", None)
    tree.mapping.setattr("objects", range(len(tree)))
    M = {}
    ordering = {}
    def _optOrdering(tree):
        if len(tree)==1:
            for leaf in tree:
                M[tree, leaf, leaf] = 0
##                print "adding:", tree, leaf, leaf
        else:
            _optOrdering(tree.left)
            _optOrdering(tree.right)
##            print "ordering", [i for i in tree]
            Vl = set(tree.left)
            Vr = set(tree.right)
            Vlr = set(tree.left.right or tree.left)
            Vll = set(tree.left.left or tree.left)
            Vrr = set(tree.right.right or tree.right)
            Vrl = set(tree.right.left or tree.right)
            other = lambda e, V1, V2: V2 if e in V1 else V1
            for u in Vl:
                for w in Vr:
                    if True: #Improved search
                        C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
                        orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree.left, u, m])
                        orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree.right, w, k])
                        k0 = orderedKs[0]
                        curMin = 1e30000 
                        curMK = ()
                        for m in orderedMs:
                            if M[tree.left, u, m] + M[tree.right, w, k0] + C >= curMin:
                                break
                            for k in  orderedKs:
                                if M[tree.left, u, m] + M[tree.right, w, k] + C >= curMin:
                                    break
                                if curMin > M[tree.left, u, m] + M[tree.right, w, k] + matrix[m, k]:
                                    curMin = M[tree.left, u, m] + M[tree.right, w, k] + matrix[m, k]
                                    curMK = (m, k)
                        M[tree, u, w] = M[tree, w, u] = curMin
                        ordering[tree, u, w] = (tree.left, u, curMK[0], tree.right, w, curMK[1])
                        ordering[tree, w, u] = (tree.right, w, curMK[1], tree.left, u, curMK[0])
                    else:
                        def MFunc((m, k)):
                            return M[tree.left, u, m] + M[tree.right, w, k] + matrix[m, k]
                        m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
                        M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
                        ordering[tree, u, w] = (tree.left, u, m, tree.right, w, k)
                        ordering[tree, w, u] = (tree.right, w, k, tree.left, u, m)
        
    _optOrdering(tree)

    def _order(tree, u, w):
        if len(tree)==1:
            return
        left, u, m, right, w, k = ordering[tree, u, w]
        if len(left)>1 and m not in left.right:
            left.swap()
        _order(left, u, m)
##        if u!=left[0] or m!=left[-1]:
##            print "error 4:", u, m, list(left)
        if len(right)>1 and k not in right.left:
            right.swap()
        _order(right, k, w)
##        if k!=right[0] or w!=right[-1]:
##            print "error 5:", k, w, list(right)
    
    u, w = min([(u, w) for u in tree.left for w in tree.right], key=lambda (u, w): M[tree, u, w])
    
##    print "M(v) =", M[tree, u, w]
    
    _order(tree, u, w)

    def _check(tree, u, w):
        if len(tree)==1:
            return
        left, u, m, right, w, k = ordering[tree, u, w]
        if tree[0] == u and tree[-1] == w:
            _check(left, u, m)
            _check(right, k, w)
        else:
            print "Error:", u, w, tree[0], tree[-1]

    _check(tree, u ,w)            
        

    if objects:
        tree.mapping.setattr("objects", objects)

import Image, ImageDraw, ImageFont

class DendrogramPlot(object):
    defaultFontSize = 12
    defaultTreeColor = (0, 0, 0)
    defaultTextColor = (100, 100, 100)
    defaultMatrixOutlineColor = (240, 240, 240)
    def __init__(self, tree, data=None, labels=None, width=None, height=None, treeAreaWidth=None, textAreaWidth=None, matrixAreaWidth=None, fontSize=None, lineWidth=2, painter=None, clusterColors={}):
        self.tree = tree
        self.data = data
        self.labels = labels
        self.width = width
        self.height = height
        self.painter = painter
        self.treeAreaWidth = treeAreaWidth
        self.textAreaWidth = textAreaWidth
        self.matrixAreaWidth = matrixAreaWidth
        self.fontSize = fontSize
        self.lineWidth = lineWidth
        self.clusterColors = clusterColors
        self.lowColor = (0, 0, 0)
        self.hiColor = (255, 255, 255)
        if not self.labels:
            self.labels = [str(m) for m in getattr(self.tree.mapping, "objects", [""]*len(tree))]

    def _getTextSizeHint(self, text):
        if type(text)==str:
            return self.font.getsize(text)
        elif type(text)==list:
            return (max([self.font.getsize(t)[0] for t in text]), max([self.font.getsize(t)[1] for t in text]))

    def _getMatrixRowSizeHint(self, height, max=None):
        if not self.data:
            return (0, 0)
        if max==None:
            return (len(self.data.domain.attributes)*height, height)
        else:
            return (min(len(self.data.domain.attributes)*height, max), height)
            
    def _getLayout(self, labels):
        fontSize = self.fontSize or self.defaultFontSize
        if self.height:
            height = self.height
            fontSize = (height-20)/len(labels)
        else:
            height = 20+fontSize*len(labels)
        try:
            self.font = ImageFont.truetype("cour.ttf", fontSize)
        except:
            self.font = ImageFont.load_default()
            fontSize = self._getTextSizeHint("ABCDEF")[1]
        emptySpace = 4*10
        textWidth, textHeight = self._getTextSizeHint(labels)
        if self.width:
            width = self.width
            textAreaWidth = min(textWidth, (width-emptySpace)/3)
            matrixAreaWidth = self._getMatrixRowSizeHint(fontSize, (width-emptySpace)/3)[0]
            treeAreaWidth = width-emptySpace-textAreaWidth-matrixAreaWidth
        else:
            matrixAreaWidth = self._getMatrixRowSizeHint(fontSize, 400)[0]
            textAreaWidth = textWidth
            treeAreaWidth = 400
            width = treeAreaWidth+textAreaWidth+matrixAreaWidth+emptySpace
        return width, height, treeAreaWidth, textAreaWidth, matrixAreaWidth, fontSize, fontSize

    def SetLayout(self, width=None, height=None): #, treeAreaWidth=None, textAreaWidth=None, matrixAreaWidth=None):
        """Set the layout of the dendrogram. 
        """
        self.height = height
        self.width = width
##        self.treeAreaWidth = treeAreaWidth
##        self.textAreaWidth = textAreaWidth
##        self.matrixAreaWidth = matrixAreaWidth

    def SetMatrixColorScheme(self, low, hi):
        """Set the matrix color scheme. low and hi must be (r, g, b) tuples
        """
        self.lowColor = low
        self.hiColor = hi

    def SetClusterColors(self, clusterColors={}):
        """clusterColors must be a dictionary with cluster instances as keys and (r, g, b) tuples as items.
        """
        self.clusterColors = clusterColors
        
    def _getColorScheme(self, gamma=1.0):
        vals = [float(val) for ex in self.data for val in ex if not val.isSpecial() and val.variable.varType==orange.VarTypes.Continuous] or [0]
        avg = sum(vals)/len(vals)
        maxVal, minVal = max(vals), min(vals)
        def colorScheme(val):
            if val.isSpecial():
                return None
            elif val.variable.varType==orange.VarTypes.Continuous:
##                r = g = b = int(255.0*(float(val)-avg)/abs(maxVal-minVal))
                r, g, b = [int(self.lowColor[i]+(self.hiColor[i]-self.lowColor[i])*(float(val)-minVal)/abs(maxVal-minVal)) for i in range(3)]
            elif val.variable.varType==orange.VarTypes.Discrete:
                r = g = b = int(255.0*float(val)/len(val.variable.values))
            return (r, g, b)
        return colorScheme

    def _initPainter(self, w, h):
        self.image = Image.new("RGB", (w, h), color=(255, 255, 255))
        self.painter = ImageDraw.Draw(self.image)

    def _truncText(self, text, width):
        while text:
            if self._getTextSizeHint(text)[0]>width:
                text = text[:-1]
            else:
                break
        return text
                
    
    def Plot(self, file="graph.png"):
        """Draws the dendrogram and saves it to file
        """
        if type(file)==str:
            file = open(file, "wb")
        topMargin = 10
        bottomMargin = 10
        leftMargin = 10
        rightMargin = 10
        width, height, treeAreaWidth, textAreaWidth, matrixAreaWidth, hAdvance, fontSize = self._getLayout(self.labels)
        treeAreaStart = leftMargin
        textAreaStart = treeAreaStart+treeAreaWidth+leftMargin
        matrixAreaStart = textAreaStart+textAreaWidth+leftMargin
        self.globalHeight = topMargin
        globalTreeHeight = self.tree.height
        if not self.painter:
            self._initPainter(width, height)
        def _drawTree(tree, color=None):
            treeHeight = treeAreaStart+(1-tree.height/globalTreeHeight)*treeAreaWidth
            color = self.clusterColors.get(tree, color or self.defaultTreeColor)
            if tree.branches:
                subClusterPoints = []
                for t in tree.branches:
                    point, cc = _drawTree(t, color)
                    self.painter.line([(treeHeight, point[1]), point], fill=cc, width=self.lineWidth)
                    subClusterPoints.append(point)
                self.painter.line([(treeHeight, subClusterPoints[0][1]), (treeHeight, subClusterPoints[-1][1])], fill=color, width=self.lineWidth)
                return (treeHeight, (subClusterPoints[0][1]+subClusterPoints[-1][1])/2), color
            else:
                self.globalHeight+=hAdvance
                return (treeAreaStart+treeAreaWidth, self.globalHeight-hAdvance/2), color
        _drawTree(self.tree)
        if self.data:
            colorSheme = self._getColorScheme()
            cellWidth = float(matrixAreaWidth)/len(self.data.domain.attributes)
            def _drawMatrixRow(ex, yPos):
                for i, attr in enumerate(ex.domain.attributes):
                    col = colorSheme(ex[attr])
                    if col:
                        if cellWidth>4:
                            self.painter.rectangle([(int(matrixAreaStart+i*cellWidth), yPos), (int(matrixAreaStart+(i+1)*cellWidth), yPos+hAdvance)], fill=colorSheme(ex[attr]), outline=self.defaultMatrixOutlineColor)
                        else:
                            self.painter.rectangle([(int(matrixAreaStart+i*cellWidth), yPos), (int(matrixAreaStart+(i+1)*cellWidth), yPos+hAdvance)], fill=colorSheme(ex[attr]))
                    else:
                        pass #TODO indicate a missing value
##        for i, (label, row) in enumerate(zip(labels, matrix)):

        rows = []        
        for i, el in enumerate(self.tree):
            el = self.tree.mapping[i] # in case mapping has objects and el is not an integer
            label = self.labels[el]
##            print label, el, i
            try:
                self.painter.text((textAreaStart, topMargin+i*hAdvance), self._truncText(label, textAreaWidth), font=self.font, fill=self.defaultTextColor)
            except IOError, err:
                print err
                print label
            if self.data:
                row = self.data[el]
                rows.append(row)
                _drawMatrixRow(row, topMargin+i*hAdvance)
##        if self.data:
##            import orangene
##            map = orangene.HeatmapConstructor(orange.ExampleTable(rows), None) 
        self.image.save(file, "PNG")
    

if __name__=="__main__":
##    data = orange.ExampleTable("doc//datasets//brown-selected.tab")
##    data = orange.ExampleTable("doc//datasets//iris.tab")
    data = orange.ExampleTable("doc//datasets//zoo.tab")
##    data = orange.ExampleTable("doc//datasets//titanic.tab")
##    m = [[], [ 3], [ 2, 4], [17, 5, 4], [ 2, 8, 3, 8], [ 7, 5, 10, 11, 2], [ 8, 4, 1, 5, 11, 13], [ 4, 7, 12, 8, 10, 1, 5], [13, 9, 14, 15, 7, 8, 4, 6], [12, 10, 11, 15, 2, 5, 7, 3, 1]]
##    matrix = orange.SymMatrix(m)
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
##    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])
    root = orange.HierarchicalClustering(matrix, linkage=orange.HierarchicalClustering.Average)
##    root.mapping.objects = [str(ex.getclass()) for ex in data]
    d = DendrogramPlot(root, data=data, labels=[str(ex.getclass()) for ex in data], width=500, height=2000)
    d.SetMatrixColorScheme((0, 255, 0), (255, 0, 0))
##    d.SetClusterColors({root.left:(0,255,0), root.right:(0,0,255)})
    d.Plot("graph.png")
    print "Sum:", sum([matrix[root.mapping[i], root.mapping[i+1]] for i in range(len(root.mapping)-1)])
    orderLeafs(root, matrix)
    print "Sum:", sum([matrix[root.mapping[i], root.mapping[i+1]] for i in range(len(root.mapping)-1)])
    d = DendrogramPlot(root, data=data, labels=[str(ex.getclass()) for ex in data], width=500, height=2000, lineWidth=1)
    d.SetMatrixColorScheme((0, 255, 0), (255, 0, 0))
    d.Plot("graphOrdered.png")
    
    
        
            
