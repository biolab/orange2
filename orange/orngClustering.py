import orange
import math

import Image, ImageDraw, ImageFont


class DendrogramPlot(object):
    defaultFontSize = 12
    defaultTreeColor = (255, 0, 0)
    defaultTextColor = (100, 100, 100)
    defaultMatrixOutlineColor = (240, 240, 240)
    def __init__(self, tree, data=None, width=None, height=None, treeAreaWidth=None, textAreaWidth=None, matrixAreaWidth=None, fontSize=None, painter=None, clusterColors={}):
        self.tree = tree
        self.data = data
        self.width = width
        self.height = height
        self.painter = painter
        self.treeAreaWidth = treeAreaWidth
        self.textAreaWidth = textAreaWidth
        self.matrixAreaWidth = matrixAreaWidth
        self.fontSize = fontSize
        self.clusterColors = clusterColors
        self.lowColor = (0, 0, 0)
        self.hiColor = (255, 255, 255)

    def _getTextSizeHint(self, text):
        if type(text)==str:
            return self.font.getsize(text)
        elif type(text)==list:
            return (max([self.font.getsize(t)[0] for t in text]), max([self.font.getsize(t)[1] for t in text]))

    def _getMatrixRowSizeHint(self, height, max=None):
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
        
    def _getColorScheme(self):
        vals = [float(val) for ex in self.data for val in ex if not val.isSpecial() and val.variable.varType==orange.VarTypes.Continuous] or [0]
        avg = sum(vals)/len(vals)
        maxVal, minVal = max(vals), min(vals)
        def colorScheme(val):
            if val.isSpecial():
                return None
            elif val.variable.varType==orange.VarTypes.Continuous:
##                r = g = b = int(255.0*(float(val)-avg)/abs(maxVal-minVal))
                r, g, b = [int(self.lowColor[i]+(self.hiColor[i]-self.lowColor[i])*(float(val)-avg)/abs(maxVal-minVal)) for i in range(3)]
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
        labels = [str(ex.getclass()) for ex in data] # TODO: get arbitrary labels
        colorSheme = self._getColorScheme()
        width, height, treeAreaWidth, textAreaWidth, matrixAreaWidth, hAdvance, fontSize = self._getLayout(labels)
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
                    point = _drawTree(t, color)
                    self.painter.line([(treeHeight, point[1]), point], fill=color, width=2)
                    subClusterPoints.append(point)
                self.painter.line([(treeHeight, subClusterPoints[0][1]), (treeHeight, subClusterPoints[-1][1])], fill=color, width=2)
                return (treeHeight, (subClusterPoints[0][1]+subClusterPoints[-1][1])/2)
            else:
                self.globalHeight+=hAdvance
                return (treeAreaStart+treeAreaWidth, self.globalHeight-hAdvance/2)
        _drawTree(self.tree)
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
        for i in self.tree:
            label = labels[i]
            row = self.data[i]
            self.painter.text((textAreaStart, topMargin+i*hAdvance), self._truncText(label, textAreaWidth), font=self.font, fill=self.defaultTextColor)
            _drawMatrixRow(row, topMargin+i*hAdvance)

        self.image.save(file, "PNG")
    

if __name__=="__main__":
    data = orange.ExampleTable("doc//datasets//brown-selected.tab")
##    data = orange.ExampleTable("doc//datasets//iris.tab")
##    data = orange.ExampleTable("doc//datasets//zoo.tab")
##    data = orange.ExampleTable("doc//datasets//titanic.tab")
##    m = [[], [ 3], [ 2, 4], [17, 5, 4], [ 2, 8, 3, 8], [ 7, 5, 10, 11, 2], [ 8, 4, 1, 5, 11, 13], [ 4, 7, 12, 8, 10, 1, 5], [13, 9, 14, 15, 7, 8, 4, 6], [12, 10, 11, 15, 2, 5, 7, 3, 1]]
##    matrix = orange.SymMatrix(m)
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])
    root = orange.HierarchicalClustering(matrix, linkage=orange.HierarchicalClustering.Average)
    d = DendrogramPlot(root, data=data)#, fontSize=20) #height=1000)
    d.SetMatrixColorScheme((0, 255, 0), (255, 0, 0))
    d.SetClusterColors({root.left:(0,255,0), root.right:(0,0,255)})
    d.Plot()
    
    
        
            