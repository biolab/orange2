import math
import sys
import orange
import random

try:
    import matplotlib
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt #import plot, show
except ImportError:
    matplotlib = None
    
##############################################################################
# miscellaneous functions (used across this module)

def data_center(data):
    """Return the central - average - point in the data set"""
    atts = data.domain.attributes
    astats = orange.DomainBasicAttrStat(data)
    center = [astats[a].avg if a.varType == orange.VarTypes.Continuous \
              else max(enumerate(orange.Distribution(a, data)), key=lambda x:x[1])[0] if a.varType == orange.VarTypes.Discrete
              else None
              for a in atts]
    if data.domain.classVar:
        center.append(0)
    return orange.Example(data.domain, center)

def minindex(x):
    """Return the index of the minimum element"""
    return x.index(min(x))

def avg(x):
    """Return the average (mean) of a given list"""
    return (float(sum(x)) / len(x)) if x else 0

##############################################################################
# k-means clustering

# clustering scoring functions 

def score_distance_to_centroids(km):
    """Return the sum of distances from cluster elements to their centroids."""
    return sum([km.distance(km.centroids[km.clusters[i]], d) for i,d in enumerate(km.data)])

score_distance_to_centroids.minimize = True

def score_conditionalEntropy(km):
    """cluster quality measured by conditional entropy"""
    pass

def score_withinClusterDistance(data, clusters, _, distance):
    """weighted average within-cluster pairwise distance"""
    pass

score_withinClusterDistance.minimize = True

def score_silhouette(km, index=None):
    """Return the silhouette score (of a specific example if index is specified)"""
    if index == None:
        return avg([score_silhouette(km, i) for i in range(len(km.data))])
    cind = km.clusters[index]
    a = avg([km.distance(km.data[index], ex) for i, ex in enumerate(km.data) if
             km.clusters[i] == cind and i != index])
    b = min(avg([km.distance(km.data[index], ex) for i, ex in enumerate(km.data) if
                 km.clusters[i] == c])
            for c in range(len(km.centroids)) if c != cind)
    return float(b - a) / max(a, b)

def plot_silhouette(km, filename='tmp.png'):
    if not matplotlib:
        raise Exception("Need matplotlib library!")
    plt.figure()
    scores = [[] for i in range(km.k)]
    for i, c in enumerate(km.clusters):
        scores[c].append(score_silhouette(km, i))
    csizes = map(len, scores)
    cpositions = [sum(csizes[:i]) + (i+1)*3 + csizes[i]/2 for i in range(km.k)]
    scores = reduce(lambda x,y: x + [0]*3 + sorted(y), scores, [])
    plt.barh(range(len(scores)), scores, linewidth=0, color='c')
    plt.yticks(cpositions, map(str, range(km.k)))
    #plt.title('Silhouette plot')
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette value')
    plt.savefig(filename)

# clustering initialization (seeds)
# initialization functions should be of the type f(data, k, distfun)

def kmeans_init_random(data, k, _):
    """Return arbitrary k data instances from the data set.""" 
    return data.getitems(random.sample(range(len(data)), k))

def kmeans_init_diversity(data, k, distfun):
    """Return k most distant (heuristics) data instances."""
    center = data_center(data)
    # the first seed should be the farthest point from the center
    seeds = [max([(distfun(d, center), d) for d in data])[1]]
    # other seeds are added iteratively, and are data points that are farthest from the current set of seeds
    for i in range(1,k):
        seeds.append(max([(min([distfun(d, s) for s in seeds]), d) for d in data if d not in seeds])[1])
    return seeds

class KMeans_init_hierarchicalClustering():
    """Return centers of k clusters obtained by hierachical clustering.""" 
    def __init__(self, n=100):
        self.n = n

    def __call__(self, data, k, disfun):
        sample = orange.ExampleTable(random.sample(data, min(self.n, len(data))))
        root = hierarchicalClustering(sample)
        cmap = hierarchicalClustering_topClusters(root, k)
        return [data_center(orange.ExampleTable([sample[e] for e in cl])) for cl in cmap]
    
# k-means clustering, main implementation

class KMeans:
    def __init__(self, data=None, centroids=3, maxiters=None, minscorechange=None,
                 stopchanges=0, nstart=1, initialization=kmeans_init_random,
                 distance=orange.ExamplesDistanceConstructor_Euclidean,
                 scoring=score_distance_to_centroids, inner_callback = None,
                 outer_callback = None, initialize_only = False):
        self.data = data
        self.k = centroids if type(centroids)==int else len(centroids)
        self.centroids = centroids if type(centroids) == orange.ExampleTable else None
        self.maxiters = maxiters
        self.minscorechange = minscorechange
        self.stopchanges = stopchanges
        self.nstart = nstart
        self.initialization = initialization
        self.distance_constructor = distance
        self.distance = self.distance_constructor(self.data)
        self.scoring = scoring
        self.minimize_score = True if hasattr(scoring, 'minimize') else False
        self.inner_callback = inner_callback
        self.outer_callback = outer_callback
        if self.data and not initialize_only:
            self.run()
        
    def __call__(self, data = None):
        if data:
            print "NEW DATA"
            self.data = data
        self.run()
    
    def init_centroids(self):
        """initialize cluster centroids"""
        if self.centroids and not self.nstart > 1: # centroids were specified
            return
        self.centroids = self.initialization(self.data, self.k, self.distance) 
        
    def compute_centeroid(self, data):
        """Return a centroid of the data set."""
        return data_center(data)
    
    def compute_cluster(self):
        """calculate membership in clusters"""
        return [minindex([self.distance(s, d) for s in self.centroids]) for d in self.data]
    
    def runone(self):
        """run a single iteration of k-means clustering"""
        self.centroids = [self.compute_centeroid(self.data.getitems(
            [i for i, c in enumerate(self.clusters) if c == cl])) for cl in range(self.k)]
        self.clusters = self.compute_cluster()
        
    def run(self):
        """run a central k-means clustering loop"""
        self.winner = None
        for startindx in range(self.nstart):
            self.init_centroids()
            self.clusters = old_cluster = self.compute_cluster()
            self.score = old_score = self.scoring(self)
            self.nchanges = len(self.data)
            self.iteration = 0
            stopcondition = False
            if self.inner_callback:
                self.inner_callback(self)
            while not stopcondition:
                self.iteration += 1
                self.runone()
                self.nchanges = sum(map(lambda x,y: x!=y, old_cluster, self.clusters))
                old_cluster = self.clusters
                if self.minscorechange != None:
                    self.score = self.scoring(self)
                    scorechange = (self.score - old_score) / old_score
                    if self.minimize_score:
                        scorechange = -scorechange
                    old_score = self.score
                stopcondition = (self.nchanges <= self.stopchanges or
                                 self.iteration == self.maxiters or
                                 (self.minscorechange != None and
                                  scorechange <= self.minscorechange))
                if self.inner_callback:
                    self.inner_callback(self)
            if self.minscorechange == None:
                self.score = self.scoring(self)
            if self.nstart > 1:
                if not self.winner or (self.score < self.winner[0] if
                        self.minimize_score else self.score > self.winner[0]):
                    self.winner = (self.score, self.clusters, self.centroids)
                if self.outer_callback:
                    self.outer_callback(self)

        if self.nstart > 1:
            self.score, self.clusters, self.centroids = self.winner

##############################################################################
# hierarhical clustering

def hierarchicalClustering(data,
                           distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean,
                           linkage=orange.HierarchicalClustering.Average,
                           order=False,
                           progressCallback=None):
    """Return a hierarhical clustering of the data set."""
    distance = distanceConstructor(data)
    matrix = orange.SymMatrix(len(data))
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = distance(data[i], data[j])
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=progressCallback)
    if order:
        orderLeaves(root, matrix, progressCallback=progressCallback)
    return root

def hierarchicalClustering_attributes(data, distance=None, linkage=orange.HierarchicalClustering.Average, order=False, progressCallback=None):
    """Return hierarhical clustering of attributes in the data set."""
    matrix = orange.SymMatrix(len(data.domain.attributes))
    for a1 in range(len(data.domain.attributes)):
        for a2 in range(a1):
            matrix[a1, a2] = orange.PearsonCorrelation(a1, a2, self.data, 0).p
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=progressCallback)
    if order:
        orderLeaves(root, matrix, progressCallback=progressCallback)
    return root

def hierarchicalClustering_clusterList(node, prune=None):
    """Return a list of clusters down from the node of hierarchical clustering."""
    if prune:
        if len(node) <= prune:
            return [] 
    if node.branches:
        return [node] + hierarchicalClustering_clusterList(node.left, prune) + hierarchicalClustering_clusterList(node.right, prune)
    return [node]

def hierarchicalClustering_topClusters(root, k):
    """Return k topmost clusters from hierarchical clustering."""
    candidates = set([root])
    while len(candidates) < k:
        repl = max([(max(c.left.height, c.right.height), c) for c in candidates if c.branches])[1]
        candidates.discard(repl)
        candidates.add(repl.left)
        candidates.add(repl.right)
    return candidates

def hierarhicalClustering_topClustersMembership(root, k):
    """Return data instances' cluster membership (list of indices) to k topmost clusters."""
    clist = hierarchicalClustering_topClusters(root, k)
    cmap = [None] * len(root)
    for i, c in enumerate(clist):
        for e in c:
            cmap[e] = i
    return cmap

def orderLeaves(tree, matrix, progressCallback=None):
    """Order the leaves in the clustering tree.

    (based on Ziv Bar-Joseph et al. (Fast optimal leaf ordering for herarchical clustering')
    Arguments:
        tree   --binary hierarchical clustering tree of type orange.HierarchicalCluster
        matrix --orange.SymMatrix that was used to compute the clustering
        progressCallback --function used to report progress
    """
    objects = getattr(tree.mapping, "objects", None)
    tree.mapping.setattr("objects", range(len(tree)))
    M = {}
    ordering = {}
    visitedClusters = set()
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

            if progressCallback:
                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
                visitedClusters.add(tree)
        
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

try:
    import Image, ImageDraw, ImageFont
    _hasPIL = True
except ImportError:
    _hasPIL = False

class DendrogramPlot(object):
    defaultFontSize = 12
    defaultTreeColor = (0, 0, 0)
    defaultTextColor = (100, 100, 100)
    defaultMatrixOutlineColor = (240, 240, 240)
    def __init__(self, tree, data=None, labels=None, width=None, height=None, treeAreaWidth=None, textAreaWidth=None, matrixAreaWidth=None, fontSize=None, lineWidth=2, painter=None, clusterColors={}):
        if not _hasPIL:
            raise ImportError("Could not import PIL (Python Imaging Library). Please make sure PIL is installed on your system.")
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

    def setMatrixColorScheme(self, low, hi):
        """Set the matrix color scheme. low and hi must be (r, g, b) tuples
        """
        self.lowColor = low
        self.hiColor = hi

    def setClusterColors(self, clusterColors={}):
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
                
    def plot(self, filename="graph.png"):
        """Draw the dendrogram and save it to file."""
        file = open(filename, "wb")
        topMargin = 10
        bottomMargin = 10
        leftMargin = 10
        rightMargin = 10
        width, height, treeAreaWidth, textAreaWidth, matrixAreaWidth, hAdvance, fontSize = self._getLayout(self.labels)
        treeAreaStart = leftMargin
##        textAreaStart = treeAreaStart+treeAreaWidth+leftMargin
##        matrixAreaStart = textAreaStart+textAreaWidth+leftMargin
        matrixAreaStart = treeAreaStart + treeAreaWidth + leftMargin
        textAreaStart = matrixAreaStart + matrixAreaWidth + leftMargin
        
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
        self.image.save(file)

try:
    import numpy
except ImportError:
    numpy = None

try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.table import Table, Cell
    from matplotlib.text import Text
    from matplotlib.artist import Artist
    import  matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    Text , Artist, Table, Cell = object, object, obejct, object

##class ScalableText(Text):
##    _max_height = 1.0
##    cachedFontSize = None
##    def draw(self, renderer):
##        fontSize = self.fontsizesetter(renderer)
##        if fontSize != self.get_fontsize():
##            self.set_fontsize(fontSize)
##        Text.draw(self, renderer)
##
##    def fontsizesetter(self, renderer):
##        if ScalableText.cachedFontSize != None:
##            return ScalableText.cachedFontSize
##        transform = plt.gca().transData #.inverted()
##        _, y1 = transform.transform_point((0,0))
##        _, y2 = transform.transform_point((0,1))
##        fontsize = max(int(abs(y1 - y2)*0.85), 1)
##        ScalableText.cachedFontSize = fontsize
##        return fontsize
##
##    def get_width(self, renderer):
##        fontSize = self.fontsizesetter(renderer)
##        if fontSize != self.get_fontsize():
##            self.set_fontsize(fontSize)
##        l, b, w, h = self.get_window_extent(renderer).bounds
##        transform = plt.gca().transData.inverted()
##        x1, _ = transform.transform_point((0, 0))
##        x2, _ = transform.transform_point((w, 0))
##        w = abs(x1 - x2)*1.05
##        return w

class TableCell(Cell):
    PAD = 0.05
    def __init__(self, *args, **kwargs):
        Cell.__init__(self, *args, **kwargs)
        self._text.set_clip_on(True)

class TablePlot(Table):
    max_fontsize = 12
    def __init__(self, xy, axes=None, bbox=None):
        Table.__init__(self, axes or plt.gca(), bbox=bbox)
        self.xy = xy
        self.set_transform(self._axes.transData)
        self._fixed_widhts = None
        self.max_fontsize = plt.rcParams.get("font.size", 12)

    def add_cell(self, row, col, *args, **kwargs):
        xy = (0,0)

        cell = TableCell(xy, *args, **kwargs)
        cell.set_figure(self.figure)
        cell.set_transform(self.get_transform())

        cell.set_clip_on(True)
        cell.set_clip_box(self._axes.bbox)
        cell._text.set_clip_box(self._axes.bbox)
        self._cells[(row, col)] = cell

    def draw(self, renderer):
        if not self.get_visible(): return
        self._update_positions(renderer)

        keys = self._cells.keys()
        keys.sort()
        for key in keys:
            self._cells[key].draw(renderer)

    def _update_positions(self, renderer):
        keys = numpy.array(self._cells.keys())
        cells = numpy.array([[self._cells.get((row, col), None) for col in range(max(keys[:, 1] + 1))] \
                             for row in range(max(keys[:, 0] + 1))])
        
        widths = self._get_column_widths(renderer)
        x = self.xy[0] + numpy.array([numpy.sum(widths[:i]) for i in range(len(widths))])
        y = self.xy[1] - numpy.arange(cells.shape[0]) - 0.5
        
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                cells[i, j].set_xy((x[j], y[i]))
                cells[i, j].set_width(widths[j])
                cells[i, j].set_height(1.0)

        self._width = numpy.sum(widths)
        self._height = cells.shape[0]

        self.pchanged()

    def _get_column_widths(self, renderer):
        keys = numpy.array(self._cells.keys())
        widths = numpy.zeros(len(keys)).reshape((numpy.max(keys[:,0]+1), numpy.max(keys[:,1]+1)))
        fontSize = self._calc_fontsize(renderer)
        for (row, col), cell in self._cells.items():
            cell.set_fontsize(fontSize)
            l, b, w, h = cell._text.get_window_extent(renderer).bounds
            transform = self._axes.transData.inverted()
            x1, _ = transform.transform_point((0, 0))
            x2, _ = transform.transform_point((w + w*TableCell.PAD + 10, 0))
            w = abs(x1 - x2)
            widths[row, col] = w
        return numpy.max(widths, 0)

    def _calc_fontsize(self, renderer):
        transform = self._axes.transData
        _, y1 = transform.transform_point((0, 0))
        _, y2 = transform.transform_point((0, 1))
        return min(max(int(abs(y1 - y2)*0.85) ,4), self.max_fontsize)

    def get_children(self):
        return self._cells.values()

    def get_bbox(self):
        return matplotlib.transform.Bbox([self.xy[0], self.xy[1], self.xy[0] + 10, self.xy[1] + 180])
    
##class TableTextLayout(Artist):
##    def __init__(self, xy=(0, 0), tableText=[], font=None, fontSize=10, *args, **kwargs):
##        Artist.__init__(self, *args, **kwargs)
##        self.xy = xy
##        self.tableText = numpy.array(tableText)
##        self.font = font
##        self.fontSize = fontSize
##        self.cells = numpy.array([[ScalableText(0,0, self.tableText[i,j], figure=plt.gcf()) \
##                                  for j in range(self.tableText.shape[1])] \
##                                      for i in range(self.tableText.shape[0])])
##        for text in self.cells.flat:
##            plt.gca().add_artist(text)
##
##    def set_layout(self, renderer):
##        widths = numpy.array([[cell.get_width(renderer) for cell in row] for row in self.cells])
##        widths = numpy.max(widths, 0)
##        x = self.xy[0] + numpy.array([numpy.sum(widths[:i]) for i in range(widths.size)])
##        y = self.xy[1] - numpy.arange(self.cells.shape[0]) - 0.5
##        print x,y
##        for i in range(self.cells.shape[0]):
##            for j in range(self.cells.shape[1]):
##                self.cells[i,j].set_position((x[j], y[i]))
##    
##    def draw(self, renderer):
##        self.set_layout(renderer)
##        ScalableText.cachedFontSize = None
##        for cell in self.cells.flat:
##            cell.draw(renderer)
##
##    def get_children(self):
##        return list(self.cells.flat)

class DendrogramPlotPylab(object):
    def __init__(self, root, data=None, labels=None, dendrogram_width=None, heatmap_width=None, label_width=None, space_width=None, border_width=0.05, plot_attr_names=False, cmap=None, params={}):
        if not matplotlib:
            raise ImportError("Could not import matplotlib module. Please make sure matplotlib is installed on your system.")
        self.root = root
        self.data = data
        self.labels = labels if labels else [str(i) for i in range(len(root))]
        self.dendrogram_width = dendrogram_width
        self.heatmap_width = heatmap_width
        self.label_width = label_width
        self.space_width = space_width
        self.border_width = border_width
        self.params = params
        self.plot_attr_names = plot_attr_names

    def plotDendrogram(self):
        self.text_items = []
        def draw_tree(tree):
            if tree.branches:
                points = []
                for branch in tree.branches:
                    center = draw_tree(branch)
                    plt.plot([center[0], tree.height], [center[1], center[1]], color="black")
                    points.append(center)
                plt.plot([tree.height, tree.height], [points[0][1], points[-1][1]], color="black")
                return (tree.height, (points[0][1] + points[-1][1])/2.0)
            else:
                return (0.0, tree.first)
        draw_tree(self.root)
        
    def plotHeatMap(self):
        import numpy.ma as ma
        import numpy
        dx, dy = self.root.height, 0
        fx, fy = self.root.height/len(self.data.domain.attributes), 1.0
        data, c, w = self.data.toNumpyMA()
        data = (data - ma.min(data))/(ma.max(data) - ma.min(data))
        x = numpy.arange(data.shape[1] + 1)/float(numpy.max(data.shape))
        y = numpy.arange(data.shape[0] + 1)/float(numpy.max(data.shape))*len(self.root)
        self.heatmap_width = numpy.max(x)

        X, Y = numpy.meshgrid(x, y - 0.5)

        self.meshXOffset = numpy.max(X)

        plt.jet()
        mesh = plt.pcolormesh(X, Y, data[self.root.mapping], edgecolor="b", linewidth=2)

        if self.plot_attr_names:
            names = [attr.name for attr in self.data.domain.attributes]
            plt.xticks(numpy.arange(data.shape[1] + 1)/float(numpy.max(data.shape)), names)
        plt.gca().xaxis.tick_top()
        for label in plt.gca().xaxis.get_ticklabels():
            label.set_rotation(45)

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False

        
    
    def plotLabels_(self):
        import numpy
##        plt.yticks(numpy.arange(len(self.labels) - 1, 0, -1), self.labels)
##        for tick in plt.gca().yaxis.get_major_ticks():
##            tick.tick1On = False
##            tick.label1On = False
##            tick.label2On = True
##        text = TableTextLayout(xy=(self.meshXOffset+1, len(self.root)), tableText=[[label] for label in self.labels])
        text = TableTextLayout(xy=(self.meshXOffset*1.005, len(self.root) - 1), tableText=[[label] for label in self.labels])
        text.set_figure(plt.gcf())
        plt.gca().add_artist(text)
        plt.gca()._set_artist_props(text)

    def plotLabels(self):
##        table = TablePlot(xy=(self.meshXOffset*1.005, len(self.root) -1), axes=plt.gca())
        table = TablePlot(xy=(0, len(self.root) -1), axes=plt.gca())
        table.set_figure(plt.gcf())
        for i,label in enumerate(self.labels):
            table.add_cell(i, 0, width=1, height=1, text=label, loc="left", edgecolor="w")
        table.set_zorder(0)
        plt.gca().add_artist(table)
        plt.gca()._set_artist_props(table)
    
    def plot(self, filename=None, show=False):
        plt.rcParams.update(self.params)
        labelLen = max(len(label) for label in self.labels)
        w, h = 800, 600
        space = 0.01 if self.space_width == None else self.space_width
        border = self.border_width
        width = 1.0 - 2*border
        height = 1.0 - 2*border
        textLineHeight = min(max(h/len(self.labels), 4), plt.rcParams.get("font.size", 12))
        maxTextLineWidthEstimate = textLineHeight*labelLen
##        print maxTextLineWidthEstimate
        textAxisWidthRatio = 2.0*maxTextLineWidthEstimate/w
##        print textAxisWidthRatio
        labelsAreaRatio = min(textAxisWidthRatio, 0.4) if self.label_width == None else self.label_width
        x, y = len(self.data.domain.attributes), len(self.data)

        heatmapAreaRatio = min(1.0*y/h*x/w, 0.3) if self.heatmap_width == None else self.heatmap_width
        dendrogramAreaRatio = 1.0 - labelsAreaRatio - heatmapAreaRatio - 2*space if self.dendrogram_width == None else self.dendrogram_width

        self.fig = plt.figure()
        self.labels_offset = self.root.height/20.0
        dendrogramAxes = plt.axes([border, border, width*dendrogramAreaRatio, height])
        dendrogramAxes.xaxis.grid(True)
        import matplotlib.ticker as ticker

        dendrogramAxes.yaxis.set_major_locator(ticker.NullLocator())
        dendrogramAxes.yaxis.set_minor_locator(ticker.NullLocator())
        dendrogramAxes.invert_xaxis()
        self.plotDendrogram()
        heatmapAxes = plt.axes([border + width*dendrogramAreaRatio + space, border, width*heatmapAreaRatio, height], sharey=dendrogramAxes)

        heatmapAxes.xaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.xaxis.set_minor_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_minor_locator(ticker.NullLocator())
        
        self.plotHeatMap()
        labelsAxes = plt.axes([border + width*(dendrogramAreaRatio + heatmapAreaRatio + 2*space), border, width*labelsAreaRatio, height], sharey=dendrogramAxes)
        self.plotLabels()
        labelsAxes.set_axis_off()
        labelsAxes.xaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.xaxis.set_minor_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_minor_locator(ticker.NullLocator())
        if filename:
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(self.fig)
            canvas.print_figure(filename)
        if show:
            plt.show()

if __name__=="__main__":
    data = orange.ExampleTable("doc//datasets//brown-selected.tab")
    root = hierarchicalClustering(data, order=True) #, linkage=orange.HierarchicalClustering.Single)
    print root
    d = DendrogramPlotPylab(root, data=data, labels=[str(ex.getclass()) for ex in data], dendrogram_width=0.4, heatmap_width=0.3,  params={}, cmap=None)
    d.plot(show=True, filename="graph.png")
    
