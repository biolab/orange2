"""
***********************
Hierarchical clustering
***********************

.. index::
   single: clustering, kmeans
.. index:: aglomerative clustering

Examples
========

An example.

.. automethod:: Orange.clustering.hierarchical.clustering

"""
import orange
import Orange
from Orange.core import HierarchicalClustering, \
                        HierarchicalCluster, \
                        HierarchicalClusterList
                        
from Orange.misc import progressBarMilestones
                        
import sys


def clustering(data,
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
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        order_leaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def clustering_features(data, distance=None, linkage=orange.HierarchicalClustering.Average, order=False, progressCallback=None):
    """Return hierarhical clustering of attributes in the data set."""
    matrix = orange.SymMatrix(len(data.domain.attributes))
    for a1 in range(len(data.domain.attributes)):
        for a2 in range(a1):
            matrix[a1, a2] = (1.0 - orange.PearsonCorrelation(a1, a2, data, 0).r) / 2.0
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        order_leaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def cluster_to_list(node, prune=None):
    """Return a list of clusters down from the node of hierarchical clustering."""
    if prune:
        if len(node) <= prune:
            return [] 
    if node.branches:
        return [node] + cluster_to_list(node.left, prune) + cluster_to_list(node.right, prune)
    return [node]

def top_clusters(root, k):
    """Return k topmost clusters from hierarchical clustering."""
    candidates = set([root])
    while len(candidates) < k:
        repl = max([(max(c.left.height, c.right.height), c) for c in candidates if c.branches])[1]
        candidates.discard(repl)
        candidates.add(repl.left)
        candidates.add(repl.right)
    return candidates

def top_cluster_membership(root, k):
    """Return data instances' cluster membership (list of indices) to k topmost clusters."""
    clist = top_clusters(root, k)
    cmap = [None] * len(root)
    for i, c in enumerate(clist):
        for e in c:
            cmap[e] = i
    return cmap

def order_leaves(tree, matrix, progressCallback=None):
    """Order the leaves in the clustering tree.

    (based on Ziv Bar-Joseph et al. (Fast optimal leaf ordering for hierarchical clustering')
    Arguments:
        tree   --binary hierarchical clustering tree of type orange.HierarchicalCluster
        matrix --orange.SymMatrix that was used to compute the clustering
        progressCallback --function used to report progress
    """
#    from Orange.misc import recursion_limit
    
    objects = getattr(tree.mapping, "objects", None)
    tree.mapping.setattr("objects", range(len(tree)))
    M = {}
    ordering = {}
    visitedClusters = set()
    
    def _optOrderingRecursive(tree):
        if len(tree)==1:
            for leaf in tree:
                M[tree, leaf, leaf] = 0
        else:
            _optOrderingRecursive(tree.left)
            _optOrderingRecursive(tree.right)
            
            Vl = set(tree.left)
            Vr = set(tree.right)
            Vlr = set(tree.left.right or tree.left)
            Vll = set(tree.left.left or tree.left)
            Vrr = set(tree.right.right or tree.right)
            Vrl = set(tree.right.left or tree.right)
            other = lambda e, V1, V2: V2 if e in V1 else V1
            tree_left, tree_right = tree.left, tree.right
            for u in Vl:
                for w in Vr:
#                    if True: #Improved search
                    C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
                    orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree_left, u, m])
                    orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree_right, w, k])
                    k0 = orderedKs[0]
                    curMin = 1e30000 
                    curM = curK = None
                    for m in orderedMs:
                        if M[tree_left, u, m] + M[tree_right, w, k0] + C >= curMin:
                            break
                        for k in  orderedKs:
                            if M[tree_left, u, m] + M[tree_right, w, k] + C >= curMin:
                                break
                            testMin = M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
                            if curMin > testMin:
                                curMin = testMin
                                curM = m
                                curK = k
                    M[tree, u, w] = M[tree, w, u] = curMin
                    ordering[tree, u, w] = (tree_left, u, curM, tree_right, w, curK)
                    ordering[tree, w, u] = (tree_right, w, curK, tree_left, u, curM)
#                    else:
#                    def MFunc((m, k)):
#                        return M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
#                    m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
#                    M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
#                    ordering[tree, u, w] = (tree_left, u, m, tree_right, w, k)
#                    ordering[tree, w, u] = (tree_right, w, k, tree_left, u, m)

            if progressCallback:
                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
                visitedClusters.add(tree)
    
    if False:
        with recursion_limit(sys.getrecursionlimit() + len(tree)):
            _optOrderingRecursive(tree)
        
    def _optOrderingIterative(tree):
        if len(tree)==1:
            for leaf in tree:
                M[tree, leaf, leaf] = 0
        else:
#            _optOrdering(tree.left)
#            _optOrdering(tree.right)
            
            Vl = set(tree.left)
            Vr = set(tree.right)
            Vlr = set(tree.left.right or tree.left)
            Vll = set(tree.left.left or tree.left)
            Vrr = set(tree.right.right or tree.right)
            Vrl = set(tree.right.left or tree.right)
            other = lambda e, V1, V2: V2 if e in V1 else V1
            tree_left, tree_right = tree.left, tree.right
            for u in Vl:
                for w in Vr:
#                    if True: #Improved search
                        C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
                        orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree_left, u, m])
                        orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree_right, w, k])
                        k0 = orderedKs[0]
                        curMin = 1e30000 
                        curM = curK = None
                        for m in orderedMs:
                            if M[tree_left, u, m] + M[tree_right, w, k0] + C >= curMin:
                                break
                            for k in  orderedKs:
                                if M[tree_left, u, m] + M[tree_right, w, k] + C >= curMin:
                                    break
                                testMin = M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
                                if curMin > testMin:
                                    curMin = testMin
                                    curM = m
                                    curK = k
                        M[tree, u, w] = M[tree, w, u] = curMin
                        ordering[tree, u, w] = (tree_left, u, curM, tree_right, w, curK)
                        ordering[tree, w, u] = (tree_right, w, curK, tree_left, u, curM)
#                    else:
#                        def MFunc((m, k)):
#                            return M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
#                        m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
#                        M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
#                        ordering[tree, u, w] = (tree_left, u, m, tree_right, w, k)
#                        ordering[tree, w, u] = (tree_right, w, k, tree_left, u, m)

#            if progressCallback:
#                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
#                visitedClusters.add(tree)
    from Orange.misc import progressBarMilestones
    
    subtrees = postorder(tree)
    milestones = progressBarMilestones(len(subtrees), 1000)
    
    for i, subtree in enumerate(subtrees):
        _optOrderingIterative(subtree)
        if progressCallback and i in milestones:
            progressCallback(100.0 * i / len(subtrees))

#    def _orderRecursive(tree, u, w):
#        """ Order the tree based on the computed optimal ordering. 
#        """
#        if len(tree)==1:
#            return
#        left, u, m, right, w, k = ordering[tree, u, w]
#        if len(left)>1 and m not in left.right:
#            left.swap()
#        _orderRecursive(left, u, m)
#        
#        if len(right)>1 and k not in right.left:
#            right.swap()
#        _orderRecursive(right, k, w)
        
    def _orderIterative(tree, u, w):
        """ Order the tree based on the computed optimal ordering. 
        """
        opt_uw = {tree: (u, w)}
        for subtree in preorder(tree):
            if subtree.branches:
                u, w = opt_uw[subtree]
                left, u, m, right, w, k = ordering[subtree, u, w]
                opt_uw[left] = (u, m)
                opt_uw[right] = (k, w)
                
                if left.branches and m not in left.right:
                    left.swap()
                
                if right.branches and k not in right.left:
                    right.swap()
            
#        if len(tree) == 1:
#            return
#        left, u, m, right, w, k = ordering[tree, u, w]
#        if len(left) > 1 and m not in left.right:
#            left.swap()
#        _orderIterative(left, u, m)
#        
#        if len(right) > 1 and k not in right.left:
#            right.swap()
#        _orderIterative(right, k, w)
    
    u, w = min([(u, w) for u in tree.left for w in tree.right], key=lambda (u, w): M[tree, u, w])
    
##    print "M(v) =", M[tree, u, w]
    
    if False:
        with recursion_limit(sys.getrecursionlimit() + len(tree)):
            _orderRecursive(tree, u, w)
            
    _orderIterative(tree, u, w)
            

#    def _check(tree, u, w):
#        if len(tree)==1:
#            return
#        left, u, m, right, w, k = ordering[tree, u, w]
#        if tree[0] == u and tree[-1] == w:
#            _check(left, u, m)
#            _check(right, k, w)
#        else:
#            print "Error:", u, w, tree[0], tree[-1]
#
#    with recursion_limit(sys.getrecursionlimit() + len(tree)):
#        _check(tree, u ,w)

    if objects:
        tree.mapping.setattr("objects", objects)

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
##    import  matplotlib.pyplot as plt
except (ImportError, IOError), ex:
    matplotlib = None
    Text , Artist, Table, Cell = object, object, object, object

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
        import matplotlib.pyplot as plt
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

class DendrogramPlotPylab(object):
    def __init__(self, root, data=None, labels=None, dendrogram_width=None, heatmap_width=None, label_width=None, space_width=None, border_width=0.05, plot_attr_names=False, cmap=None, params={}):
        if not matplotlib:
            raise ImportError("Could not import matplotlib module. Please make sure matplotlib is installed on your system.")
        import matplotlib.pyplot as plt
        self.plt = plt
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
                    self.plt.plot([center[0], tree.height], [center[1], center[1]], color="black")
                    points.append(center)
                self.plt.plot([tree.height, tree.height], [points[0][1], points[-1][1]], color="black")
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

        self.plt.jet()
        mesh = self.plt.pcolormesh(X, Y, data[self.root.mapping], edgecolor="b", linewidth=2)

        if self.plot_attr_names:
            names = [attr.name for attr in self.data.domain.attributes]
            self.plt.xticks(numpy.arange(data.shape[1] + 1)/float(numpy.max(data.shape)), names)
        self.plt.gca().xaxis.tick_top()
        for label in self.plt.gca().xaxis.get_ticklabels():
            label.set_rotation(45)

        for tick in self.plt.gca().xaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False

    def plotLabels_(self):
        import numpy
##        self.plt.yticks(numpy.arange(len(self.labels) - 1, 0, -1), self.labels)
##        for tick in self.plt.gca().yaxis.get_major_ticks():
##            tick.tick1On = False
##            tick.label1On = False
##            tick.label2On = True
##        text = TableTextLayout(xy=(self.meshXOffset+1, len(self.root)), tableText=[[label] for label in self.labels])
        text = TableTextLayout(xy=(self.meshXOffset*1.005, len(self.root) - 1), tableText=[[label] for label in self.labels])
        text.set_figure(self.plt.gcf())
        self.plt.gca().add_artist(text)
        self.plt.gca()._set_artist_props(text)

    def plotLabels(self):
##        table = TablePlot(xy=(self.meshXOffset*1.005, len(self.root) -1), axes=self.plt.gca())
        table = TablePlot(xy=(0, len(self.root) -1), axes=self.plt.gca())
        table.set_figure(self.plt.gcf())
        for i,label in enumerate(self.labels):
            table.add_cell(i, 0, width=1, height=1, text=label, loc="left", edgecolor="w")
        table.set_zorder(0)
        self.plt.gca().add_artist(table)
        self.plt.gca()._set_artist_props(table)
    
    def plot(self, filename=None, show=False):
        self.plt.rcParams.update(self.params)
        labelLen = max(len(label) for label in self.labels)
        w, h = 800, 600
        space = 0.01 if self.space_width == None else self.space_width
        border = self.border_width
        width = 1.0 - 2*border
        height = 1.0 - 2*border
        textLineHeight = min(max(h/len(self.labels), 4), self.plt.rcParams.get("font.size", 12))
        maxTextLineWidthEstimate = textLineHeight*labelLen
##        print maxTextLineWidthEstimate
        textAxisWidthRatio = 2.0*maxTextLineWidthEstimate/w
##        print textAxisWidthRatio
        labelsAreaRatio = min(textAxisWidthRatio, 0.4) if self.label_width == None else self.label_width
        x, y = len(self.data.domain.attributes), len(self.data)

        heatmapAreaRatio = min(1.0*y/h*x/w, 0.3) if self.heatmap_width == None else self.heatmap_width
        dendrogramAreaRatio = 1.0 - labelsAreaRatio - heatmapAreaRatio - 2*space if self.dendrogram_width == None else self.dendrogram_width

        self.fig = self.plt.figure()
        self.labels_offset = self.root.height/20.0
        dendrogramAxes = self.plt.axes([border, border, width*dendrogramAreaRatio, height])
        dendrogramAxes.xaxis.grid(True)
        import matplotlib.ticker as ticker

        dendrogramAxes.yaxis.set_major_locator(ticker.NullLocator())
        dendrogramAxes.yaxis.set_minor_locator(ticker.NullLocator())
        dendrogramAxes.invert_xaxis()
        self.plotDendrogram()
        heatmapAxes = self.plt.axes([border + width*dendrogramAreaRatio + space, border, width*heatmapAreaRatio, height], sharey=dendrogramAxes)

        heatmapAxes.xaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.xaxis.set_minor_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_minor_locator(ticker.NullLocator())
        
        self.plotHeatMap()
        labelsAxes = self.plt.axes([border + width*(dendrogramAreaRatio + heatmapAreaRatio + 2*space), border, width*labelsAreaRatio, height], sharey=dendrogramAxes)
        self.plotLabels()
        labelsAxes.set_axis_off()
        labelsAxes.xaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.xaxis.set_minor_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_minor_locator(ticker.NullLocator())
        if filename:
            import matplotlib.backends.backend_agg
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(self.fig)
            canvas.print_figure(filename)
        if show:
            self.plt.show()
        
        
from orngMisc import ColorPalette, EPSRenderer
class DendrogramPlot(object):
    """ A class for drawing dendrograms
    Example:
    >>> a = DendrogramPlot(tree)
    """
    def __init__(self, tree, attr_tree = None, labels=None, data=None, width=None, height=None, tree_height=None, heatmap_width=None, text_width=None, 
                 spacing=2, cluster_colors={}, color_palette=ColorPalette([(255, 0, 0), (0, 255, 0)]), maxv=None, minv=None, gamma=None, renderer=EPSRenderer):
        self.tree = tree
        self.attr_tree = attr_tree
        self.labels = [str(ex.getclass()) for ex in data] if not labels and data and data.domain.classVar else (labels or [])
#        self.attr_labels = [str(attr.name) for attr in data.domain.attributes] if not attr_labels and data else attr_labels or []
        self.data = data
        self.width, self.height = float(width) if width else None, float(height) if height else None
        self.tree_height = tree_height
        self.heatmap_width = heatmap_width
        self.text_width = text_width
        self.font_size = 10.0
        self.linespacing = 0.0
        self.cluster_colors = cluster_colors
        self.horizontal_margin = 10.0
        self.vertical_margin = 10.0
        self.spacing = float(spacing) if spacing else None
        self.color_palette = color_palette
        self.minv = minv
        self.maxv = maxv
        self.gamma = gamma
        self.set_matrix_color_schema(color_palette, minv, maxv, gamma)
        self.renderer = renderer
        
    def set_matrix_color_schema(self, color_palette, minv, maxv, gamma=None):
        """ Set the matrix color scheme.
        """
        if isinstance(color_palette, ColorPalette):
            self.color_palette = color_palette
        else:
            self.color_palette = ColorPalette(color_palette)
        self.minv = minv
        self.maxv = maxv
        self.gamma = gamma
        
    def color_shema(self):
        vals = [float(val) for ex in self.data for val in ex if not val.isSpecial() and val.variable.varType==orange.VarTypes.Continuous] or [0]
        avg = sum(vals)/len(vals)
        
        maxVal = self.maxv if self.maxv else max(vals)
        minVal = self.minv if self.minv else min(vals)
        
        def _colorSchema(val):
            if val.isSpecial():
                return self.color_palette(None)
            elif val.variable.varType==orange.VarTypes.Continuous:
                r, g, b = self.color_palette((float(val) - minVal) / abs(maxVal - minVal), gamma=self.gamma)
            elif val.variable.varType==orange.VarTypes.Discrete:
                r = g = b = int(255.0*float(val)/len(val.variable.values))
            return (r, g, b)
        return _colorSchema
    
    def layout(self):
        height_final = False
        width_final = False
        tree_height = self.tree_height or 100
        if self.height:
            height, height_final = self.height, True
            heatmap_height = height - (tree_height + self.spacing if self.attr_tree else 0) - 2 * self.horizontal_margin
            font_size =  heatmap_height / len(self.labels) #self.font_size or (height - (tree_height + self.spacing if self.attr_tree else 0) - 2 * self.horizontal_margin) / len(self.labels)
        else:
            font_size = self.font_size
            heatmap_height = font_size * len(self.labels)
            height = heatmap_height + (tree_height + self.spacing if self.attr_tree else 0) + 2 * self.horizontal_margin
             
        text_width = self.text_width or max([len(label) for label in self.labels] + [0]) * font_size #max([self.renderer.string_size_hint(label) for label in self.labels])
        
        if self.width:
            width = self.width
            heatmap_width = width - 2 * self.vertical_margin - tree_height - (2 if self.data else 1) * self.spacing - text_width if self.data else 0
        else:
            heatmap_width = len(self.data.domain.attributes) * heatmap_height / len(self.data) if self.data else 0
            width = 2 * self.vertical_margin + tree_height + (heatmap_width + self.spacing if self.data else 0) + self.spacing + text_width
            
        return width, height, tree_height, heatmap_width, heatmap_height, text_width, font_size
    
    def plot(self, filename="graph.eps"):
        width, height, tree_height, heatmap_width, heatmap_height, text_width, font_size = self.layout()
        heatmap_cell_height = heatmap_height / len(self.labels)

        heatmap_cell_width = 0.0 if not self.data else heatmap_width / len(self.data.domain.attributes)
        
        self.renderer = self.renderer(width, height)
        
        def draw_tree(cluster, root, treeheight, treewidth, color):
            height = treeheight * cluster.height / root.height
            if cluster.branches:
                centers = []
                for branch in cluster.branches:
                    center = draw_tree(branch, root, treeheight, treewidth, self.cluster_colors.get(branch, color))
                    centers.append(center)
                    self.renderer.draw_line(center[0], center[1], center[0], height, stroke_color = self.cluster_colors.get(branch, color))
                    
                self.renderer.draw_line(centers[0][0], height, centers[-1][0], height, stroke_color = self.cluster_colors.get(cluster, color))
                return (centers[0][0] + centers[-1][0]) / 2.0, height
            else:
                return float(treewidth) * cluster.first / len(root), 0.0
        self.renderer.save_render_state()
        self.renderer.translate(self.vertical_margin + tree_height, self.horizontal_margin + (tree_height + self.spacing if self.attr_tree else 0) + heatmap_cell_height / 2.0)
        self.renderer.rotate(90)
#        print self.renderer.transform()
        draw_tree(self.tree, self.tree, tree_height, heatmap_height, self.cluster_colors.get(self.tree, (0,0,0)))
        self.renderer.restore_render_state()
        if self.attr_tree:
            self.renderer.save_render_state()
            self.renderer.translate(self.vertical_margin + tree_height + self.spacing + heatmap_cell_width / 2.0, self.horizontal_margin + tree_height)
            self.renderer.scale(1.0, -1.0)
#            print self.renderer.transform()
            draw_tree(self.attr_tree, self.attr_tree, tree_height, heatmap_width, self.cluster_colors.get(self.attr_tree, (0,0,0)))
            self.renderer.restore_render_state()
        
        self.renderer.save_render_state()
        self.renderer.translate(self.vertical_margin + tree_height + self.spacing, self.horizontal_margin + (tree_height + self.spacing if self.attr_tree else 0))
#        print self.renderer.transform()
        if self.data:
            colorSchema = self.color_shema()
            for i, ii in enumerate(self.tree):
                ex = self.data[ii]
                for j, jj in enumerate((self.attr_tree if self.attr_tree else range(len(self.data.domain.attributes)))):
                    r, g, b = colorSchema(ex[jj])
                    self.renderer.draw_rect(j * heatmap_cell_width, i * heatmap_cell_height, heatmap_cell_width, heatmap_cell_height, fill_color=(r, g, b), stroke_color=(255, 255, 255))
        
        self.renderer.translate(heatmap_width + self.spacing, heatmap_cell_height)
#        print self.renderer.transform()
        self.renderer.set_font("Times-Roman", font_size)
        for index in self.tree: #label in self.labels:
            self.renderer.draw_text(0.0, 0.0, self.labels[index])
            self.renderer.translate(0.0, heatmap_cell_height)
        self.renderer.restore_render_state()
        self.renderer.save(filename)
        
def dendrogram_draw(filename, *args, **kwargs):
    import os
    from orngMisc import PILRenderer, EPSRenderer, SVGRenderer
    name, ext = os.path.splitext(filename)
    kwargs["renderer"] = {".eps":EPSRenderer, ".svg":SVGRenderer, ".png":PILRenderer}.get(ext.lower(), PILRenderer)
#    print kwargs["renderer"], ext
    d = DendrogramPlot(*args, **kwargs)
    d.plot(filename)
    
    
"""
Utility functions
=================

"""
    
def postorder(cluster):
    """ Return a post order list of clusters.
    
    :param cluster: Cluster
    :type cluster: :class:`HierarchicalCluster`
    
    """
    order = []
    visited = set()
    stack = [cluster]
    while stack:
        cluster = stack.pop(0)
        
        if cluster.branches:
            if cluster in visited:
                order.append(cluster)
            else:
                stack = cluster.branches + [cluster] + stack
                visited.add(cluster)
        else:
            order.append(cluster)
            visited.add(cluster)
    return order
    
    
def preorder(cluster):
    """ Return a pre order list of clusters.
    
    :param cluster: Cluster
    :type cluster: :class:`HierarchicalCluster`
    
    """
    order = []
    stack = [cluster]
    while stack:
        cluster = stack.pop(0)
        order.append(cluster)
        if cluster.branches:
            stack = cluster.branches + stack
    return order
    
    
def dendrogram_layout(root_cluster, expand_leaves=False):
    """ Return a layout of the cluster dendrogram on a 2D plane. The return 
    value if a list of (subcluster, (start, center, end)) tuples where
    subcluster is an instance of :class:`HierarchicalCluster` and start,
    end are the two end points for the cluster branch. The list is sorted
    in post-order.
    
    :param root_cluster: Cluster to layout.
    :type root_cluster: :class:`HierarchicalCluster`
    
    :param expand_leaves: If True leaves will span proportional to the number
        of items they map, else all leaves will be the same width. 
     
    """
    result = []
    cluster_geometry = {}
    leaf_idx = 0
    for cluster in postorder(root_cluster):
        if not cluster.branches:
            if expand_leaves:
                start, end = float(cluster.first), float(cluster.last)
            else:
                start = end = leaf_idx
                leaf_idx += 1
            center = (start + end) / 2.0
            cluster_geometry[cluster] = (start, center, end)
            result.append((cluster, (start, center, end)))
        else:
            left = cluster.branches[0]
            right = cluster.branches[1]
            left_center = cluster_geometry[left][1]
            right_center = cluster_geometry[right][1]
            start, end = left_center, right_center
            center = (start + end) / 2.0
            cluster_geometry[cluster] = (start, center, end)
            result.append((cluster, (start, center, end)))
            
    return result
    
    
def pruned(root_cluster, level=None, height=None, condition=None):
    """ Return a new pruned clustering instance.
    
    .. note:: This uses `copy.deepcopy` to create a copy of the root_cluster
        instance.
    
    :param cluster: Cluster to prune.
    :type cluster: :class:`HierarchicalCluster`
    
    :param level: If not None prune all clusters deeper then `level`.
    :type level: int
    
    :param height: If not None prune all clusters with height lower then
        `height`.
    :type height: float
    
    :param condition: If not None condition must be a function taking a
        single :class:`HierarchicalCluster` instance and returning a True 
        or False value indicating if the cluster should be pruned.
    :type condition: function 
    
    """
    import copy
    
    # XXX This is unsafe HierarchicalCluster should take care of copying
    if hasattr(root_cluster.mapping, "objects"):
        objects = root_cluster.mapping.objects
        root_cluster.mapping.objects = None
        has_objects = True
    else:
        has_objects = False
        
    root_cluster = copy.deepcopy(root_cluster)
    
    prune(root_cluster, level, height, condition)
    
    if has_objects:
        root_cluster.mapping.objects = objects
        
    return root_cluster
    
    
def prune(root_cluster, level=None, height=None, condition=None):
    """ Prune clustering instance in place
    
    :param cluster: Cluster to prune.
    :type cluster: :class:`HierarchicalCluster`
    
    :param level: If not None prune all clusters deeper then `level`.
    :type level: int
    
    :param height: If not None prune all clusters with height lower then
        `height`.
    :type height: float
    
    :param condition: If not None condition must be a function taking a
        single :class:`HierarchicalCluster` instance and returning a True 
        or False value indicating if the cluster should be pruned.
    :type condition: function
    
    """
    if not any(arg is not None for arg in [level, height, condition]):
        raise ValueError("At least one pruning argument must be supplied")
    
    level_check = height_check = condition_check = lambda cl: False
    cluster_depth = cluster_depths(root_cluster)
    
    if level is not None:
        level_check = lambda cl: cluster_depth[cl] > level
        
    if height is not None:
        height_check = lambda cl: cl.height < height

    if condition is not None:
        condition_check = condition
        
    pruned_set = set()
    
    def check_all(cl):
        return any([check(cl) for check in [level_check, height_check,
                                            condition_check]])
        
    for cluster in preorder(root_cluster):
        if cluster not in pruned_set:
            if check_all(cluster):
                cluster.branches = None
                pruned_set.update(set(preorder(cluster)))
            else:
                pass
    
    
def cluster_depths(root_cluster):
    """ Return a dictionary mapping :class:`HierarchicalCluster` instances to
    their depths in the `root_cluster` hierarchy.
    
    """
    depths = {}
    depths[root_cluster] = 0
    for cluster in preorder(root_cluster):
        cl_depth = depths[cluster]
        if cluster.branches:
            depths.update(dict.fromkeys(cluster.branches, cl_depth + 1))
    return depths


def instance_distance_matrix(data,
            distance_constructor=orange.ExamplesDistanceConstructor_Euclidean,
            progress_callback=None):
    """ A helper function that computes an :class:`Orange.core.SymMatrix` of all
    pairwise distances between instances in `data`.
    
    :param data: A data table
    :type data: :class:`Orange.data.Table`
    
    :param distance_constructor: An ExamplesDistance_Constructor instance.
    :type distance_constructor: :class:`Orange.distances.ExampleDistConstructor`
    
    """
    matrix = orange.SymMatrix(len(data))
    dist = distance_constructor(data)
    
    iter_count = matrix.dim * (matrix.dim - 1) / 2
    milestones = progressBarMilestones(iter_count, 100)
    
    for count, ((i, ex1), (j, ex2)) in enumerate(_pairs(enumerate(data))):
        matrix[i, j] = dist(ex1, ex2)
        if progress_callback and count in milestones:
            progress_callback(100.0 * count / iter_count)
            
    return matrix 


def feature_distance_matrix(data, distance=None, progress_callback=None):
    """ A helper function that computes an :class:`Orange.core.SymMatrix` of
    all pairwise distances between features in `data`.
    
    :param data: A data table
    :type data: :class:`Orange.data.Table`
    :param distance: a function taking two lists and returning the distance.
    :type distance: function
     
    """
    attributes = data.domain.attributes
    matrix = orange.SymMatrix(len(attributes))
    iter_count = matrix.dim * (matrix.dim - 1) / 2
    milestones = progressBarMilestones(iter_count, 100)
    
    for count, ((i, a1), (j, a2)) in enumerate(_pairs(enumerate(attributes))):
        matrix[i, j] = (1.0 - orange.PearsonCorrelation(a1, a2, data, 0).r) / 2.0
        if progress_callback and count in milestones:
            progress_callback(100.0 * count / iter_count)
            
    return matrix


def _pairs(seq, same = False):
    """ Return all pairs from elements of `seq`.
    """
    seq = list(seq)
    same = 0 if same else 1
    for i in range(len(seq)):
        for j in range(i + same, len(seq)):
            yield seq[i], seq[j]
    
    
def joining_cluster(root_cluster, item1, item2):
    """ Return the cluster where `item1` and `item2` are first joined
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
    :param item1: Index of the item or an element of `root_cluster.objects`
    :param item2: Index of the item or an element of `root_cluster.objects`
    
    """
    cluster = root_cluster
    while True:
        if cluster.branches:
            for branch in cluster.branches:
                if item1 in branch and item2 in branch:
                    cluster = branch
                    break
            else:
                return cluster
        else:
            return cluster
        

def cophenetic_distances(root_cluster):
    """ Return the cophenetic distance matrix between items in clustering.
    Cophenetic distance is defined as the height of the cluster where the 
    two items are first joined.
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
     
    """

    mapping = root_cluster.mapping  
    matrix = Orange.core.SymMatrix(len(mapping))
    for cluster in postorder(root_cluster):
        if cluster.branches:
            for branch1, branch2 in _pairs(cluster.branches):
                for idx1 in mapping[branch1.first: branch1.last]:
                    for idx2 in mapping[branch2.first: branch2.last]:
                        matrix[idx1, idx2] = cluster.height
                
        else:
            for ind1, ind2 in _pairs(mapping[cluster.first: cluster.last]):
                matrix[ind1, ind2] = cluster.height
    
    return matrix
    
def cophenetic_correlation(root_cluster, matrix):
    """ Return the `cophenetic correlation coefficient
    <http://en.wikipedia.org/wiki/Cophenetic_correlation>`_ of the given
    clustering.
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
    
    :param matrix: The distance matrix from which the `root_cluster` was
        derived.
     
    """
    import numpy
    cophenetic = cophenetic_distances(root_cluster)
    cophenetic = [list(row) for row in cophenetic]
    original = [list(row) for row in matrix]
    cophenetic = numpy.ravel(cophenetic)
    original = numpy.ravel(original)
    return numpy.corrcoef(cophenetic, original)[0, 1]
    
    
if __name__=="__main__":
    data = orange.ExampleTable("doc//datasets//brown-selected.tab")
#    data = orange.ExampleTable("doc//datasets//iris.tab")
    root = hierarchicalClustering(data, order=True) #, linkage=orange.HierarchicalClustering.Single)
    attr_root = hierarchicalClustering_attributes(data, order=True)
#    print root
#    d = DendrogramPlotPylab(root, data=data, labels=[str(ex.getclass()) for ex in data], dendrogram_width=0.4, heatmap_width=0.3,  params={}, cmap=None)
#    d.plot(show=True, filename="graph.png")

    dendrogram_draw("graph.eps", root, attr_tree=attr_root, data=data, labels=[str(e.getclass()) for e in data], tree_height=50, #width=500, height=500,
                          cluster_colors={root.right:(255,0,0), root.right.right:(0,255,0)}, 
                          color_palette=ColorPalette([(255, 0, 0), (0,0,0), (0, 255,0)], gamma=0.5, 
                                                     overflow=(255, 255, 255), underflow=(255, 255, 255))) #, minv=-0.5, maxv=0.5)
