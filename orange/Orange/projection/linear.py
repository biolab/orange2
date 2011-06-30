'''
##############################
Linear projection (``linear``)
##############################

.. index:: linear projection

.. index::
   single: projection; linear

This module contains the FreeViz algorithm
`(Demsar et al, 2005) <http://www.ailab.si/idamap/idamap2005/papers/12%20Demsar%20CR.pdf>`_
[1], which finds a good two-dimensional projection of the given data, where the
quality is defined by a separation of the data from different classes and the
proximity of the instances from the same class. FreeViz would normally be used
through a widget since it is primarily a method for graphical exploration of
the data. About the only case where one would like to use this module directly
is to tests the classification aspects of the method, that is, to verify the
accuracy of the resulting kNN-like classifiers on a set of benchmark data sets.

Description of the method itself is far beyond the scope of this page. See the
above paper for the original version of the method; at the moment of writing
the method has been largely extended and not published yet, though the basic
principles are the same.

[1] Janez Demsar, Gregor Leban, Blaz Zupan: FreeViz - An Intelligent
Visualization Approach for Class-Labeled Multidimensional Data Sets,
Proceedings of IDAMAP 2005, Edinburgh. 

***********************
Projection Optimization
***********************

.. autoclass:: Orange.projection.linear.FreeViz
   :members:
   :show-inheritance:
   :exclude-members: attractG, attractG, autoSetParameters, cancelOptimization,
      classPermutationList, classPermutationList, findProjection,
      forceBalancing, forceSigma, getShownAttributeList, mirrorSymmetry,
      optimizeSeparation, optimize_FAST_Separation, optimize_LDA_Separation,
      optimize_SLOW_Separation, radialAnchors, randomAnchors, repelG,
      s2nMixAnchors, s2nMixData, s2nPlaceAttributes, s2nSpread,
      setStatusBarText, showAllAttributes, stepsBeforeUpdate,
      useGeneralizedEigenvectors

**********************
Learner and Classifier
**********************

.. autoclass:: Orange.projection.linear.FreeVizLearner
   :members:
   :show-inheritance:

.. autoclass:: Orange.projection.linear.FreeVizClassifier
   :members:
   :show-inheritance:

.. autoclass:: Orange.projection.linear.S2NHeuristicLearner
   :members:
   :show-inheritance:

'''


import Orange
import orangeom
import math
import random
import numpy

from numpy.linalg import inv, pinv, eig      # matrix inverse and eigenvectors
from orngScaleLinProjData import orngScaleLinProjData
import orngVisFuncts
from Orange.misc import deprecated_keywords
from Orange.misc import deprecated_members

try:
    import numpy.ma as MA
except:
    import numpy.core.ma as MA

#implementation
FAST_IMPLEMENTATION = 0
SLOW_IMPLEMENTATION = 1
LDA_IMPLEMENTATION = 2

LAW_LINEAR = 0
LAW_SQUARE = 1
LAW_GAUSSIAN = 2
LAW_KNN = 3
LAW_LINEAR_PLUS = 4

DR_PCA = 0
DR_SPCA = 1
DR_PLS = 2

def normalize(x):
    return x / numpy.linalg.norm(x)

def center(matrix):
    '''centers all variables, i.e. subtracts averages in colomns
    and divides them by their standard deviations'''
    n,m = numpy.shape(matrix)
    return (matrix - numpy.multiply(matrix.mean(axis = 0),
                                    numpy.ones((n,m))))/numpy.std(matrix,
                                                                  axis = 0)


class FreeViz:
    """
    Contains an easy-to-use interface to the core of the method, which is
    written in C++.
    
    .. attribute:: attract_g
    
    Coefficient for the attractive forces. By increasing or decreasing the ratio
    between :obj:`attract_g` and :obj:`repel_g`, you can make one kind of the
    forces stronger. Default: 1.
    
    .. attribute:: repel_g
    
    Coefficient for the repulsive forces. By increasing or decreasing the ratio
    between :obj:`attract_g` and :obj:`repel_g`, you can make one kind of the
    forces stronger. Default: 1.
    
    .. attribute:: force_balancing
    
    If set (default is False), the forces are balanced so that the total sum of
    the attractive equals the total of repulsive, before they are multiplied by
    the above factors. (By our experience, this gives bad results so you may
    want to leave this alone.)
    
    .. attribute:: law
    
    Can be LAW_LINEAR, LAW_SQUARE, LAW_GAUSSIAN, LAW_KNN or LAW_LINEAR_PLUS.
    Default is LAW_LINEAR, which means that the attractive forces increase
    linearly by the distance and the repulsive forces are inversely
    proportional to the distance. LAW_SQUARE would make them rise or fall with
    the square of the distance, LAW_GAUSSIAN is based on a kind of
    log-likelihood estimation, LAW_KNN tries to directly optimize the
    classification accuracy of the kNN classifier in the projection space, and
    in LAW_LINEAR_PLUS both forces rise with the square of the distance,
    yielding a method that is somewhat similar to PCA. We found the first law
    perform the best, with the second to not far behind.
    
    .. attribute:: force_sigma
    
    The sigma to be used in LAW_GAUSSIAN and LAW_KNN.
    
    .. attribute:: mirror_symmetry
    
    If enabled, it keeps the projection of the second attribute on the upper
    side of the graph (the first is always on the right-hand x-axis). This is
    useful when comparing whether two projections are the same, but has no
    effect on the projection's clarity or its classification accuracy.

    There are some more, undescribed, methods of a more internal nature.
    
    """
    
    def __init__(self, graph = None):
        if not graph:
            graph = orngScaleLinProjData()
        self.graph = graph

        self.implementation = 0
        self.attract_g = 1.0
        self.repel_g = 1.0
        self.law = LAW_LINEAR
        self.restrain = 0
        self.force_balancing = 0
        self.force_sigma = 1.0
        self.mirror_symmetry = 1
        self.use_generalized_eigenvectors = 1

        # s2n heuristics parameters
        self.steps_before_update = 10
        self.s2n_spread = 5
        self.s2n_place_attributes = 50
        self.s2n_mix_data = None
        self.auto_set_parameters = 1
        self.class_permutation_list = None
        self.attrs_num = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750,
                          1000]

    def clear_data(self):
        self.s2n_mix_data = None
        self.class_permutation_list = None
        
    clearData = clear_data

    def set_statusbar_text(self, *args):
        pass
    
    setStatusBarText = set_statusbar_text

    def show_all_attributes(self):
        self.graph.anchorData = [(0,0, a.name)
                                 for a in self.graph.dataDomain.attributes]
        self.radial_anchors()
        
    showAllAttributes = show_all_attributes

    def get_shown_attribute_list(self):
        return [anchor[2] for anchor in self.graph.anchorData]

    getShownAttributeList = get_shown_attribute_list

    def radial_anchors(self):
        """
        Reset the projection so that the anchors (projections of attributes)
        are placed evenly around the circle.
        
        """
        attr_list = self.get_shown_attribute_list()
        if not attr_list: return
        phi = 2*math.pi/len(attr_list)
        self.graph.anchorData = [(math.cos(i*phi), math.sin(i*phi), a)
                                 for i, a in enumerate(attr_list)]

    radialAnchors = radial_anchors

    def random_anchors(self):
        """
        Set the projection to a random one.
        
        """
        if not self.graph.haveData: return
        attr_list = self.get_shown_attribute_list()
        if not attr_list: return

        if self.restrain == 0:
            def ranch(i, label):
                r = 0.3+0.7*random.random()
                phi = 2*math.pi*random.random()
                return (r*math.cos(phi), r*math.sin(phi), label)

        elif self.restrain == 1:
            def ranch(i, label):
                phi = 2*math.pi*random.random()
                return (math.cos(phi), math.sin(phi), label)

        else:
            def ranch(i, label):
                r = 0.3+0.7*random.random()
                phi = 2*math.pi * i / max(1, len(attr_list))
                return (r*math.cos(phi), r*math.sin(phi), label)

        anchors = [ranch(*a) for a in enumerate(attr_list)]

        if not self.restrain == 1:
            maxdist = math.sqrt(max([x[0]**2+x[1]**2 for x in anchors]))
            anchors = [(x[0]/maxdist, x[1]/maxdist, x[2]) for x in anchors]

        if not self.restrain == 2 and self.mirror_symmetry:
            #### Need to rotate and mirror here
            pass

        self.graph.anchorData = anchors

    randomAnchors = random_anchors

    @deprecated_keywords({"singleStep": "single_step"})
    def optimize_separation(self, steps = 10, single_step = False, distances=None):
        """
        Optimize the class separation. If you did not change any of the settings
        which are not documented above, it will call a fast C++ routine which
        will make :obj:`steps` optimization steps at a time, after which the
        graph (if one is given) is updated. If :obj:`single_step` is True, it
        will do that only once,
        otherwise it calls it on and on, and compares the current positions of
        the anchors with those 50 calls ago. If no anchor moved for more than
        1e-3, it stops. In Orange Canvas the optimization is also stopped if
        someone outside (namely, the stop button) manages to set the FreeViz's
        flag attribute
        :obj:`Orange.projection.linear.FreeViz.cancel_optimization`.
        """
        # check if we have data and a discrete class
        if (not self.graph.haveData or len(self.graph.rawData) == 0
            or not (self.graph.dataHasClass or distances)):
            return
        ai = self.graph.attributeNameIndex
        attr_indices = [ai[label] for label in self.get_shown_attribute_list()]
        if not attr_indices: return

        if self.implementation == FAST_IMPLEMENTATION:
            return self.optimize_fast_separation(steps, single_step, distances)

        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp
        if single_step: steps = 1
        if self.implementation == SLOW_IMPLEMENTATION:
            impl = self.optimize_slow_separation
        elif self.implementation == LDA_IMPLEMENTATION:
            impl = self.optimize_lda_separation
        xanchors = None; yanchors = None

        for c in range((single_step and 1) or 50):
            for i in range(steps):
                if self.__class__ != FreeViz and self.cancel_optimization == 1:
                    return
                self.graph.anchorData, (xanchors, yanchors) = impl(attr_indices,
                                                                   self.graph.anchorData,
                                                                   xanchors,
                                                                   yanchors)
            if self.__class__ != FreeViz: qApp.processEvents()
            if hasattr(self.graph, "updateGraph"): self.graph.updateData()
            #self.recomputeEnergy()

    optimizeSeparation = optimize_separation

    @deprecated_keywords({"singleStep": "single_step"})
    def optimize_fast_separation(self, steps = 10, single_step = False, distances=None):
        optimizer = [orangeom.optimizeAnchors, orangeom.optimizeAnchorsRadial,
                     orangeom.optimizeAnchorsR][self.restrain]
        ai = self.graph.attributeNameIndex
        attr_indices = [ai[label] for label in self.get_shown_attribute_list()]
        if not attr_indices: return

        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        positions = [numpy.array([x[:2] for x in self.graph.anchorData])]
        needed_steps = 0

        valid_data = self.graph.getValidList(attr_indices)
        n_valid = sum(valid_data) 
        if not n_valid:
            return 0

        data = numpy.compress(valid_data, self.graph.noJitteringScaledData,
                              axis=1)
        data = numpy.transpose(data).tolist()
        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp

        if distances:
            if n_valid != len(valid_data):
                classes = Orange.core.SymMatrix(n_valid)
                r = 0
                for ro, vr in enumerate(valid_data):
                    if not vr:
                        continue
                    c = 0
                    for co, vr in enumerate(valid_data):
                        if vr:
                            classes[r, c] = distances[ro, co]
                            c += 1
                    r += 1  
            else:
                classes = distances
        else:
            classes = numpy.compress(valid_data,
                                     self.graph.originalData[self.graph.dataClassIndex]).tolist()
        while 1:
            self.graph.anchorData = optimizer(data, classes,
                                              self.graph.anchorData,
                                              attr_indices,
                                              attractG = self.attract_g,
                                              repelG = self.repel_g,
                                              law = self.law,
                                              sigma2 = self.force_sigma,
                                              dynamicBalancing = self.force_balancing,
                                              steps = steps,
                                              normalizeExamples = self.graph.normalizeExamples,
                                              contClass = 2 if distances
                                              else self.graph.dataHasContinuousClass,
                                              mirrorSymmetry = self.mirror_symmetry)
            needed_steps += steps

            if self.__class__ != FreeViz:
                qApp.processEvents()

            if hasattr(self.graph, "updateData"):
                self.graph.potentialsBmp = None
                self.graph.updateData()

            positions = positions[-49:]+[numpy.array([x[:2] for x
                                                      in self.graph.anchorData])]
            if len(positions)==50:
                m = max(numpy.sum((positions[0]-positions[49])**2), 0)
                if m < 1e-3: break
            if single_step or (self.__class__ != FreeViz
                               and self.cancel_optimization):
                break
        return needed_steps

    optimize_FAST_Separation = optimize_fast_separation

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "anchorData": "anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def optimize_lda_separation(self, attr_indices, anchor_data, xanchors = None, yanchors = None):
        if (not self.graph.haveData or len(self.graph.rawData) == 0
            or not self.graph.dataHasDiscreteClass): 
            return anchor_data, (xanchors, yanchors)
        class_count = len(self.graph.dataDomain.classVar.values)
        valid_data = self.graph.getValidList(attr_indices)
        selected_data = numpy.compress(valid_data,
                                       numpy.take(self.graph.noJitteringScaledData,
                                                  attr_indices, axis = 0),
                                       axis = 1)

        if xanchors == None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors == None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.createProjectionAsNumericArray(attr_indices,
                                                                    valid_data = valid_data,
                                                                    xanchors = xanchors,
                                                                    yanchors = yanchors,
                                                                    scaleFactor = self.graph.scaleFactor,
                                                                    normalize = self.graph.normalizeExamples,
                                                                    useAnchorData = 1)
        if trans_proj_data == None:
            return anchor_data, (xanchors, yanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions, y_positions, classData = (proj_data[0], proj_data[1],
                                               proj_data[2])

        averages = []
        for i in range(class_count):
            ind = classData == i
            xpos = numpy.compress(ind, x_positions)
            ypos = numpy.compress(ind, y_positions)
            xave = numpy.sum(xpos)/len(xpos)
            yave = numpy.sum(ypos)/len(ypos)
            averages.append((xave, yave))

        # compute the positions of all the points. we will try to move all points so that the center will be in the (0,0)
        x_center_vector = -numpy.sum(x_positions) / len(x_positions)
        y_center_vector = -numpy.sum(y_positions) / len(y_positions)
        center_vector_length = math.sqrt(x_center_vector*x_center_vector +
                                         y_center_vector*y_center_vector)

        mean_destination_vectors = []

        for i in range(class_count):
            xdir = 0.0; ydir = 0.0; rs = 0.0
            for j in range(class_count):
                if i==j: continue
                r = math.sqrt((averages[i][0] - averages[j][0])**2 +
                              (averages[i][1] - averages[j][1])**2)
                if r == 0.0:
                    xdir += math.cos((i/float(class_count))*2*math.pi)
                    ydir += math.sin((i/float(class_count))*2*math.pi)
                    r = 0.0001
                else:
                    xdir += (1/r**3) * ((averages[i][0] - averages[j][0]))
                    ydir += (1/r**3) * ((averages[i][1] - averages[j][1]))
                #rs += 1/r
            #actualDirAmpl = math.sqrt(xDir**2 + yDir**2)
            #s = abs(xDir)+abs(yDir)
            #xDir = rs * (xDir/s)
            #yDir = rs * (yDir/s)
            mean_destination_vectors.append((xdir, ydir))


        maxlength = math.sqrt(max([x**2 + y**2 for (x,y)
                                   in mean_destination_vectors]))
        mean_destination_vectors = [(x/(2*maxlength), y/(2*maxlength)) for (x,y)
                                    in mean_destination_vectors]     # normalize destination vectors to some normal values
        mean_destination_vectors = [(mean_destination_vectors[i][0]+averages[i][0],
                                     mean_destination_vectors[i][1]+averages[i][1])
                                    for i in range(len(mean_destination_vectors))]    # add destination vectors to the class averages
        #mean_destination_vectors = [(x + x_center_vector/5, y + y_center_vector/5) for (x,y) in mean_destination_vectors]   # center mean values
        mean_destination_vectors = [(x + x_center_vector, y + y_center_vector)
                                    for (x,y) in mean_destination_vectors]   # center mean values

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)

        for c in range(class_count):
            ind = (classData == c)
            numpy.putmask(fxs, ind, mean_destination_vectors[c][0]-x_positions)
            numpy.putmask(fys, ind, mean_destination_vectors[c][1]-y_positions)

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)))
        gxs /= (20*m); gys /= (20*m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors**2 + newyanchors**2))
        newxanchors /= m
        newyanchors /= m

        #self.parentWidget.updateGraph()

        """
        for a in range(len(anchor_data)):
            x = anchor_data[a][0]; y = anchor_data[a][1];
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [x, x+gxs[a]], yData = [y, y+gys[a]], forceFilledSymbols = 1, lineWidth=3)

        for i in range(class_count):
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [averages[i][0], mean_destination_vectors[i][0]], yData = [averages[i][1], mean_destination_vectors[i][1]], forceFilledSymbols = 1, lineWidth=3)
            self.parentWidget.graph.addCurve("lll%i" % i, QColor(0, 0, 0), QColor(0, 0, 0), 10, style = QwtPlotCurve.Lines, xData = [averages[i][0], averages[i][0]], yData = [averages[i][1], averages[i][1]], forceFilledSymbols = 1, lineWidth=5)
        """
        #self.parentWidget.graph.repaint()
        #self.graph.anchor_data = [(newxanchors[i], newyanchors[i], anchor_data[i][2]) for i in range(len(anchor_data))]
        #self.graph.updateData(attrs, 0)
        return [(newxanchors[i], newyanchors[i], anchor_data[i][2])
                for i in range(len(anchor_data))], (newxanchors, newyanchors)

    optimize_LDA_Separation = optimize_lda_separation

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "anchorData": "anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def optimize_slow_separation(self, attr_indices, anchor_data, xanchors = None, yanchors = None):
        if (not self.graph.haveData or len(self.graph.rawData) == 0
            or not self.graph.dataHasDiscreteClass): 
            return anchor_data, (xanchors, yanchors)
        valid_data = self.graph.getValidList(attr_indices)
        selected_data = numpy.compress(valid_data, numpy.take(self.graph.noJitteringScaledData,
                                                              attr_indices,
                                                              axis = 0),
                                       axis = 1)

        if xanchors == None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors == None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.createProjectionAsNumericArray(attr_indices,
                                                                    valid_data = valid_data,
                                                                    xanchors = xanchors,
                                                                    yanchors = yanchors,
                                                                    scaleFactor = self.graph.scaleFactor,
                                                                    normalize = self.graph.normalizeExamples,
                                                                    useAnchorData = 1)
        if trans_proj_data == None:
            return anchor_data, (xanchors, yanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions = proj_data[0]; x_positions2 = numpy.array(x_positions)
        y_positions = proj_data[1]; y_positions2 = numpy.array(y_positions)
        class_data = proj_data[2]  ; class_data2 = numpy.array(class_data)

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)
        gxs = numpy.zeros(len(anchor_data), numpy.float)        # gradients
        gys = numpy.zeros(len(anchor_data), numpy.float)

        rotate_array = range(len(x_positions))
        rotate_array = rotate_array[1:] + [0]
        for i in range(len(x_positions)-1):
            x_positions2 = numpy.take(x_positions2, rotate_array)
            y_positions2 = numpy.take(y_positions2, rotate_array)
            class_data2 = numpy.take(class_data2, rotate_array)
            dx = x_positions2 - x_positions
            dy = y_positions2 - y_positions
            rs2 = dx**2 + dy**2
            rs2 += numpy.where(rs2 == 0.0, 0.0001, 0.0)    # replace zeros to avoid divisions by zero
            rs = numpy.sqrt(rs2)

            F = numpy.zeros(len(x_positions), numpy.float)
            classDiff = numpy.where(class_data == class_data2, 1, 0)
            numpy.putmask(F, classDiff, 150*self.attract_g*rs2)
            numpy.putmask(F, 1-classDiff, -self.repel_g/rs2)
            fxs += F * dx / rs
            fys += F * dy / rs

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)))
        gxs /= (20*m); gys /= (20*m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors**2 + newyanchors**2))
        newxanchors /= m
        newyanchors /= m
        return [(newxanchors[i], newyanchors[i], anchor_data[i][2])
                for i in range(len(anchor_data))], (newxanchors, newyanchors)

    optimize_SLOW_Separation = optimize_slow_separation

    # ###############################################################
    # S2N HEURISTIC FUNCTIONS
    # ###############################################################



    # place a subset of attributes around the circle. this subset must contain "good" attributes for each of the class values
    @deprecated_keywords({"setAttributeListInRadviz":
                          "set_attribute_list_in_radviz"})
    def s2n_mix_anchors(self, set_attribute_list_in_radviz = 1):
        # check if we have data and a discrete class
        if (not self.graph.haveData or len(self.graph.rawData) == 0
            or not self.graph.dataHasDiscreteClass): 
            self.set_statusbar_text("S2N only works on data with a discrete class value")
            return

        # compute the quality of attributes only once
        if self.s2n_mix_data == None:
            ranked_attrs, ranked_attrs_by_class = orngVisFuncts.findAttributeGroupsForRadviz(self.graph.rawData,
                                                                                             orngVisFuncts.S2NMeasureMix())
            self.s2n_mix_data = (ranked_attrs, ranked_attrs_by_class)
            class_count = len(ranked_attrs_by_class)
            attrs = ranked_attrs[:(self.s2n_place_attributes/class_count)*
                                 class_count]    # select appropriate number of attributes
        else:
            class_count = len(self.s2n_mix_data[1])
            attrs = self.s2n_mix_data[0][:(self.s2n_place_attributes/class_count)*
                                         class_count]

        if len(attrs) == 0:
            self.set_statusbar_text("No discrete attributes found")
            return 0

        arr = [0]       # array that will tell where to put the next attribute
        for i in range(1,len(attrs)/2): arr += [i,-i]

        phi = (2*math.pi*self.s2n_spread)/(len(attrs)*10.0)
        anchor_data = []; start = []
        arr2 = arr[:(len(attrs)/class_count)+1]
        for cls in range(class_count):
            start_pos = (2*math.pi*cls)/class_count
            if self.class_permutation_list: cls = self.class_permutation_list[cls]
            attrs_cls = attrs[cls::class_count]
            temp_data = [(arr2[i], math.cos(start_pos + arr2[i]*phi),
                          math.sin(start_pos + arr2[i]*phi),
                          attrs_cls[i]) for i in
                          range(min(len(arr2), len(attrs_cls)))]
            start.append(len(anchor_data) + len(arr2)/2) # starting indices for each class value
            temp_data.sort()
            anchor_data += [(x, y, name) for (i, x, y, name) in temp_data]

        anchor_data = anchor_data[(len(attrs)/(2*class_count)):] + anchor_data[:(len(attrs)/(2*class_count))]
        self.graph.anchorData = anchor_data
        attrNames = [anchor[2] for anchor in anchor_data]

        if self.__class__ != FreeViz:
            if set_attribute_list_in_radviz:
                self.parentWidget.setShownAttributeList(attrNames)
            self.graph.updateData(attrNames)
            self.graph.repaint()
        return 1

    s2nMixAnchors = s2n_mix_anchors

    # find interesting linear projection using PCA, SPCA, or PLS
    @deprecated_keywords({"attrIndices": "attr_indices",
                          "setAnchors": "set_anchors",
                          "percentDataUsed": "percent_data_used"})
    def find_projection(self, method, attr_indices = None, set_anchors = 0, percent_data_used = 100):
        if not self.graph.haveData: return
        ai = self.graph.attributeNameIndex
        if attr_indices == None:
            attributes = self.get_shown_attribute_list()
            attr_indices = [ai[label] for label in attributes]
        if len(attr_indices) == 0: return None

        valid_data = self.graph.getValidList(attr_indices)
        if sum(valid_data) == 0: return None

        data_matrix = numpy.compress(valid_data, numpy.take(self.graph.noJitteringScaledData,
                                                            attr_indices,
                                                            axis = 0),
                                     axis = 1)
        if self.graph.dataHasClass:
            class_array = numpy.compress(valid_data,
                                         self.graph.noJitteringScaledData[self.graph.dataClassIndex])

        if percent_data_used != 100:
            indices = Orange.data.sample.SubsetIndices2(self.graph.rawData,
                                                1.0-(float(percent_data_used)/100.0))
            try:
                data_matrix = numpy.compress(indices, data_matrix, axis = 1)
            except:
                pass
            if self.graph.dataHasClass:
                class_array = numpy.compress(indices, class_array)

        vectors = None
        if method == DR_PCA:
            vals, vectors = create_pca_projection(data_matrix, ncomps = 2,
                                                  use_generalized_eigenvectors = self.use_generalized_eigenvectors)
        elif method == DR_SPCA and self.graph.dataHasClass:
            vals, vectors = create_pca_projection(data_matrix, class_array,
                                                  ncomps = 2,
                                                  use_generalized_eigenvectors = self.use_generalized_eigenvectors)
        elif method == DR_PLS and self.graph.dataHasClass:
            data_matrix = data_matrix.transpose()
            class_matrix = numpy.transpose(numpy.matrix(class_array))
            vectors = create_pls_projection(data_matrix, class_matrix, 2)
            vectors = vectors.T

        # test if all values are 0, if there is an invalid number in the array and if there are complex numbers in the array
        if (vectors == None or not vectors.any() or
            False in numpy.isfinite(vectors) or False in numpy.isreal(vectors)):
            self.set_statusbar_text("Unable to compute anchor positions for the selected attributes")  
            return None

        xanchors = vectors[0]
        yanchors = vectors[1]

        m = math.sqrt(max(xanchors**2 + yanchors**2))

        xanchors /= m
        yanchors /= m
        names = self.graph.attributeNames
        attributes = [names[attr_indices[i]] for i in range(len(attr_indices))]

        if set_anchors:
            self.graph.setAnchors(list(xanchors), list(yanchors), attributes)
            self.graph.updateData()
            self.graph.repaint()
        return xanchors, yanchors, (attributes, attr_indices)

    findProjection = find_projection


FreeViz = deprecated_members({"attractG": "attract_g",
                              "repelG": "repel_g",
                              "forceBalancing": "force_balancing",
                              "forceSigma": "force_sigma",
                              "mirrorSymmetry": "mirror_symmetry",
                              "useGeneralizedEigenvectors": "use_generalized_eigenvectors",
                              "stepsBeforeUpdate": "steps_before_update",
                              "s2nSpread": "s2n_spread",
                              "s2nPlaceAttributes": "s2n_place_attributes",
                              "s2nMixData": "s2n_mix_data",
                              "autoSetParameters": "auto_set_parameters",
                              "classPermutationList": "class_permutation_list",
                              "attrsNum": "attrs_num",
                              "cancelOptimization": "cancel_optimization"})(FreeViz)


@deprecated_keywords({"X": "x", "Y": "y", "Ncomp": "ncomp"})
def create_pls_projection(x,y, ncomp = 2):
    '''Predict y from x using first ncomp principal components'''

    # data dimensions
    n, mx = numpy.shape(x)
    my = numpy.shape(y)[1]

    # Z-scores of original matrices
    ymean = y.mean()
    x,y = center(x), center(y)

    p = numpy.empty((mx,ncomp))
    w = numpy.empty((mx,ncomp))
    c = numpy.empty((my,ncomp))
    t = numpy.empty((n,ncomp))
    u = numpy.empty((n,ncomp))
    b = numpy.zeros((ncomp,ncomp))

    e,f = x,y

    # main algorithm
    for i in range(ncomp):

        u = numpy.random.random_sample((n,1))
        w = normalize(numpy.dot(e.T,u))
        t = normalize(numpy.dot(e,w))
        c = normalize(numpy.dot(f.T,t))

        dif = t
        # iterations for loading vector t
        while numpy.linalg.norm(dif) > 10e-16:
            c = normalize(numpy.dot(f.T,t))
            u = numpy.dot(f,c)
            w = normalize(numpy.dot(e.T,u))
            t0 = normalize(numpy.dot(e,w))
            dif = t - t0
            t = t0

        t[:,i] = t.T
        u[:,i] = u.T
        c[:,i] = c.T
        w[:,i] = w.T

        b = numpy.dot(t.T,u)[0,0]
        b[i][i] = b
        p = numpy.dot(e.T,t)
        p[:,i] = p.T
        e = e - numpy.dot(t,p.T)
        xx = b * numpy.dot(t,c.T)
        f = f - xx

    # esimated y
    #YE = numpy.dot(numpy.dot(t,b),c.t)*numpy.std(y, axis = 0) + ymean
    #y = y*numpy.std(y, axis = 0)+ ymean
    #BPls = numpy.dot(numpy.dot(numpy.linalg.pinv(p.t),b),c.t)

    return w

createPLSProjection = create_pls_projection

# if no class data is provided we create PCA projection
# if there is class data then create SPCA projection
@deprecated_keywords({"dataMatrix": "data_matrix",
                      "classArray": "class_array",
                      "NComps": "ncomps",
                      "useGeneralizedEigenvectors": "use_generalized_eigenvectors"})
def create_pca_projection(data_matrix, class_array = None, ncomps = -1, use_generalized_eigenvectors = 1):
    if type(data_matrix) == numpy.ma.core.MaskedArray:
        data_matrix = numpy.array(data_matrix)
    if class_array != None and type(class_array) == numpy.ma.core.MaskedArray:
        class_array = numpy.array(class_array)
        
    data_matrix = numpy.transpose(data_matrix)

    s = numpy.sum(data_matrix, axis=0)/float(len(data_matrix))
    data_matrix -= s       # substract average value to get zero mean

    if class_array != None and use_generalized_eigenvectors:
        covarMatrix = numpy.dot(numpy.transpose(data_matrix), data_matrix)
        try:
            matrix = inv(covarMatrix)
        except:
            return None, None
        matrix = numpy.dot(matrix, numpy.transpose(data_matrix))
    else:
        matrix = numpy.transpose(data_matrix)

    # compute dataMatrixT * L * dataMatrix
    if class_array != None:
        # define the Laplacian matrix
        l = numpy.zeros((len(data_matrix), len(data_matrix)))
        for i in range(len(data_matrix)):
            for j in range(i+1, len(data_matrix)):
                l[i,j] = -int(class_array[i] != class_array[j])
                l[j,i] = -int(class_array[i] != class_array[j])

        s = numpy.sum(l, axis=0)      # doesn't matter which axis since the matrix l is symmetrical
        for i in range(len(data_matrix)):
            l[i,i] = -s[i]

        matrix = numpy.dot(matrix, l)

    matrix = numpy.dot(matrix, data_matrix)

    vals, vectors = eig(matrix)
    if vals.dtype.kind == "c":       # if eigenvalues are complex numbers then do nothing
         return None, None
    vals = list(vals)
    
    if ncomps == -1:
        ncomps = len(vals)
    ncomps = min(ncomps, len(vals))
    
    ret_vals = []
    ret_indices = []
    for i in range(ncomps):
        ret_vals.append(max(vals))
        bestind = vals.index(max(vals))
        ret_indices.append(bestind)
        vals[bestind] = -1
    
    return ret_vals, numpy.take(vectors.T, ret_indices, axis = 0)         # i-th eigenvector is the i-th column in vectors so we have to transpose the array

createPCAProjection = create_pca_projection


# #############################################################################
# class that represents freeviz classifier
class FreeVizClassifier(Orange.classification.Classifier):
    """
    A kNN classifier on the 2D projection of the data, optimized by FreeViz.
    
    Usually the learner
    (:class:`Orange.projection.linear.FreeVizLearner`) is used to construct the
    classifier.
    
    When constructing the classifier manually, the following parameters can
    be passed:
    
    :param data: table of data instances to project to a 2D plane and use for
        classification.
    :type data: :class:`Orange.data.Table`
    
    :param freeviz: the FreeViz algorithm instance to use to optimize the 2D
        projection.
    :type freeviz: :class:`Orange.projection.linear.FreeViz`
    
    """
    
    def __init__(self, data, freeviz):
        self.freeviz = freeviz

        if self.freeviz.__class__ != FreeViz:
            self.freeviz.parentWidget.setData(data)
            self.freeviz.parentWidget.show_all_attributes = 1
        else:
            self.freeviz.graph.setData(data)
            self.freeviz.show_all_attributes()

        #self.FreeViz.randomAnchors()
        self.freeviz.radial_anchors()
        self.freeviz.optimize_separation()

        graph = self.freeviz.graph
        ai = graph.attributeNameIndex
        labels = [a[2] for a in graph.anchorData]
        indices = [ai[label] for label in labels]

        valid_data = graph.getValidList(indices)
        domain = Orange.data.Domain([graph.dataDomain[i].name for i in indices]+
                               [graph.dataDomain.classVar.name],
                               graph.dataDomain)
        offsets = [graph.attrValues[graph.attributeNames[i]][0]
                   for i in indices]
        normalizers = [graph.getMinMaxVal(i) for i in indices]
        selected_data = numpy.take(graph.originalData, indices, axis = 0)
        averages = numpy.average(numpy.compress(valid_data, selected_data,
                                                axis=1), 1)
        class_data = numpy.compress(valid_data,
                                    graph.originalData[graph.dataClassIndex])

        graph.createProjectionAsNumericArray(indices, useAnchorData = 1,
                                             removeMissingData = 0,
                                             valid_data = valid_data,
                                             jitterSize = -1)
        self.classifier = Orange.classification.knn.P2NN(domain,
                                      numpy.transpose(numpy.array([numpy.compress(valid_data,
                                                                                  graph.unscaled_x_positions),
                                                                   numpy.compress(valid_data,
                                                                                  graph.unscaled_y_positions),
                                                                   class_data])),
                                      graph.anchorData, offsets, normalizers,
                                      averages, graph.normalizeExamples, law=1)

    # for a given instance run argumentation and find out to which class it most often fall
    @deprecated_keywords({"example": "instance", "returnType": "return_type"})
    def __call__(self, instance, return_type=Orange.classification.Classifier.GetValue):
        #instance.setclass(0)
        return self.classifier(instance, returntype)

FreeVizClassifier = deprecated_members({"FreeViz":"freeviz"})(FreeVizClassifier)

class FreeVizLearner(Orange.classification.Learner):
    """
    A learner that builds a :class:`FreeVizClassifier` on given data. An
    instance of :class:`FreeViz` can be passed to the constructor as a
    keyword argument :obj:`freeviz`.    

    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.
    
    """
    def __new__(cls, freeviz = None, instances = None, weight_id = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(freeviz, **argkw)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, freeviz = None):
        if not freeviz:
            freeviz = FreeViz()
        self.freeviz = freeviz
        self.name = "freeviz Learner"

    @deprecated_keywords({"examples": "instances", "weightID": "weight_id"})
    def __call__(self, instances, weight_id = 0):
        return FreeVizClassifier(instances, self.freeviz)

FreeVizLearner = deprecated_members({"FreeViz":"freeviz"})(FreeVizLearner)


class S2NHeuristicLearner(Orange.classification.Learner):
    """
    This class is not documented yet.
    
    """
    def __new__(cls, freeviz = None, instances = None, weight_id = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(freeviz, **argkw)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, freeviz = None):
        if not freeviz:
            freeviz = FreeViz()
        self.freeviz = freeviz
        self.name = "S2N Feature Selection Learner"

    @deprecated_keywords({"examples": "instances", "weightID": "weight_id"})
    def __call__(self, instances, weight_id = 0):
        return S2NHeuristicClassifier(instances, self.freeviz)

S2NHeuristicLearner = deprecated_members({"FreeViz":
                                          "freeviz"})(S2NHeuristicLearner)
