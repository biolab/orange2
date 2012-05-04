#TODO: eliminate create_pls_projection (transform into a class)
#TODO: Projector as a preprocessor

import Orange
from Orange import orangeom
import math
import random
import numpy

from Orange import classification, data, feature
from Orange.classification import knn

from Orange.data.preprocess.scaling import ScaleLinProjData
from Orange.orng import orngVisFuncts as visfuncts
from Orange.utils import deprecated_keywords, deprecated_members

try:
    import numpy.ma as MA
except ImportError:
    import numpy.core.ma as MA

class enum(int):
    def set_name(self, name):
        self.name = name
        return self

    def __repr__(self):
        return getattr(self, "name", str(self))


#implementation
FAST_IMPLEMENTATION = 0
SLOW_IMPLEMENTATION = 1
LDA_IMPLEMENTATION = 2

LAW_LINEAR = enum(0).set_name("LAW_LINEAR")
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
    """centers all variables, i.e. subtracts averages in colomns
    and divides them by their standard deviations"""
    n, m = numpy.shape(matrix)
    return (matrix - numpy.multiply(matrix.mean(axis=0),
                                    numpy.ones((n, m)))) / numpy.std(matrix,
                                                                     axis=0)


class FreeViz:
    """
    Contains an easy-to-use interface to the core of the method, which is
    written in C++. Differs from other linear projection optimizers in that it itself can store the data
    to make iterative optimization and visualization possible. It can, however, still be used as any other
    projection optimizer by calling (:obj:`~Orange.projection.linear.FreeViz.__call__`) it.
    """

    #: Coefficient for the attractive forces. By increasing or decreasing the ratio
    #: between :obj:`attract_g` and :obj:`repel_g`, you can make one kind of the
    #: forces stronger.
    attract_g = 1.

    #: Coefficient for the repulsive forces. By increasing or decreasing the ratio
    #: between :obj:`attract_g` and :obj:`repel_g`, you can make one kind of the
    #: forces stronger.
    repel_g = 1.

    #: If set, the forces are balanced so that the total sum of
    #: the attractive equals the total of repulsive, before they are multiplied by
    #: the above factors. (By our experience, this gives bad results so you may
    #: want to leave this alone.)
    force_balancing = False

    #: Can be LAW_LINEAR, LAW_SQUARE, LAW_GAUSSIAN, LAW_KNN or LAW_LINEAR_PLUS.
    #: Default is LAW_LINEAR, which means that the attractive forces increase
    #: linearly by the distance and the repulsive forces are inversely
    #: proportional to the distance. LAW_SQUARE would make them rise or fall with
    #: the square of the distance, LAW_GAUSSIAN is based on a kind of
    #: log-likelihood estimation, LAW_KNN tries to directly optimize the
    #: classification accuracy of the kNN classifier in the projection space, and
    #: in LAW_LINEAR_PLUS both forces rise with the square of the distance,
    #: yielding a method that is somewhat similar to PCA. We found the first law
    #: perform the best, with the second to not far behind.
    law = LAW_LINEAR

    #: The sigma to be used in LAW_GAUSSIAN and LAW_KNN.
    force_sigma = 1.

    #: If enabled, it keeps the projection of the second attribute on the upper
    #: side of the graph (the first is always on the right-hand x-axis). This is
    #: useful when comparing whether two projections are the same, but has no
    #: effect on the projection's clarity or its classification accuracy.

    #: There are some more, undescribed, methods of a more internal nature.
    mirror_symmetry = True

    implementation = FAST_IMPLEMENTATION
    restrain = False
    use_generalized_eigenvectors = True

    # s2n heuristics parameters
    steps_before_update = 10
    s2n_spread = 5
    s2n_place_attributes = 50
    s2n_mix_data = None
    auto_set_parameters = 1
    class_permutation_list = None
    attrs_num = (5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000)

    cancel_optimization = False


    def __init__(self, graph=None):
        if not graph:
            graph = ScaleLinProjData()
        self.graph = graph

    def __call__(self, dataset=None):
        """
        Perform FreeViz optimization on the dataset, if given, and return a resulting
        linear :class:`~Orange.projection.linear.Projector`. If no dataset is given,
        the projection currently stored within the FreeViz object is returned as
        a :class:`~Orange.projection.linear.Projector`.

        :param dataset: input data set.
        :type dataset: :class:`Orange.data.Table`

        :rtype: :class:`~Orange.projection.linear.Projector`
        """
        if dataset:
            self.graph.setData(dataset)
            self.show_all_attributes()

            self.radial_anchors()
            self.optimize_separation()

            X = dataset.to_numpy_MA("a")[0]
            Xm = numpy.mean(X, axis=0)
            Xd = X - Xm
            stdev = numpy.std(Xd, axis=0)
        else:
            Xm = numpy.zeros(len(self.graph.anchor_data))
            stdev = None

        graph = self.graph

        U = numpy.array([val[:2] for val in self.graph.anchor_data]).T

        domain = graph.data_domain
        if len(domain) > len(self.graph.anchor_data):
            domain = data.Domain([graph.data_domain[a]
                                  for _, _, a in self.graph.anchor_data],
                                                                        graph.data_domain.class_var)

        return Projector(input_domain=domain,
                         center=Xm,
                         scale=stdev,
                         standardize=False,
                         projection=U)


    def clear_data(self):
        self.s2n_mix_data = None
        self.class_permutation_list = None

    clearData = clear_data

    def set_statusbar_text(self, *args):
        pass

    setStatusBarText = set_statusbar_text

    def show_all_attributes(self):
        self.graph.anchor_data = [(0, 0, a.name)
        for a in self.graph.data_domain.attributes]
        self.radial_anchors()

    showAllAttributes = show_all_attributes

    def get_shown_attribute_list(self):
        return [anchor[2] for anchor in self.graph.anchor_data]

    getShownAttributeList = get_shown_attribute_list

    def radial_anchors(self):
        """
        Reset the projection so that the anchors (projections of attributes)
        are placed evenly around the circle.
        
        """
        attr_list = self.get_shown_attribute_list()
        if not attr_list:
            return
        if "3d" in getattr(self, "parentName", "").lower():
            self.graph.anchor_data = self.graph.create_anchors(len(attr_list), attr_list)
            return
        phi = 2 * math.pi / len(attr_list)
        self.graph.anchor_data = [(math.cos(i * phi), math.sin(i * phi), a)
        for i, a in enumerate(attr_list)]

    radialAnchors = radial_anchors

    def random_anchors(self):
        """
        Set the projection to a random one.
        
        """
        if not self.graph.have_data:
            return
        attr_list = self.get_shown_attribute_list()
        if not attr_list:
            return
        if "3d" in getattr(self, "parentName", "").lower():
            if self.restrain == 0:
                def ranch(i, label):
                    r = 0.3 + 0.7 * random.random()
                    phi = 2 * math.pi * random.random()
                    theta = math.pi * random.random()
                    return (r * math.sin(theta) * math.cos(phi),
                            r * math.sin(theta) * math.sin(phi),
                            r * math.cos(theta),
                            label)
            elif self.restrain == 1:
                def ranch(i, label):
                    phi = 2 * math.pi * random.random()
                    theta = math.pi * random.random()
                    r = 1.
                    return (r * math.sin(theta) * math.cos(phi),
                            r * math.sin(theta) * math.sin(phi),
                            r * math.cos(theta),
                            label)
            else:
                self.graph.anchor_data = self.graph.create_anchors(len(attr_list), attr_list)

                def ranch(i, label):
                    r = 0.3 + 0.7 * random.random()
                    return (r * self.graph.anchor_data[i][0],
                            r * self.graph.anchor_data[i][1],
                            r * self.graph.anchor_data[i][2],
                            label)

            anchors = [ranch(*a) for a in enumerate(attr_list)]

            if not self.restrain == 1:
                maxdist = math.sqrt(max([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 for x in anchors]))
                anchors = [(x[0] / maxdist, x[1] / maxdist, x[2] / maxdist, x[3]) for x in anchors]

            self.graph.anchor_data = anchors
            return

        if self.restrain == 0:
            def ranch(i, label):
                r = 0.3 + 0.7 * random.random()
                phi = 2 * math.pi * random.random()
                return r * math.cos(phi), r * math.sin(phi), label

        elif self.restrain == 1:
            def ranch(i, label):
                phi = 2 * math.pi * random.random()
                return math.cos(phi), math.sin(phi), label

        else:
            def ranch(i, label):
                r = 0.3 + 0.7 * random.random()
                phi = 2 * math.pi * i / max(1, len(attr_list))
                return r * math.cos(phi), r * math.sin(phi), label

        anchors = [ranch(*a) for a in enumerate(attr_list)]

        if not self.restrain == 1:
            maxdist = math.sqrt(max([x[0] ** 2 + x[1] ** 2 for x in anchors]))
            anchors = [(x[0] / maxdist, x[1] / maxdist, x[2]) for x in anchors]

        if not self.restrain == 2 and self.mirror_symmetry:
            #TODO: Need to rotate and mirror here
            pass

        self.graph.anchor_data = anchors

    randomAnchors = random_anchors

    @deprecated_keywords({"singleStep": "single_step"})
    def optimize_separation(self, steps=10, single_step=False, distances=None):
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
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not (self.graph.data_has_class or distances)):
            return
        ai = self.graph.attribute_name_index
        attr_indices = [ai[label] for label in self.get_shown_attribute_list()]
        if not attr_indices: return

        if self.implementation == FAST_IMPLEMENTATION and not hasattr(self, '_use_3D'): # TODO
            return self.optimize_fast_separation(steps, single_step, distances)
        elif self.implementation == LDA_IMPLEMENTATION:
            impl = self.optimize_lda_separation
        else:
            impl = self.optimize_slow_separation

        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp
        if single_step: steps = 1
        xanchors = None
        yanchors = None
        zanchors = None

        if hasattr(self, '_use_3D'):
            if self.implementation == SLOW_IMPLEMENTATION:
                impl = self.optimize_slow_separation_3D
            elif self.implementation == LDA_IMPLEMENTATION:
                impl = self.optimize_lda_separation_3D
            else:
                print('Unimplemented method!')
                return

            for c in range((single_step and 1) or 50):
                for i in range(steps):
                    if self.__class__ != FreeViz and self.cancel_optimization == 1:
                        return
                    self.graph.anchor_data, (xanchors, yanchors, zanchors) = impl(attr_indices,
                                                                                  self.graph.anchor_data,
                                                                                  xanchors,
                                                                                  yanchors,
                                                                                  zanchors)
                if self.__class__ != FreeViz: qApp.processEvents()
                if hasattr(self.graph, "updateGraph"): self.graph.updateData()
        else:
            for c in range((single_step and 1) or 50):
                for i in range(steps):
                    if self.__class__ != FreeViz and self.cancel_optimization == 1:
                        return
                    self.graph.anchor_data, (xanchors, yanchors) = impl(attr_indices,
                                                                        self.graph.anchor_data,
                                                                        xanchors,
                                                                        yanchors)
                if self.__class__ != FreeViz: qApp.processEvents()
                if hasattr(self.graph, "updateGraph"): self.graph.updateData()

    optimizeSeparation = optimize_separation

    @deprecated_keywords({"singleStep": "single_step"})
    def optimize_fast_separation(self, steps=10, single_step=False, distances=None):
        optimizer = [orangeom.optimizeAnchors, orangeom.optimizeAnchorsRadial,
                     orangeom.optimizeAnchorsR][self.restrain]
        ai = self.graph.attribute_name_index
        attr_indices = [ai[label] for label in self.get_shown_attribute_list()]
        if not attr_indices: return

        # repeat until less than 1% energy decrease in 5 consecutive iterations*steps steps
        positions = [numpy.array([x[:2] for x in self.graph.anchor_data])]
        needed_steps = 0

        valid_data = self.graph.get_valid_list(attr_indices)
        n_valid = sum(valid_data)
        if not n_valid:
            return 0

        dataset = numpy.compress(valid_data, self.graph.no_jittering_scaled_data,
                                 axis=1)
        dataset = numpy.transpose(dataset).tolist()
        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp

        if distances:
            if n_valid != len(valid_data):
                classes = Orange.misc.SymMatrix(n_valid)
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
                                     self.graph.original_data[self.graph.data_class_index]).tolist()
        while 1:
            self.graph.anchor_data = optimizer(dataset, classes,
                                               self.graph.anchor_data,
                                               attr_indices,
                                               attractG=self.attract_g,
                                               repelG=self.repel_g,
                                               law=self.law,
                                               sigma2=self.force_sigma,
                                               dynamicBalancing=self.force_balancing,
                                               steps=steps,
                                               normalizeExamples=self.graph.normalize_examples,
                                               contClass=2 if distances
                                               else self.graph.data_has_continuous_class,
                                               mirrorSymmetry=self.mirror_symmetry)
            needed_steps += steps

            if self.__class__ != FreeViz:
                qApp.processEvents()

            if hasattr(self.graph, "updateData"):
                self.graph.potentials_emp = None
                self.graph.updateData()

            positions = positions[-49:] + [numpy.array([x[:2] for x
                                                        in self.graph.anchor_data])]
            if len(positions) == 50:
                m = max(numpy.sum((positions[0] - positions[49]) ** 2), 0)
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
    def optimize_lda_separation(self, attr_indices, anchor_data, xanchors=None, yanchors=None):
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not self.graph.data_has_discrete_class):
            return anchor_data, (xanchors, yanchors)
        class_count = len(self.graph.data_domain.classVar.values)
        valid_data = self.graph.get_valid_list(attr_indices)
        selected_data = numpy.compress(valid_data,
                                       numpy.take(self.graph.no_jittering_scaled_data,
                                                  attr_indices, axis=0),
                                       axis=1)

        if xanchors is None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors is None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.create_projection_as_numeric_array(attr_indices,
                                                                        validData=valid_data,
                                                                        xanchors=xanchors,
                                                                        yanchors=yanchors,
                                                                        scaleFactor=self.graph.scale_factor,
                                                                        normalize=self.graph.normalize_examples,
                                                                        useAnchorData=1)
        if trans_proj_data is None:
            return anchor_data, (xanchors, yanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions, y_positions, classData = (proj_data[0], proj_data[1],
                                               proj_data[2])

        averages = []
        for i in range(class_count):
            ind = classData == i
            xpos = numpy.compress(ind, x_positions)
            ypos = numpy.compress(ind, y_positions)
            xave = numpy.sum(xpos) / len(xpos)
            yave = numpy.sum(ypos) / len(ypos)
            averages.append((xave, yave))

        # compute the positions of all the points. we will try to move all points so that the center will be in the (0,0)
        x_center_vector = -numpy.sum(x_positions) / len(x_positions)
        y_center_vector = -numpy.sum(y_positions) / len(y_positions)

        mean_destination_vectors = []

        for i in range(class_count):
            xdir, ydir = 0., 0.
            for j in range(class_count):
                if i == j: continue
                r = math.sqrt((averages[i][0] - averages[j][0]) ** 2 +
                              (averages[i][1] - averages[j][1]) ** 2)
                if r == 0.0:
                    xdir += math.cos((i / float(class_count)) * 2 * math.pi)
                    ydir += math.sin((i / float(class_count)) * 2 * math.pi)
                else:
                    xdir += (1 / r ** 3) * ((averages[i][0] - averages[j][0]))
                    ydir += (1 / r ** 3) * ((averages[i][1] - averages[j][1]))

            mean_destination_vectors.append((xdir, ydir))

        maxlength = math.sqrt(max([x ** 2 + y ** 2 for (x, y)
                                   in mean_destination_vectors]))
        mean_destination_vectors = [(x / (2 * maxlength), y / (2 * maxlength))
        for (x, y) in mean_destination_vectors]     #normalize destination vectors to some normal values
        mean_destination_vectors = [(mean_destination_vectors[i][0] + averages[i][0],
                                     mean_destination_vectors[i][1] + averages[i][1])
        for i in range(len(mean_destination_vectors))]    # add destination vectors to the class averages
        mean_destination_vectors = [(x + x_center_vector, y + y_center_vector)
        for (x, y) in mean_destination_vectors]   # center mean values

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)

        for c in range(class_count):
            ind = (classData == c)
            numpy.putmask(fxs, ind, mean_destination_vectors[c][0] - x_positions)
            numpy.putmask(fys, ind, mean_destination_vectors[c][1] - y_positions)

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)))
        gxs /= (20 * m)
        gys /= (20 * m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors ** 2 + newyanchors ** 2))
        newxanchors /= m
        newyanchors /= m

        return [(newxanchors[i], newyanchors[i], anchor_data[i][2])
        for i in range(len(anchor_data))], (newxanchors, newyanchors)

    optimize_LDA_Separation = optimize_lda_separation

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "anchorData": "anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def optimize_slow_separation(self, attr_indices, anchor_data, xanchors=None, yanchors=None):
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not self.graph.data_has_discrete_class):
            return anchor_data, (xanchors, yanchors)
        valid_data = self.graph.get_valid_list(attr_indices)
        selected_data = numpy.compress(valid_data, numpy.take(self.graph.no_jittering_scaled_data,
                                                              attr_indices,
                                                              axis=0),
                                       axis=1)

        if xanchors is None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors is None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.create_projection_as_numeric_array(attr_indices,
                                                                        validData=valid_data,
                                                                        xanchors=xanchors,
                                                                        yanchors=yanchors,
                                                                        scaleFactor=self.graph.scale_factor,
                                                                        normalize=self.graph.normalize_examples,
                                                                        useAnchorData=1)
        if trans_proj_data is None:
            return anchor_data, (xanchors, yanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions = proj_data[0]
        x_positions2 = numpy.array(x_positions)
        y_positions = proj_data[1]
        y_positions2 = numpy.array(y_positions)
        class_data = proj_data[2]
        class_data2 = numpy.array(class_data)

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)

        rotate_array = range(len(x_positions))
        rotate_array = rotate_array[1:] + [0]
        for i in range(len(x_positions) - 1):
            x_positions2 = numpy.take(x_positions2, rotate_array)
            y_positions2 = numpy.take(y_positions2, rotate_array)
            class_data2 = numpy.take(class_data2, rotate_array)
            dx = x_positions2 - x_positions
            dy = y_positions2 - y_positions
            rs2 = dx ** 2 + dy ** 2
            rs2 += numpy.where(rs2 == 0.0, 0.0001, 0.0)    # replace zeros to avoid divisions by zero
            rs = numpy.sqrt(rs2)

            F = numpy.zeros(len(x_positions), numpy.float)
            classDiff = numpy.where(class_data == class_data2, 1, 0)
            numpy.putmask(F, classDiff, 150 * self.attract_g * rs2)
            numpy.putmask(F, 1 - classDiff, -self.repel_g / rs2)
            fxs += F * dx / rs
            fys += F * dy / rs

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)))
        gxs /= (20 * m)
        gys /= (20 * m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors ** 2 + newyanchors ** 2))
        newxanchors /= m
        newyanchors /= m
        return [(newxanchors[i], newyanchors[i], anchor_data[i][2])
        for i in range(len(anchor_data))], (newxanchors, newyanchors)

    optimize_SLOW_Separation = optimize_slow_separation


    @deprecated_keywords({"attrIndices": "attr_indices",
                          "anchorData": "anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def optimize_lda_separation_3D(self, attr_indices, anchor_data, xanchors=None, yanchors=None, zanchors=None):
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not self.graph.data_has_discrete_class):
            return anchor_data, (xanchors, yanchors, zanchors)
        class_count = len(self.graph.data_domain.classVar.values)
        valid_data = self.graph.get_valid_list(attr_indices)
        selected_data = numpy.compress(valid_data,
                                       numpy.take(self.graph.no_jittering_scaled_data,
                                                  attr_indices, axis=0),
                                       axis=1)

        if xanchors is None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors is None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)
        if zanchors is None:
            zanchors = numpy.array([a[2] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.create_projection_as_numeric_array(attr_indices,
                                                                        validData=valid_data,
                                                                        xanchors=xanchors,
                                                                        yanchors=yanchors,
                                                                        zanchors=zanchors,
                                                                        scaleFactor=self.graph.scale_factor,
                                                                        normalize=self.graph.normalize_examples,
                                                                        useAnchorData=1)
        if trans_proj_data is None:
            return anchor_data, (xanchors, yanchors, zanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions, y_positions, z_positions, classData = (proj_data[0],
                                                            proj_data[1],
                                                            proj_data[2],
                                                            proj_data[3])

        averages = []
        for i in range(class_count):
            ind = classData == i
            xpos = numpy.compress(ind, x_positions)
            ypos = numpy.compress(ind, y_positions)
            zpos = numpy.compress(ind, z_positions)
            xave = numpy.sum(xpos) / len(xpos)
            yave = numpy.sum(ypos) / len(ypos)
            zave = numpy.sum(zpos) / len(zpos)
            averages.append((xave, yave, zave))

        # compute the positions of all the points. we will try to move all points so that the center will be in the (0,0)
        x_center_vector = -numpy.sum(x_positions) / len(x_positions)
        y_center_vector = -numpy.sum(y_positions) / len(y_positions)

        mean_destination_vectors = []

        for i in range(class_count):
            xdir, ydir = 0., 0.
            for j in range(class_count):
                if i == j: continue
                r = math.sqrt((averages[i][0] - averages[j][0]) ** 2 +
                              (averages[i][1] - averages[j][1]) ** 2)
                if r == 0.0:
                    xdir += math.cos((i / float(class_count)) * 2 * math.pi)
                    ydir += math.sin((i / float(class_count)) * 2 * math.pi)
                else:
                    xdir += (1 / r ** 3) * ((averages[i][0] - averages[j][0]))
                    ydir += (1 / r ** 3) * ((averages[i][1] - averages[j][1]))

            mean_destination_vectors.append((xdir, ydir))

        maxlength = math.sqrt(max([x ** 2 + y ** 2 for (x, y)
                                   in mean_destination_vectors]))
        mean_destination_vectors = [(x / (2 * maxlength), y / (2 * maxlength)) for (x, y)
                                                                               in
                                                                               mean_destination_vectors]     # normalize destination vectors to some normal values
        mean_destination_vectors = [(mean_destination_vectors[i][0] + averages[i][0],
                                     mean_destination_vectors[i][1] + averages[i][1])
        for i in range(len(mean_destination_vectors))]    # add destination vectors to the class averages
        mean_destination_vectors = [(x + x_center_vector, y + y_center_vector)
        for (x, y) in mean_destination_vectors]   # center mean values

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)

        for c in range(class_count):
            ind = (classData == c)
            numpy.putmask(fxs, ind, mean_destination_vectors[c][0] - x_positions)
            numpy.putmask(fys, ind, mean_destination_vectors[c][1] - y_positions)

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)))
        gxs /= (20 * m)
        gys /= (20 * m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors ** 2 + newyanchors ** 2))
        newxanchors /= m
        newyanchors /= m

        return [(newxanchors[i], newyanchors[i], anchor_data[i][2])
        for i in range(len(anchor_data))], (newxanchors, newyanchors)

    optimize_LDA_Separation_3D = optimize_lda_separation_3D

    @deprecated_keywords({"attrIndices": "attr_indices",
                          "anchorData": "anchor_data",
                          "XAnchors": "xanchors",
                          "YAnchors": "yanchors"})
    def optimize_slow_separation_3D(self, attr_indices, anchor_data, xanchors=None, yanchors=None, zanchors=None):
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not self.graph.data_has_discrete_class):
            return anchor_data, (xanchors, yanchors, zanchors)
        valid_data = self.graph.get_valid_list(attr_indices)
        selected_data = numpy.compress(valid_data, numpy.take(self.graph.no_jittering_scaled_data,
                                                              attr_indices,
                                                              axis=0),
                                       axis=1)

        if xanchors is None:
            xanchors = numpy.array([a[0] for a in anchor_data], numpy.float)
        if yanchors is None:
            yanchors = numpy.array([a[1] for a in anchor_data], numpy.float)
        if zanchors is None:
            zanchors = numpy.array([a[2] for a in anchor_data], numpy.float)

        trans_proj_data = self.graph.create_projection_as_numeric_array(attr_indices,
                                                                        validData=valid_data,
                                                                        XAnchors=xanchors,
                                                                        YAnchors=yanchors,
                                                                        ZAnchors=zanchors,
                                                                        scaleFactor=self.graph.scale_factor,
                                                                        normalize=self.graph.normalize_examples,
                                                                        useAnchorData=1)
        if trans_proj_data is None:
            return anchor_data, (xanchors, yanchors, zanchors)

        proj_data = numpy.transpose(trans_proj_data)
        x_positions = proj_data[0]
        x_positions2 = numpy.array(x_positions)
        y_positions = proj_data[1]
        y_positions2 = numpy.array(y_positions)
        z_positions = proj_data[2]
        z_positions2 = numpy.array(z_positions)
        class_data = proj_data[3]
        class_data2 = numpy.array(class_data)

        fxs = numpy.zeros(len(x_positions), numpy.float)        # forces
        fys = numpy.zeros(len(x_positions), numpy.float)
        fzs = numpy.zeros(len(x_positions), numpy.float)

        rotate_array = range(len(x_positions))
        rotate_array = rotate_array[1:] + [0]
        for i in range(len(x_positions) - 1):
            x_positions2 = numpy.take(x_positions2, rotate_array)
            y_positions2 = numpy.take(y_positions2, rotate_array)
            z_positions2 = numpy.take(z_positions2, rotate_array)
            class_data2 = numpy.take(class_data2, rotate_array)
            dx = x_positions2 - x_positions
            dy = y_positions2 - y_positions
            dz = z_positions2 - z_positions
            rs2 = dx ** 2 + dy ** 2 + dz ** 2
            rs2 += numpy.where(rs2 == 0.0, 0.0001, 0.0)    # replace zeros to avoid divisions by zero
            rs = numpy.sqrt(rs2)

            F = numpy.zeros(len(x_positions), numpy.float)
            classDiff = numpy.where(class_data == class_data2, 1, 0)
            numpy.putmask(F, classDiff, 150 * self.attract_g * rs2)
            numpy.putmask(F, 1 - classDiff, -self.repel_g / rs2)
            fxs += F * dx / rs
            fys += F * dy / rs
            fzs += F * dz / rs

        # compute gradient for all anchors
        gxs = numpy.array([sum(fxs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gys = numpy.array([sum(fys * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)
        gzs = numpy.array([sum(fzs * selected_data[i])
                           for i in range(len(anchor_data))], numpy.float)

        m = max(max(abs(gxs)), max(abs(gys)), max(abs(gzs)))
        gxs /= (20 * m)
        gys /= (20 * m)
        gzs /= (20 * m)

        newxanchors = xanchors + gxs
        newyanchors = yanchors + gys
        newzanchors = zanchors + gzs

        # normalize so that the anchor most far away will lie on the circle
        m = math.sqrt(max(newxanchors ** 2 + newyanchors ** 2 + newzanchors ** 2))
        newxanchors /= m
        newyanchors /= m
        newzanchors /= m
        return [(newxanchors[i], newyanchors[i], newzanchors[i], anchor_data[i][3])
        for i in range(len(anchor_data))], (newxanchors, newyanchors, newzanchors)

    optimize_SLOW_Separation_3D = optimize_slow_separation_3D



    # ###############################################################
    # S2N HEURISTIC FUNCTIONS
    # ###############################################################



    # place a subset of attributes around the circle. this subset must contain "good" attributes for each of the class values
    @deprecated_keywords({"setAttributeListInRadviz":
                              "set_attribute_list_in_radviz"})
    def s2n_mix_anchors(self, set_attribute_list_in_radviz=1):
        # check if we have data and a discrete class
        if (not self.graph.have_data or len(self.graph.raw_data) == 0
            or not self.graph.data_has_discrete_class):
            self.set_statusbar_text("S2N only works on data with a discrete class value")
            return

        # compute the quality of attributes only once
        if self.s2n_mix_data is None:
            ranked_attrs, ranked_attrs_by_class = visfuncts.findAttributeGroupsForRadviz(self.graph.raw_data,
                                                                                         visfuncts.S2NMeasureMix())
            self.s2n_mix_data = (ranked_attrs, ranked_attrs_by_class)
            class_count = len(ranked_attrs_by_class)
            attrs = ranked_attrs[:(self.s2n_place_attributes / class_count) *
                                  class_count]    # select appropriate number of attributes
        else:
            class_count = len(self.s2n_mix_data[1])
            attrs = self.s2n_mix_data[0][:(self.s2n_place_attributes / class_count) *
                                          class_count]

        if not len(attrs):
            self.set_statusbar_text("No discrete attributes found")
            return 0

        arr = [0]       # array that will tell where to put the next attribute
        for i in range(1, len(attrs) / 2): arr += [i, -i]

        phi = (2 * math.pi * self.s2n_spread) / (len(attrs) * 10.0)
        anchor_data = [];
        start = []
        arr2 = arr[:(len(attrs) / class_count) + 1]
        for cls in range(class_count):
            start_pos = (2 * math.pi * cls) / class_count
            if self.class_permutation_list: cls = self.class_permutation_list[cls]
            attrs_cls = attrs[cls::class_count]
            temp_data = [(arr2[i], math.cos(start_pos + arr2[i] * phi),
                          math.sin(start_pos + arr2[i] * phi),
                          attrs_cls[i]) for i in
                                        range(min(len(arr2), len(attrs_cls)))]
            start.append(len(anchor_data) + len(arr2) / 2) # starting indices for each class value
            temp_data.sort()
            anchor_data += [(x, y, name) for (i, x, y, name) in temp_data]

        anchor_data = anchor_data[(len(attrs) / (2 * class_count)):] + anchor_data[:(len(attrs) / (2 * class_count))]
        self.graph.anchor_data = anchor_data
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
    def find_projection(self, method, attr_indices=None, set_anchors=0, percent_data_used=100):
        if not self.graph.have_data: return
        ai = self.graph.attribute_name_index
        if attr_indices is None:
            attributes = self.get_shown_attribute_list()
            attr_indices = [ai[label] for label in attributes]
        if not len(attr_indices): return None

        valid_data = self.graph.get_valid_list(attr_indices)
        if sum(valid_data) == 0: return None

        data_matrix = numpy.compress(valid_data, numpy.take(self.graph.no_jittering_scaled_data,
                                                            attr_indices,
                                                            axis=0),
                                     axis=1)
        if self.graph.data_has_class:
            class_array = numpy.compress(valid_data,
                                         self.graph.no_jittering_scaled_data[self.graph.data_class_index])

        if percent_data_used != 100:
            indices = data.sample.SubsetIndices2(self.graph.raw_data,
                                                 1.0 - (float(percent_data_used) / 100.0))
            try:
                data_matrix = numpy.compress(indices, data_matrix, axis=1)
            except ValueError:
                pass
            if self.graph.data_has_class:
                class_array = numpy.compress(indices, class_array)

        ncomps = 3 if hasattr(self, '_use_3D') else 2
        vectors = None
        if method == DR_PCA:
            pca = Pca(standardize=False, max_components=ncomps,
                      use_generalized_eigenvectors=False, ddof=0)
            domain = data.Domain([feature.Continuous("g%d" % i) for i
                                  in xrange(len(data_matrix))], False)
            pca = pca(data.Table(domain, data_matrix.T))
            vals, vectors = pca.variances, pca.projection
        elif method == DR_SPCA and self.graph.data_has_class:
            pca = Spca(standardize=False, max_components=ncomps,
                       use_generalized_eigenvectors=self.use_generalized_eigenvectors, ddof=0)
            domain = data.Domain([feature.Continuous("g%d" % i) for i
                                  in xrange(len(data_matrix))], feature.Continuous("c"))
            pca = pca(data.Table(domain,
                                 numpy.hstack([data_matrix.T, numpy.array(class_array, ndmin=2).T])))
            vals, vectors = pca.variances, pca.projection
        elif method == DR_PLS and self.graph.data_has_class:
            data_matrix = data_matrix.transpose()
            class_matrix = numpy.transpose(numpy.matrix(class_array))
            vectors = create_pls_projection(data_matrix, class_matrix, ncomps)
            vectors = vectors.T

        # test if all values are 0, if there is an invalid number in the array and if there are complex numbers in the array
        if (vectors is None or not vectors.any() or
            False in numpy.isfinite(vectors) or False in numpy.isreal(vectors)):
            self.set_statusbar_text("Unable to compute anchor positions for the selected attributes")
            return None

        xanchors = vectors[0]
        yanchors = vectors[1]

        if ncomps == 3:
            zanchors = vectors[2]
            m = math.sqrt(max(xanchors ** 2 + yanchors ** 2 + zanchors ** 2))
            zanchors /= m
        else:
            m = math.sqrt(max(xanchors ** 2 + yanchors ** 2))

        xanchors /= m
        yanchors /= m
        names = self.graph.attribute_names
        attributes = [names[attr_indices[i]] for i in range(len(attr_indices))]

        if set_anchors:
            if ncomps == 3:
                self.graph.set_anchors(list(xanchors), list(yanchors), list(zanchors), attributes)
            else:
                self.graph.set_anchors(list(xanchors), list(yanchors), attributes)
            if hasattr(self.graph, "updateData"):
                self.graph.updateData()
            if hasattr(self.graph, "repaint"):
                self.graph.repaint()

        if ncomps == 3:
            return xanchors, yanchors, zanchors, (attributes, attr_indices)
        else:
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
def create_pls_projection(x, y, ncomp=2):
    """Predict y from x using first ncomp principal components"""

    # data dimensions
    n, mx = numpy.shape(x)
    my = numpy.shape(y)[1]

    # Z-scores of original matrices
    ymean = y.mean()
    x, y = center(x), center(y)

    p = numpy.empty((mx, ncomp))
    w = numpy.empty((mx, ncomp))
    c = numpy.empty((my, ncomp))
    t = numpy.empty((n, ncomp))
    u = numpy.empty((n, ncomp))
    b = numpy.zeros((ncomp, ncomp))

    e, f = x, y

    # main algorithm
    for i in range(ncomp):
        u = numpy.random.random_sample((n, 1))
        w = normalize(numpy.dot(e.T, u))
        t = normalize(numpy.dot(e, w))
        c = normalize(numpy.dot(f.T, t))

        dif = t
        # iterations for loading vector t
        while numpy.linalg.norm(dif) > 10e-16:
            c = normalize(numpy.dot(f.T, t))
            u = numpy.dot(f, c)
            w = normalize(numpy.dot(e.T, u))
            t0 = normalize(numpy.dot(e, w))
            dif = t - t0
            t = t0

        t[:, i] = t.T
        u[:, i] = u.T
        c[:, i] = c.T
        w[:, i] = w.T

        b = numpy.dot(t.T, u)[0, 0]
        b[i][i] = b
        p = numpy.dot(e.T, t)
        p[:, i] = p.T
        e = e - numpy.dot(t, p.T)
        xx = b * numpy.dot(t, c.T)
        f = f - xx

    return w

createPLSProjection = create_pls_projection

# #############################################################################
# class that represents freeviz classifier
class FreeVizClassifier(classification.Classifier):
    """
    A kNN classifier on the 2D projection of the data, optimized by FreeViz.
    
    Usually the learner
    (:class:`Orange.projection.linear.FreeVizLearner`) is used to construct the
    classifier.
    
    When constructing the classifier manually, the following parameters can
    be passed:
    
    :param dataset: table of data instances to project to a 2D plane and use for
        classification.
    :type dataset: :class:`Orange.data.Table`
    
    :param freeviz: the FreeViz algorithm instance to use to optimize the 2D
        projection.
    :type freeviz: :class:`Orange.projection.linear.FreeViz`
    
    """

    def __init__(self, dataset, freeviz):
        self.freeviz = freeviz

        if self.freeviz.__class__ != FreeViz:
            self.freeviz.parentWidget.setData(dataset)
            self.freeviz.parentWidget.showAllAttributes = 1
        else:
            self.freeviz.graph.set_data(dataset)
            self.freeviz.show_all_attributes()

        self.freeviz.radial_anchors()
        self.freeviz.optimize_separation()

        graph = self.freeviz.graph
        ai = graph.attribute_name_index
        labels = [a[2] for a in graph.anchor_data]
        indices = [ai[label] for label in labels]

        valid_data = graph.get_valid_list(indices)
        domain = data.Domain([graph.data_domain[i].name for i in indices] +
                             [graph.data_domain.classVar.name],
                             graph.data_domain)
        offsets = [graph.attr_values[graph.attribute_names[i]][0]
                   for i in indices]
        normalizers = [graph.get_min_max_val(i) for i in indices]
        selected_data = numpy.take(graph.original_data, indices, axis=0)
        averages = numpy.average(numpy.compress(valid_data, selected_data,
                                                axis=1), 1)
        class_data = numpy.compress(valid_data,
                                    graph.original_data[graph.data_class_index])

        graph.create_projection_as_numeric_array(indices, use_anchor_data=1,
                                                 remove_missing_data=0,
                                                 valid_data=valid_data,
                                                 jitter_size=-1)
        self.classifier = knn.P2NN(domain,
                                   numpy.transpose(numpy.array([numpy.compress(valid_data,
                                                                               graph.unscaled_x_positions),
                                                                numpy.compress(valid_data,
                                                                               graph.unscaled_y_positions),
                                                                class_data])),
                                   graph.anchor_data, offsets, normalizers,
                                   averages, graph.normalize_examples, law=1)

    # for a given instance run argumentation and find out to which class it most often fall
    @deprecated_keywords({"example": "instance", "returnType": "return_type"})
    def __call__(self, instance, return_type=classification.Classifier.GetValue):
        return self.classifier(instance, return_type)

FreeVizClassifier = deprecated_members({"FreeViz": "freeviz"})(FreeVizClassifier)

class FreeVizLearner(classification.Learner):
    """
    A learner that builds a :class:`FreeVizClassifier` on given data. An
    instance of :class:`FreeViz` can be passed to the constructor as a
    keyword argument :obj:`freeviz`.    

    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.
    
    """

    def __new__(cls, freeviz=None, instances=None, weight_id=0, **argkw):
        self = classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(freeviz, **argkw)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, freeviz=None, **kwd):
        if not freeviz:
            freeviz = FreeViz()
        self.freeviz = freeviz
        self.name = "freeviz Learner"

    @deprecated_keywords({"examples": "instances", "weightID": "weight_id"})
    def __call__(self, instances, weight_id=0):
        return FreeVizClassifier(instances, self.freeviz)

FreeVizLearner = deprecated_members({"FreeViz": "freeviz"})(FreeVizLearner)


class S2NHeuristicLearner(classification.Learner):
    """
    This class is not documented yet.
    
    """

    def __new__(cls, freeviz=None, instances=None, weight_id=0, **argkw):
        self = classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(freeviz, **argkw)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, freeviz=None, **kwd):
        if not freeviz:
            freeviz = FreeViz()
        self.freeviz = freeviz
        self.name = "S2N Feature Selection Learner"

    @deprecated_keywords({"examples": "instances", "weightID": "weight_id"})
    def __call__(self, instances, weight_id=0):
        return S2NHeuristicClassifier(instances, self.freeviz)

S2NHeuristicLearner = deprecated_members({"FreeViz":
                                              "freeviz"})(S2NHeuristicLearner)

class Projector(object):
    """
    Stores a linear projection of data and uses it to transform any given data with matching input domain.
    """
    #: Domain of the data set that was used to construct principal component subspace.
    input_domain = None

    #: Domain used in returned data sets. This domain has a continuous
    #: variable for each axis in the projected space,
    #: and no class variable(s).
    output_domain = None

    #: Array containing means of each variable in the data set that was used
    #: to construct the projection.
    center = numpy.array(())

    #: An array containing standard deviations of each variable in the data
    #: set that was used to construct the projection.
    scale = numpy.array(())

    #: True, if standardization was used when constructing the projection. If
    #: set, instances will be standardized before being projected.
    standardize = True

    #: Array containing projection (vectors that describe the
    #: transformation from input to output domain).
    projection = numpy.array(()).reshape(0, 0)

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

        features = []
        for i in range(len(self.projection)):
            f = feature.Continuous("Comp.%d" % (i + 1))
            f.get_value_from = _ProjectSingleComponent(self, f, i)
            features.append(f)

        self.output_domain = Orange.data.Domain(features,
                                                self.input_domain.class_var,
                                                class_vars=self.input_domain.class_vars)

    def __call__(self, dataset):
        """
        Project data.

        :param dataset: input data set
        :type dataset: :class:`Orange.data.Table`

        :rtype: :class:`Orange.data.Table`
        """
        if not isinstance(dataset, data.Table):
            dataset = data.Table([dataset])
        if len(self.projection.T) != len(dataset.domain.features):
            dataset = data.Table(self.input_domain, dataset)

        X, = dataset.to_numpy_MA("a")
        Xm, U = self.center, self.projection
        n, m = X.shape

        if m != len(self.projection.T):
            raise ValueError, "Invalid number of features"

        Xd = X - Xm

        if self.standardize:
            Xd /= self.scale

        self.A = numpy.dot(Xd, U.T)

        class_, classes = dataset.to_numpy_MA("c")[0], dataset.to_numpy_MA("m")[0]
        return data.Table(self.output_domain, numpy.hstack((self.A, class_, classes)))

class _ProjectSingleComponent():
    def __init__(self, projector, feature, idx):
        self.projector = projector
        self.feature = feature
        self.idx = idx

    def __call__(self, example, return_what):
        if len(self.projector.projection.T) != len(example.domain.features):
            ex = Orange.data.Table(self.projector.input_domain, [example])
        else:
            ex = Orange.data.Table([example])
        ex = ex.to_numpy_MA("a")[0]
        ex -= self.projector.center
        if self.projector.standardize:
            ex /= self.projector.scale
        return self.feature(numpy.dot(self.projector.projection[self.idx, :], ex.T)[0])


#color table for biplot
Colors = ['bo', 'go', 'yo', 'co', 'mo']

class PCA(object):
    """
    Orthogonal transformation of data into a set of uncorrelated variables called
    principal components. This transformation is defined in such a way that the
    first variable has as high variance as possible.

    :param standardize: perform standardization of the data set.
    :type standardize: boolean
    :param max_components: maximum number of retained components.
    :type max_components: int
    :param variance_covered: percent of the variance to cover with components.
    :type variance_covered: float
    :param use_generalized_eigenvectors: use generalized eigenvectors (ie.
        multiply data matrix with inverse of its covariance matrix).
    :type use_generalized_eigenvectors: boolean
    """

    #: Delta degrees of freedom used for numpy operations.
    #: 1 means normalization with (N-1) in cov and std operations
    ddof = 1

    def __new__(cls, dataset=None, **kwds):
        optimizer = object.__new__(cls)

        if dataset:
            optimizer.__init__(**kwds)
            return optimizer(dataset)
        else:
            return optimizer

    def __init__(self, standardize=True, max_components=0, variance_covered=1,
                 use_generalized_eigenvectors=0, ddof=1):
        self.standardize = standardize
        self.max_components = max_components
        self.variance_covered = min(1, variance_covered)
        self.use_generalized_eigenvectors = use_generalized_eigenvectors
        self.ddof=ddof

    def __call__(self, dataset):
        """
        Perform a PCA analysis on a data set and return a linear projector
        that maps data into principal component subspace.

        :param dataset: input data set.
        :type dataset: :class:`Orange.data.Table`

        :rtype: :class:`~Orange.projection.linear.PcaProjector`
        """
        Xd, stdev, mean, relevant_features = self._normalize_data(dataset)

        #use generalized eigenvectors
        if self.use_generalized_eigenvectors:
            inv_covar = numpy.linalg.inv(numpy.dot(Xd.T, Xd))
            Xg = numpy.dot(inv_covar, Xd.T)
        else:
            Xg = Xd.T

        components, variances = self._perform_pca(dataset, Xd, Xg)
        components = self._insert_zeros_for_constant_features(len(dataset.domain.features),
                                                              components,
                                                              relevant_features)

        variances, components, variance_sum = self._select_components(variances, components)

        n, m = components.shape

        return PcaProjector(input_domain=dataset.domain,
                            center=mean,
                            scale=stdev,
                            standardize=self.standardize,
                            projection=components,
                            variances=variances,
                            variance_sum=variance_sum)

    def _normalize_data(self, dataset):
        if not len(dataset) or not len(dataset.domain.features):
            raise ValueError("Empty dataset")
        X = dataset.to_numpy_MA("a")[0]

        Xm = numpy.mean(X, axis=0)
        Xd = X - Xm

        if not Xd.any():
            raise ValueError("All features are constant")

        #take care of the constant features
        stdev = numpy.std(Xd, axis=0, ddof=self.ddof)
        stdev[stdev == 0] = 1. # to prevent division by zero
        relevant_features = stdev != 0
        Xd = Xd[:, relevant_features]
        if self.standardize:
            Xd /= stdev[relevant_features]
        return Xd, stdev, Xm, relevant_features

    def _perform_pca(self, dataset, Xd, Xg):
        n, m = Xd.shape
        if n < m:
            _, D, U = numpy.linalg.svd(Xd, full_matrices=0)
            D *= D / (n - self.ddof)
        else:
            C = numpy.dot(Xg, Xd) / (n - self.ddof)
            U, D, T = numpy.linalg.svd(C)
            U = U.T  # eigenvectors are now in rows
        return U, D

    def _select_components(self, D, U):
        variance_sum = D.sum()
        #select eigen vectors
        if self.variance_covered != 1:
            nfeatures = numpy.searchsorted(numpy.cumsum(D) / variance_sum,
                                           self.variance_covered) + 1
            U = U[:nfeatures, :]
            D = D[:nfeatures]
        if self.max_components > 0:
            U = U[:self.max_components, :]
            D = D[:self.max_components]
        return D, U, variance_sum

    def _insert_zeros_for_constant_features(self, original_dimension, U, relevant_features):
        n, m = U.shape
        if m != original_dimension:
            U_ = numpy.zeros((n, original_dimension))
            U_[:, relevant_features] = U
            U = U_
        return U

Pca = PCA


class Spca(PCA):
    def _perform_pca(self, dataset, Xd, Xg):
        # define the Laplacian matrix
        c = dataset.to_numpy("c")[0]
        l = -numpy.array(numpy.hstack([(c != v) for v in c]), dtype='f')
        l -= numpy.diagflat(numpy.sum(l, axis=0))

        Xg = numpy.dot(Xg, l)

        return Pca._perform_pca(self, dataset, Xd, Xg)


@deprecated_members({"pc_domain": "output_domain"})
class PcaProjector(Projector):
    #: Array containing variances of principal components.
    variances = numpy.array(())

    #: Sum of all variances in the data set that was used to construct the PCA space.
    variance_sum = 0.

    def __str__(self):
        ncomponents = 10
        s = self.variance_sum
        cs = numpy.cumsum(self.variances) / s
        stdev = numpy.sqrt(self.variances)
        return "\n".join([
            "PCA SUMMARY",
            "",
            "Std. deviation of components:",
            " ".join(["              "] +
                     ["%10s" % a.name for a in self.output_domain.attributes]),
            " ".join(["Std. deviation"] +
                     ["%10.3f" % a for a in stdev]),
            " ".join(["Proportion Var"] +
                     ["%10.3f" % a for a in self.variances / s * 100]),
            " ".join(["Cumulative Var"] +
                     ["%10.3f" % a for a in cs * 100]),
            "",
            ]) if len(self.output_domain) <= ncomponents else\
        "\n".join([
            "PCA SUMMARY",
            "",
            "Std. deviation of components:",
            " ".join(["              "] +
                     ["%10s" % a.name for a in self.output_domain.attributes[:ncomponents]] +
                     ["%10s" % "..."] +
                     ["%10s" % self.output_domain.attributes[-1].name]),
            " ".join(["Std. deviation"] +
                     ["%10.3f" % a for a in stdev[:ncomponents]] +
                     ["%10s" % ""] +
                     ["%10.3f" % stdev[-1]]),
            " ".join(["Proportion Var"] +
                     ["%10.3f" % a for a in self.variances[:ncomponents] / s * 100] +
                     ["%10s" % ""] +
                     ["%10.3f" % (self.variances[-1] / s * 100)]),
            " ".join(["Cumulative Var"] +
                     ["%10.3f" % a for a in cs[:ncomponents] * 100] +
                     ["%10s" % ""] +
                     ["%10.3f" % (cs[-1] * 100)]),
            "",
            ])


    def scree_plot(self, filename=None, title='Scree Plot'):
        """
        Draw a scree plot of principal components

        :param filename: Name of the file to which the plot will be saved.
                         If None, plot will be displayed instead.
        :type filename: str
        :param title: Plot title
        :type title: str
        """
        import pylab as plt

        s = self.variance_sum
        vc = self.variances / s
        cs = numpy.cumsum(self.variances) / s

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x_axis = range(len(self.variances))
        plt.grid(True)

        ax.set_xlabel('Principal Component Number')
        ax.set_ylabel('Proportion of Variance')
        ax.set_title(title + "\n")
        ax.plot(x_axis, vc, color="red")
        ax.scatter(x_axis, vc, color="red", label="Variance")

        ax.plot(x_axis, cs, color="orange")
        ax.scatter(x_axis, cs, color="orange", label="Cumulative Variance")
        ax.legend(loc=0)

        ax.axis([-0.5, len(self.variances) - 0.5, 0, 1])

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def biplot(self, filename=None, components=(0, 1), title='Biplot'):
        """
        Draw biplot for PCA. Actual projection must be performed via pca(data)
        before bipot can be used.

        :param filename: Name of the file to which the plot will be saved.
                         If None, plot will be displayed instead.
        :type filename: str
        :param components: List of two components to plot.
        :type components: list
        :param title: Plot title
        :type title: str
        """
        import pylab as plt

        if len(components) < 2:
            raise ValueError, 'Two components are needed for biplot'

        if not (0 <= min(components) <= max(components) < len(self.variances)):
            raise ValueError, 'Invalid components'

        X = self.A[:, components[0]]
        Y = self.A[:, components[1]]

        vectorsX = self.variances[:, components[0]]
        vectorsY = self.variances[:, components[1]]

        max_load_value = self.variances.max() * 1.5

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title(title + "\n")
        ax1.set_xlabel("PC%s (%d%%)" % (components[0], self.variances[components[0]] / self.variance_sum * 100))
        ax1.set_ylabel("PC%s (%d%%)" % (components[1], self.variances[components[1]] / self.variance_sum * 100))
        ax1.xaxis.set_label_position('bottom')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('left')

        ax1.plot(X, Y, Colors[0])


        #eliminate double axis on right
        ax0 = ax1.twinx()
        ax0.yaxis.set_visible(False)

        ax2 = ax0.twiny()
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.set_ticks_position('top')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        arrowprops = dict(facecolor='red', edgecolor='red', width=1, headwidth=4)

        for (x, y, a) in zip(vectorsX, vectorsY, self.input_domain.attributes):
            if max(x, y) < 0.1:
                continue
            print x, y, a
            ax2.annotate('', (x, y), (0, 0), arrowprops=arrowprops)
            ax2.text(x * 1.1, y * 1.2, a.name, color='red')

        ax2.set_xlim(-max_load_value, max_load_value)
        ax2.set_ylim(-max_load_value, max_load_value)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()


class Fda(object):
    """
    Construct a linear projection of data using FDA. When using this projection optimization method, data is always
    standardized prior to being projected.

    If data instances are provided to the constructor,
    the optimization algorithm is called and the resulting projector
    (:class:`~Orange.projection.linear.FdaProjector`) is
    returned instead of the optimizer (instance of this class).

    :rtype: :class:`~Orange.projection.linear.Fda` or
            :class:`~Orange.projection.linear.FdaProjector`
    """

    def __new__(cls, dataset=None):
        self = object.__new__(cls)
        if dataset:
            self.__init__()
            return self.__call__(dataset)
        else:
            return self

    def __call__(self, dataset):
        """
        Perform a FDA analysis on a data set and return a linear projector
        that maps data into another vector space.

        :param dataset: input data set.
        :type dataset: :class:`Orange.data.Table`

        :rtype: :class:`~Orange.projection.linear.FdaProjector`
        """
        X, Y = dataset.to_numpy_MA("a/c")

        Xm = numpy.mean(X, axis=0)
        X -= Xm

        #take care of the constant features
        stdev = numpy.std(X, axis=0)
        relevant_features = stdev != 0
        stdev[stdev == 0] = 1.
        X /= stdev
        X = X[:, relevant_features]

        instances, features = X.shape
        class_count = len(set(Y))
        # special case when we have two classes
        if class_count == 2:
            data1 = MA.take(X, numpy.argwhere(Y == 0).flatten(), axis=0)
            data2 = MA.take(X, numpy.argwhere(Y != 0).flatten(), axis=0)
            miDiff = MA.average(data1, axis=1) - MA.average(data2, axis=1)
            covMatrix = (MA.dot(data1.T, data1) + MA.dot(data2.T, data2)) / instances
            U = numpy.linalg.inv(covMatrix) * miDiff
            D = numpy.array([1])
        else:
            # compute means and average covariances of examples in each class group
            Sw = MA.zeros([features, features])
            for v in set(Y):
                d = MA.take(X, numpy.argwhere(Y == v).flatten(), axis=0)
                d -= numpy.mean(d, axis=0)
                Sw += MA.dot(d.T, d)
            Sw /= instances
            total = MA.dot(X.T, X) / float(instances)
            Sb = total - Sw

            matrix = numpy.linalg.inv(Sw) * Sb
            D, U = numpy.linalg.eigh(matrix)

        sorted_indices = [i for _, i in sorted([(ev, i)
        for i, ev in enumerate(D)], reverse=True)]
        U = numpy.take(U, sorted_indices, axis=1)
        D = numpy.take(D, sorted_indices)

        #insert zeros for constant features
        n, m = U.shape
        if m != M:
            U_ = numpy.zeros((n, M))
            U_[:, relevant_features] = U
            U = U_

        out_domain = data.Domain([feature.Continuous("Comp.%d" %
                                                     (i + 1)) for
                                  i in range(len(D))], False)

        return FdaProjector(input_domain=dataset.domain,
                            output_domain=out_domain,
                            center=Xm,
                            scale=stdev,
                            standardize=True,
                            eigen_vectors=U,
                            projection=U,
                            eigen_values=D)


class FdaProjector(Projector):
    """
    .. attribute:: eigen_vectors

        Synonymous for :obj:`~Orange.projection.linear.Projector.projection`.

    .. attribute:: eigen_values

        Array containing eigenvalues corresponding to eigenvectors.

    """


@deprecated_keywords({"dataMatrix": "data_matrix",
                      "classArray": "class_array",
                      "NComps": "ncomps",
                      "useGeneralizedEigenvectors": "use_generalized_eigenvectors"})
def create_pca_projection(data_matrix, class_array=None, ncomps=-1, use_generalized_eigenvectors=True):
    import warnings

    warnings.warn("Deprecated in favour of Orange"
                  ".projection.linear.Pca.",
                  DeprecationWarning)
    if isinstance(data_matrix, numpy.ma.core.MaskedArray):
        data_matrix = numpy.array(data_matrix)
    if isinstance(class_array, numpy.ma.core.MaskedArray):
        class_array = numpy.array(class_array)

    data_matrix = numpy.transpose(data_matrix)

    s = numpy.sum(data_matrix, axis=0) / float(len(data_matrix))
    data_matrix -= s       # substract average value to get zero mean

    if class_array is not None and use_generalized_eigenvectors:
        covarMatrix = numpy.dot(numpy.transpose(data_matrix), data_matrix)
        try:
            matrix = numpy.linalg.inv(covarMatrix)
        except numpy.linalg.LinAlgError:
            return None, None
        matrix = numpy.dot(matrix, numpy.transpose(data_matrix))
    else:
        matrix = numpy.transpose(data_matrix)

    # compute dataMatrixT * L * dataMatrix
    if class_array is not None:
        # define the Laplacian matrix
        l = numpy.zeros((len(data_matrix), len(data_matrix)))
        for i in range(len(data_matrix)):
            for j in range(i + 1, len(data_matrix)):
                l[i, j] = -int(class_array[i] != class_array[j])
                l[j, i] = -int(class_array[i] != class_array[j])

        s = numpy.sum(l, axis=0)      # doesn't matter which axis since the matrix l is symmetrical
        for i in range(len(data_matrix)):
            l[i, i] = -s[i]

        matrix = numpy.dot(matrix, l)

    matrix = numpy.dot(matrix, data_matrix)

    vals, vectors = numpy.linalg.eig(matrix)
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

    return ret_vals, numpy.take(vectors.T, ret_indices,
                                axis=0)         # i-th eigenvector is the i-th column in vectors so we have to transpose the array

createPCAProjection = create_pca_projection
