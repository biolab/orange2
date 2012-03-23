# Description: FreeViz projector
# Category:    projection
# Uses:        zoo
# Referenced:  Orange.projection.linear
# Classes:     Orange.projection.linear.FreeViz, Orange.projection.linear.Projector

import Orange
import numpy as np

tab = Orange.data.Table('titanic')

ind = Orange.data.sample.SubsetIndices2(p0=0.99)(tab)
train, test = tab.select(ind, 0), tab.select(ind, 1)

freeviz = Orange.projection.linear.FreeViz()
freeviz.graph.set_data(train)
freeviz.show_all_attributes()

def mirror(tab):
    a = tab.to_numpy("a")[0]
    rotate = np.diagflat([-1 if val<0 else 1 for val in a[0]])
    a = np.dot(a, rotate)
    return Orange.data.Table(Orange.data.Domain(tab.domain.features), a)

print "PCA"
freeviz.find_projection(Orange.projection.linear.DR_PCA, set_anchors=True)
projector = freeviz()
for e, projected in zip(test, mirror(projector(test))):
    print e, projected

print "SPCA"
freeviz.find_projection(Orange.projection.linear.DR_SPCA, set_anchors=True)
projector = freeviz()
for e, projected in zip(test, mirror(projector(test))):
    print e, projected

print "SPCA w/out generalization"
freeviz.use_generalized_eigenvectors = False
freeviz.find_projection(Orange.projection.linear.DR_SPCA, set_anchors=True)
projector = freeviz()
for e, projected in zip(test, mirror(projector(test))):
    print e, projected

print "PCA with 2 attributes"
freeviz.graph.anchor_data = [(0,0, a.name) for a in freeviz.graph.data_domain
                                                    .attributes[:2]]
freeviz.find_projection(Orange.projection.linear.DR_PCA, set_anchors=True)
projector = freeviz()
for e, projected in zip(test, mirror(projector(test))):
    print e, projected
