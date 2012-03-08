# Description: FreeViz projector
# Category:    projection
# Uses:        zoo
# Referenced:  Orange.projection.linear
# Classes:     Orange.projection.linear.FreeViz, Orange.projection.linear.Projector

import Orange
zoo = Orange.data.Table('zoo')

optimizer = Orange.projection.linear.FreeViz()
projector = optimizer(zoo)

for e, projected in zip(zoo, projector(zoo))[:10]:
    print e, projected