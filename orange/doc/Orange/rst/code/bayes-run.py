import Orange
som = Orange.projection.som.SOMLearner(map_shape=(10, 20), initialize=Orange.projection.som.InitializeRandom)
map = som(Orange.data.Table("iris.tab"))
for n in map:
    print "node:", n.pos[0], n.pos[1]
    for e in n.examples:
        print "\t",e
