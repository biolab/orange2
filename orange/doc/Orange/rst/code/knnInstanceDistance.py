import Orange

table = Orange.data.Table("lenses")

nnc = Orange.classifier.knn.FindNearestConstructore()
nnc.distanceConstructor = Orange.core.ExamplesDistanceConstructor_Euclidean()

did = Orange.core.newmetaid()
nn = nnc(table, 0, did)

print "*** Reference instance: ", table[0]
for inst in nn(table[0], 5):
    print inst