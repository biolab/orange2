import Orange

table = Orange.data.Table("lenses")

nnc = Orange.classification.knn.FindNearestConstructor()
nnc.distanceConstructor = Orange.core.ExamplesDistanceConstructor_Euclidean()

did = Orange.data.new_meta_id()
nn = nnc(table, 0, did)

print "*** Reference instance: ", table[0]
for inst in nn(table[0], 5):
    print inst
