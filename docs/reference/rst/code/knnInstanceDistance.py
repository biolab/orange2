import Orange

lenses = Orange.data.Table("lenses")

nnc = Orange.classification.knn.FindNearestConstructor()
nnc.distanceConstructor = Orange.distance.Euclidean()

did = Orange.data.new_meta_id()
nn = nnc(lenses, 0, did)

print "*** Reference instance: ", lenses[0]
for inst in nn(lenses[0], 5):
    print inst
