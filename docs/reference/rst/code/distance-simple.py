import Orange

data = Orange.data.Table("iris")
#build a distance with a DistanceConstructor
measure = Orange.distance.Euclidean(data)
print "Distance between first two examples:", \
    measure(data[0], data[1]) #use the Distance

matrix = Orange.distance.distance_matrix(data)
print "Distance between first two examples:", \
    matrix[0,1]
