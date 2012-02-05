import Orange
    
iris = Orange.data.Table("iris")
km = Orange.clustering.kmeans.Clustering(iris, 3)
print km.clusters[-10:]
