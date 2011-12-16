import Orange
    
table = Orange.data.Table("iris")
km = Orange.clustering.kmeans.Clustering(table, 3)
print km.clusters[-10:]
