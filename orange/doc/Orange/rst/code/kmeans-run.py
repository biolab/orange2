import orange
import Orange.cluster
    
data = orange.ExampleTable("iris")
km = Orange.cluster.KMeans(data, 3)
print km.clusters[-10:]
