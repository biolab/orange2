import orange
import orngClustering
    
data = orange.ExampleTable("iris")
km = orngClustering.KMeans(data, 3)
print km.clusters[-10:]