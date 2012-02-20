import orange
import orngClustering

import random
random.seed(42)

data = orange.ExampleTable("iris")
km = orngClustering.KMeans(data, 3)
print km.clusters[-10:]

