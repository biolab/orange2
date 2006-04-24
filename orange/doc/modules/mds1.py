import orange
import orngMDS
import math
data=orange.ExampleTable("../datasets/iris.tab")
dist = orange.ExamplesDistanceConstructor_Euclidean(data)
matrix = orange.SymMatrix(len(data))
for i in range(len(data)):
   for j in range(i+1):
       matrix[i, j] = dist(data[i], data[j])
mds=orngMDS.MDS(matrix)
#mds.Torgerson()
mds.getStress(orngMDS.KruskalStress)
i=0
while 100>i:
    i+=1
    oldStress=mds.avgStress
    for j in range(10): mds.SMACOFstep()
    mds.getStress(orngMDS.KruskalStress)
    if oldStress*1e-3 > math.fabs(oldStress-mds.avgStress):
        break;
for (p, e) in zip(mds.points, data):
    print p, e 
