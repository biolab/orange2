# Description: Example of advanced use of MDS 
# Category:    association
# Classes:     orngMDS.MDS
# Referenced:  orngMDS.htm
# Uses:        iris.tab

import orange, orngMDS, math

data=orange.ExampleTable("../datasets/iris.tab")
dist = orange.ExamplesDistanceConstructor_Euclidean(data)
matrix = orange.SymMatrix(len(data))
for i in range(len(data)-1):
   for j in range(i+1, len(data)):
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
    print "<%4.2f, %4.2f> %s" % (p[0], p[1], e)
