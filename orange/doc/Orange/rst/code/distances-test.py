import os
os.chdir("C:\\Python26\\Lib\\site-packages\\orange\\Orange\\distances")

import Orange
reload(Orange.distances)

# Read some data
table = Orange.data.Table("iris.tab")

# Euclidean distance constructor
d2Constr = Orange.distances.EuclideanConstructor()
d2 = d2Constr(table)

# Constructs 
dPears = Orange.distances.PearsonRConstructor(table)

#reference instance
ref = table[0]

print "Euclidean distances from the first data instance: "

for ins in table[:5]:
    print "%5.4f" % d2(ins, ref),
print 

print "Pearson correlation distance from the first data instance: "

for ins in table[:5]:
    print "%5.4f" % dPears(ins, ref),
print 