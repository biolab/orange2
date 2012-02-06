# Description: Demonstrates the use of attribute construction
# Category:    feature induction
# Classes:     FeatureByMinComplexity, FeatureByIM, FeatureByKramer, FeatureByCartesianProduct
# Uses:        monks-2.tab

import orange
import orngCI

data = orange.ExampleTable("../datasets/monks-2")

ab, quality = orngCI.FeatureByMinComplexity(data, ["a", "b"])
print "Quality: %.3f" % quality  
print "Values", ab.values

data2 = orngCI.addAnAttribute(ab, data)

c = orange.ContingencyAttrClass(ab, data)
for i in c:
    print i
    
    
ab, quality = orngCI.FeatureByIM(data, ["a", "b"])
print "Quality: %.3f" % quality  
print "Values", ab.values

data2 = orngCI.addAnAttribute(ab, data)

c = orange.ContingencyAttrClass(ab, data)
for i in c:
    print i

    
ab, quality = orngCI.FeatureByKramer(data, ["a", "b"])
print "Quality: %.3f" % quality  
print "Values", ab.values

data2 = orngCI.addAnAttribute(ab, data)

c = orange.ContingencyAttrClass(ab, data)
for i in c:
    print i
 
#Does not work
#ab, quality = orngCI.FeatureByRandom(data, ["a", "b"])
#print "Quality: %.3f" % quality  
#print "Values", ab.values
#
#data2 = orngCI.addAnAttribute(ab, data)
#
#c = orange.ContingencyAttrClass(ab, data)
#for i in c:
#    print i 
    
    
ab, quality = orngCI.FeatureByCartesianProduct(data, ["a", "b"])
print "Quality: %.3f" % quality  
print "Values", ab.values

data2 = orngCI.addAnAttribute(ab, data)

c = orange.ContingencyAttrClass(ab, data)
for i in c:
    print i 