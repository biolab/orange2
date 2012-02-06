# Description: Shows how to assess the quality of attributes
# Category:    attribute quality
# Classes:     MeasureAttribute, MeasureAttribute_info, 
# Uses:        lenses
# Referenced:  MeasureAttribute.htm

import orange, random
data = orange.ExampleTable("lenses")

meas = orange.MeasureAttribute_info()

astigm = data.domain["astigmatic"]
print "Information gain of 'astigmatic': %6.4f" % meas(astigm, data)

classdistr = orange.Distribution(data.domain.classVar, data)
cont = orange.ContingencyAttrClass("tear_rate", data)
print "Information gain of 'tear_rate': %6.4f" % meas(cont, classdistr)

dcont = orange.DomainContingency(data)
print "Information gain of the first attribute: %6.4f" % meas(0, dcont)
print

print "*** A set of more exhaustive tests for different way of passing arguments to MeasureAttribute ***"

names = [a.name for a in data.domain.attributes]
attrs = len(names)

print ("%30s"+"%15s"*attrs) % (("",) + tuple(names))

fstr = "%30s" + "%15.4f"*attrs


print "Computing information gain directly from examples"
print fstr % (("- by attribute number:",) + tuple([meas(i, data) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, data) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, data) for i in data.domain.attributes]))
print

dcont = orange.DomainContingency(data)
print "Computing information gain from DomainContingency"
print fstr % (("- by attribute number:",) + tuple([meas(i, dcont) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, dcont) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, dcont) for i in data.domain.attributes]))
print

print "Computing information gain from DomainContingency"
cdist = orange.Distribution(data.domain.classVar, data)
print fstr % (("- by attribute number:",) + tuple([meas(orange.ContingencyAttrClass(i, data), cdist) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(orange.ContingencyAttrClass(i, data), cdist) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(orange.ContingencyAttrClass(i, data), cdist) for i in data.domain.attributes]))
print

values = ["v%i" % i for i in range(len(data.domain[2].values)*len(data.domain[3].values))]
cartesian = orange.EnumVariable("cart", values = values)
cartesian.getValueFrom = orange.ClassifierByLookupTable(cartesian, data.domain[2], data.domain[3], values)

print "Information gain of Cartesian product of %s and %s: %6.4f" % (data.domain[2].name, data.domain[3].name, meas(cartesian, data))

mid = orange.newmetaid()
data.domain.addmeta(mid, orange.EnumVariable(values = ["v0", "v1"]))
data.addMetaAttribute(mid)

rg = random.Random()
rg.seed(0)
for ex in data:
    ex[mid] = orange.Value(rg.randint(0, 1))

print "Information gain for a random meta attribute: %6.4f" % meas(mid, data)
