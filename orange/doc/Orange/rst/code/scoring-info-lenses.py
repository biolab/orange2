# Description: Shows how to assess the quality of attributes
# Category:    feature scoring
# Uses:        lenses
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.Measure, Orange.features.scoring.Info

import Orange
import random
table = Orange.data.Table("lenses")

meas = Orange.feature.scoring.InfoGain()

astigm = table.domain["astigmatic"]
print "Information gain of 'astigmatic': %6.4f" % meas(astigm, table)

classdistr = Orange.data.value.Distribution(table.domain.classVar, table)
cont = Orange.probability.distributions.ContingencyAttrClass("tear_rate", table)
print "Information gain of 'tear_rate': %6.4f" % meas(cont, classdistr)

dcont = Orange.probability.distributions.DomainContingency(table)
print "Information gain of the first attribute: %6.4f" % meas(0, dcont)
print

print "*** A set of more exhaustive tests for different way of passing arguments to MeasureAttribute ***"

names = [a.name for a in table.domain.attributes]
attrs = len(names)

print ("%30s"+"%15s"*attrs) % (("",) + tuple(names))

fstr = "%30s" + "%15.4f"*attrs


print "Computing information gain directly from examples"
print fstr % (("- by attribute number:",) + tuple([meas(i, table) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, table) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, table) for i in table.domain.attributes]))
print

dcont = Orange.probability.distributions.DomainContingency(table)
print "Computing information gain from DomainContingency"
print fstr % (("- by attribute number:",) + tuple([meas(i, dcont) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, dcont) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, dcont) for i in table.domain.attributes]))
print

print "Computing information gain from DomainContingency"
cdist = Orange.data.value.Distribution(table.domain.classVar, table)
print fstr % (("- by attribute number:",) + tuple([meas(Orange.probability.distributions.ContingencyAttrClass(i, table), cdist) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(Orange.probability.distributions.ContingencyAttrClass(i, table), cdist) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(Orange.probability.distributions.ContingencyAttrClass(i, table), cdist) for i in table.domain.attributes]))
print

values = ["v%i" % i for i in range(len(table.domain[2].values)*len(table.domain[3].values))]
cartesian = Orange.data.variable.Discrete("cart", values = values)
cartesian.getValueFrom = Orange.classification.lookup.ClassifierByLookupTable(cartesian, table.domain[2], table.domain[3], values)

print "Information gain of Cartesian product of %s and %s: %6.4f" % (table.domain[2].name, table.domain[3].name, meas(cartesian, table))

mid = Orange.core.newmetaid()
table.domain.addmeta(mid, Orange.data.variable.Discrete(values = ["v0", "v1"]))
table.addMetaAttribute(mid)

rg = random.Random()
rg.seed(0)
for ex in table:
    ex[mid] = Orange.data.value.Value(rg.randint(0, 1))

print "Information gain for a random meta attribute: %6.4f" % meas(mid, table)
