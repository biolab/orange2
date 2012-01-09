# Description: Shows how to assess the quality of attributes
# Category:    feature scoring
# Uses:        lenses
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.Measure, Orange.features.scoring.Info

import Orange, random

table = Orange.data.Table("lenses")

meas = Orange.feature.scoring.InfoGain()

astigm = table.domain["astigmatic"]
print "Information gain of 'astigmatic': %6.4f" % meas(astigm, table)

classdistr = Orange.statistics.distribution.Distribution(table.domain.class_var, table)
cont = Orange.statistics.contingency.VarClass("tear_rate", table)
print "Information gain of 'tear_rate': %6.4f" % meas(cont, classdistr)

dcont = Orange.statistics.contingency.Domain(table)
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

dcont = Orange.statistics.contingency.Domain(table)
print "Computing information gain from DomainContingency"
print fstr % (("- by attribute number:",) + tuple([meas(i, dcont) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, dcont) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, dcont) for i in table.domain.attributes]))
print

print "Computing information gain from DomainContingency"
cdist = Orange.statistics.distribution.Distribution(table.domain.class_var, table)
print fstr % (("- by attribute number:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, table), cdist) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, table), cdist) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, table), cdist) for i in table.domain.attributes]))
print

values = ["v%i" % i for i in range(len(table.domain[2].values)*len(table.domain[3].values))]
cartesian = Orange.data.variable.Discrete("cart", values = values)
cartesian.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(cartesian, table.domain[2], table.domain[3], values)

print "Information gain of Cartesian product of %s and %s: %6.4f" % (table.domain[2].name, table.domain[3].name, meas(cartesian, table))

mid = Orange.data.new_meta_id()
table.domain.add_meta(mid, Orange.data.variable.Discrete(values = ["v0", "v1"]))
table.add_meta_attribute(mid)

rg = random.Random()
rg.seed(0)
for ex in table:
    ex[mid] = Orange.data.Value(rg.randint(0, 1))

print "Information gain for a random meta attribute: %6.4f" % meas(mid, table)
