# Description: Shows how to assess the quality of attributes
# Category:    feature scoring
# Uses:        lenses
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.Measure, Orange.features.scoring.Info

import Orange, random

lenses = Orange.data.Table("lenses")

meas = Orange.feature.scoring.InfoGain()

astigm = lenses.domain["astigmatic"]
print "Information gain of 'astigmatic': %6.4f" % meas(astigm, lenses)

classdistr = Orange.statistics.distribution.Distribution(lenses.domain.class_var, lenses)
cont = Orange.statistics.contingency.VarClass("tear_rate", lenses)
print "Information gain of 'tear_rate': %6.4f" % meas(cont, classdistr)

dcont = Orange.statistics.contingency.Domain(lenses)
print "Information gain of the first attribute: %6.4f" % meas(0, dcont)
print

print "*** A set of more exhaustive tests for different way of passing arguments to MeasureAttribute ***"

names = [a.name for a in lenses.domain.attributes]
attrs = len(names)

print ("%30s"+"%15s"*attrs) % (("",) + tuple(names))

fstr = "%30s" + "%15.4f"*attrs


print "Computing information gain directly from examples"
print fstr % (("- by attribute number:",) + tuple([meas(i, lenses) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, lenses) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, lenses) for i in lenses.domain.attributes]))
print

dcont = Orange.statistics.contingency.Domain(lenses)
print "Computing information gain from DomainContingency"
print fstr % (("- by attribute number:",) + tuple([meas(i, dcont) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(i, dcont) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(i, dcont) for i in lenses.domain.attributes]))
print

print "Computing information gain from DomainContingency"
cdist = Orange.statistics.distribution.Distribution(lenses.domain.class_var, lenses)
print fstr % (("- by attribute number:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, lenses), cdist) for i in range(attrs)]))
print fstr % (("- by attribute name:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, lenses), cdist) for i in names]))
print fstr % (("- by attribute descriptor:",) + tuple([meas(Orange.statistics.contingency.VarClass(i, lenses), cdist) for i in lenses.domain.attributes]))
print

values = ["v%i" % i for i in range(len(lenses.domain[2].values)*len(lenses.domain[3].values))]
cartesian = Orange.feature.Discrete("cart", values = values)
cartesian.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(cartesian, lenses.domain[2], lenses.domain[3], values)

print "Information gain of Cartesian product of %s and %s: %6.4f" % (lenses.domain[2].name, lenses.domain[3].name, meas(cartesian, lenses))

mid = Orange.feature.Descriptor.new_meta_id()
lenses.domain.add_meta(mid, Orange.feature.Discrete(values = ["v0", "v1"]))
lenses.add_meta_attribute(mid)

rg = random.Random()
rg.seed(0)
for ex in lenses:
    ex[mid] = Orange.data.Value(rg.randint(0, 1))

print "Information gain for a random meta attribute: %6.4f" % meas(mid, lenses)
