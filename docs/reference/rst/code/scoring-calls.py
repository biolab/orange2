# Description: Demonstation of different type of calls of the scoring method.
# Category:    feature scoring
# Uses:        titanic
# Referenced:
# Classes:

import Orange
titanic = Orange.data.Table("titanic")
meas = Orange.feature.scoring.GainRatio()

print "Call with variable and data table"
print meas(0, titanic)

print "Call with variable and domain contingency"
domain_cont = Orange.statistics.contingency.Domain(titanic)
print meas(0, domain_cont)

print "Call with contingency and class distribution"
cont = Orange.statistics.contingency.VarClass(0, titanic)
class_dist = Orange.statistics.distribution.Distribution( \
    titanic.domain.class_var, titanic)
print meas(cont, class_dist)
