# Description: Demonstation of different type of calls of the scoring method.
# Category:    feature scoring
# Uses:        titanic
# Referenced:
# Classes:

import Orange
table = Orange.data.Table("titanic")
meas = Orange.feature.scoring.GainRatio()

print "Call with variable and data table"
print meas(0, table)

print "Call with variable and domain contingency"
domain_cont = Orange.statistics.contingency.Domain(table)
print meas(0, domain_cont)

print "Call with contingency and class distribution"
cont = Orange.statistics.contingency.VarClass(0, table)
class_dist = Orange.statistics.distribution.Distribution( \
    table.domain.class_var, table)
print meas(cont, class_dist)
