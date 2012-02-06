# Description: Demonstrates the use of correspondence analysis
# Category:    correspondence, projection
# Classes:     CA
# Uses:        smokers_ct.tab
import Orange, numpy
from Orange.projection import correspondence
from Orange.statistics import contingency

data = Orange.data.Table("smokers_ct.tab")
staff = data.domain["Staff group"]
smoking = data.domain["Smoking category"]

# Compute the contingency
cont = contingency.VarVar(staff, smoking, data)

c = correspondence.CA(cont, staff.values, smoking.values)

print "Row profiles"
print c.row_profiles()
print
print "Column profiles"
print c.column_profiles()

c.plot_biplot()

print "Singular values: " + str(numpy.diag(c.D))
print "Eigen values: " + str(numpy.square(numpy.diag(c.D)))
print "Percentage of Inertia:" + str(c.inertia_of_axes() / sum(c.inertia_of_axes()) * 100.0)
print

print "Principal row coordinates:"
print c.row_factors()
print
print "Decomposition Of Inertia:"
print c.column_inertia()

c.plot_scree_diagram()
