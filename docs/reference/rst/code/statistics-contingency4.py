import Orange.statistics.contingency

table = Orange.data.Table("monks-1.tab")
cont = Orange.statistics.contingency.ClassVar("e", table)

print "Inner variable: ", cont.inner_variable.name
print "Outer variable: ", cont.outer_variable.name
print
print "Class variable: ", cont.class_var.name
print "Attribute:      ", cont.variable.name
print

print "Distributions:"
for val in cont.class_var:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print

first_value = Orange.data.Value(cont.variable, 0)
first_native = first_value.native()
print "Probabilities for e='%s'" % first_native
for val in cont.class_var:
    print "  p(%s|%s) = %5.3f" % (first_native, val.native(), cont.p_attr(first_value, val))
print

cont = Orange.statistics.contingency.ClassVar(table.domain["e"], table.domain.class_var)
for ins in table:
    cont.add_var_class(ins["e"], ins.get_class())

print "Distributions from a matrix computed manually:"
for val in cont.class_var:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
