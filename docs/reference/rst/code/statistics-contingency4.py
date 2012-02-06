import Orange.statistics.contingency

monks = Orange.data.Table("monks-1.tab")
cont = Orange.statistics.contingency.ClassVar("e", monks)

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

cont = Orange.statistics.contingency.ClassVar(monks.domain["e"], monks.domain.class_var)
for ins in monks:
    cont.add_var_class(ins["e"], ins.get_class())

print "Distributions from a matrix computed manually:"
for val in cont.class_var:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
