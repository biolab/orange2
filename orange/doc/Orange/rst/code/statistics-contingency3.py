import Orange.statistics.contingency

table = Orange.data.Table("monks-1.tab")
cont = Orange.statistics.contingency.VarClass("e", table)

print "Inner variable: ", cont.inner_variable.name
print "Outer variable: ", cont.outer_variable.name
print
print "Class variable: ", cont.class_var.name
print "Feature:      ", cont.variable.name
print

print "Distributions:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print

first_class = Orange.data.Value(cont.class_var, 1)
first_native = first_class.native()
print "Probabilities of class '%s'" % first_native
for val in cont.variable:
    print "  p(%s|%s) = %5.3f" % (first_native, val.native(), 
                                  cont.p_class(val, first_class))
print

cont = Orange.statistics.contingency.VarClass(table.domain["e"],
                                              table.domain.class_var)
for ins in table:
    cont.add_var_class(ins["e"], ins.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print
