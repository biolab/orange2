import Orange

iris = Orange.data.Table("iris")
cont = Orange.statistics.contingency.ClassVar("sepal length", iris)

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

print "Estimated for e=5.5"
for val in cont.class_var:
    print "  f(%s|%s) = %5.3f" % (5.5, val.native(), cont.p_attr(5.5, val))
print

cont = Orange.statistics.contingency.ClassVar(iris.domain["sepal length"], 
                                              iris.domain.class_var)
for ins in iris:
    cont.add_var_class(ins["sepal length"], ins.get_class())

print "Distributions from a matrix computed manually:"
for val in cont.class_var:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
