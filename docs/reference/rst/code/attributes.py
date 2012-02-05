import Orange
titanic = Orange.data.Table("titanic.tab")
var = titanic.domain[0]
print var
print "Attributes", var.attributes
var.attributes["a"] = "12"
print "Set a=12"
print "Attributes", var.attributes
