import Orange
data = Orange.data.Table("unusedValues.tab")

new_variables = [Orange.preprocess.RemoveUnusedValues(var, data) for var in data.domain.variables]

print
for variable in range(len(data.domain)):
    print data.domain[variable],
    if new_variables[variable] == data.domain[variable]:
        print "retained as is"
    elif new_variables[variable]:
        print "reduced, new values are", new_variables[variable].values
    else:
        print "removed"
