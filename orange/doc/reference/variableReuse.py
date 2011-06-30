import orange

# Creates a new variable
v1, s = orange.Variable.make("a", orange.VarTypes.Discrete, ["a", "b"])
print s
print v1.values
print

# Reuses v1
v2, s = orange.Variable.make("a", orange.VarTypes.Discrete, ["a"], ["c"])
print s
print "v1.values: ", v1.values
print "v2 is v1: ", v2 is v1
print

# Reuses v1
v3, s = orange.Variable.make("a", orange.VarTypes.Discrete, ["a", "b", "c", "d"])
print s
print "v1.values: ", v1.values
print "v3 is v1: ", v3 is v1
print

# Creates a new one due to incompatibility
v4, s = orange.Variable.make("a", orange.VarTypes.Discrete, ["b"])
print s
print "v1.values: ", v1.values
print "v4.values: ", v4.values
print "v4 is v1: ", v4 is v1
print

# Can reuse - the order is not prescribed
v5, s = orange.Variable.make("a", orange.VarTypes.Discrete, None, ["c", "a"])
print s
print "v1.values: ", v1.values
print "v5.values: ", v5.values
print "v5 is v1: ", v5 is v1
print

# Can reuse despite missing and unrecognized values - the order is not prescribed
v6, s = orange.Variable.make("a", orange.VarTypes.Discrete, None, ["e"])
print s
print "v1.values: ", v1.values
print "v6.values: ", v6.values
print "v6 is v1: ", v6 is v1
print

# Can't reuse due to unrecognized values
v7, s = orange.Variable.make("a", orange.VarTypes.Discrete, None, ["f"], orange.Variable.MakeStatus.NoRecognizedValues)
print s
print "v1.values: ", v1.values
print "v7.values: ", v7.values
print "v7 is v1: ", v7 is v1
print

# No reuse
v8, s = orange.Variable.make("a", orange.VarTypes.Discrete, ["a", "b", "c", "d", "e"], None, orange.Variable.MakeStatus.OK)
print s
print "v1.values: ", v1.values
print "v8.values: ", v8.values
print "v8 is v1: ", v8 is v1

