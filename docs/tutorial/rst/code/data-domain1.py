import Orange

data = Orange.data.Table("imports-85.tab")
m = len(data.domain.features)
m_cont = sum(1 for x in data.domain.features if x.var_type==Orange.feature.Type.Continuous)
m_disc = sum(1 for x in data.domain.features if x.var_type==Orange.feature.Type.Discrete)
m_disc = len(data.domain.features)
print "%d features, %d continuous and %d discrete" % (m, m_cont, m-m_cont)

print "First three features:"
for i in range(3):
    print "   ", data.domain.features[i].name

print "First three features (again):"
for x in data.domain.features[:3]:
    print "   ", x.name

print "Class:", data.domain.class_var.name
