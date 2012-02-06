import Orange

monks = Orange.data.Table("monks-1.tab")
cont = Orange.statistics.contingency.Table(monks.domain["e"], monks.domain.classVar)
for ins in monks:
    cont [ins["e"]] [ins.get_class()] += 1

print "Contingency items:"

for val, dist in cont.items():
    print val, dist
print

for ins in monks:
    cont.add(ins["e"], ins.get_class())