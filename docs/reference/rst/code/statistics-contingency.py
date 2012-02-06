import Orange

monks = Orange.data.Table("monks-1.tab")
cont = Orange.statistics.contingency.VarClass("e", monks)
for val, dist in cont.items():
    print val, dist

print cont.keys()
print cont.values()
print cont.items()

print cont[0]
print cont["1"]

for i in cont:
    print i

cont.normalize()
for val, dist in cont.items():
    print val, dist     