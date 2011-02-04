import Orange
import distributions

table = Orange.data.Table("monks-1.tab")
cont = distributions.ContingencyAttrClass("e", table)
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