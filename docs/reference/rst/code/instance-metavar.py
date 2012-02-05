import random
import Orange
lenses = Orange.data.Table("lenses")
id = Orange.data.new_meta_id()
for inst in lenses:
    inst[id] = random.random()
print lenses[0]
