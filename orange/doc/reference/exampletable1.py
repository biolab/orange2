# Description: Shows how to construct an orange.ExampleTable out of nothing
# Category:    basic classes
# Classes:     ExampleTable, Domain
# Uses:        
# Referenced:  ExampleTable.htm

import orange, random
random.seed(0)

card = [3, 3, 2, 3, 4, 2]
values = ["1", "2", "3", "4"]

attributes = [orange.EnumVariable(chr(97+i), values = values[:card[i]])
              for i in range(6)]

classattr = orange.EnumVariable("y", values = ["0", "1"])
                                
domain = orange.Domain(attributes + [classattr])

data = orange.ExampleTable(domain)
for i in range(5):
    ex = [random.randint(0, c-1) for c in card]
    ex.append(ex[0]==ex[1] or ex[4]==0)
    data.append(ex)
for ex in data:
    print ex

loe = [
    ["3", "1", "1", "2", "1", "1",  "1"],
    ["3", "1", "1", "2", "2", "1",  "0"],
    ["3", "3", "1", "2", "2", "1",  "1"]]

d2 = orange.ExampleTable(domain, loe)
d2[0] = ["1", "1", 1, "1", "1", "1", "1"]