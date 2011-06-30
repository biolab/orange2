# Description: Shows how to use classes that systematically generate subsets of attributes
# Category:    feature subset selection, constructive induction
# Classes:     SubsetsGenerator, SubsetsGenerator_constSize, SubsetsGenerator_minMaxSize, SubsetsGenerator_constant
# Uses:        monk1
# Referenced:  SubsetsGenerator.htm

import orange

data = orange.ExampleTable("monk1")

print "\n\nAttributes by call-constructed subsets generator"
gen1 = orange.SubsetsGenerator_constSize(data.domain.attributes, B=3)
for attrs in gen1:
  print attrs

print "\n\nAttributes through list comprehension"
print [ss for ss in gen1]

print "\n\nSubsets by ordinary subsets generator"
gen2 = orange.SubsetsGenerator_constSize(B=3)
for attrs in gen2(data.domain.attributes):
  print attrs

def f(gen, data):
  for attrs in gen(data.domain.attributes):
    print attrs

print "\n\nSubsets by pre-constructed bound set"
gen3 = orange.SubsetsGenerator_constSize(B=3)
f(gen3, data)

print "\n\nSubsets by min-max generator"
gen4 = orange.SubsetsGenerator_minMaxSize(min=1, max=3)
for attrs in gen4(data.domain.attributes):
  print attrs


print "\n\nSubsets by constant generator"
gen5 = orange.SubsetsGenerator_constant()
gen5.constant = data.domain[:3]
for attrs in gen5(data.domain.attributes):
  print attrs

print "\n\n... once more: subsets by constant generator"
gen5 = orange.SubsetsGenerator_constant(data.domain.attributes)
gen5.constant = data.domain[:3]
for attrs in gen5:
  print attrs
