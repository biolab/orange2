# Description: Shows how to remove or select examples by values of attributes
# Category:    preprocessing
# Classes:     Preprocessor, Preprocessor_take, Preprocessor_drop
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

print "\n\nSelecting examples that have prescription 'hypermetrope' and are 'young' or 'pre-presbyopic'\n"
pp = orange.Preprocessor_take()
pp.values[prescr] = "hypermetrope"
pp.values[age] = ["young", "pre-presbyopic"]
data2 = pp(data)

for ex in data2:
    print ex

print "\n\nRemoving examples that have prescription 'hypermetrope' and are 'young' or 'pre-presbyopic'\n"
pp = orange.Preprocessor_drop()
pp.values[prescr] = "hypermetrope"
pp.values[age] = ["young", "pre-presbyopic"]
data2 = pp(data)

for ex in data2:
    print ex
