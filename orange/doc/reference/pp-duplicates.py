# Description: Shows how to remove or merge duplicate example
# Category:    preprocessing, duplicate examples
# Classes:     Preprocessor, Preprocessor_removeDuplicates, Preprocessor_ignore
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

print "\n\nPreprocessor_removeDuplicates\n"

print "Before removal\n"
data2 = orange.Preprocessor_ignore(data, attributes = [age])
for ex in data2:
    print ex

print "After removal\n"
data2, weightID = orange.Preprocessor_removeDuplicates(data2)
# we add a meta attribute so that output is always the same
# (else, the meta id would depend upon the number of meta attributes
# constructed, which would trigger suspicions about randomness in testing scripts
data2.domain.addmeta(weightID, orange.FloatVariable("#"))
for ex in data2:
    print ex


