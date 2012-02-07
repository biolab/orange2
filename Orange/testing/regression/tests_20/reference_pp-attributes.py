# Description: Shows how to (manually) select a subset of attributes
# Category:    preprocessing, feature subset selection
# Classes:     Preprocessor, Preprocessor_select, Preprocessor_ignore
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

## Preprocessor_select

print "\n\nPreprocessor_select\n"
pp = orange.Preprocessor_select()
pp.attributes = [age, tears]

data2 = pp(data)
print "Attributes: %s, classVar %s" % (data2.domain.attributes, data2.domain.classVar)
for i in data2[:5]:
    print i    

## Preprocessor_ignore

print "\n\nPreprocessor_ignore\n"    
data2 = orange.Preprocessor_ignore(data, attributes = [age, tears])
print "Attributes: %s, classVar %s" % (data2.domain.attributes, data2.domain.classVar)
for i in data2[:5]:
    print i    


