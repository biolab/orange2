# Description: Shows how to assess the quality of attributes not in the dataset
# Category:    attribute quality
# Classes:     EntropyDiscretization, MeasureAttribute, MeasureAttribute_info
# Uses:        iris
# Referenced:  MeasureAttribute.htm

import orange, random
data = orange.ExampleTable("iris")

d1 = orange.EntropyDiscretization("petal length", data)
print orange.MeasureAttribute_info(d1, data)
