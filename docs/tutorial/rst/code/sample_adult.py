# Description: Read 'adult' data set and select 3% of instances (use stratified sampling)
# Category:    preprocessing
# Uses:        adult.tab
# Classes:     ExampleTable, MakeRandomIndices2
# Referenced:  basic_exploration.htm

import orange
data = orange.ExampleTable("adult_sample.tab")
selection = orange.MakeRandomIndices2(data, 0.03)
sample = data.select(selection, 0)
sample.save("adult_sample.tab")
