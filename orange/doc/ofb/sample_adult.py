# Author:      B Zupan
# Version:     1.0
# Description: Read 'adult' data set and select 3% of instances (use stratified sampling)
# Category:    preprocessing
# Uses:        adult.tab

import orange
data = orange.ExampleTable("../datasets/adult_sample")
selection = orange.MakeRandomIndices2(data, 0.03)
sample = data.select(selection, 0)
orange.saveTabDelimited("adult_sample_sample.tab", sample)
