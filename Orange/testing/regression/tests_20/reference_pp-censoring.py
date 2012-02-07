# Description: Shows how to add weight that reflect the confidence in examples in presence of censoring
# Category:    preprocessing, survival analysis, censoring, weighting
# Classes:     Preprocessor, Preprocessor_addCensorWeight
# Uses:        wpbc
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("wpbc")

time = data.domain["time"]
fail = data.domain.classVar.values.index("R")

data2, weightID = orange.Preprocessor_addCensorWeight(
   data, 0, # 0 = no initial weights
   eventValue = fail, timeVar=time, maxTime = 20,
   method = orange.Preprocessor_addCensorWeight.KM)
   
print "class\ttime\tweight"
for ex in data2.select(recur="N", time=(0, 10)):
    print "%s\t%5.2f\t%5.3f" % (ex.getclass(), float(ex["time"]), ex.getmeta(weightID))
print