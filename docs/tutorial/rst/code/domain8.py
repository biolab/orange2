# Description: Shows how to add class noise to data
# Category:    preprocessing
# Uses:        imports-85
# Classes:     Preprocessor_addClassNoise, orngTest.crossValidation
# Referenced:  domain.htm

import orange, orngTest, orngStat

filename = "promoters.tab"
data = orange.ExampleTable(filename)
data.name = "unspoiled"
datasets = [data]

add_noise = orange.Preprocessor_addClassNoise()
for noiselevel in (0.2, 0.4, 0.6):
  add_noise.proportion = noiselevel
  add_noise.random_generator = 42
  d = add_noise(data)
  d.name = "class noise %4.2f" % noiselevel
  datasets.append(d)

learner = orange.BayesLearner()

for d in datasets:
  results = orngTest.crossValidation([learner], d, folds=10)
  print "%20s   %5.3f" % (d.name, orngStat.CA(results)[0])
