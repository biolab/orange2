# Description: using your own imputer and continuizer in PCA
# Category:    projection
# Uses:        adult_sample
# Referenced:  orngPCA.htm
# Classes:     orngPCA.PCA

import orange, orngPCA

data = orange.ExampleTable("bridges.tab")

imputer = orange.ImputerConstructor_maximal

continuizer = orange.DomainContinuizer()
continuizer.multinomialTreatment = continuizer.AsNormalizedOrdinal
continuizer.classTreatment = continuizer.Ignore
continuizer.continuousTreatment = continuizer.Leave

pca = orngPCA.PCA(data, standardize = True, imputer = imputer, continuizer = continuizer)
print pca