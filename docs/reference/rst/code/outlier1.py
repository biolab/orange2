import Orange

bridges = Orange.data.Table("bridges")
outlierDet = Orange.preprocess.outliers.OutlierDetection()
outlierDet.set_examples(bridges)
print outlierDet.z_values()
