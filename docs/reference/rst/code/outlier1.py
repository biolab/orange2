import Orange

bridges = Orange.data.Table("bridges")
outlierDet = Orange.data.outliers.OutlierDetection()
outlierDet.set_examples(bridges)
print outlierDet.z_values()
