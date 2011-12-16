import Orange

data = Orange.data.Table("bridges")
outlierDet = Orange.preprocess.outliers.OutlierDetection()
outlierDet.set_examples(data)
print outlierDet.z_values()
