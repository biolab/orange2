import Orange

bridges = Orange.data.Table("bridges")
outlierDet = Orange.data.outliers.OutlierDetection()
outlierDet.set_examples(bridges)
print ", ".join("%.8f" % val for val in outlierDet.z_values())
